"""whitelab.py — tie축 헤드 입력의 **스케일 프로파일**을 연다. 한 번도 안 건드린 축.

현행 계약(§3.6 [4])
    통합 SVD-512 → 성분별 TR-RMS 로 나눔(화이트닝, 좌표별 RMS 1)
    근거는 코드에 없다. **대조군이 한 번도 없었다** — 물려받은 설계 선택이다.

왜 지금 여는가 (진단 실측, `log_whitelab_diag` 참조)
  ① 화이트닝은 균일 weight decay 하에서 **성분별 릿지의 한 설정**이다. 입력 열을
     s_j^(1−α) 로 두면 (α=1 현행) 출력 기여에 걸리는 유효 벌점이 pen_j ∝ s_j^(−2(1−α)).
     α=1 은 그 족에서 **전 방향 무차등**을 고른 것 — 정교한 선택이 아니라 중립 선택이다.
  ② 방향당 SNR 은 분산 서열을 **강하게** 따른다. 짝: 상위16 SNR 11.9 → 꼬리 1.2
     (잡음바닥의 1.2배 = 거의 잡음). log r² ~ log s² 기울기 γ = +0.613(짝)/+0.742(홀).
     선형-릿지 최적을 그대로 옮기면 **α* ≈ 0.39(짝) / 0.26(홀)** — 현행 1 과 멀다.
  ③ 소스별 방향 예산이 신호와 어긋나 있다. 짝 기준 tb1 은 512 방향 중 **11.5개**만
     받는데 신호의 20.5% 를 담는다(신호/방향 9.11). tb2 는 245방향에 0.75.
     통합 SVD 가 "유효 랭크가 큰 소스"에 예산을 몰아준다 — 신뢰도와 무관한 기준이다.
  ! ②③ 은 단변량 선형 상관이다. 헤드는 곱 유닛(TANHTANH·BILIN)이라 상호작용을 쓰고,
    그건 이 진단이 못 본다. 방향 근거이지 크기 근거가 아니다.

팔 (전부 §3.6 확정 헤드 · h128 · 2R · 같은 로짓 직결 · 같은 결합기 · 입력 폭 1024 고정)
  [통합 기저 — 캐시만, 재계산 없음]
    J-A1     현행 화이트닝 (α=1)                              ← 기준
    J-A50    부분 화이트닝 α=0.5
    J-A25    α=0.25   (짝 α* 0.39 · 홀 0.26 을 사이에 둔다)
    J-A0     화이트닝 없음 (원 분산 서열 유지)
    J-A125 · J-A150 · J-A200   **과화이트닝** α>1 — 꼭짓점 확인용 반대편
  [소스별 기저 — raw 4소스 재적재 + 소스별 SVD-128 (합 512, 폭 동일)]
    P-WHITE  소스별 SVD-128 + 화이트닝     '섞임 제거'만 분리 (예산 균등 128×4)
    P-RAW    소스별 SVD-128, 화이트닝 없음
    P-SHRINK 소스별 릿지 e/(e+λ_s) — **λ_s 는 소스마다 TR 3-fold CV 로 따로 고른다**
             (= 전문가별 가변 릿지. 라벨은 TR tie 만 본다, stage2 무접촉)
    P-ARD    좌표별 적응 릿지 pen_j ← λ(1/(w²+ε))/평균, 5회 (winlab t_ard 규칙)

읽는 법: J-A* 는 **화이트닝 단독** 효과(기저 불변). P-* 는 기저까지 바꾼다 —
J-A1 vs P-WHITE 가 '섞임/예산', P-WHITE vs P-RAW 가 그 위의 '화이트닝'이다.

비용: 캐시 팔만이면 ~7분. 소스별 팔 포함이면 raw 적재(~2분)+SVD 8회(~4분)+ARD(옵션,
      느림) 로 ~25분 (ARD 포함 ~40분).
"""
import argparse
import gc
import json
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.extmath import randomized_svd

import tie_ladder as TL
import penlab as PL
import readout as RO
import winlab as WL

FINAL = TL.FINAL
SRCS = WL.SRCS
K_SRC = 128                                  # 소스당 방향 (×4 = 512, 통합과 폭 동일)
LAM_S = [10.0, 31.6228, 100.0, 316.228, 1000.0, 3162.28]   # λ_s 후보 (TR CV)
ARMS_J = ('J-A1', 'J-A50', 'J-A25', 'J-A0', 'J-A125', 'J-A150', 'J-A200')
ARMS_P = ('P-WHITE', 'P-RAW', 'P-SHRINK', 'P-ARD')
ALPHA = {'J-A1': 1.0, 'J-A50': 0.5, 'J-A25': 0.25, 'J-A0': 0.0,
         # α>1 = **과화이트닝** — 저분산 방향을 화이트닝보다 더 키운다. α<1 이 단조로
         # 졌으므로(2026-08-12) α=1 이 꼭짓점인지 단조가 계속되는지를 이쪽에서 닫는다.
         # K·h 가 양방향 확인으로 닫힌 것과 같은 규약.
         'J-A125': 1.25, 'J-A150': 1.5, 'J-A200': 2.0}


# ── 헤드 (§3.6 TB 체인) — 입력 블록만 갈아끼운다 ────────────────────────────
def tb_cols(A, CUt, CVt, CUs, CVs, seed):
    """penlab.tb_cols 와 동일하되 몸통 입력 4개를 인자로 받는다.

    열 배치는 그대로 [CU | Lu][CV | Lv] → 판독 skip = Lu·Lv, 몸통 = CU·CV.
    폭이 512 로 고정이므로 파라미터 수(262,537×2)도 현행과 같다.
    """
    K = CUt.shape[1]
    nu, nv = A['Lu_tr'].shape[1], A['Lv_tr'].shape[1]
    assert K == CVt.shape[1] == CUs.shape[1] == CVs.shape[1], '네 블록의 폭이 같아야 한다'
    sk = list(range(K, K + nu)) + list(range(2 * K + nu, 2 * K + nu + nv))
    pool = list(range(0, K)) + list(range(K + nu, 2 * K + nu))
    ct = np.ascontiguousarray(np.hstack([CUt, A['Lu_tr']]), np.float32)
    cv = np.ascontiguousarray(np.hstack([CVt, A['Lv_tr']]), np.float32)
    cs = np.ascontiguousarray(np.hstack([CUs, A['Lu_s2']]), np.float32)
    vs = np.ascontiguousarray(np.hstack([CVs, A['Lv_s2']]), np.float32)
    z, cols = None, []
    for unit in ('TANHTANH', 'BILIN'):
        net = RO.make_factory(unit, 'LIN', sk, pool)
        r, aux = TL.fit_gat(ct, cv, A['y01m'], z, True, seed, cs, vs,
                            net=net, sym_fn=RO.sym_post)
        cols.append(r)
        z = aux['z_tr']
    return np.column_stack(cols)


# ── 통합 기저 팔 — 캐시의 U_sc/V_sc 로 화이트닝을 되감는다 ───────────────────
def joint_alpha(Z, alpha):
    """CUt 는 이미 Ct/sc 다. Ct/sc^α = CUt · sc^(1−α) — SVD 재계산 없음.

    α=1 이면 곱이 1 이라 **캐시와 비트 단위로 같다**(항등 검증 대상).
    """
    out = []
    for c, k in (('CUt', 'U_sc'), ('CVt', 'V_sc'), ('CUs', 'U_sc'), ('CVs', 'V_sc')):
        g = (Z[k] ** (1.0 - alpha)).astype(np.float32)
        out.append(np.ascontiguousarray(Z[c].astype(np.float32) * g))
    return out                                   # CUt, CVt, CUs, CVs


# ── 소스별 기저 팔 ──────────────────────────────────────────────────────────
def src_svd(RAW, k=K_SRC):
    """소스별 SVD-k. 성분과 **성분 RMS(e)** 를 같이 돌려준다 (화이트닝은 나중에 선택).

    짝(u)은 TR 중심화 · 홀(v)은 무중심 — §3.6 [4] 계약과 같다.
    """
    out = {}
    for s, R in RAW.items():
        t0, d = time.time(), {}
        for p, center in (('u', True), ('v', False)):
            tr, s2 = R[f'{p}_tr'], R[f'{p}_s2']
            mu = tr.mean(0) if center else np.zeros(tr.shape[1], np.float32)
            _, _, Vt = randomized_svd((tr - mu).astype(np.float64), n_components=k,
                                      random_state=TL.SEED)
            W = Vt.T.astype(np.float32)
            ct, cs = (tr - mu) @ W, (s2 - mu) @ W
            e = np.clip(np.sqrt((ct.astype(np.float64) ** 2).mean(0)), 1e-9,
                        None).astype(np.float32)
            d[f'{p}_tr'], d[f'{p}_s2'], d[f'{p}_e'] = ct, cs, e
        out[s] = d
        print(f'    [psvd·{s}] SVD-{k} ×(u,v) [{time.time() - t0:.0f}s]')
    return out


def pick_lam(P, y01, p, s):
    """소스 s·패리티 p 의 λ_s 를 **TR 3-fold CV** 로 고른다 (stage2 무접촉).

    성분을 화이트닝한 판 위에서 릿지를 걸어 CV 손실이 최소인 λ 를 고른다 —
    수축 프로파일 e/(e+λ) 의 λ 와 같은 눈금이 되도록 릿지도 성분 e 척도에서 건다.
    """
    X = P[s][f'{p}_tr'].astype(np.float64)
    n = len(y01)
    fold = np.zeros(n, int)
    for j, (_, iv) in enumerate(StratifiedKFold(3, shuffle=True, random_state=TL.SEED)
                                .split(np.zeros((n, 1)), y01)):
        fold[iv] = j
    best, warm = None, [None] * 3
    for lam in sorted(LAM_S, reverse=True):
        tot = 0.0
        for j in range(3):
            tr, te = fold != j, fold == j
            v, _ = TL._fit(X[tr], y01[tr], np.full(X.shape[1], lam), True, warm[j])
            warm[j] = v
            z = X[te] @ v[:X.shape[1]] + v[X.shape[1]]
            tot += float(TL.rowloss(z, y01[te]).sum())
        if best is None or tot < best[1]:
            best = (lam, tot)
    return best[0]


def build_p(P, arm, A, y01):
    """소스별 성분에 스케일 프로파일을 입혀 [4소스 × 128] 을 잇는다."""
    blocks = {'u_tr': [], 'v_tr': [], 'u_s2': [], 'v_s2': []}
    diag = {}
    for s in SRCS:
        for p in ('u', 'v'):
            e = P[s][f'{p}_e'].astype(np.float64)
            if arm == 'P-WHITE':
                g = 1.0 / e
            elif arm == 'P-RAW':
                g = np.ones_like(e)
            elif arm == 'P-SHRINK':
                lam = pick_lam(P, y01, p, s)
                g = (e ** 2) / (e ** 2 + lam) / e         # 수축 후 화이트닝 눈금 유지
                diag[f'{s}·{p}'] = f'λ {lam:g} · 수축중앙 {np.median((e**2)/(e**2+lam)):.3f}'
            elif arm == 'P-ARD':
                g = ard_gain(P[s][f'{p}_tr'], y01, e)
                diag[f'{s}·{p}'] = f'ard 이득 범위 {g.min()/g.max():.3f}~1'
            else:
                raise ValueError(arm)
            g = g.astype(np.float32)
            blocks[f'{p}_tr'].append(P[s][f'{p}_tr'] * g)
            blocks[f'{p}_s2'].append(P[s][f'{p}_s2'] * g)
    out = [np.ascontiguousarray(np.hstack(blocks[k]), np.float32)
           for k in ('u_tr', 'v_tr', 'u_s2', 'v_s2')]
    return out, diag


def ard_gain(Xtr, y01, e, iters=5, eps=1e-8, lam=100.0):
    """좌표별 적응 릿지로 관련도 |w_j| 를 배워 그 크기로 성분을 재척도 (run_ard 규칙).

    화이트닝한 판에서 적합한다 — 그래야 w_j 가 순수 관련도이지 분산 대리가 아니다.
    재정규화하지 않는다(하면 재척도가 상쇄된다).
    """
    X = (Xtr / e).astype(np.float64)
    pen, v = np.full(X.shape[1], lam), None
    for _ in range(iters):
        v, _ = TL._fit(X, y01, pen, True, v)
        w = v[:X.shape[1]]
        inv = 1.0 / (w ** 2 + eps)
        pen = lam * inv / inv.mean()
    w = np.abs(v[:X.shape[1]])
    return (w / max(w.max(), 1e-12)) / e            # 관련도 이득 × 화이트닝


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bundle', default='temp/heads_repro')
    ap.add_argument('--arms', default=','.join(ARMS_J + ARMS_P[:3]),
                    help=f'{ARMS_J + ARMS_P} · P-ARD 는 느리다')
    ap.add_argument('--seeds', default='42,1,2')
    ap.add_argument('--out', default='temp/whitelab')
    args = ap.parse_args()
    arms = [a for a in args.arms.split(',') if a]
    seeds = [int(s) for s in args.seeds.split(',') if s]
    assert set(arms) <= set(ARMS_J + ARMS_P), f'모르는 팔 {set(arms) - set(ARMS_J + ARMS_P)}'
    assert 'J-A1' in arms, 'J-A1 이 기준이라 반드시 있어야 한다'

    print('[적재]')
    U = TL.build(FINAL / args.bundle)
    y_t = (U['y'] == 2).astype(np.float64)
    hold, cu = U['hold'], U['cu']
    A = TL.gat_assets(U)
    Z = TL.load_svd_cache()

    INP, diags = {}, {}
    for a in arms:
        if a in ARMS_J:
            INP[a] = joint_alpha(Z, ALPHA[a])
    # 항등 검증 — α=1 은 캐시와 비트 동일해야 한다
    d0 = np.abs(INP['J-A1'][0] - Z['CUt'].astype(np.float32)).max()
    assert d0 == 0.0, f'J-A1 이 캐시와 다르다 max|Δ| {d0:.3e}'
    print(f'  통합 기저 팔 {len([a for a in arms if a in ARMS_J])}개 · J-A1 항등 검증 통과')

    if any(a in ARMS_P for a in arms):
        t0 = time.time()
        RAW = WL.raw_uv(U)
        P = src_svd(RAW)
        del RAW
        gc.collect()
        print(f'  소스별 기저 준비 [{time.time() - t0:.0f}s]')
        for a in arms:
            if a in ARMS_P:
                INP[a], diags[a] = build_p(P, a, A, A['y01m'])
        del P
        gc.collect()

    res, rows_ = {}, {}
    for a in arms:
        CUt, CVt, CUs, CVs = INP[a]
        for sd in seeds:
            t0 = time.time()
            G = tb_cols(A, CUt, CVt, CUs, CVs, sd)
            X = np.hstack([U['tie_base'], G])
            z, dep = PL.crossfit_struct(X, y_t, U['fold'], cu, True, 5, 'base')
            ll = TL.rowloss(z[cu], y_t[cu])
            h = hold[cu]
            tag = f'{a}-s{sd}'
            res[tag] = dict(arm=a, seed=sd, ll=float(ll[h].mean()),
                            picked=dep['picked'], v=dep['v'], sec=round(time.time() - t0, 1))
            rows_[tag] = ll[h]
            print(f'  [{tag}] tie ll(hold) {ll[h].mean():.5f}  ({time.time() - t0:.0f}s)')
        del INP[a]
        gc.collect()

    print(f'\n{"=" * 78}\n[대조] J-A1(현행 화이트닝) 기준 · 같은 헤드·결합기 · paired\n{"=" * 78}')
    m0 = np.mean([rows_[f'J-A1-s{s}'] for s in seeds], axis=0)
    b0 = np.array([res[f'J-A1-s{s}']['ll'] for s in seeds])
    print(f'  {"팔":10s}{"ll(hold)":>11s}   J-A1 대비 (시드평균 paired)      시드부호')
    for a in arms:
        v = np.array([res[f'{a}-s{s}']['ll'] for s in seeds])
        if a == 'J-A1':
            print(f'  {a:10s}{v.mean():11.5f}   기준')
            continue
        cur = np.mean([rows_[f'{a}-s{s}'] for s in seeds], axis=0)
        m, lo, hi = TL.paired_ci(cur, m0)
        print(f'  {a:10s}{v.mean():11.5f}   {TL.ci_str(m, lo, hi)}  '
              f'{int(((v - b0) < 0).sum())}/{len(v)}')

    if diags:
        print(f'\n{"=" * 78}\n[진단] 소스별 스케일 선택\n{"=" * 78}')
        for a, d in diags.items():
            print(f'  {a}')
            for k, s in d.items():
                print(f'    {k:12s} {s}')

    print('\n  ! J-A* 는 기저 불변·스케일만 — 화이트닝 단독 효과.'
          '\n    P-* 는 기저까지 바뀐다: J-A1 vs P-WHITE = 섞임/예산, P-WHITE vs P-RAW = 화이트닝.')

    op = FINAL / args.out
    np.savez_compressed(op.with_suffix('.rowloss.npz'), **rows_)
    with open(op.with_suffix('.json'), 'w') as f:
        json.dump({'cells': res}, f, ensure_ascii=False, indent=1)
    print(f'\n저장: {op}.rowloss.npz / {op}.json')


if __name__ == '__main__':
    main()
