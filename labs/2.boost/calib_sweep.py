"""calib_sweep.py — 상위 조합 x 보정기 6계열 **전수조사** (2026-08-07).

무엇을 하나
  headgrid_final_v2 의 상위 --top 조합(축별 자유 조합) 각각에 대해, 제안된 보정기 전부를
  **결합기와 같은 5-fold 로 교차적합**해 무보정 대비 paired Δ 를 잰다. 게이트(calib_gate)의
  십분위표는 간접 판정이었고 이것이 직접 판정이다. 판정 문턱 = +0.0005 (계약).

보정기 (파라미터 수) — 전부 3-class 합성 후 파이프라인 맨 끝에 적용
  A 전역T(1)      : L' = s·L. 3-class 로짓 표준 온도. 결합기가 이미 흡수했을 1차원.
  B 축별T(2)      : 합성 전 z_w·s_w, z_t·s_t. 축 결합기는 자기 축 이진 손실만 최소화하고
                    3-class 합성 손실은 아니라는 틈을 재는 팔.
  C r블록(2)      : z' = z + (s-1)·z_r — λ 벌점을 받은 표현 열(기둥+셀) 기여만 스케일.
                    적합점 기울기 -2λ|w_r|²/n < 0 (릿지 수축)이라 이론 근거가 가장 명확.
                    로짓 열·절편 기여는 건드리지 않는다.
  V 벡터대칭(3)   : [s_ab·La, s_ab·Lb, s_t·Lt + b] — 스왑 등변 유지.
  M 매트릭스대칭(6): 스왑 등변 3x3 혼합 + tie 절편. 구계열 매트릭스(7par)의 대응물.
  F2 확신도(3)    : z' = z·(c + a·tanh(b·|z|)) 축별 적용 — "애매한 건 부드럽게, 확신은
                    날카롭게"의 매끄러운 2자유도 판. 십분위 10par 와 달리 미시험이었다.
  (클리핑은 적합하지 않는다 — ε 1e-5/1e-6 에 걸리는 행 수만 보고. 방어용 고정 상수.)

평가 위생
  stage2 7,748 은 결합기 적합에 이미 쓰였다. 보정기를 같은 데이터 전체로 고르면 낙관이
  낀다(꼬리 샤프닝 실측: 같은 데이터 +0.0001 vs 교차적합 -0.0027). 그래서 fold k 의
  보정 파라미터는 fold k 를 뺀 행에서 적합한다. **자기적합 Δ 도 같이 찍는다** — 이 값은
  낙관 상한이고, 교차적합 Δ 가 판정이다.

의존성 — 재학습 없음
  헤드 출력 열은 head_cols.npz(동결)에서 읽는다. 없으면 calib_gate.py 를 먼저 실행.
  결합기 z 는 RH.crossfit 재실행(결정론) + fold 별 선택 λ 로 예측 우주를 clean 전체로
  확장(승부축을 tie 행에도 예측 — 3-벡터 정의에 필요). r 블록 기여는 같은 적합의
  열 부분합으로 얻는다.

사용 (~10-15분, GPU 불필요)
  python Boost/calib_sweep.py --top 5 2>&1 | tee Boost/log_calib_sweep.txt
"""
import argparse
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

import fwls as FW
import reprhead as RH
import repr5 as R5
import headgrid as HG
import headgrid_final_v2 as V

sys.stdout.reconfigure(line_buffering=True)

COLS_NPZ = FW.BOOST / 'head_cols.npz'
THR = 0.0005


# ══════════════════════════════════════════════════════ 합성·손실
def compose(zw, zt):
    la, lb = zw / 2.0, -zw / 2.0
    return la, lb, zt + np.logaddexp(la, lb)


def rowll3(la, lb, lt, y3):
    M = np.column_stack([la, lb, lt])
    return logsumexp(M, axis=1) - M[np.arange(len(y3)), y3]


# ══════════════════════════════════════════════════════ 보정기
def cal_A(x, zw, zt, zwr, ztr):
    la, lb, lt = compose(zw, zt)
    return x[0] * la, x[0] * lb, x[0] * lt


def cal_B(x, zw, zt, zwr, ztr):
    return compose(x[0] * zw, x[1] * zt)


def cal_C(x, zw, zt, zwr, ztr):
    return compose(zw + (x[0] - 1.0) * zwr, zt + (x[1] - 1.0) * ztr)


def cal_V(x, zw, zt, zwr, ztr):
    la, lb, lt = compose(zw, zt)
    return x[0] * la, x[0] * lb, x[1] * lt + x[2]


def cal_M(x, zw, zt, zwr, ztr):
    la, lb, lt = compose(zw, zt)
    a, b, g, d, e, f = x
    return (a * la + b * lb + g * lt, b * la + a * lb + g * lt,
            d * (la + lb) + e * lt + f)


def cal_F(x, zw, zt, zwr, ztr):
    gw = x[0] + x[1] * np.tanh(x[2] * np.abs(zw))
    gt = x[0] + x[1] * np.tanh(x[2] * np.abs(zt))
    return compose(zw * gw, zt * gt)


CALS = [
    ('A 전역T', cal_A, np.array([1.0])),
    ('B 축별T', cal_B, np.array([1.0, 1.0])),
    ('C r블록', cal_C, np.array([1.0, 1.0])),
    ('V 벡터대칭', cal_V, np.array([1.0, 1.0, 0.0])),
    ('M 매트릭스대칭', cal_M, np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])),
    ('F2 확신도', cal_F, np.array([1.0, 0.0, 1.0])),
]


def fit_cal(fn, x0, D, mask):
    """mask 행의 평균 3-class NLL 을 최소화 (Nelder-Mead — 파라미터 <= 6)."""
    zw, zt, zwr, ztr, y3 = (D[k] for k in ('zw', 'zt', 'zwr', 'ztr', 'y3'))

    def obj(x):
        la, lb, lt = fn(x, zw[mask], zt[mask], zwr[mask], ztr[mask])
        v = rowll3(la, lb, lt, y3[mask])
        return float(v.mean()) if np.isfinite(v).all() else 1e6

    res = minimize(obj, x0, method='Nelder-Mead',
                   options=dict(xatol=1e-4, fatol=1e-7, maxiter=4000, maxfev=8000))
    return res.x


# ══════════════════════════════════════════════════════ 결합기 z (결정론 재실행)
def axis_scores(M, cols, pillar, pred_rows):
    """OOF z 와 r블록(기둥+셀) 기여. 예측 우주는 pred_rows (승부축도 tie 행 포함).
    fold k 의 재적합은 crossfit 이 고른 λ·같은 정규화라 rows 우주에선 z 가 정확히 같다."""
    parts = (pillar or []) + cols
    Lbase = M['base']
    Xi = np.column_stack([c for _, c in parts])
    X = np.hstack([Lbase, Xi])
    nl = Lbase.shape[1]
    bias = M['bias']
    n_even = X.shape[1] if bias else 0
    y01, fold, rows = M['y01'], M['fold'], M['rows']
    _z_ev, _zs, picked, n_bad, _ne, _c, _w = RH.crossfit(
        X, y01, fold, bias, FW.LAMS, rows=rows, n_even=n_even, n_logit=nl)
    if n_bad:
        print(f'    [주의] 결합기 미수렴 {n_bad}건')
    zp = np.full(len(y01), np.nan)
    zr = np.full(len(y01), np.nan)
    p = X.shape[1]
    for k, lam in enumerate(picked):
        tr = rows & (fold != k)
        te = pred_rows & (fold == k)
        A_, B_ = RH._z(X[tr], X[te], n_even)
        v, _ok = RH._fit(A_, y01[tr], RH._penvec(p, nl, lam), bias)
        zf, _ = RH._scores(v, B_, n_even, bias, False)
        zp[te] = zf
        zr[te] = B_[:, nl:] @ v[nl:p]
    return zp, zr


# ══════════════════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()

    A = R5.load_repr(single_best=True)
    y, nz, hold, cu, fold = A['y'], A['nz'], A['hold'], A['cu'], A['fold']

    assert COLS_NPZ.exists(), f'{COLS_NPZ} 없음 — 먼저 python Boost/calib_gate.py 실행'
    cache = np.load(COLS_NPZ)
    gz = np.load(HG.OOF_NPZ)
    z2 = np.load(V.OOF_NPZ)

    AX = {
        'tie': dict(base=A['T5'], bias=True, rows=cu, y01=(y == 2).astype(np.float64),
                    fold=fold),
        'win': dict(base=A['M3'], bias=False, rows=cu & nz,
                    y01=(y == 0).astype(np.float64), fold=fold),
    }
    PIL = {ax: [(f'p{i}', gz[f'pil__{ax}__{i}']) for i in range(4)] for ax in AX}

    # ── 조합 순위 (저장 ll 만으로) ──
    M1 = {ax: gz[f'll__{ax}__M1'] for ax in AX}
    have = {}
    for k in z2.files:
        pre, ax, nm = k.split('__')[0], k.split('__')[1], k.split('__')[2]
        have.setdefault(ax, {})[(nm, '얹기' if pre == 'll' else '단독')] = z2[k]
    clean = ~np.isnan(M1['tie'])

    def combo_ll(lt, lw, msk):
        mt, mw = msk & ~np.isnan(lt), msk & ~np.isnan(lw)
        return float(lt[mt].mean() + mw.sum() / mt.sum() * lw[mw].mean())

    rows_c = sorted(
        ((combo_ll(tv, wv, clean), combo_ll(tv, wv, clean & hold), (tn, tm), (wn, wm))
         for (tn, tm), tv in have['tie'].items()
         for (wn, wm), wv in have['win'].items()),
        key=lambda r: r[0])[:args.top]

    # ── 필요한 (축, 셀, 모드)의 z 를 한 번씩만 계산 ──
    zcache = {}
    for _c, _h, (tn, tm), (wn, wm) in rows_c:
        for ax, nm, mode in (('tie', tn, tm), ('win', wn, wm)):
            key = (ax, nm, mode)
            if key in zcache:
                continue
            ck = f'col__{ax}__{nm}'
            assert ck in cache.files, f'{ck} 동결 없음 — calib_gate.py 를 먼저 실행'
            Cm = cache[ck]
            cols = [(f'{nm}·c{i}', Cm[:, i]) for i in range(Cm.shape[1])]
            pil = PIL[ax] if (mode == '얹기' and not nm.startswith('LIN')) else None
            t_ = time.time()
            zcache[key] = axis_scores(AX[ax], cols, pil, cu)
            # 자기검증 — 저장 행별 ll 재현
            m = AX[ax]['rows']
            ll_now = RH.rowloss(zcache[key][0][m], AX[ax]['y01'][m])
            d_ = float(np.nanmean(np.abs(ll_now - have[ax][(nm, mode)][m])))
            print(f'  [z] {ax}·{nm}({mode})  {(time.time() - t_):.0f}s  '
                  f'재현차 {d_:.6f}' + ('  OK' if d_ < 5e-3 else '  *** 불일치'))
            assert d_ < 5e-3, f'{ax}·{nm} 재현 실패'

    # ── 조합별 전수조사 ──
    y3 = y.astype(int)
    summary = []
    for rank, (c_, h_, (tn, tm), (wn, wm)) in enumerate(rows_c, 1):
        RH.head(f'보정{rank}', f'#{rank}  tie {tn} ({tm})  x  win {wn} ({wm})')
        zt_full, ztr_full = zcache[('tie', tn, tm)]
        zw_full, zwr_full = zcache[('win', wn, wm)]
        D = dict(zw=zw_full[cu], zt=zt_full[cu], zwr=zwr_full[cu], ztr=ztr_full[cu],
                 y3=y3[cu])
        fold_c, hold_c = fold[cu], hold[cu]
        nC = int(cu.sum())

        la, lb, lt = compose(D['zw'], D['zt'])
        ll_u = rowll3(la, lb, lt, D['y3'])
        print(f'  무보정  clean {ll_u.mean():.4f} | holdout {ll_u[hold_c].mean():.4f}'
              f'   (표 대조 {c_:.4f} / {h_:.4f})')
        P = np.exp(np.column_stack([la, lb, lt])
                   - logsumexp(np.column_stack([la, lb, lt]), axis=1, keepdims=True))
        mn = P.min(1)
        print(f'  클리핑(적합 없음)  min p {mn.min():.6f} | <1e-5: {int((mn < 1e-5).sum())}행'
              f' | <1e-6: {int((mn < 1e-6).sum())}행  -> 걸리는 행 없으면 항등\n')

        rows_t = []
        best = None
        for nm_c, fn, x0 in CALS:
            # 자기적합 (낙관 상한) + 파라미터
            xf = fit_cal(fn, x0, D, np.ones(nC, bool))
            ls, lbs, lts = fn(xf, D['zw'], D['zt'], D['zwr'], D['ztr'])
            ll_self = rowll3(ls, lbs, lts, D['y3'])
            d_self = float(ll_u.mean() - ll_self.mean())
            # 교차적합 (판정)
            ll_cv = np.full(nC, np.nan)
            for k in range(RH.K):
                te = fold_c == k
                xk = fit_cal(fn, x0, D, ~te)
                lak, lbk, ltk = fn(xk, D['zw'][te], D['zt'][te], D['zwr'][te],
                                   D['ztr'][te])
                ll_cv[te] = rowll3(lak, lbk, ltk, D['y3'][te])
            d, lo, hi = RH.paired_ci(ll_u, ll_cv)
            dh, loh, hih = RH.paired_ci(ll_u[hold_c], ll_cv[hold_c])
            par = np.array2string(xf, precision=3, separator=',',
                                  suppress_small=True)
            rows_t.append([nm_c, f'{len(x0)}', f'{d_self:+.4f}',
                           RH.ci_str(d, lo, hi), RH.ci_str(dh, loh, hih), par])
            if best is None or d > best[1]:
                best = (nm_c, d, lo, hi)
        RH.table(['보정기', 'par', '자기적합Δ(낙관상한)', '교차적합Δ (판정)',
                  'holdout Δ', '전체적합 파라미터'],
                 rows_t, ['l', 'r', 'r', 'l', 'l', 'l'])
        mark = '문턱 통과' if best[1] >= THR and best[2] > 0 else \
            f'문턱(+{THR}) 미달'
        print(f'  -> 이 조합 최선: {best[0]}  Δ {best[1]:+.4f} — {mark}')
        summary.append((rank, f'tie {tn}({tm}) x win {wn}({wm})', best))

    RH.head('총괄', f'조합 {len(summary)}개 x 보정기 {len(CALS)}계열 — 최선 요약'
                    f' (판정 문턱 +{THR})')
    RH.table(['#', '조합', '최선 보정기', '교차적합Δ', 'CI'],
             [[str(r), nm, b[0], f'{b[1]:+.4f}', f'[{b[2]:+.4f},{b[3]:+.4f}]']
              for r, nm, b in summary], ['r', 'l', 'l', 'r', 'l'])
    print(f'\n총 {(time.time() - t0) / 60:.1f}분')
    print('판정(보정기 채택 여부·최종 조합 확정)은 사용자.')


if __name__ == '__main__':
    main()
