"""tierowlab.py — 승부 헤드 학습 우주에 tie 행 투입 (α 가중) · 사전등록 프로브

현행: 승부 헤드는 `m = y_tr != 2` 로 **비-tie 행에서만** 학습한다 (destnation.win_head).
제안: tie 행을 y=0.5·가중치 α 로 함께 넣으면 "애매하면 평평하게" 를 더 배울 수 있는가.

기전 (중요) — tie 행에서 평평해지는 것 자체는 **채점되지 않는다**. tie 행의 3-class
손실은 −log p₀ 이고 z_win 이 등장하지 않기 때문이다(destnation.score 참조). 따라서 이
팔이 이길 수 있는 경로는 단 하나, **전이**다: tie 행이 "애매함의 모양" 을 가르치고 그것이
비-tie 애매 행으로 넘어가야 한다. 즉 신호 추가가 아니라 **데이터 증강·정규화** 가설이며,
승부 헤드의 학습 행이 ~32.3k → 46.7k (+45%) 로 늘어나는 것이 실제 레버다.

패리티 (확인함) — y=0.5 는 A↔B 교환 불변인 **유일한** 라벨이다. 교환 시 z→−z, y→1−y
이므로 대칭이려면 y=1−y. 실제로 L(z)=softplus(z)−0.5z 는 L(−z)=L(z) 이고 기울기
σ(z)−0.5 도 홀함수다. **홀 축 계약이 정확히 보존**되므로 새 가드가 필요 없다.

이웃 항목 (plan.md §5, 둘 다 전멸 — 이 프로브는 좌표가 다르다)
    combinelab  z_w 를 p_tie·tie 로짓 분위로 **사후 변조**        0/5 시드
    asmlab      재조립에서 교차축 항 (T+ / W+ / TW)               0/3 시드
    → 두 프로브 모두 "tie 정보로 승부 눈금을 건드린다" 였고, 여기는 **학습 우주** 다.

━━━━━━━━━━━━━━━━━━━━━━━━ 사전등록 ━━━━━━━━━━━━━━━━━━━━━━━━
팔        α ∈ {0.25, 0.5, 1.0}   (α=0 = 현행 채택 모델)
라벨      tie 행 y=0.5 · 가중치 α  |  비-tie 행 y∈{0,1} · 가중치 1
손실      sum(w·l)/sum(w)  — α 가 바뀌어도 눈금이 유지되어 AdamW 의 WD 대비
          기울기 비율이 안 흔들린다. α=0 경로는 무가중 평균(비트 재현용).
ES        **전체 행** (사용자 결정) · TR 15% · 학습과 같은 가중 목적
시드      42, 1, 2  (선별 집합 — 기준선이 기록돼 있어 페어 비교가 된다)
주 지표   holdout 5,748 행 **총 3-class ll** vs 현행 채택 모델
부 지표   holdout∩비-tie 3,972 행 win ll  (기전 확인용)
판정      paired 행부트스트랩 95% CI (10,000) · 문턱 0.0005 · 시드 부호 3/3
무료 가드 tie ll 이 REF 와 비트 일치해야 한다 (z_tie 재사용 → 어긋나면 파일 오적재)
이식 가드 --guard: 시드 42 를 α=0 경로로 돌려 REF['win'] 를 1e-9 로 재현하는지 확인.
          fit_head_w 가 destnation.fit_head 와 같은 RNG 경로를 걷는지 검증한다.

예측 (사전 기록) — 작거나 없음, null 쪽. 로지스틱 손실은 이미 비-tie 행만으로
불확실성을 가르치고(접전 행 라벨이 50/50 → 기울기가 이미 z→0), 추정 대상이
P(A|¬tie) 에서 수축 혼합으로 **이동**하므로 편향이 붙는다. 다만 "+45% 행" 은 앞선 두
프로브가 못 건드린 레버이고 대상(W4)이 신규 설계라 한 세트의 값어치가 있다.

실행:  cd final && python temp/tierowlab.py            (ML2 · GPU · 예상 25~35분)
       빠른 확인:  --alphas 1.0 --seeds 42             (약 4분)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import destnation as DN                                          # noqa: E402

NBOOT, THR = 10_000, 5e-4
BASE_NPZ = DN.FINAL / 'temp/destnation.rowloss.npz'


# ══════════════════════════════════════════ 층 1 — 가중 승부 헤드
def fit_head_w(CUt_, CVt_, y01, w_row, seed, CUs_, CVs_, net, hidden=DN.H):
    """승부 헤드 1개 (1R · 홀 축 sgn=−1). destnation.fit_head 의 가중 판.

    w_row=None 이면 무가중 — 그때 destnation.fit_head 와 **완전히 같은 경로**다
    (앵커 없음 분기까지 포함). 구성 순서·RNG 소비를 그대로 유지해야 α=0 재현이 선다.
    """
    torch.manual_seed(seed)
    d = CUt_.shape[1] + CVt_.shape[1]
    f = net(d, hidden)
    nn.init.zeros_(f.w.weight)
    nn.init.zeros_(f.w.bias)
    f = f.to(DN.DEVICE)
    a = nn.Parameter(torch.tensor(1.0, device=DN.DEVICE))        # 1R 이라 s0=0, 그래도 동일 구성
    opt = torch.optim.AdamW([{'params': f.parameters(), 'weight_decay': DN.WD},
                             {'params': [a], 'weight_decay': 0.0}], lr=DN.LR)
    s0z = np.zeros(len(y01))
    T = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device=DN.DEVICE)
    U_, V_, S_, Y_ = T(CUt_), T(CVt_), T(s0z), T(y01)
    out = lambda Ux, Vx: DN.sym_post(f, Ux, Vx, -1)               # 승부 = 홀

    if w_row is None:
        base = nn.BCEWithLogitsLoss()
        mk = lambda _idx: base                                    # 무가중 = 캠페인 경로
    else:
        raw = nn.BCEWithLogitsLoss(reduction='none')
        W_ = T(w_row)

        def mk(idx):
            wi = W_[idx]
            sw = wi.sum()
            assert float(sw) > 0, 'ES/학습 분할에 가중치 양수인 행이 없다'
            return lambda z, y: (raw(z, y) * wi).sum() / sw

    eidx, tidx = DN._es_split(len(y01), seed)
    Ue, Ve, Se, Ye = U_[eidx], V_[eidx], S_[eidx], Y_[eidx]
    Ut, Vt, St, Yt = U_[tidx], V_[tidx], S_[tidx], Y_[tidx]
    loss_t, loss_e = mk(tidx), mk(eidx)

    best, best_sd, bad, used = np.inf, None, 0, 0
    for ep in range(DN.EPOCHS):
        f.train()
        opt.zero_grad()
        loss_t(a * St + out(Ut, Vt), Yt).backward()
        opt.step()
        f.eval()
        with torch.no_grad():
            v = float(loss_e(a * Se + out(Ue, Ve), Ye))
        if v < best - 1e-6:
            best, bad, used = v, 0, ep + 1
            best_sd = {k2: t.detach().clone() for k2, t in f.state_dict().items()}
        elif (bad := bad + 1) >= DN.PATIENCE:
            break
    f.load_state_dict(best_sd)
    with torch.no_grad():
        r_s2 = out(T(CUs_), T(CVs_)).cpu().numpy().astype(np.float64)
    return r_s2, dict(es_ll=best, es_epoch=used)


def win_head_a(A, RAW, seed, alpha, full=False):
    """§3.8 승부 헤드 4소스 — α=0 이면 비-tie 행만(현행), α>0 이면 전체 행 + tie y=0.5·α.

    full=True 는 **대조군**: α=0 인데 행 우주는 전체다. tie 행 가중치가 0 이라 학습
    신호는 기준선과 같고, ES 분할만 α>0 팔과 같아진다. 두 효과를 가르는 유일한 팔이다.
      (ctrl − 기준선) = 순수 ES 분할 효과   |   (α − ctrl) = 순수 α 효과
    """
    nu, nv = A['Lu_tr'].shape[1], A['Lv_tr'].shape[1]
    y_tr = A['y_tr']
    tie = y_tr == 2
    if alpha == 0.0 and not full:
        keep = ~tie
        y01w = (y_tr[keep] == 0).astype(np.float64)
        w_row = None
    else:
        keep = np.ones(len(y_tr), bool)
        y01w = np.where(tie, 0.5, (y_tr == 0).astype(np.float64))
        w_row = np.where(tie, float(alpha), 1.0)
    cols, aux_all = [], []
    for si, s in enumerate(DN.SRCS):
        R = RAW[s]
        du, dv = R['u_tr'].shape[1], R['v_tr'].shape[1]
        sk = list(range(du, du + nu)) + list(range(du + nu + dv, du + nu + dv + nv))
        pool = list(range(0, du)) + list(range(du + nu, du + nu + dv))
        cu_t = np.ascontiguousarray(np.hstack([R['u_tr'][keep], A['Lu_tr'][keep]]), np.float32)
        cv_t = np.ascontiguousarray(np.hstack([R['v_tr'][keep], A['Lv_tr'][keep]]), np.float32)
        cu_s = np.ascontiguousarray(np.hstack([R['u_s2'], A['Lu_s2']]), np.float32)
        cv_s = np.ascontiguousarray(np.hstack([R['v_s2'], A['Lv_s2']]), np.float32)
        net = lambda d, h, _s=sk, _p=pool, _du=du: DN.WinNet(d, h, _s, _p, _du)
        r_s2, aux = fit_head_w(cu_t, cv_t, y01w, w_row, seed + si, cu_s, cv_s, net)
        cols.append(r_s2)
        aux_all.append(aux)
        print(f'      win {s:5s} es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]}')
        del cu_t, cv_t, cu_s, cv_s
    return np.column_stack(cols), aux_all


# ══════════════════════════════════════════ 판정
def paired_boot(d, nboot=NBOOT, seed=0):
    """행 부트스트랩 — d 는 행별 (팔 − 기준) 손실차, 시드 평균."""
    rng = np.random.default_rng(seed)
    n, acc = len(d), np.empty(nboot)
    for i in range(0, nboot, 1000):
        j = min(i + 1000, nboot)
        acc[i:j] = d[rng.integers(0, n, size=(j - i, n))].mean(1)
    return float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seeds', default='42,1,2')
    ap.add_argument('--alphas', default='0.25,0.5,1.0')
    ap.add_argument('--guard', action='store_true',
                    help='시드 42 를 α=0 으로 돌려 REF 비트 재현 확인 (약 +2분)')
    ap.add_argument('--ctrl', action='store_true',
                    help='대조군 α0·전체행 ES 를 함께 돌려 ES 분할 교체 효과를 분리 (시드당 +40초)')
    ap.add_argument('--out', default='temp/tierowlab')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s]
    alphas = [float(a) for a in args.alphas.split(',') if a]
    t_all = time.time()

    print('[적재]')
    U = DN.build()
    y, fold, cu, nz, hold = U['y'], U['fold'], U['cu'], U['nz'], U['hold']
    yA = (y == 0).astype(np.float64)
    mw = cu & nz
    Z = DN.load_svd_cache()
    A = DN.head_assets(U, Z)
    RAW = DN.raw_uv(Z)
    del Z

    assert BASE_NPZ.exists(), f'기준선 없음 — 먼저 destnation.py 를 돌려라 ({BASE_NPZ})'
    B = np.load(BASE_NPZ)
    for sd in seeds:
        assert f'z_tie_s{sd}' in B and f'row_s{sd}' in B, f'기준선에 시드 {sd} 없음'
    n_tie_tr = int((A['y_tr'] == 2).sum())
    print(f'  기준선 {BASE_NPZ.name} — 시드 {seeds}\n'
          f'  TR {len(A["y_tr"]):,}행 = 비-tie {len(A["y_tr"]) - n_tie_tr:,} + tie {n_tie_tr:,} '
          f'(α>0 이면 승부 헤드 학습 행 +{n_tie_tr / (len(A["y_tr"]) - n_tie_tr):.0%})')

    if args.guard:
        print(f'\n{"=" * 78}\n[이식 가드] 시드 42 · α=0 — REF 재현\n{"=" * 78}')
        W0, _ = win_head_a(A, RAW, 42, 0.0)
        zw0, _ = DN.win_combiner(np.hstack([U['win_base'], W0]), yA, fold, mw, 3)
        _, _, wl0 = DN.score(U, B['z_tie_s42'], zw0, hold)
        d0 = wl0 - DN.REF[42]['win']
        print(f'  win {wl0:.10f} vs REF {DN.REF[42]["win"]:.10f}  Δ {d0:+.2e}')
        assert abs(d0) < 1e-9, f'*** 이식 불일치 — fit_head_w 가 캠페인 경로와 다르다 (Δ {d0:.2e})'
        print('  → 비트 재현. fit_head_w 는 destnation.fit_head 와 같은 경로다.')

    # 팔 = (태그, α, full). ctrl 은 α=0·전체행 ES — ES 분할 교체 효과만 뽑는 대조군.
    arms = ([('ctrl', 0.0, True)] if args.ctrl else []) + \
           [(f'a{al:g}', al, False) for al in alphas]
    res, delta = {}, {}
    for tag, al, full in arms:
        drow = []
        for sd in seeds:
            print(f'\n{"=" * 78}\n[{tag} (α {al:g}) · 시드 {sd}]\n{"=" * 78}')
            t0 = time.time()
            W, _ = win_head_a(A, RAW, sd, al, full)
            z_win, dep = DN.win_combiner(np.hstack([U['win_base'], W]), yA, fold, mw, 3)
            z_tie = B[f'z_tie_s{sd}']
            row, tie_ll, win_ll = DN.score(U, z_tie, z_win, hold)
            assert abs(tie_ll - DN.REF[sd]['tie']) < 1e-9, \
                f'*** tie 축이 움직였다 ({tie_ll:.10f} vs {DN.REF[sd]["tie"]:.10f}) — 파일 오적재'
            base = B[f'row_s{sd}']
            assert base.shape == row.shape, '기준선 행 수 불일치'
            drow.append(row - base)
            res[f'{tag}_s{sd}'] = dict(
                arm=tag, alpha=al, full=full, seed=sd, tie=tie_ll, win=win_ll,
                tot=float(row.mean()),
                base_tot=float(base.mean()), base_win=DN.REF[sd]['win'],
                d_tot=float(row.mean() - base.mean()),
                d_win=float(win_ll - DN.REF[sd]['win']),
                win_lam=dep['lam'], sec=round(time.time() - t0, 1))
            c = res[f'{tag}_s{sd}']
            print(f'  tie {tie_ll:.5f}(고정) · win {win_ll:.5f} ({c["d_win"]:+.5f}) '
                  f'→ 총 {c["tot"]:.5f} ({c["d_tot"]:+.5f})  [{c["sec"]:.0f}s]')
        delta[tag] = np.mean(drow, 0)                              # 시드 평균 행 손실차

    op = DN.FINAL / args.out
    np.savez_compressed(op.with_suffix('.npz'), **{k: v for k, v in delta.items()})
    with open(op.with_suffix('.json'), 'w') as f:
        json.dump({'cells': res, 'seeds': seeds, 'alphas': alphas,
                   'arms': [a[0] for a in arms], 'ctrl': bool(args.ctrl),
                   'thr': THR, 'nboot': NBOOT}, f, ensure_ascii=False, indent=1)

    print(f'\n{"=" * 78}\n[판정] holdout 5,748 · 기준 = 현행 채택 모델 · '
          f'양수 = 악화\n{"=" * 78}')
    print(f'  {"팔":>6s}{"총 ll":>10s}{"Δ총":>10s}{"95% CI":>22s}{"win ll":>10s}'
          f'{"Δwin":>10s}   부호   판정')
    b0 = np.mean([res[f'{arms[0][0]}_s{sd}']['base_tot'] for sd in seeds])
    for tag, al, _f in arms:
        cs = [res[f'{tag}_s{sd}'] for sd in seeds]
        dt = np.mean([c['d_tot'] for c in cs])
        lo, hi = paired_boot(delta[tag])
        sign = sum(c['d_tot'] < 0 for c in cs)
        verd = 'ho' if (hi < -THR and sign == len(seeds)) else \
               ('ha' if lo > THR else 'noise')
        v = {'ho': '★ 채택 후보', 'ha': '기각 (유의 악화)', 'noise': '노이즈'}[verd]
        print(f'  {tag:>6s}{np.mean([c["tot"] for c in cs]):10.5f}{dt:+10.5f}'
              f'  [{lo:+.5f},{hi:+.5f}]{np.mean([c["win"] for c in cs]):10.5f}'
              f'{np.mean([c["d_win"] for c in cs]):+10.5f}   {sign}/{len(seeds)}   {v}')
    print(f'  {"기준":>6s}{b0:10.5f}   ← 현행 채택 모델 (destnation, 같은 시드)')
    print(f'\n  문턱 {THR} · CI 는 시드 평균 행손실차의 행 부트스트랩 {NBOOT:,}회.\n'
          f'  채택 요건: CI 상단 < −{THR} **그리고** 시드 부호 {len(seeds)}/{len(seeds)}.')

    # ── 교란 분해. ctrl 이 있어야만 α 순효과를 뽑을 수 있다.
    #    기준선은 비-tie 우주(n≈32.3k)로 ES 를 뽑고 α>0 팔은 전체(n=46.7k)로 뽑는다 —
    #    분할 자체가 다르므로 "기준선 대비" 에는 α 효과와 분할 교체 효과가 겹쳐 있다.
    if args.ctrl:
        dc = delta['ctrl']
        cc = [res[f'ctrl_s{sd}'] for sd in seeds]
        lo, hi = paired_boot(dc)
        print(f'\n{"=" * 78}\n[교란 분해] ctrl = α0·전체행 ES (학습 신호는 기준선과 동일)'
              f'\n{"=" * 78}')
        print(f'  ES 분할 교체 효과  (ctrl − 기준선)  {np.mean([c["d_tot"] for c in cc]):+.5f}'
              f'  [{lo:+.5f},{hi:+.5f}]  부호 {sum(c["d_tot"] < 0 for c in cc)}/{len(seeds)}')
        print(f'\n  {"α":>6s}{"α 순효과":>12s}{"95% CI":>22s}   부호   판정 (vs ctrl)')
        for tag, al, _f in arms[1:]:
            d = delta[tag] - dc                                    # 같은 행 → 페어드
            cs = [res[f'{tag}_s{sd}'] for sd in seeds]
            dd = [c['d_tot'] - c2['d_tot'] for c, c2 in zip(cs, cc)]
            lo2, hi2 = paired_boot(d)
            sg = sum(x < 0 for x in dd)
            v2 = '★ 채택 후보' if (hi2 < -THR and sg == len(seeds)) else \
                 ('기각 (유의 악화)' if lo2 > THR else '노이즈')
            print(f'  {al:>6g}{float(np.mean(dd)):+12.5f}  [{lo2:+.5f},{hi2:+.5f}]'
                  f'   {sg}/{len(seeds)}   {v2}')
        print('\n  ! 이 표가 α 족의 진짜 판정이다. 위 표의 "Δ총" 은 분할 교체가 섞여 있다.')
    print(f'\n저장: {op}.npz / {op}.json    총 {((time.time() - t_all) / 60):.1f}분')


if __name__ == '__main__':
    main()
