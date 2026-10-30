"""nested logit 상위항 — 무승부 확률을 승부축 확신도로 변조한다 (결합기만, 헤드 재학습 없음).

가설: 승부가 결정적일수록 무승부가 덜 나와야 한다. 지금 조립부는 그 결합을 **구조적으로
금지**한다 — P(tie) = σ(z_tie) 가 z_win 과 정확히 무관하다.

nested logit 의 λ 를 우리 좌표로 풀면 두 조각이고, 앞 조각은 이미 닫혀 있다.

    ① 하위      q = σ(z_win / λ)                        = 승부축 전역 온도
                → combinelab TEMP 팔에서 기각 (+0.00002 [−0.00004,+0.00009] 0/5)
    ② 상위      ℓ_tie = z_tie + λ·log(2cosh(z_win/2λ))  = 이 실험

②만 남겨 우리 좌표로 정리하면 놀랍도록 단순해진다. ℓ_tie 에 (1+c)·L 을 얹으면
softmax 분모에서 e^L 이 인수분해돼 **나머지가 전부 상쇄**되고, 남는 것은

    z_tie' = z_tie + c·L,      L = log(2cosh(z_win/2)) = logaddexp(z_win/2, −z_win/2)

뿐이다 (c=0 이 현행 λ=1). q 도 P(A)/P(B) 의 비율도 손대지 않는다.
→ **승부축 ll 은 비트 단위로 불변이고, 판정은 tie축 ll 하나로 닫힌다** (Δ총점 = Δtie ll).

파리티: cosh 가 짝함수라 L 은 스왑 불변(짝)이다. 짝 축인 z_tie 에 **더할 수** 있다.
        (승부축이었다면 곱해야 했다 — combinelab 의 z_win × exp(c·t̂) 참조)

L 의 뜻: z_win=0 에서 log2, |z_win| 이 크면 |z_win|/2 로 수렴하는 **부드러운 |z_win|/2**.
         즉 "승부축이 얼마나 결정적인가"의 짝 통계다.

팔 (전부 c=0 에서 BASE 로 정확히 환원)
    SHIFT   z_tie + c₀              절편만 — **대조군**. tie 결합기가 이미 절편을 적합했으니
                                    0 이어야 정상이다. 아니면 하네스를 의심한다
    NEST1   z_tie + c₀ + c₁·L̃       ★ 본 가설
    NEST2   z_tie + c₀ + c₁·L̃ + c₂·L̃²   곡률 허용

c 의 적합 우주 (`--fitmode`) — combinelab 의 교훈 그대로
    cv     (기본) 바깥 폴드마다 학습행 안에서 다시 K-fold 를 돌려 **중첩 OOF** (z_tie, L) 을
           만들고 그 위에서 c 를 적합, 테스트행에 적용. 판정은 여전히 OOF
    split  비-holdout clean 2,000행에서 한 번 적합 (참고용 — 1차 실패 방식)

승부축 z 는 **비-tie 행에서 적합하되 전 clean 행에 예측**한다. tie 행에도 L 이 필요하고,
배포 경로가 원래 그렇다(테스트에서 어느 행이 tie 인지 모른다). 겹치는 행에서 표준 경로와
비트 동일함을 assert 로 확인한다.

진단표: L 5분위별 (실제 tie율 · 예측 tie율 · 잔차 · 그 분위를 보정하는 절편 c₀*).
        **적합 이전에 가설의 생사를 보여준다** — holdout 쪽 잔차가 분위와 함께 안 움직이면
        그 결합은 원리적으로 죽었다.
"""
import argparse
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

import tie_ladder as TL

FINAL = TL.FINAL
ARMS_ALL = ['SHIFT', 'NEST1', 'NEST2']
DEG = {'SHIFT': 0, 'NEST1': 1, 'NEST2': 2}


# ══════════════════════════════════════════════ 결합기 (예측 우주 확장판)
def cross_pred(X, y01, fold, fit_mask, pred_mask, bias, n_logit):
    """폴드별 적합(fit_mask) → 예측(pred_mask). λ 는 표준 경로가 폴드별로 고른 값 그대로.

    fit ⊊ pred 인 경우(승부축: 비-tie 에서 적합, 전 clean 에 예측)를 위해 분리했다.
    겹치는 행에서 TL.crossfit 과 일치함을 확인한다.
    """
    n_even = X.shape[1] if bias else 0
    lams = TL.LAMS if n_logit < X.shape[1] else [TL.LAMS[0]]
    z_std, picked, n_bad, _ = TL.crossfit(X, y01, fold, bias, lams, rows=fit_mask,
                                          n_even=n_even, n_logit=n_logit)
    z = np.full(len(y01), np.nan)
    for k in range(TL.K):
        tr, te = (fold != k) & fit_mask, (fold == k) & pred_mask
        if not te.any():
            continue
        A, B = TL._z(X[tr], X[te], n_even)
        v, ok = TL._fit(A, y01[tr], TL._penvec(A.shape[1], n_logit, picked[k]), bias)
        n_bad += (not ok)
        z[te] = TL._scores(v, B, n_even, bias)
    ov = np.isfinite(z_std) & np.isfinite(z)
    assert ov.any(), '표준 경로와 겹치는 행이 없다'
    d = float(np.abs(z[ov] - z_std[ov]).max())
    assert d < 1e-9, f'표준 경로 재현 실패 max|Δ| {d:.3e}'
    return z, [float(p) for p in picked], int(n_bad)


def inclusive(zw):
    """L = log(2·cosh(z_win/2)) — 승부축 결정성의 짝 통계. 스왑 불변.

    zw 는 예측 우주 밖에서 NaN 이다. 그 자리는 NaN 으로 남기되 경고는 내지 않는다.
    """
    out = np.full(len(zw), np.nan)
    m = np.isfinite(zw)
    out[m] = np.logaddexp(zw[m] / 2.0, -zw[m] / 2.0)
    return out


# ══════════════════════════════════════════════ c 적합
def fit_shift(z, that, y01, mask, deg):
    """z' = z + c₀ + Σ_d c_d·t̂^d 의 c 를 mask 행에서 적합 (오프셋 z 는 고정)."""
    zm, ym = z[mask], y01[mask]
    P = np.column_stack([that[mask] ** (d + 1) for d in range(deg)]) if deg else None
    # NaN 이 하나라도 섞이면 L-BFGS 가 초기점에서 멈춰 c=0 을 조용히 돌려준다.
    assert np.isfinite(zm).all(), 'c 적합 입력 z 에 비유한'
    assert P is None or np.isfinite(P).all(), 'c 적합 입력 t̂ 에 비유한'

    def fg(c):
        s = zm + c[0] + (P @ c[1:] if deg else 0.0)
        val = float(np.mean(np.logaddexp(0, s) - ym * s))
        r = (expit(s) - ym) / len(ym)
        grad = [r.sum()] + [float(r @ P[:, d]) for d in range(deg)]
        return val, np.asarray(grad)

    res = minimize(fg, np.zeros(deg + 1), jac=True, method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-9})
    return res.x, bool(res.success)


def apply_shift(z, that, c):
    return z + c[0] + sum(c[d + 1] * that ** (d + 1) for d in range(len(c) - 1))


def mod_stats(raw, fit_mask):
    """표준화 상수. 라벨을 안 쓰고 적합 우주의 L 값 분포에서만 뽑는다."""
    return (float(raw[fit_mask].mean()),
            float(np.clip(raw[fit_mask].std(), 1e-9, None)))


def modulator(raw, mu, sd):
    """중첩 L 과 바깥 L 에 **같은 상수**를 쓴다 — 안 그러면 c 의 축척이 안 옮겨간다."""
    return (raw - mu) / sd


# ══════════════════════════════════════════════ 중첩 재료
def nested_axes(U, W):
    """바깥 폴드별 — 학습행의 중첩 OOF (z_tie, z_win) 와 테스트행의 바깥 OOF.

    L 이 승부축에서 나오므로 두 축 모두 중첩해야 c 가 fold k 의 라벨을 안 본다.
    """
    Xt = np.hstack([U['tie_base'], U['GAT']])
    Xw = np.hstack([U['win_base'], W])
    nlt, nlw = U['tie_base'].shape[1], U['win_base'].shape[1]
    yt = (U['y'] == 2).astype(np.float64)
    yw = (U['y'] == 0).astype(np.float64)
    cu, nz = U['cu'], U['cu'] & U['nz']
    parts = []
    for k in range(TL.K):
        tr, te = (U['fold'] != k) & cu, (U['fold'] == k) & cu
        idx = np.flatnonzero(tr)
        sub = np.full(U['n'], -1, int)
        skf = StratifiedKFold(TL.K, shuffle=True, random_state=TL.SEED + 200 + k)
        for j, (_, iv) in enumerate(skf.split(np.zeros((len(idx), 1)), yt[idx])):
            sub[idx[iv]] = j
        zt_in, _, _ = cross_pred(Xt, yt, sub, tr, tr, True, nlt)
        zw_in, _, _ = cross_pred(Xw, yw, sub, tr & nz, tr, False, nlw)
        assert np.isfinite(zt_in[tr]).all() and np.isfinite(zw_in[tr]).all(), \
            f'fold {k} 중첩 OOF 비유한'
        assert not (tr & te).any(), '폴드 누수'
        parts.append(dict(tr=tr, te=te, zt_in=zt_in, zw_in=zw_in))
    return parts


# ══════════════════════════════════════════════ 진단
def shift_table(z, y01, var, mask, label, nq=5):
    """변조자 5분위별 (실제 tie율 · 예측 tie율 · 잔차 · 보정 절편 c₀*)."""
    zz, yy, vv = z[mask], y01[mask], var[mask]
    edges = np.quantile(vv, np.linspace(0, 1, nq + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    print(f'    [{label}]  n={int(mask.sum()):,}')
    print(f'      {"분위":>4}{"행":>7}{"평균 L":>9}{"실제 tie":>10}{"예측 tie":>10}'
          f'{"잔차":>9}{"c0*":>9}')
    out = []
    for i in range(nq):
        b = (vv >= edges[i]) & (vv < edges[i + 1])
        if b.sum() < 30:
            continue
        one = np.ones(int(b.sum()), bool)
        c, ok = fit_shift(zz[b], np.zeros(int(b.sum())), yy[b], one, 0)
        act, pred = float(yy[b].mean()), float(expit(zz[b]).mean())
        print(f'      {i + 1:>4}{int(b.sum()):7d}{vv[b].mean():9.3f}{act:10.4f}'
              f'{pred:10.4f}{act - pred:+9.4f}{c[0]:+9.4f}'
              + ('' if ok else '  ** 미수렴'))
        out.append(dict(q=i + 1, n=int(b.sum()), L=float(vv[b].mean()), act=act,
                        pred=pred, resid=act - pred, c0=float(c[0])))
    return out


# ══════════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bundle', default='temp/heads_repro')
    ap.add_argument('--cols', default='temp/winconf.rowloss.npz')
    ap.add_argument('--tag', default='SK-RAW')
    ap.add_argument('--seeds', default='5,6,7,8,9')
    ap.add_argument('--rounds', type=int, default=1, help='쓸 헤드 라운드 수 (확정 1R)')
    ap.add_argument('--fitmode', default='cv', choices=['cv', 'split'])
    ap.add_argument('--arms', default=','.join(ARMS_ALL))
    ap.add_argument('--out', default='temp/nestlab')
    args = ap.parse_args()

    arms = [a for a in args.arms.split(',') if a]
    seeds = [int(s) for s in args.seeds.split(',') if s]
    assert set(arms) <= set(ARMS_ALL), f'모르는 팔 {set(arms) - set(ARMS_ALL)}'

    print(f'[적재]  fitmode={args.fitmode}')
    U = TL.build(FINAL / args.bundle)
    cu, nz = U['cu'], U['cu'] & U['nz']
    yt = (U['y'] == 2).astype(np.float64)
    yw = (U['y'] == 0).astype(np.float64)
    fit_m, ev_m = cu & ~U['hold'], cu & U['hold']
    print(f'  행: tie 결합기 {int(cu.sum()):,} · 승부 적합 {int(nz.sum()):,} '
          f'· c 적합(split) {int(fit_m.sum()):,} · 판정 {int(ev_m.sum()):,}')

    # tie 축 — 저장 GAT 고정, 학습 없음
    Xt = np.hstack([U['tie_base'], U['GAT']])
    zt0, dep_t = TL.combiner(Xt, yt, U['fold'], cu, bias=True,
                             n_logit=U['tie_base'].shape[1])
    ll0 = TL.rowloss(zt0[ev_m], yt[ev_m])
    print(f'  tie 고정 (저장 GAT · 학습 없음) — λ {dep_t["lam"]:g} '
          f'· tie(hold) {ll0.mean():.5f}')

    Z = np.load(FINAL / args.cols)
    nc = args.rounds * 4
    res, rowl, diag, cinfo = {}, {}, {}, {}

    def score(tag, z, note=''):
        ll = TL.rowloss(z[ev_m], yt[ev_m])
        assert np.isfinite(ll).all(), f'{tag} 비유한 손실'
        res[tag] = dict(tie=float(ll.mean()))
        rowl[tag] = ll
        print(f'    [{tag:16s}] tie(hold) {ll.mean():.5f}{note}')

    score('BASE', zt0, '  ← c=0 (현행 λ=1)')

    for si, sd in enumerate(seeds):
        key = f'cols__{args.tag}-s{sd}'
        assert key in Z.files, f'{key} 없음 — 저장 열 {Z.files[:4]}…'
        W = Z[key][:, :nc].astype(np.float64)
        print(f'\n{"-" * 78}\n[시드 {sd}] 헤드 열 {W.shape[1]}개 ({args.rounds}R)\n{"-" * 78}')

        Xw = np.hstack([U['win_base'], W])
        zw, lamw, nb = cross_pred(Xw, yw, U['fold'], nz, cu, False,
                                  U['win_base'].shape[1])
        assert np.isfinite(zw[cu]).all(), '승부 z 가 전 clean 행을 못 덮었다'
        L = inclusive(zw)
        L_mu, L_sd = mod_stats(L, fit_m)          # 시드 변수 sd 를 가리지 않게 이름 분리
        that = modulator(L, L_mu, L_sd)
        wll = TL.rowloss(zw[ev_m & nz], yw[ev_m & nz]).mean()
        print(f'    승부축 λ {lamw} · win(hold) {wll:.5f} (전 팔에서 불변) '
              f'· L 적합우주 [{L[fit_m].min():.2f}, {L[fit_m].max():.2f}] '
              f'평균 {L[fit_m].mean():.3f}')

        if si == 0:
            diag['fit'] = shift_table(zt0, yt, L, fit_m, 'L 분위 · 적합우주(비-holdout)')
            diag['hold'] = shift_table(zt0, yt, L, ev_m, 'L 분위 · holdout — 재현되나')

        parts = nested_axes(U, W) if args.fitmode == 'cv' else None
        if parts is not None:                       # 중첩 경로 자체의 기준선
            zb = np.full(U['n'], np.nan)
            for P in parts:
                zb[P['te']] = zt0[P['te']]
            score(f'BASEn-s{sd}', np.where(np.isfinite(zb), zb, zt0),
                  '  (중첩 경로 기준 · BASE 와 같아야 정상)')

        for a in arms:
            deg = DEG[a]
            tag = f'{a}-s{sd}'
            if args.fitmode == 'cv':
                z = np.full(U['n'], np.nan)
                cs = []
                for P in parts:
                    Lin = modulator(inclusive(P['zw_in']), L_mu, L_sd)
                    c, ok = fit_shift(P['zt_in'], Lin, yt, P['tr'], deg)
                    z[P['te']] = apply_shift(zt0[P['te']], that[P['te']], c)
                    cs.append((np.round(c, 4).tolist(), ok))
            else:
                c, ok = fit_shift(zt0, that, yt, fit_m, deg)
                z = apply_shift(zt0, that, c)
                cs = [(np.round(c, 4).tolist(), ok)]
            cinfo[tag] = cs
            score(tag, z, f'  c(폴드별) {[x[0] for x in cs][:2]}…')

    op = FINAL / args.out
    np.savez_compressed(op.with_suffix('.rowloss.npz'), **rowl)
    with open(op.with_suffix('.json'), 'w') as f:
        json.dump({'cells': res, 'diag': diag, 'c': cinfo}, f,
                  ensure_ascii=False, indent=1)
    print(f'\n저장: {op}.rowloss.npz / {op}.json')

    print(f'\n{"=" * 78}\n[요약] tie 전용 ll · {len(seeds)}시드 · BASE {res["BASE"]["tie"]:.5f}'
          f'\n{"=" * 78}')
    print('  팔                tie ll   BASE 대비 (행 paired)')
    for a in arms:
        v = np.array([res[f'{a}-s{s}']['tie'] for s in seeds])
        r = np.mean([rowl[f'{a}-s{s}'] for s in seeds], axis=0)
        m, lo, hi = TL.paired_ci(r, rowl['BASE'])
        d = v - res['BASE']['tie']
        print(f'  {a:12s}{v.mean():10.5f}   {TL.ci_str(m, lo, hi)}  '
              f'시드 {d.mean():+.5f} {int((d < 0).sum())}/{len(d)}')
    print('\n  ! 승부축 ll 은 전 팔에서 불변이므로 Δ총점 = Δ(tie ll) 이다')
    print('  ! SHIFT 가 0 이 아니면 하네스 의심 — tie 결합기가 이미 절편을 적합했다')
    print('  ! 진단표 holdout 쪽 잔차가 분위와 함께 안 움직이면 c 적합 결과와 무관하게 기각')


if __name__ == '__main__':
    main()
