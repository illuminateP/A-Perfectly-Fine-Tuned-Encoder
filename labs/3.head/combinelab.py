"""승부축 온도의 조건부 변조 — 결합기만 다시 적합한다 (헤드 재학습 없음).

가설: 어떤 행에서는 승부 신호가 잡음이므로 z_win 을 눌러야 한다. 무엇으로 그 행을
가려낼 것인가가 **변조자** t̂ 다.

파리티 계약 — z_win 은 **홀**이어야 한다. t̂ 는 짝이므로 더할 수 없고 곱할 수만 있다:
    z_win + c·t̂      ✗ 짝이 더해져 반대칭 붕괴 (스왑 시 P(A)↔P(B) 깨짐)
    z_win × exp(c·t̂) ✓ 홀 × 짝 = 홀

변조자 (`--tmod`, 쉼표로 여러 개)
    p   σ(z_tie)          무승부스러움. **1차 실험에서 기각** — holdout T* 가 평평했다
    z   z_tie             같은 것의 로그오즈 링크
    d   소스 불일치도      4개 소스 점수를 RMS 정렬한 뒤 행별 표준편차.
                          각 열이 홀이라 스왑 시 전부 부호반전 → 표준편차는 불변(짝) → 합법.
                          **선형 결합기는 분산을 표현할 수 없으므로 층 2 에 진짜 새 정보다**
                          (p 는 승부 헤드가 이미 같은 입력에서 보던 것이라 새 정보가 아니었다)

팔 (전부 c=0 에서 BASE 로 정확히 환원)
    TEMP   z·exp(c₀)              전역 온도 — §5 대조군 (t̂ 무관)
    MOD1   z·exp(c₁·t̂)            ★ 본 가설
    MOD2   z·exp(c₁·t̂ + c₂·t̂²)    곡률 허용
    MODF   X=[m3|W4|m3⊙t̂|W4⊙t̂]    열별 변조 (+7 파라미터, 기존 crossfit 이 적합)

c 의 적합 우주 (`--fitmode`) — **1차 실험의 실패 원인이 여기 있었다**
    cv     (기본) 폴드마다, 결합기를 적합하는 **바로 그 학습행**에서 c 를 적합한다.
           과신을 피하려고 학습행 안에서 다시 K-fold 를 돌려 OOF z 를 만들고 그 위에서 적합.
           판정은 여전히 OOF. MODF 와 같은 프로토콜이 되어 BASE 와 조건이 맞는다.
    split  1차 실험 방식 — 비-holdout 1,382행에서 한 번 적합. 참고용으로 남긴다.
           (그 우주가 판정우주를 대표 못 했다: 파라미터 1개짜리 TEMP 조차 +0.00104)

진단표: 변조자 5분위별 최적 온도 T*. **적합 이전에** 가설의 생사를 보여준다 —
holdout 쪽 T* 가 분위와 함께 움직이지 않으면 그 변조자는 원리적으로 죽는다.
"""
import argparse
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

import tie_ladder as TL

FINAL = TL.FINAL
ARMS_ALL = ['TEMP', 'MOD1', 'MOD2', 'MODF']
TMODS = ['p', 'z', 'd']


# ══════════════════════════════════════════════ 축 점수
def tie_z(U):
    """tie 축 — 학습 없음. 저장 배포 GAT 3열로 결합기 1회 → OOF z_tie."""
    X = np.hstack([U['tie_base'], U['GAT']])
    rows = U['cu']
    y01 = (U['y'] == 2).astype(np.float64)
    z, dep = TL.combiner(X, y01, U['fold'], rows, bias=True,
                         n_logit=U['tie_base'].shape[1])
    ll = TL.rowloss(z[rows], y01[rows])
    print(f'  tie 고정 (저장 GAT · 학습 없음) — λ {dep["lam"]:g} '
          f'· tie(hold) {ll[U["hold"][rows]].mean():.5f}')
    return z


def win_z(U, W, n_logit=None):
    """승부축 결합기(기존 경로) → OOF z_win. W=None 이면 로짓만."""
    X = U['win_base'] if W is None else np.hstack([U['win_base'], W])
    rows = U['cu'] & U['nz']
    y01 = (U['y'] == 0).astype(np.float64)
    nl = U['win_base'].shape[1] if n_logit is None else n_logit
    z, dep = TL.combiner(X, y01, U['fold'], rows, bias=False, n_logit=nl)
    return z, dep


def nested_parts(U, W, n_logit=None):
    """폴드별 (학습행 OOF z, 테스트행 z). c 를 결합기와 같은 우주에서 적합하기 위한 재료.

    바깥 fold k 마다 — ① 학습행 안에서 다시 K-fold 를 돌려 그 행들의 OOF z 를 만들고
    ② 학습행 전체로 배포 적합(λ 는 안쪽 fold 선택의 중앙값)해 테스트행 z 를 낸다.
    ①은 c 적합용(과신 없음), ②는 판정용. 두 정보 모두 fold k 의 라벨을 안 본다.
    """
    X = np.hstack([U['win_base'], W])
    rows = U['cu'] & U['nz']
    y01 = (U['y'] == 0).astype(np.float64)
    nl = U['win_base'].shape[1] if n_logit is None else n_logit
    parts, lams = [], []
    for k in range(TL.K):
        tr, te = (U['fold'] != k) & rows, (U['fold'] == k) & rows
        idx = np.flatnonzero(tr)
        sub = np.full(U['n'], -1, int)
        skf = StratifiedKFold(TL.K, shuffle=True, random_state=TL.SEED + 100 + k)
        for j, (_, iv) in enumerate(skf.split(np.zeros((len(idx), 1)), y01[idx])):
            sub[idx[iv]] = j
        z_in, picked, _, _ = TL.crossfit(X, y01, sub, False, TL.LAMS,
                                         rows=tr, n_even=0, n_logit=nl)
        assert np.isfinite(z_in[tr]).all(), f'fold {k} 중첩 OOF z 비유한'
        lam = sorted(picked)[len(picked) // 2]
        A, B = TL._z(X[tr], X[te], 0)
        v, _ = TL._fit(A, y01[tr], TL._penvec(A.shape[1], nl, lam), False)
        z_te = np.full(U['n'], np.nan)
        z_te[te] = TL._scores(v, B, 0, False)
        parts.append(dict(tr=tr, te=te, z_in=z_in, z_te=z_te))
        lams.append(float(lam))
    return parts, lams


# ══════════════════════════════════════════════ 변조
def modulator(raw, fit_mask):
    """짝 통계를 표준화한 변조자. 상수는 **적합 우주에서만** 뽑는다 (holdout 미접촉)."""
    mu = float(raw[fit_mask].mean())
    sd = float(np.clip(raw[fit_mask].std(), 1e-9, None))
    return (raw - mu) / sd


def disagree(W, fit_mask):
    """소스 불일치도 — 각 열을 적합우주 RMS 로 정렬한 뒤 행별 표준편차 (짝)."""
    rms = np.clip(np.sqrt((W[fit_mask] ** 2).mean(0)), 1e-9, None)
    return (W / rms).std(1)


def fit_gain(z, that, y01, mask, deg):
    """z' = z·exp(c₀ + Σ_d c_d·t̂^d) 의 c 를 mask 행에서 적합. deg=0 이면 전역 온도."""
    P = np.column_stack([that[mask] ** (d + 1) for d in range(deg)]) if deg else None
    zm, ym = z[mask], y01[mask]

    def fg(c):
        g = np.exp(c[0] + (P @ c[1:] if deg else 0.0))
        zz = zm * g
        val = float(np.mean(np.logaddexp(0, zz) - ym * zz))
        r = (expit(zz) - ym) * zz / len(ym)          # d/dc_d = r·t̂^d
        grad = [r.sum()] + [float(r @ P[:, d]) for d in range(deg)]
        return val, np.asarray(grad)

    res = minimize(fg, np.zeros(deg + 1), jac=True, method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-9})
    return res.x, bool(res.success)


def apply_gain(z, that, c):
    p = sum(c[d + 1] * that ** (d + 1) for d in range(len(c) - 1))
    return z * np.exp(c[0] + p)


def gain_cv(parts, that, y01, deg, n):
    """폴드마다 학습행에서 c 를 적합해 테스트행에 적용."""
    z = np.full(n, np.nan)
    cs = []
    for P in parts:
        c, ok = fit_gain(P['z_in'], that, y01, P['tr'], deg)
        z[P['te']] = apply_gain(P['z_te'][P['te']], that[P['te']], c)
        cs.append((np.round(c, 4).tolist(), ok))
    return z, cs


# ══════════════════════════════════════════════ 진단
def temp_table(z, y01, var, mask, label, nq=5):
    """변조자 5분위별 최적 온도. T* 가 분위와 함께 움직이면 가설이 산다."""
    zz, yy, vv = z[mask], y01[mask], var[mask]
    edges = np.quantile(vv, np.linspace(0, 1, nq + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    print(f'    [{label}]  n={int(mask.sum()):,}')
    print(f'      {"분위":>6}{"행":>7}{"승부 ll":>10}{"T*":>8}{"평균|z|":>9}')
    out = []
    for i in range(nq):
        b = (vv >= edges[i]) & (vv < edges[i + 1])
        if b.sum() < 30:
            continue
        c, ok = fit_gain(zz[b], np.zeros(int(b.sum())), yy[b],
                         np.ones(int(b.sum()), bool), 0)
        T = float(np.exp(-c[0]))                       # z'=z·exp(c₀) ⇒ T = 1/exp(c₀)
        ll = float(TL.rowloss(zz[b], yy[b]).mean())
        print(f'      {i + 1:>6}{int(b.sum()):7d}{ll:10.5f}{T:8.3f}'
              f'{np.abs(zz[b]).mean():9.3f}' + ('' if ok else '  ** 미수렴'))
        out.append(dict(q=i + 1, n=int(b.sum()), ll=ll, T=T))
    return out


# ══════════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bundle', default='temp/heads_repro')
    ap.add_argument('--cols', default='temp/winconf.rowloss.npz')
    ap.add_argument('--tag', default='SK-RAW')
    ap.add_argument('--seeds', default='5,6,7,8,9')
    ap.add_argument('--rounds', type=int, default=1, help='쓸 헤드 라운드 수 (확정 1R)')
    ap.add_argument('--tmod', default='p,d', help='변조자 쉼표 목록: ' + ' '.join(TMODS))
    ap.add_argument('--fitmode', default='cv', choices=['cv', 'split'])
    ap.add_argument('--arms', default=','.join(ARMS_ALL))
    ap.add_argument('--out', default='temp/combinelab2')
    args = ap.parse_args()

    arms = [a for a in args.arms.split(',') if a]
    tmods = [t for t in args.tmod.split(',') if t]
    seeds = [int(s) for s in args.seeds.split(',') if s]
    assert set(arms) <= set(ARMS_ALL), f'모르는 팔 {set(arms) - set(ARMS_ALL)}'
    assert set(tmods) <= set(TMODS), f'모르는 변조자 {set(tmods) - set(TMODS)}'

    print(f'[적재]  fitmode={args.fitmode} · 변조자 {tmods}')
    U = TL.build(FINAL / args.bundle)
    zt = tie_z(U)

    rows = U['cu'] & U['nz']
    y01 = (U['y'] == 0).astype(np.float64)
    fit_m = rows & ~U['hold']
    ev_m = rows & U['hold']
    print(f'  행: 결합기 {int(rows.sum()):,} · split 모드 c 적합 {int(fit_m.sum()):,} '
          f'· 판정 {int(ev_m.sum()):,}')

    Z = np.load(FINAL / args.cols)
    nc = args.rounds * 4
    res, roww, diag, cinfo = {}, {}, {}, {}

    def score(tag, z, note=''):
        ll = TL.rowloss(z[ev_m], y01[ev_m])
        assert np.isfinite(ll).all(), f'{tag} 비유한 손실'
        res[tag] = dict(win=float(ll.mean()))
        roww[tag] = ll
        print(f'    [{tag:16s}] 승부(hold) {ll.mean():.5f}{note}')

    print(f'\n{"="*78}\n[기준선] 결정론\n{"="*78}')
    score('LIN-S', win_z(U, U['LIN'])[0], '  ← 현행 소스별 3R')
    score('logit', win_z(U, None)[0], '  ← 로짓만 [m3]')

    for si, sd in enumerate(seeds):
        key = f'cols__{args.tag}-s{sd}'
        assert key in Z.files, f'{key} 없음 — 저장 열 {Z.files[:4]}…'
        W = Z[key][:, :nc].astype(np.float64)
        print(f'\n{"-"*78}\n[시드 {sd}] 헤드 열 {W.shape[1]}개 ({args.rounds}R)\n{"-"*78}')

        z0, dep0 = win_z(U, W)
        score(f'BASE0-s{sd}', z0, f'  (기존 경로 · λ {dep0["lam"]:g})')

        if args.fitmode == 'cv':
            parts, lams = nested_parts(U, W)
            zb = np.full(U['n'], np.nan)
            for P in parts:
                zb[P['te']] = P['z_te'][P['te']]
                assert not P['tr'][P['te']].any(), '폴드 누수'
            score(f'BASE-s{sd}', zb, f'  (중첩 경로 · λ {lams})')
        else:
            parts, zb = None, z0
            score(f'BASE-s{sd}', zb, '  (= BASE0)')

        MOD = {}
        for tm in tmods:
            raw = {'p': expit(zt), 'z': zt, 'd': disagree(W, fit_m)}[tm]
            that = modulator(raw, fit_m)
            MOD[tm] = that
            print(f'    변조자 t̂[{tm}] — 적합우주 sd {raw[fit_m].std():.4f} '
                  f'· 판정우주 평균 {that[ev_m].mean():+.4f} (표준화 후)')
            if si == 0:
                diag[f'{tm}-fit'] = temp_table(zb, y01, raw, fit_m,
                                               f't̂[{tm}] · 적합우주(비-holdout)')
                diag[f'{tm}-hold'] = temp_table(zb, y01, raw, ev_m,
                                                f't̂[{tm}] · holdout — 재현되나')

        for a in arms:
            for tm in (['-'] if a == 'TEMP' else tmods):
                tag = f'{a}-s{sd}' if a == 'TEMP' else f'{a}@{tm}-s{sd}'
                that = np.zeros(U['n']) if a == 'TEMP' else MOD[tm]
                if a == 'MODF':
                    M = U['win_base'].shape[1]
                    Wi = np.hstack([W, U['win_base'] * that[:, None], W * that[:, None]])
                    zi, di = win_z(U, Wi, n_logit=M)
                    score(tag, zi, f'  (열 {M + Wi.shape[1]} · λ {di["lam"]:g})')
                    continue
                deg = {'TEMP': 0, 'MOD1': 1, 'MOD2': 2}[a]
                if args.fitmode == 'cv':
                    zm, cs = gain_cv(parts, that, y01, deg, U['n'])
                    cinfo[tag] = cs
                    note = f'  c(폴드별) {[c for c, _ in cs][:2]}…'
                else:
                    c, ok = fit_gain(zb, that, y01, fit_m, deg)
                    zm = apply_gain(zb, that, c)
                    cinfo[tag] = [(np.round(c, 4).tolist(), ok)]
                    note = f'  c {np.round(c, 4)}' + ('' if ok else ' ** 미수렴')
                score(tag, zm, note)

    op = FINAL / args.out
    with open(op.with_suffix('.json'), 'w') as f:
        json.dump({'cells': res, 'diag': diag, 'c': cinfo, 'tmods': tmods,
                   'fitmode': args.fitmode, 'seeds': seeds, 'rounds': args.rounds},
                  f, ensure_ascii=False, indent=1)
    np.savez_compressed(op.with_suffix('.rowloss.npz'), **roww)
    print(f'\n저장: {op}.json / {op}.rowloss.npz')

    mean = lambda t: np.mean([res[f'{t}-s{s}']['win'] for s in seeds])
    vec = lambda t: np.array([res[f'{t}-s{s}']['win'] for s in seeds])
    rw = lambda t: np.mean([roww[f'{t}-s{s}'] for s in seeds], axis=0)

    print(f'\n{"="*78}\n[요약] 승부 전용 ll · {len(seeds)}시드 · BASE {mean("BASE"):.5f} '
          f'(기존경로 {mean("BASE0"):.5f}) · 현행 LIN-S {res["LIN-S"]["win"]:.5f}\n{"="*78}')
    print(f'  {"팔":14s}{"승부 ll":>10}   BASE 대비 (같은 시드 paired)')
    tags = ['BASE0'] + [a if a == 'TEMP' else f'{a}@{tm}'
                        for a in arms for tm in (['-'] if a == 'TEMP' else tmods)]
    for t in tags:
        m, lo, hi = TL.paired_ci(rw(t), rw('BASE'))
        d = vec(t) - vec('BASE')
        print(f'  {t:14s}{mean(t):10.5f}   {TL.ci_str(m, lo, hi)}  '
              f'시드 {d.mean():+.5f} {int((d < 0).sum())}/{len(d)}')

    print('\n  ! BASE0−BASE 는 프로토콜 차이(중첩 crossfit)일 뿐 — 0 근처여야 정상')
    print('  ! TEMP 가 0 이 아니면 하네스 의심 — §5 는 전역 온도를 이미 닫았다')
    print('  ! 진단표에서 holdout T* 가 분위와 함께 안 움직이면 그 변조자는 죽은 것이다')


if __name__ == '__main__':
    main()
