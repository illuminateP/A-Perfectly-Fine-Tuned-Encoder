"""asmlab — 재조립(assembly) 교차축 프로브.

사전등록 (결과 확인 전 고정):
  가설 T+ : P(tie) 가 승부 마진 크기에 의존한다 — 접전(|z_w| 작음)일수록 tie ↑.
            tie축 z 를 [z_t, logcosh(z_w/2)] + 절편으로 재적합 (둘 다 짝 → 합법).
  가설 W+ : 조건부 A:B 눈금이 tie 정도에 의존한다 — z_w 를 [z_w, z_w·z_t] 로 재적합
            (홀·짝×홀=홀 → 합법, 절편 금지 유지).
  가설 TW : 둘 다. T+/W+ 단독 유의가 전제인 참고 팔.

  기준    : 현행 체인 조립 (z_t, z_w 그대로). holdout 3-class ll = 0.9604444353 재현 필수.
  판정    : holdout 5,748 행별 3-class 손실 paired bootstrap 95% CI (B=10,000, seed 42).
            채택 후보 = 개선 방향 + CI 0 제외 + |Δ| ≥ 0.0005. 그 외 기각. clean OOF 는 참고.
  규율    : 결합기와 동일 — 5-fold 교차적합(U.fold), 짝 열만 중심화, 홀 축 절편 금지,
            프로브 열은 전부 무벌점(로짓 취급 → λ 격자 없음).
  z_w 확장: 승부 결합기는 cu∩nz 적합이라 무승부 행 z_w 가 없다. fold k 의 전 행을
            "fold k 제외 ∩ cu∩nz" 적합(λ = crossfit 이 fold별로 고른 값)으로 예측해 확장.
            cu∩nz 행에서 기존 OOF 와 일치해야 한다 (assert).

실행: cd final && python temp/asmlab.py   (ML2, 예상 2~5분 — 기준 결합기 내부 λ CV 가 대부분)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tie_ladder as TL

BASE_HOLD = 0.9604444353
NBOOT, THR = 10_000, 5e-4
LOG2 = float(np.log(2.0))


def crossfit_ext(X, y01, fold, m, bias, n_logit, picked):
    """fold k 전 행(적합 우주 밖 포함)을 "fold k 제외 ∩ m" 적합으로 예측 — OOF 의 전행 확장."""
    n_even = X.shape[1] if bias else 0
    z = np.full(len(y01), np.nan)
    for k in range(TL.K):
        tr = (fold != k) & m
        te = fold == k
        A, B = TL._z(X[tr], X[te], n_even)
        v, ok = TL._fit(A, y01[tr], TL._penvec(X.shape[1], n_logit, picked[k]), bias)
        assert ok, f'ext fold {k} 미수렴'
        z[te] = TL._scores(v, B, n_even, bias)
    return z


def probe(X, y01, fold, rows, bias, names, tag):
    """프로브 결합기 — 전 열 무벌점(n_logit=p → λ 한 점, 내부 CV 없음)."""
    z, dep = TL.combiner(X, y01, fold, rows, bias=bias, n_logit=X.shape[1])
    w = dep['v']
    coef = '  '.join(f'{nm} {w[i]:+.4f}' for i, nm in enumerate(names))
    if bias:
        coef += f'  절편 {w[len(names)]:+.4f}'
    print(f'  [{tag}] 배포 계수(표준화 좌표): {coef} | 미수렴 {dep["n_bad"]}')
    return z


def axis_ll(U, zt_, zw_, seg):
    """seg 행 우주의 행별 3-class 손실(체인 회계)과 (tie ll, win ll)."""
    y, nz = U['y'], U['nz']
    yT, yA = (y == 2).astype(np.float64), (y == 0).astype(np.float64)
    lt = TL.rowloss(zt_[seg], yT[seg])
    lw = TL.rowloss(zw_[seg & nz], yA[seg & nz])
    row = lt.astype(np.float64).copy()
    row[nz[seg]] += lw
    return row, float(lt.mean()), float(lw.mean())


def main():
    U = TL.build(TL.FINAL / 'temp/heads_repro')
    y, fold, cu, nz, hold = U['y'], U['fold'], U['cu'], U['nz'], U['hold']
    assert cu[hold].all(), 'hold ⊄ cu — 회계 전제 붕괴'
    yT = (y == 2).astype(np.float64)
    yA = (y == 0).astype(np.float64)
    mw = cu & nz

    # ── 기준 결합기 (검증 런과 동일 호출)
    Xw = np.hstack([U['win_base'], U['LIN']])
    Xt = np.hstack([U['tie_base'], U['GAT']])
    zw, dw = TL.combiner(Xw, yA, fold, mw, bias=False, n_logit=U['win_base'].shape[1])
    zt, dt = TL.combiner(Xt, yT, fold, cu, bias=True, n_logit=U['tie_base'].shape[1])
    print(f"기준 재구성 — 승부 λ {dw['picked']} | tie λ {dt['picked']}")

    row0, lt0, lw0 = axis_ll(U, zt, zw, hold)
    ll0 = float(row0.mean())
    assert abs(ll0 - BASE_HOLD) < 1e-6, f'기준 불일치 {ll0:.10f} != {BASE_HOLD}'

    # ── z_w 전행 확장 + 일치 가드
    zw_ext = crossfit_ext(Xw, yA, fold, mw, bias=False,
                          n_logit=U['win_base'].shape[1], picked=dw['picked'])
    assert np.allclose(zw_ext[mw], zw[mw], atol=1e-8), 'ext 가 기존 OOF 와 불일치'
    assert np.isfinite(zw_ext[cu]).all(), 'cu 행에 비유한 z_w'

    # ── 교차축 진단 (채택 판정과 무관한 참고)
    lc = np.logaddexp(zw_ext / 2.0, -zw_ext / 2.0) - LOG2        # logcosh(z_w/2), 짝·안정
    q = np.quantile(np.abs(zw_ext[cu]), [0.25, 0.5, 0.75])
    b = np.digitize(np.abs(zw_ext[cu]), q)
    rates = ' '.join(f'{yT[cu][b == i].mean():.3f}' for i in range(4))
    print(f'[진단] corr(logcosh, z_t)|cu = {np.corrcoef(lc[cu], zt[cu])[0, 1]:+.4f} | '
          f'|z_w| 사분위(저→고) tie율 {rates}')

    # ── 프로브 팔
    zt2 = probe(np.column_stack([zt, lc]), yT, fold, cu,
                bias=True, names=['z_t', 'logcosh'], tag='T+')
    zw2 = probe(np.column_stack([zw_ext, zw_ext * zt]), yA, fold, mw,
                bias=False, names=['z_w', 'z_w·z_t'], tag='W+')

    # ── 판정
    print(f'\n기준  hold 3cls {ll0:.6f}  (tie {lt0:.5f} | win {lw0:.5f})  '
          f'clean {axis_ll(U, zt, zw, cu)[0].mean():.6f}')
    for tag, (a, w_) in [('T+', (zt2, zw)), ('W+', (zt, zw2)), ('TW', (zt2, zw2))]:
        row, lt, lw = axis_ll(U, a, w_, hold)
        d, lo, hi = TL.paired_ci(row, row0, n_boot=NBOOT)
        v = ('채택 후보 ★' if hi < 0 and abs(d) >= THR else
             '유의 개선(문턱 미달)' if hi < 0 else
             '유의 악화' if lo > 0 else '노이즈')
        print(f'{tag:3s} hold 3cls {row.mean():.6f}  Δ {d:+.5f} [{lo:+.5f},{hi:+.5f}]  '
              f'(tie {lt:.5f} | win {lw:.5f})  clean {axis_ll(U, a, w_, cu)[0].mean():.6f}  → {v}')


if __name__ == '__main__':
    main()
