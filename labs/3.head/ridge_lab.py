"""ridge_lab.py — 기둥(tie축) 릿지 구조를 네 모드로 비교한다. 같은 판정 틀·같은 재료.

무엇을 묻나
    지금 기둥은 **소스마다 따로** 릿지를 적합하고, λ 는 그 소스의 1024열에 **균일**하다.
        X_s = [s0 (무벌점) | P_s 1024 (λ_s)]      4번 따로
    차원별 차등은 릿지 기하 σₖ²/(σₖ²+λ) 가 자동으로 만들 뿐 **학습된 게 아니다**.
    그리고 소스끼리 서로를 못 봐서 **풀링 간 중복**을 아무도 걸러내지 않는다.

네 모드
    indep  현행 그대로. 소스별 1-D λ.            ← 검산용. holdout 0.95970 을 내야 한다
    joint  4소스를 한 설계행렬에. λ 벡터 격자 전수.
             X = [s0 | P_A | P_B | P_C | P_D]  p=4097
             pen = [0 | λ_A×1024 | … | λ_D×1024]
           β_B 의 뜻이 "s0 를 고정했을 때"에서 "s0 **와 A·C·D 를** 고정했을 때"로 바뀐다.
    band   소스별. 그 소스의 특이기저로 회전해 **스펙트럼 구간마다 λ**.
             감쇠 곡선의 **모양**을 학습한다. 하드 절단이 아니라서
             "SVD-512 절단이 FULL 릿지에게 졌다"는 기록과 모순되지 않는다.
    ard    소스별. **좌표마다 λ** 를 적응적으로 학습 (adaptive ridge).
             pen_j ∝ 1/w_j²  로 갱신하면 작았던 열이 더 눌려 결국 닫힌다 — L0 대리.
             전역 세기 λ 하나만 CV 로 고르고 나머지는 데이터가 정한다.

공통 규약 — 여기를 어기면 비교가 성립하지 않는다
    · 정규화·앵커·CV 분할·최종 재적합 절차는 `fit_heads.fit_beta` 와 같다
      (짝축이라 중심화+표준화, s0 는 RMS only, train 내부 StratifiedKFold(3, seed 42))
    · 산출은 항상 **기둥 4열**(stage2 10,748) — 결합기 인터페이스 불변
    · GAT·LIN 은 모드와 무관하므로 **1회만** 학습해 전 모드가 공유
    · 판정은 holdout 5,748 행별 손실 paired CI vs `factorial_heads.rowloss.npz[tr0_p4_r3]`

사용
    python -u temp/ridge_lab.py --modes indep                    # 검산 (0.95970)
    python -u temp/ridge_lab.py --modes joint --closed
    python -u temp/ridge_lab.py --modes band,ard
    python -u temp/ridge_lab.py --modes indep,joint,band,ard     # 전부
"""
import argparse
import gc
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_F = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_F / 'temp'))
sys.path.insert(0, str(_F / 'models'))
import fit_heads as FH                                          # noqa: E402

TEMP = _F / 'temp'
THRESH = 0.0005
BASE_ROWLOSS = TEMP / 'factorial_heads.rowloss.npz'
BASE_CELL = 'tr0_p4_r3'


def verdict(lo, hi):
    if lo > THRESH:
        return '유의 손해 ★'
    if hi < -THRESH:
        return '유의 이득 ★'
    if lo > 0:
        return '손해(문턱 미달)'
    if hi < 0:
        return '이득(문턱 미달)'
    return '노이즈'


# ══════════════════════════════════════════════ 공통 재료
class Design:
    """tie축 기둥의 설계 재료. 정규화·앵커는 fit_beta 와 같은 규약."""

    def __init__(self):
        Mt = FH.Ms['tie']
        self.srcs = list(Mt['src'])
        self.y = Mt['y01m']                       # train tie 라벨
        self.bias = Mt['bias']                    # True
        self.rows_s2 = Mt['rows']                 # clean 7,748 (앵커 적합용)

        # s0 — tie 로짓 5열의 고정 결합. fit_anchor 는 stage2 clean 에서 적합한다
        self.w_anc, self.b_anc = FH.fit_anchor(
            [FH.S2[nm] for nm in FH.tie_names], Mt['y01'], Mt['rows'], Mt['bias'], FH.tie_names)
        s0 = np.column_stack([FH.TR[nm] for nm in FH.tie_names]) @ self.w_anc + self.b_anc
        self.s0z = s0 / np.clip(np.sqrt((s0 ** 2).mean()), 1e-9, None)

        # 풀링 — 소스별 1024 를 이어 붙인다. mu/sc 는 **열별**이라 소스별로 낸 것과 같다
        P = np.hstack([FH.TR[s] for s in self.srcs])
        self.mu = P.mean(0)                       # 짝축 -> 중심화 + 표준화
        self.sc = np.clip(P.std(0), 1e-9, None)
        self.Pz = (P - self.mu) / self.sc
        del P
        gc.collect()
        P2 = np.hstack([FH.PS2[s] for s in self.srcs])
        self.P2z = (P2 - self.mu) / self.sc
        del P2
        gc.collect()

        self.d = self.Pz.shape[1] // len(self.srcs)
        assert self.d == 1024, f'소스당 차원 {self.d}'
        self.sl = {s: slice(i * self.d, (i + 1) * self.d) for i, s in enumerate(self.srcs)}

        # 내부 3-fold — fit_beta 와 같은 분할
        from sklearn.model_selection import StratifiedKFold
        self.inner = np.zeros(len(self.y), int)
        for j, (_, iv) in enumerate(StratifiedKFold(3, shuffle=True, random_state=FH.SEED)
                                    .split(np.zeros((len(self.y), 1)), self.y)):
            self.inner[iv] = j
        print(f'설계: train {self.Pz.shape[0]:,}행 x 풀링 {self.Pz.shape[1]:,}열 '
              f'({len(self.srcs)}소스 x {self.d}) | stage2 {self.P2z.shape[0]:,}행 '
              f'| X 메모리 {self.Pz.nbytes / 1e9:.2f}GB')

    # ── 소스 하나짜리 설계 (indep·band·ard 용) ──
    def X_src(self, s):
        return np.hstack([self.s0z[:, None], self.Pz[:, self.sl[s]]])

    def X_joint(self):
        return np.hstack([self.s0z[:, None], self.Pz])

    # ── 적합 + CV ──
    def cv_loss(self, X, pen, warm=None):
        """내부 3-fold held-out 로그손실 합. warm 은 fold 별 웜스타트 리스트."""
        tot, warm = 0.0, (warm or [None] * 3)
        for j in range(3):
            tr, te = self.inner != j, self.inner == j
            v, _ = FH._fit(X[tr], self.y[tr], pen, self.bias, warm[j])
            warm[j] = v
            z = X[te] @ v[:X.shape[1]] + (v[X.shape[1]] if self.bias else 0.0)
            tot += float((np.logaddexp(0, z) - self.y[te] * z).sum())
        return tot, warm

    def final(self, X, pen, warm=None):
        v, ok = FH._fit(X, self.y, pen, self.bias, warm)
        return v, ok

    def column(self, s, beta):
        """소스 s 의 β(1024) -> stage2 기둥 열."""
        return self.P2z[:, self.sl[s]] @ beta


def pen_src(d, lam, n_free=1):
    pen = np.full(d + n_free, float(lam))
    pen[:n_free] = 0.0
    return pen


# ══════════════════════════════════════════════ 모드
def run_indep(D, grid=None):
    """현행 재현 — 소스별 1-D λ. holdout 0.95970 을 내야 한다."""
    out, diag = [], {}
    for i, s in enumerate(D.srcs):
        X = D.X_src(s)
        lams = grid or FH.LAMS_TR
        score, warm = {}, None
        t0 = time.time()
        for lam in sorted(lams, reverse=True):
            score[lam], warm = D.cv_loss(X, pen_src(D.d, lam), warm)
        lam = min(score, key=score.get)
        v, ok = D.final(X, pen_src(D.d, lam))
        out.append((f'p{i}', D.column(s, v[1:1 + D.d])))
        diag[s] = {'lam': float(lam), 'grid': {float(k): v2 for k, v2 in score.items()}}
        print(f'  [indep·{s}] λ {lam:g} | {time.time() - t0:.0f}s'
              + ('' if ok else ' ** 미수렴'))
        del X
        gc.collect()
    return out, diag


def _gray(dims):
    """뱀 순회(boustrophedon) — 이웃 조합이 한 좌표만 달라 웜스타트가 유효하다.

    혼합 진법에서 앞 좌표들의 합이 홀수인 행은 방향을 뒤집는다. 그러면 연속한 두 조합의
    해밍거리가 항상 1 이라, 이전 해가 다음 적합의 좋은 출발점이 된다.
    """
    order = list(itertools.product(*[range(n) for n in dims]))
    return sorted(order, key=lambda c: tuple(
        (c[i] if sum(c[:i]) % 2 == 0 else dims[i] - 1 - c[i]) for i in range(len(dims))))


def run_joint(D, closed=False, span=1, max_ext=2):
    """안 1 — 4소스 한 설계행렬, λ 벡터 격자 전수. 경계 도달 시 자동 확장."""
    L = FH.LAMS_TR
    base = {}
    for s in D.srcs:
        hint = FH._LAMS_ALL.get('base', {}).get(f'기둥·{s}')
        base[s] = hint if hint else 1000.0
    cand = {}
    for s in D.srcs:
        i = min(range(len(L)), key=lambda j: abs(np.log10(L[j]) - np.log10(base[s])))
        c = [L[j] for j in range(max(0, i - span), min(len(L), i + span + 1))]
        if closed:
            c = c + [1e8]
        cand[s] = sorted(set(c))
    print('  격자:')
    for s in D.srcs:
        print(f'    {s:10s} {[f"{x:g}" for x in cand[s]]}')

    X = D.X_joint()
    P_TOT = X.shape[1]
    print(f'  X {X.shape} {X.nbytes / 1e9:.2f}GB')

    def penvec(v):
        pen = np.zeros(P_TOT)
        for si, s in enumerate(D.srcs):
            pen[1 + D.sl[s].start:1 + D.sl[s].stop] = v[si]
        return pen

    score, seen, warm = {}, set(), None
    t0 = time.time()

    def sweep(grids, Xm):
        nonlocal warm
        dims = [len(grids[s]) for s in D.srcs]
        for combo in _gray(dims):
            key = tuple(grids[s][combo[si]] for si, s in enumerate(D.srcs))
            if key in seen:
                continue
            seen.add(key)
            score[key], warm = D.cv_loss(Xm, penvec(key), warm)
            print(f'    λ {[f"{x:g}" for x in key]}  CV {score[key] / len(D.y):.5f}'
                  f'  [{len(seen)}조합 {(time.time() - t0) / 60:.1f}분]')

    sweep(cand, X)
    for _ in range(max_ext):                      # 경계 자동 확장
        best = min(score, key=score.get)
        grew = False
        for si, s in enumerate(D.srcs):
            c = cand[s]
            if best[si] == max(c) and max(c) < L[-1]:
                nxt = [x for x in L if x > max(c)]
                if nxt:
                    cand[s] = sorted(c + [nxt[0]]); grew = True
            if best[si] == min(c) and min(c) > L[0]:
                prv = [x for x in L if x < min(c)]
                if prv:
                    cand[s] = sorted(cand[s] + [prv[-1]]); grew = True
        if not grew:
            break
        print('  경계 도달 -> 격자 확장')
        sweep(cand, X)

    best = min(score, key=score.get)
    print(f'  최적 λ {[f"{x:g}" for x in best]} | 조합 {len(score)} | {(time.time() - t0) / 60:.1f}분')
    v, ok = D.final(X, penvec(best))
    out = [(f'p{i}', D.column(s, v[1 + D.sl[s].start:1 + D.sl[s].stop]))
           for i, s in enumerate(D.srcs)]
    diag = {'best': {s: float(best[i]) for i, s in enumerate(D.srcs)},
            'surface': {'|'.join(f'{x:g}' for x in k): float(v2) for k, v2 in score.items()},
            'indep': {s: float(base[s]) for s in D.srcs}, 'ok': bool(ok)}
    del X
    gc.collect()
    return out, diag


def run_band(D, bands=(64, 256), mult=(0.1, 1.0, 10.0)):
    """(나) 소스별 — 특이기저로 회전해 스펙트럼 구간마다 λ. 감쇠 곡선의 모양을 학습."""
    out, diag = [], {}
    edges = [0] + list(bands) + [D.d]
    for i, s in enumerate(D.srcs):
        t0 = time.time()
        Pz = D.Pz[:, D.sl[s]]
        G = (Pz.T @ Pz) / len(Pz)
        ev, V = np.linalg.eigh(G)
        o = np.argsort(-ev)
        ev, V = ev[o], V[:, o]                    # 특이값² 내림차순
        R = Pz @ V                                # 회전된 설계 (train)
        R2 = D.P2z[:, D.sl[s]] @ V                # 〃 (stage2)
        X = np.hstack([D.s0z[:, None], R])
        P_TOT = X.shape[1]
        base = FH._LAMS_ALL.get('base', {}).get(f'기둥·{s}', 1000.0)

        def penvec(ls, _p=P_TOT):
            pen = np.zeros(_p)
            for b in range(len(edges) - 1):
                pen[1 + edges[b]:1 + edges[b + 1]] = ls[b]
            return pen

        score, warm = {}, None
        for combo in itertools.product(*[[base * m for m in mult]] * (len(edges) - 1)):
            score[combo], warm = D.cv_loss(X, penvec(combo), warm)
        best = min(score, key=score.get)
        v, ok = D.final(X, penvec(best))
        beta_rot = v[1:1 + D.d]
        out.append((f'p{i}', R2 @ beta_rot))
        diag[s] = {'base': float(base), 'bands': list(edges),
                   'best': [float(x) for x in best],
                   'ev_top8': [float(x) for x in ev[:8]],
                   'grid': {'|'.join(f'{x:g}' for x in k): float(v2) for k, v2 in score.items()}}
        print(f'  [band·{s}] 구간 {edges} λ {[f"{x:g}" for x in best]} '
              f'| {len(score)}조합 {time.time() - t0:.0f}s' + ('' if ok else ' ** 미수렴'))
        del Pz, G, V, R, R2, X
        gc.collect()
    return out, diag


def run_ard(D, iters=5, span=1, eps=1e-8):
    """(가) 소스별 — 좌표마다 λ 를 적응적으로 학습 (adaptive ridge).

    pen_j <- λ · (1/(w_j²+ε)) / mean_k(1/(w_k²+ε))   평균 벌점을 λ 로 유지한다.
    작았던 열이 더 눌려 결국 닫히는 L0 대리다. 전역 세기 λ 만 CV 로 고른다.
    """
    L = FH.LAMS_TR
    out, diag = [], {}
    for i, s in enumerate(D.srcs):
        t0 = time.time()
        X = D.X_src(s)
        base = FH._LAMS_ALL.get('base', {}).get(f'기둥·{s}', 1000.0)
        j0 = min(range(len(L)), key=lambda j: abs(np.log10(L[j]) - np.log10(base)))
        cand = [L[j] for j in range(max(0, j0 - span), min(len(L), j0 + span + 1))]

        def adapt(Xf, yf, lam, warm=None):
            """적응 반복 — 마지막 (v, pen) 을 돌려준다."""
            pen = pen_src(D.d, lam)
            v = warm
            for _ in range(iters):
                v, _ = FH._fit(Xf, yf, pen, D.bias, v)
                w = v[1:1 + D.d]
                inv = 1.0 / (w ** 2 + eps)
                pen = np.concatenate([[0.0], lam * inv / inv.mean()])
            return v, pen

        score = {}
        for lam in sorted(cand, reverse=True):
            tot = 0.0
            for j in range(3):
                tr, te = D.inner != j, D.inner == j
                v, _ = adapt(X[tr], D.y[tr], lam)
                z = X[te] @ v[:X.shape[1]] + (v[X.shape[1]] if D.bias else 0.0)
                tot += float((np.logaddexp(0, z) - D.y[te] * z).sum())
            score[lam] = tot
        lam = min(score, key=score.get)
        v, pen = adapt(X, D.y, lam)
        w = v[1:1 + D.d]
        out.append((f'p{i}', D.column(s, w)))
        # 얼마나 닫혔나 — 상위 몇 개 좌표가 |w| 의 대부분을 갖는가
        a = np.sort(np.abs(w))[::-1]
        cum = np.cumsum(a) / max(a.sum(), 1e-12)
        n90 = int(np.searchsorted(cum, 0.90) + 1)
        diag[s] = {'lam': float(lam), 'iters': iters,
                   'n_coord_90pct': n90, 'pen_ratio_max_min': float(pen[1:].max() / pen[1:].min()),
                   'grid': {float(k): float(v2) for k, v2 in score.items()}}
        print(f'  [ard·{s}] λ {lam:g} | |w| 90% 를 좌표 {n90}/{D.d} 개가 설명 '
              f'| 벌점 최대/최소 {pen[1:].max() / pen[1:].min():.1e} | {time.time() - t0:.0f}s')
        del X
        gc.collect()
    return out, diag


MODES = {'deployed': None, 'indep': run_indep, 'joint': run_joint, 'band': run_band, 'ard': run_ard}


# ══════════════════════════════════════════════ 판정
def evaluate(pillar, tag, ref):
    FH.PILLAR = pillar
    FH._fit_combiner()
    r = FH._score()
    if ref is None:
        d, lo, hi, vd = 0.0, 0.0, 0.0, '기준 없음'
    else:
        d, lo, hi = FH.paired_ci(r['row_hold'], ref)
        vd = verdict(lo, hi)
    print(f'  -> holdout {r["tot_h"]:.5f} (tie {r["tie_h"]:.5f} / win {r["win_h"]:.5f}) '
          f'| Δ {d:+.5f} [{lo:+.5f},{hi:+.5f}] {vd}')
    return dict(모드=tag, holdout=r['tot_h'], tie=r['tie_h'], win=r['win_h'],
                Δ=d, lo=lo, hi=hi, 판정=vd, ll_10k=r['tot_h_10k']), r['row_hold']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--modes', default='deployed,indep',
                    help='deployed,indep,joint,band,ard')
    ap.add_argument('--closed', action='store_true', help='joint 격자에 CLOSED(1e8) 후보 추가')
    ap.add_argument('--span', type=int, default=1, help='joint/ard 격자 반경')
    ap.add_argument('--ard-iters', type=int, default=5)
    ap.add_argument('--bands', default='64,256', help='band 경계 (특이값 내림차순 인덱스)')
    ap.add_argument('--out', default=str(TEMP / 'ridge_lab'))
    a = ap.parse_args()
    modes = [m.strip() for m in a.modes.split(',') if m.strip()]
    bad = [m for m in modes if m not in MODES]
    assert not bad, f'모르는 모드: {bad}'

    ref = None
    if BASE_ROWLOSS.exists():
        ref = np.load(BASE_ROWLOSS)[BASE_CELL]
        print(f'기준: {BASE_CELL} (holdout 0.95970) — {BASE_ROWLOSS.name}, {len(ref):,}행\n')
    else:
        print(f'** {BASE_ROWLOSS.name} 없음 — Δ/CI 를 못 낸다\n')

    print('=' * 84)
    print('[세션] 적재 -> SVD -> 축재료 -> GAT -> LIN   (모드와 무관, 1회)')
    t0 = time.time()
    FH.HeadSession().prepare()
    FH.HEADS = TEMP / 'heads_ridgelab'
    FH.HEADS.mkdir(parents=True, exist_ok=True)
    FH.set_seeds(FH.SEED)
    FH._build_axes()
    # 기둥을 **정상 순서로** 돌린다. 두 가지 이유:
    #   ① 셀 60 이 Mt·t0 를 전역으로 만들고 셀 62(_fit_gat)·64(_fit_lin) 가 그걸 쓴다.
    #      건너뛰면 NameError 다 (실측).
    #   ② 그 결과가 **배포 경로의 진짜 기준 4열**이다. 아래 indep 모드는 Design 재구현이라,
    #      둘을 수치로 대조하는 것이 이 스크립트의 가장 날카로운 검산이다.
    FH._fit_pillar()
    PILLAR_DEPLOYED = list(FH.PILLAR)
    FH._fit_gat()
    FH._fit_lin()
    print(f'[세션] 완료 {(time.time() - t0) / 60:.1f}분 | 기둥 {len(PILLAR_DEPLOYED)}열 '
          f'/ GAT {len(FH.GAT_COLS)}열 / LIN {len(FH.LIN_COLS)}열')

    D = Design()
    rows, diags, losses = [], {}, {}

    if 'deployed' in modes or 'indep' in modes:
        print('\n' + '=' * 84)
        print('[deployed] 세션이 낸 기둥 4열 — 배포 경로 그대로 (Δ 0.00000 이어야 정상)')
        row, rl = evaluate(PILLAR_DEPLOYED, 'deployed', ref)
        row['분'] = 0.0
        rows.append(row)
        losses['deployed'] = rl

    for m in modes:
        if m == 'deployed':
            continue
        print('\n' + '=' * 84)
        print(f'[{m}]')
        t1 = time.time()
        kw = {}
        if m == 'joint':
            kw = dict(closed=a.closed, span=a.span)
        elif m == 'band':
            kw = dict(bands=tuple(int(x) for x in a.bands.split(',')))
        elif m == 'ard':
            kw = dict(iters=a.ard_iters, span=a.span)
        pillar, dg = MODES[m](D, **kw)
        if m == 'indep':
            dmax = max(float(np.abs(a[1] - b[1]).max())
                       for a, b in zip(pillar, PILLAR_DEPLOYED))
            rel = dmax / max(float(np.abs(np.array([c for _, c in PILLAR_DEPLOYED])).max()), 1e-12)
            dg['vs_deployed_max_abs'] = dmax
            dg['vs_deployed_rel'] = rel
            print(f'  [검산] Design 재구현 vs 배포 기둥: max|Δ| {dmax:.3e} (상대 {rel:.2e})')
            if rel > 1e-6:
                print('  ** 재구현이 배포 경로와 다르다 — joint/band/ard 해석 금지')
        row, rl = evaluate(pillar, m, ref)
        row['분'] = round((time.time() - t1) / 60, 1)
        rows.append(row)
        diags[m] = dg
        losses[m] = rl

    R = pd.DataFrame(rows).sort_values('holdout').reset_index(drop=True)
    base = Path(a.out)
    R.to_csv(f'{base}.csv', index=False)
    np.savez_compressed(f'{base}.rowloss.npz', **losses)
    Path(f'{base}.diag.json').write_text(json.dumps(diags, ensure_ascii=False, indent=1),
                                         encoding='utf-8')
    print('\n' + '=' * 84)
    print(f'[결과] 기준 {BASE_CELL} 0.95970 | holdout 5,748 paired | 문턱 ±{THRESH}')
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(R.to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    print(f'\n저장: {base}.csv / .rowloss.npz / .diag.json')
    print('* indep 이 0.95970 을 안 내면 설계 재료가 어긋난 것 — 다른 모드 해석 금지.')


if __name__ == '__main__':
    main()
