"""penfold.py — 층2 결합기 벌점 구조 재검 (FOLD5 셋업) · 결합기만, 헤드는 dest_fold 저장 열 재사용.

dest_fold 가 남긴 헤드 열(Gt_*, W_*; 팔 BASE/FOLD5 × 시드 42,1,2,3,4)을 그대로 읽어 결합기만
다시 적합한다 — GPU 없음. 묻는 것 넷:

  1. tie 헤드 묶음(g1 g2) 격자에 0 포함            → dest_fold 부터 이미 그렇다 (LAMS0 = LAMS).
                                                     여기서도 유지하고, 선택 분포로 확인한다.
  2. 승부축도 tie 와 같은 2묶음 구조               → [λ_L × 로짓 | λ_g × W4], 둘 다 {0,…,1e4}.
                                                     지금은 로짓 λ=0 고정. 데이터가 0 을 고르면 동일.
  3. 형상 열                                         → 평균 arith 로짓에 m̄·|m̄| (홀) / t̄·|t̄| (짝)
                                                     1열을 더한다. FOLD5 의 이득이 "폴드 대비 = arith
                                                     형상 항"이면 이 1열이 그걸 회수해야 한다.
  4. 전문가별 격자 (xgb 는 폴드 5 = 전문가 5)        → 로짓 열마다 λ_j 를 격자에서 따로 고른다.
                                                     §3.7 에서 기각된 ard 는 1/w² 재가중(희소 사전)
                                                     이었고, 이건 좌표하강 격자 탐색이다: 2묶음 최적
                                                     에서 출발해 열 하나씩 격자를 훑어 안쪽 CV 가 내려
                                                     갈 때만 받는다 (2묶음 ⊆ 전문가별 이 구성상 보장).

벌점 체계 용어 (이 파일 안에서만)
    계수 w_j     열마다 하나, 데이터가 정한다 — 제약 없음
    묶음 λ       열 묶음(로짓 / 헤드)당 하나의 릿지 세기. 현행 = tie 2묶음, win 1묶음(로짓 0 고정)
    전문가별 λ_j 로짓 열마다 하나, 헤드는 묶음 유지

팔 (arith 열 · 헤드 · tie 결합기 · win 결합기)
    B1   평균 1    BASE   2묶음(현행)         로짓 0 고정 + W4 λ (현행)   ← dest_fold BASE 와 비트 일치 기대
    B2   평균 1    BASE   2묶음               2묶음 [λ_L | λ_g]
    S2   평균+형상 BASE   2묶음               2묶음
    BA   평균 1    BASE   전문가별            전문가별
    F1   폴드 5    FOLD5  2묶음(현행)         로짓 0 고정 + W4 λ (현행)   ← dest_fold FOLD5 와 비트 일치 기대
    F2   폴드 5    FOLD5  2묶음               2묶음
    FA   폴드 5    FOLD5  전문가별            전문가별
    SA   평균+형상 BASE   전문가별            전문가별
    BR   평균 1    BASE   ard                 ard            §3.7 의 1/w² 재가중 — 같은 셋업에서 재시도
    FR   폴드 5    FOLD5  ard                 ard

격자 {0, 1, 10, 100, 1e3, 1e4} — dest_fold 에서 1e5 는 0/75, 선택은 1~100 에 몰렸다.
판정: B1 대비 paired Δ (시드 5 평균의 행별 부트스트랩 CI) + 시드 부호. 단, arith 열과 stage2 행은
시드가 공유하므로 시드 부호는 헤드 상호작용만 검증한다 — 결합기 구조 자체의 증거는 CI 다.

실행:  cd final/temp && python penfold.py            (CPU 14워커 · 약 40~60분; 적합 1회 ≈ 0.2초 × 수만 회)
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')     # 워커당 BLAS 1스레드 — 프로세스 병렬과 충돌 방지
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
from sklearn.model_selection import StratifiedKFold

import dest_fold as DF

FINAL = DF.FINAL
GRID = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]          # 10배 간격 (§3.7 확정)
GRID_HALF = [0.0] + [10.0 ** (e / 2) for e in range(0, 9)]   # 반로그: 0, 1, 3.16, 10, …, 1e4 (10점) — §3.7 이유 ③ 재확인용
SWEEPS = 2
ARMS = {
    #        arith    heads    tie        win        설명 (출력에 그대로 쓴다)
    'B1': ('mean',  'BASE',  'cur',     'cur',     '평균 + 현행벌점'),
    'B2': ('mean',  'BASE',  'blk',     'blk',     '평균 + 2묶음(win도)'),
    'S2': ('shape', 'BASE',  'blk',     'blk',     '형상 + 2묶음'),
    'BA': ('mean',  'BASE',  'percol',  'percol',  '평균 + 전문가별격자'),
    'BR': ('mean',  'BASE',  'ard',     'ard',     '평균 + ard'),
    'F1': ('fold5', 'FOLD5', 'cur',     'cur',     '폴드5 + 현행벌점'),
    'F2': ('fold5', 'FOLD5', 'blk',     'blk',     '폴드5 + 2묶음(win도)'),
    'FA': ('fold5', 'FOLD5', 'percol',  'percol',  '폴드5 + 전문가별격자'),
    'FR': ('fold5', 'FOLD5', 'ard',     'ard',     '폴드5 + ard'),
    'SA': ('shape', 'BASE',  'percol',  'percol',  '형상 + 전문가별격자'),
    'BS': ('mean',  'BASE',  'spec',    'spec',    '평균 + 스펙트럼릿지'),
    'FS': ('fold5', 'FOLD5', 'spec',    'spec',    '폴드5 + 스펙트럼릿지'),
    # 묶음 분할 대안 — 전부 전수 격자 (≤4묶음 · 6^k 점). 2묶음(B2)과 같은 arith 열·같은 헤드.
    'G3': ('mean',  'BASE',  'grp:ar',   'grp:ar',   '평균 + 묶음:인코더|arith|헤드'),
    'GT': ('mean',  'BASE',  'grp:type', 'grp:type', '평균 + 묶음:타입별(TB|ABE|ar)'),
    'GG': ('mean',  'BASE',  'grp:gen',  'grp:gen',  '평균 + 묶음:세대별(1|2|ar)'),
    'GH': ('mean',  'BASE',  'grp:head', 'grp:head', '평균 + 묶음:헤드분리'),
}
# 묶음 정의 — 열 이름 집합의 리스트. 이름에 없는 열은 없어야 한다 (assert).
#   tie 열: s_tb2 s_tb1 t_abe2 t_abe1 t_ar | g1 g2        win 열: m_abe2 m_abe1 m_ar | abe2 tb2 abe1 tb1 (W4)
GROUPS = {
    'tie': {
        'ar':   [['s_tb2', 's_tb1', 't_abe2', 't_abe1'], ['t_ar'], ['g1', 'g2']],
        'type': [['s_tb2', 's_tb1'], ['t_abe2', 't_abe1'], ['t_ar'], ['g1', 'g2']],
        'gen':  [['s_tb1', 't_abe1'], ['s_tb2', 't_abe2'], ['t_ar'], ['g1', 'g2']],
        'head': [['s_tb2', 's_tb1', 't_abe2', 't_abe1', 't_ar'], ['g1'], ['g2']],
    },
    'win': {
        'ar':   [['m_abe2', 'm_abe1'], ['m_ar'], ['abe2', 'tb2', 'abe1', 'tb1']],
        'type': [['m_abe2', 'm_abe1'], ['m_ar'], ['abe2', 'abe1'], ['tb2', 'tb1']],
        'gen':  [['m_abe1'], ['m_abe2'], ['m_ar'], ['abe2', 'tb2', 'abe1', 'tb1']],
        'head': [['m_abe2', 'm_abe1', 'm_ar'], ['abe2', 'abe1'], ['tb2', 'tb1']],
    },
}
ALPHAS = [-1.0, -0.5, 0.0, 0.5, 1.0]   # 스펙트럼 릿지의 모양 — α=0 이 정확히 2묶음
LABEL = {a: f'{a}({v[4]})' for a, v in ARMS.items()}
ARD_IT, EPS = 5, 1e-8          # penlab 'ard' 규약 — 5회 재가중 · 전 열 적응
GLOSS = """  arith 열   평균   = 5폴드 마진 평균 1열 (현행 계약)
             폴드5  = 폴드별 5열 (평균 내지 않음)
             형상   = 평균 1열 + 그 제곱형 1열 (m̄·|m̄| / t̄·|t̄|)
  벌점       현행벌점     = tie [λ_L×로짓 | λ_g×헤드] · win 로짓 0 고정 + W4 λ   (노트북 구조)
             2묶음(win도) = win 에도 [λ_L×로짓 | λ_g×W4] — 로짓 λ 를 격자 {0..1e4} 에서 고름
             전문가별격자 = 로짓 열마다 λ_j 를 격자에서 따로 (2묶음 최적에서 좌표하강) · 헤드는 묶음
             ard          = §3.7 의 1/w² 재가중 (전 열 적응 · 5회) — 기각됐던 그 방법을 이 셋업에서 재시도
             스펙트럼릿지 = 로짓 블록을 PCA 기저로 돌려 성분 k 에 λ_k = λ·(s_k/ḡ)^(−α) — 방향은 공분산이 정하고
                            모양은 α 하나 (α=0 이 2묶음 · α>0 은 작은(차이) 방향을 더 세게) · 헤드는 λ_g
             묶음:*       = 2묶음의 분할을 바꾼 것 — 묶음마다 λ 하나, ≤4묶음이라 전수 격자(6^k)로 고른다
                            인코더|arith|헤드 · 타입별(TB|ABE|ar|헤드) · 세대별(1|2|ar|헤드) · 헤드분리(로짓|g1|g2 / 로짓|W4-ABE|W4-TB)
  헤드       BASE/FOLD5 = dest_fold 가 각 arith 계약으로 학습한 층1 헤드 열 (형상 팔은 BASE 헤드)"""


# ══════════════════════════════════════════ 일반 결합기 — 묶음 λ / 전문가별 λ
def _inner(ytr):
    inner = np.zeros(len(ytr), int)
    for j, (_, iv) in enumerate(StratifiedKFold(DF.INNER, shuffle=True, random_state=DF.SEED)
                                .split(np.zeros((len(ytr), 1)), ytr)):
        inner[iv] = j
    return inner


class _Inner:
    """한 바깥 fold 의 안쪽 CV 평가기 — 벌점 벡터를 주면 안쪽 OOF 손실 합을 돌려준다 (웜스타트 유지)."""

    def __init__(self, Xtr, ytr, bias, n_even):
        self.ytr, self.bias, self.n_even = ytr, bias, n_even
        self.inner = _inner(ytr)
        self.Z = [DF._z(Xtr[self.inner != j], Xtr[self.inner == j], n_even) for j in range(DF.INNER)]
        self.warm = [None] * DF.INNER
        self.cache = {}

    def loss(self, pen):
        key = tuple(np.round(pen, 12))
        if key in self.cache:
            return self.cache[key]
        tot = 0.0
        for j in range(DF.INNER):
            A, B = self.Z[j]
            v, _ = DF._fit(A, self.ytr[self.inner != j], pen, self.bias, self.warm[j])
            self.warm[j] = v
            tot += float(DF.rowloss(DF._scores(v, B, self.n_even, self.bias),
                                    self.ytr[self.inner == j]).sum())
        self.cache[key] = tot
        return tot

    def _rot(self, j, nl):
        """안쪽 fold j 의 로짓 블록을 학습행 PCA 기저로 돌린 (A_rot, B_rot, s_norm). 라벨은 안 본다."""
        if not hasattr(self, 'rot'):
            self.rot, self.warm_spec = {}, [None] * DF.INNER
        if (j, nl) not in self.rot:
            A, B = self.Z[j]
            self.rot[(j, nl)] = _rotate(A, B, nl)
        return self.rot[(j, nl)]

    def loss_spec(self, nl, lam, alpha, lam_g):
        key = ('spec', nl, float(lam), float(alpha), float(lam_g))
        if key in self.cache:
            return self.cache[key]
        tot = 0.0
        for j in range(DF.INNER):
            A, B, s_norm, _ = self._rot(j, nl)
            pen = _pen_spec(A.shape[1], nl, s_norm, lam, alpha, lam_g)
            v, _ = DF._fit(A, self.ytr[self.inner != j], pen, self.bias, self.warm_spec[j])
            self.warm_spec[j] = v
            tot += float(DF.rowloss(DF._scores(v, B, self.n_even, self.bias),
                                    self.ytr[self.inner == j]).sum())
        self.cache[key] = tot
        return tot

    def loss_ard(self, q0):
        key = ('ard', float(q0))
        if key in self.cache:
            return self.cache[key]
        tot = 0.0
        for j in range(DF.INNER):
            A, B = self.Z[j]
            v, _ = _ard_fit(A, self.ytr[self.inner != j], A.shape[1], q0, self.bias, self.warm[j])
            self.warm[j] = v
            tot += float(DF.rowloss(DF._scores(v, B, self.n_even, self.bias),
                                    self.ytr[self.inner == j]).sum())
        self.cache[key] = tot
        return tot


def _ard_fit(A, y, p, q0, bias, w0):
    """penlab 'ard': 균일 q0 에서 시작해 pen_j ∝ q0 / w_j² 로 5회 재가중 (평균이 q0 가 되게 정규화)."""
    pen = np.full(p, float(q0))
    v = w0
    for _ in range(ARD_IT):
        v, _ = DF._fit(A, y, pen, bias, v)
        inv = 1.0 / (v[:p] ** 2 + EPS)
        pen = q0 * inv / inv.mean()
    return v, pen


def _rotate(A, B, nl):
    """로짓 블록(앞 nl 열)을 A 의 PCA 기저로 회전. 반환 (A_rot, B_rot, s_norm, V).
    s_norm = 성분 표준편차 / 기하평균 — λ 의 눈금이 α 에 따라 흔들리지 않게 정규화한다."""
    U_, S_, Vt = np.linalg.svd(A[:, :nl], full_matrices=False)
    V = Vt.T                                   # (nl, nl)
    s = S_ / np.sqrt(A.shape[0])
    s_norm = s / np.exp(np.log(np.clip(s, 1e-12, None)).mean())
    A2, B2 = A.copy(), B.copy()
    A2[:, :nl] = A[:, :nl] @ V
    B2[:, :nl] = B[:, :nl] @ V
    return A2, B2, s_norm, V


def _pen_spec(p, nl, s_norm, lam, alpha, lam_g):
    """성분 k 벌점 λ·(s_k/ḡ)^(−α). α=0 → 전 성분 λ (= 2묶음, 릿지는 회전 불변)."""
    return np.concatenate([lam * s_norm ** (-alpha), np.full(p - nl, lam_g)])


def spec_search(ev, p, nl):
    """(λ, α, λ_g) 전수 격자 — 6 × 5 × 6 = 180점. α 바깥, λ 큰 것부터 (웜스타트)."""
    best = None
    for alpha in ALPHAS:
        for lam in sorted(GRID, reverse=True):
            for lg in sorted(GRID, reverse=True):
                sc = ev.loss_spec(nl, lam, alpha, lg)
                if best is None or sc < best[0]:
                    best = (sc, (lam, alpha, lg))
    return best[1]


def _pen_groups(p, groups, lams):
    pen = np.full(p, np.nan)
    for g, lam in zip(groups, lams):
        pen[g] = lam
    assert np.isfinite(pen).all(), 'every column must belong to exactly one group'
    return pen


def group_search(ev, p, groups):
    """묶음별 λ 전수 격자 — 6^len(groups) 점, 큰 λ 부터 (웜스타트)."""
    import itertools
    best = None
    for lams in itertools.product(sorted(GRID, reverse=True), repeat=len(groups)):
        sc = ev.loss(_pen_groups(p, groups, lams))
        if best is None or sc < best[0]:
            best = (sc, lams)
    return best[1]


def _pen_blk(p, nl, q):
    return np.concatenate([np.full(nl, q[0]), np.full(p - nl, q[1])])


def blk_search(ev, p, nl):
    """2묶음 [λ_L × 로짓 nl | λ_g × 나머지] 전수 격자 — 큰 λ 부터 (웜스타트 순서는 DF 와 동일)."""
    best = None
    for a in sorted(GRID, reverse=True):
        for b in sorted(GRID, reverse=True):
            s = ev.loss(_pen_blk(p, nl, (a, b)))
            if best is None or s < best[0]:
                best = (s, (a, b))
    return best[1]


def percol_search(ev, p, nl, q0):
    """전문가별: 로짓 열마다 λ_j, 헤드는 묶음 하나. 2묶음 최적 q0 에서 출발하는 좌표하강 격자."""
    pen = _pen_blk(p, nl, q0)
    cur = ev.loss(pen)
    for _ in range(SWEEPS):
        moved = False
        for j in list(range(nl)) + ['head']:
            best_l, best_s = None, cur
            for lam in GRID:
                trial = pen.copy()
                if j == 'head':
                    trial[nl:] = lam
                else:
                    trial[j] = lam
                s = ev.loss(trial)
                if s < best_s - 1e-9:
                    best_l, best_s = lam, s
            if best_l is not None:
                if j == 'head':
                    pen[nl:] = best_l
                else:
                    pen[j] = best_l
                cur, moved = best_s, True
        if not moved:
            break
    return pen


def combiner(X, y01, fold, rows, bias, nl, kind, groups=None):
    """kind 'blk' | 'percol' | 'ard'. 반환 (OOF 점수, dict(lam=배포 벌점벡터, picked=[fold별 벌점벡터]))."""
    p = X.shape[1]
    n_even = p if bias else 0
    z = np.full(len(y01), np.nan)
    picked, q0s = [], []
    for k in range(DF.K):
        tr, te = (fold != k) & rows, (fold == k) & rows
        Xtr, ytr = X[tr], y01[tr]
        ev = _Inner(Xtr, ytr, bias, n_even)
        A, B = DF._z(Xtr, X[te], n_even)
        if kind == 'ard':
            q0 = min(sorted(GRID, reverse=True), key=lambda q: ev.loss_ard(q))
            v, pen = _ard_fit(A, ytr, p, q0, bias, None)
            q0s.append(q0)
        elif kind == 'grp':
            lams = group_search(ev, p, groups)
            pen = _pen_groups(p, groups, lams)
            v, _ = DF._fit(A, ytr, pen, bias)
            q0s.append(list(lams))
        elif kind == 'spec':
            q = spec_search(ev, p, nl)
            A, B, s_norm, _ = _rotate(A, B, nl)
            pen = _pen_spec(p, nl, s_norm, *q)
            v, _ = DF._fit(A, ytr, pen, bias)
            q0s.append(q)                                  # (λ, α, λ_g) — picked 에는 벌점벡터
        else:
            q = blk_search(ev, p, nl)
            pen = _pen_blk(p, nl, q) if kind == 'blk' else percol_search(ev, p, nl, q)
            v, _ = DF._fit(A, ytr, pen, bias)
        picked.append(pen)
        z[te] = DF._scores(v, B, n_even, bias)
    A, _ = DF._z(X[rows], X[rows], n_even)
    if kind == 'ard':                                      # 배포 = 중앙값 q0 로 전 행에서 재가중
        v, dep = _ard_fit(A, y01[rows], p, float(np.median(q0s)), bias, None)
        ok = True
    elif kind == 'spec':                                   # 배포 = (λ, α, λ_g) 성분별 중앙값, 전 행 기저
        qd = tuple(float(np.median([q[i] for q in q0s])) for i in range(3))
        A, _, s_norm, V = _rotate(A, A, nl)
        dep = _pen_spec(p, nl, s_norm, *qd)
        v, ok = DF._fit(A, y01[rows], dep, bias)
        v = np.concatenate([V @ v[:nl], v[nl:]])           # 가중치는 원 기저로 되돌려 저장
        q0s = [list(q) for q in q0s] + [['deploy', *qd]]
    else:
        dep = np.median(np.array(picked), 0)               # 배포 = 성분별 중앙값 (destnation 규약)
        v, ok = DF._fit(A, y01[rows], dep, bias)
    assert ok, 'deploy fit did not converge'
    return z, dict(lam=np.asarray(dep).tolist(), picked=[np.asarray(q).tolist() for q in picked],
                   v=[float(x) for x in v],
                   q0=(q0s if kind in ('spec', 'grp') else [float(q) for q in q0s]))


# ══════════════════════════════════════════ 워커 — 자산은 프로세스마다 한 번 만든다
_W = {}


def _assets(src):
    if 'U' in _W:
        return _W
    global GRID
    if os.environ.get('PENFOLD_GRID') == 'half':
        GRID = list(GRID_HALF)
        DF.LAMS = DF.LAMS0 = list(GRID)
    else:
        DF.set_grid(1e4)
    U = DF.build()
    fid, M = DF.arith_fold_logits()
    i_s2 = fid.get_indexer(U['ids'])
    assert (i_s2 >= 0).all(), 'arith id 커버리지'
    t5, m5 = DF.arith_cols(M[:, i_s2], 'FOLD5')
    t1, m1 = DF.arith_cols(M[:, i_s2], 'BASE')
    _W.update(U=U, HEADS=np.load(FINAL / f'{src}.rowloss.npz'),
              AR={'mean': (t1, m1), 'fold5': (t5, m5),
                  'shape': (np.hstack([t1, t1 * np.abs(t1)]), np.hstack([m1, m1 * np.abs(m1)]))})
    return _W


def fit_key(key, src):
    """(axis, arith, heads, kind, seed) 하나를 적합. 반환 (key, z, dep, 행별 holdout 손실)."""
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):          # 워커의 적재 메시지는 삼킨다
        W = _assets(src)
    U, HEADS, AR = W['U'], W['HEADS'], W['AR']
    axis, arith, heads, kind, sd = key
    y, fold, cu, nz, hold = U['y'], U['fold'], U['cu'], U['nz'], U['hold']
    t2, m2 = AR[arith]
    def _groups(cols):
        name = kind.split(':', 1)[1]
        idx = {c: i for i, c in enumerate(cols)}
        groups = [[idx[c] for c in g] for g in GROUPS[axis][name]]
        assert sorted(i for g in groups for i in g) == list(range(len(cols))), f'grouping {name} does not cover {cols}'
        return groups
    if axis == 'tie':
        X = np.hstack([U['enc_tie'], t2, HEADS[f'Gt_{heads}-s{sd}']])
        nl, yy, rows, bias = 4 + t2.shape[1], (y == 2).astype(np.float64), cu, True
        if kind in ('cur', 'blk'):            # tie 는 현행이 이미 2묶음(0 포함) — 같은 코드 경로
            z, dep = DF.tie_combiner(X, yy, fold, rows, nl)
            dep = dict(lam=_pen_blk(X.shape[1], nl, dep['lam']).tolist(),
                       picked=[_pen_blk(X.shape[1], nl, q).tolist() for q in dep['picked']], v=dep['v'])
        elif kind.startswith('grp:'):
            cols = ['s_tb2', 's_tb1', 't_abe2', 't_abe1'] + (['t_ar'] if t2.shape[1] == 1 else [f't_ar{k}' for k in range(1, 6)]) + ['g1', 'g2']
            z, dep = combiner(X, yy, fold, rows, bias, nl, 'grp', groups=_groups(cols))
        else:
            z, dep = combiner(X, yy, fold, rows, bias, nl, kind)
    else:
        X = np.hstack([U['enc_win'], m2, HEADS[f'W_{heads}-s{sd}']])
        nl, yy, rows, bias = 2 + m2.shape[1], (y == 0).astype(np.float64), cu & nz, False
        if kind == 'cur':
            z, dep = DF.win_combiner(X, yy, fold, rows, nl)
            dep = dict(lam=_pen_blk(X.shape[1], nl, (0.0, dep['lam'])).tolist(),
                       picked=[_pen_blk(X.shape[1], nl, (0.0, q)).tolist() for q in dep['picked']], v=dep['v'])
        elif kind.startswith('grp:'):
            cols = ['m_abe2', 'm_abe1'] + (['m_ar'] if m2.shape[1] == 1 else [f'm_ar{k}' for k in range(1, 6)]) + list(DF.SRCS)
            z, dep = combiner(X, yy, fold, rows, bias, nl, 'grp', groups=_groups(cols))
        else:
            z, dep = combiner(X, yy, fold, rows, bias, nl, kind)
    return key, z, dep, DF.rowloss(z[rows & hold], yy[rows & hold])


# ══════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seeds', default='42,1,2,3,4')
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--src', default='temp/dest_fold')
    ap.add_argument('--out', default='temp/penfold')
    ap.add_argument('--jobs', type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument('--grid', default='dec', choices=['dec', 'half'], help='dec = 10배 간격 (기본) · half = 반로그 10점')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s]
    arms = [a for a in args.arms.split(',') if a]
    t_all = time.time()
    global GRID
    if args.grid == 'half':
        GRID = list(GRID_HALF)
        DF.LAMS = DF.LAMS0 = list(GRID)            # DF 의 cur/tie 결합기도 같은 격자를 쓰게
    else:
        DF.set_grid(1e4)
    print(f'[격자] {[f"{g:.3g}" for g in GRID]}  (DF 결합기도 같은 격자)')

    print('[적재]')
    U = DF.build()
    hold = U['hold']
    names = {'mean': (['t_ar'], ['m_ar']), 'fold5': ([f't_ar{k}' for k in range(1, 6)], [f'm_ar{k}' for k in range(1, 6)]),
             'shape': (['t_ar', 't_ar·|t|'], ['m_ar', 'm_ar·|m|'])}
    src_json = FINAL / f'{args.src}.json'
    REF = json.load(open(src_json))['cells'] if src_json.exists() else {}

    # 고유 (축, arith, 헤드, 결합기, 시드) 만 적합 — 팔이 달라도 같은 키면 한 번
    # tie 'blk' 는 'cur' 와 같은 경로라 키를 합친다
    norm = lambda axis, kind: 'cur' if (axis == 'tie' and kind == 'blk') else kind
    keys = sorted({(axis, ARMS[a][0], ARMS[a][1], norm(axis, ARMS[a][2 if axis == 'tie' else 3]), sd)
                   for a in arms for sd in seeds for axis in ('tie', 'win')})
    print(f'  고유 적합 {len(keys)}개 (팔 {len(arms)} × 시드 {len(seeds)} × 2축 = {2 * len(arms) * len(seeds)} 칸) '
          f'· 워커 {args.jobs}')
    t0 = time.time()
    cache = {}
    os.environ['PENFOLD_GRID'] = args.grid                 # 워커 프로세스에 격자 전달
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for n, (key, z, dep, ll) in enumerate(ex.map(fit_key, keys, [args.src] * len(keys), chunksize=1), 1):
            cache[key] = (z, dep, ll)
            print(f'    [{n:3d}/{len(keys)}] {key[0]} {key[1]:6s} {key[2]:6s} {key[3]:7s} s{key[4]:<3d} '
                  f'hold {ll.mean():.5f}  [{(time.time() - t0) / 60:.1f}분]', flush=True)

    res, roww = {}, {}
    for arm in arms:
        arith, heads, kt, kw = ARMS[arm][:4]
        for sd in seeds:
            tag = f'{arm}-s{sd}'
            z_t, dep_t, lt = cache[('tie', arith, heads, norm('tie', kt), sd)]
            z_w, dep_w, lw = cache[('win', arith, heads, norm('win', kw), sd)]
            row, tie_ll, win_ll = DF.score(U, z_t, z_w, hold)
            res[tag] = dict(arm=arm, seed=sd, tot=float(row.mean()), tie=tie_ll, win=win_ll,
                            tie_lam=dep_t['lam'], win_lam=dep_w['lam'],
                            tie_picked=dep_t['picked'], win_picked=dep_w['picked'],
                            tie_w=dep_t['v'], win_w=dep_w['v'],
                            tie_q0=dep_t.get('q0', []), win_q0=dep_w.get('q0', []))
            roww[f'row_{tag}'], roww[f'tie_{tag}'], roww[f'win_{tag}'] = row, lt, lw
            line = f'  [{LABEL[arm]:22s} s{sd:<3d}] tie {tie_ll:.5f} · win {win_ll:.5f} → {row.mean():.5f}'
            ref = {'B1': 'BASE', 'F1': 'FOLD5'}.get(arm)
            if ref and f'{ref}-s{sd}' in REF:
                r = REF[f'{ref}-s{sd}']
                d = max(abs(tie_ll - r['tie']), abs(win_ll - r['win']))
                line += f'   | dest_fold {ref} 대조 {d:.1e} ' + ('비트 일치' if d < 1e-6 else '*** 불일치')
            print(line)

    op = FINAL / args.out
    with open(op.with_suffix('.json'), 'w') as f:
        json.dump({'cells': res, 'seeds': seeds, 'arms': {a: ARMS[a] for a in arms}, 'grid': GRID,
                   'names': {k: dict(tie=['s_tb2', 's_tb1', 't_abe2', 't_abe1'] + v[0] + ['g1', 'g2'],
                                     win=['m_abe2', 'm_abe1'] + v[1] + list(DF.SRCS)) for k, v in names.items()}},
                  f, ensure_ascii=False, indent=1)
    np.savez_compressed(op.with_suffix('.rowloss.npz'), **roww)
    print(f'\n저장: {op}.json / {op}.rowloss.npz')

    # ── 요약
    base = 'B1' if 'B1' in arms else arms[0]
    print(f'\n{"=" * 100}\n[요약] {len(seeds)}시드 평균 · holdout 5,748 · Δ = 팔 − {LABEL[base]} (= 현행 노트북 구조)'
          f'\n       CI = tot Δ 의 행별 부트스트랩 95% (★ 0 을 벗어남) · 시드 = Δtot<0 인 시드 수\n{"=" * 100}')
    print(GLOSS)
    print(f'\n  {"팔":24s}│{"tie":>9s}{"win":>9s}{"tot":>9s} │{"Δtie":>9s}{"Δwin":>9s}{"Δtot":>9s}  {"CI(tot)":>20s}  시드')
    rng = np.random.default_rng(DF.SEED)
    mean = lambda a, k: np.mean([res[f'{a}-s{s}'][k] for s in seeds])
    for arm in arms:
        arith, heads, kt, kw = ARMS[arm][:4]
        d_tot = np.mean([roww[f'row_{arm}-s{s}'] for s in seeds], 0) - np.mean([roww[f'row_{base}-s{s}'] for s in seeds], 0)
        bs = d_tot[rng.integers(0, len(d_tot), size=(2000, len(d_tot)))].mean(1)
        lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        per = np.array([res[f'{arm}-s{s}']['tot'] - res[f'{base}-s{s}']['tot'] for s in seeds])
        star = '★' if (lo > 0 or hi < 0) else ' '
        print(f'  {LABEL[arm]:24s}│{mean(arm, "tie"):9.5f}{mean(arm, "win"):9.5f}{mean(arm, "tot"):9.5f} │'
              f'{mean(arm, "tie") - mean(base, "tie"):+9.5f}{mean(arm, "win") - mean(base, "win"):+9.5f}{d_tot.mean():+9.5f}  '
              f'[{lo:+.5f},{hi:+.5f}]{star}  {int((per < 0).sum())}/{len(seeds)}')

    print(f'\n[선택된 λ — 묶음/열별 · 바깥 fold × 시드 전부 모아 격자값별 빈도]')
    for arm in arms:
        arith, heads, kt, kw = ARMS[arm][:4]
        for axis, kind, key in (('tie', kt, 'tie_picked'), ('win', kw, 'win_picked')):
            P = np.array([q for s in seeds for q in res[f'{arm}-s{s}'][key]])     # (folds×seeds, p)
            nl = len(names[arith][0 if axis == 'tie' else 1]) + (4 if axis == 'tie' else 2)
            cols = (['s_tb2', 's_tb1', 't_abe2', 't_abe1'] + names[arith][0] if axis == 'tie'
                    else ['m_abe2', 'm_abe1'] + names[arith][1])
            if kind.startswith('grp:'):
                gdef = GROUPS[axis][kind.split(':', 1)[1]]
                Q = [q for s_ in seeds for q in res[f'{arm}-s{s_}'][f'{axis}_q0']]
                desc = '  '.join(f'[{"+".join(g)}] ' + str({f'{v:g}': sum(1 for q in Q if q[i] == v) for v in GRID if any(q[i] == v for q in Q)})
                                 for i, g in enumerate(gdef))
            elif kind == 'spec':
                Q = [q for s_ in seeds for q in res[f'{arm}-s{s_}'][f'{axis}_q0'] if q[0] != 'deploy']
                fq = lambda i, vals: {f'{g:g}': sum(1 for q in Q if q[i] == g) for g in vals if any(q[i] == g for q in Q)}
                desc = f'λ {fq(0, GRID)}  α {fq(1, ALPHAS)}  λ_g {fq(2, GRID)}'
            elif kind in ('percol', 'ard'):
                desc = ' '.join(f'{c}:{np.median(P[:, j]):.3g}' for j, c in enumerate(cols))
                desc += f'  head:{np.median(P[:, nl:]):.3g}'
                if kind == 'ard':
                    q = [x for s_ in seeds for x in res[f'{arm}-s{s_}'].get(f'{axis}_q0', [])]
                    desc += f'   (q0 선택 {sorted(set(q))})'
            else:
                fl = lambda col: {f'{g:g}': int((P[:, col] == g).sum()) for g in GRID if (P[:, col] == g).any()}
                desc = f'logit {fl(0)}  head {fl(nl)}'
            print(f'  {LABEL[arm]:24s} {axis}  {desc}')
    print(f'\n총 {(time.time() - t_all) / 60:.1f}분')


if __name__ == '__main__':
    main()
