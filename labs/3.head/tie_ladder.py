#!/usr/bin/env python
"""tie_ladder.py — tie축 열 사다리. **결합기만** 다시 적합한다. 자립 모듈.

무엇을 재나
    tie 결합기 12열 = [로짓 5 무벌점] + [기둥 4] + [GAT 3] 중
    **기둥 블록**과 **GAT 라운드 수**를 끄고 켜며 holdout 총 ll 을 paired CI 로 비교한다.

왜 재학습이 필요 없나
    기둥열·GAT열·LIN열은 전부 `deploy.npz` 에 **stage2 10,748행 완성본**으로 저장돼 있다.
    이 열들은 train 46,729행에서 학습된 결과물이고, 어떤 부분집합을 결합기에 넣느냐는
    그 학습에 전혀 영향을 주지 않는다. 그래서 SVD·기둥·GAT·LIN 을 다시 돌릴 이유가 없다.

    부스팅 체인은 **접두사 일관**이다 — 라운드 k+1 은 라운드 k 의 누적 점수만 받으므로,
    "2라운드에서 멈춘 모델" = 지금 저장된 g1,g2 그대로다. 잘라 쓰는 것이 정확하다.
    (그래서 GAT 축은 접두사만 본다. g1 을 빼고 g2 만 쓰는 조합은 실현 가능한 모델이
     아니므로 격자에 없다.)

    승부축은 tie축 열 선택과 완전히 독립이다(결합기가 축별로 따로 적합된다). 그래서
    승부축 결합기는 **한 번만** 적합하고 전 칸이 그 행별 손실을 공유한다.

격자 (주 설계, 4 x 2 = 8칸)
                     기둥 없음        기둥 4열
    GAT 없음            L0              P          <- L0 = 로짓 5열만 (순수 기준선)
    g1                  G1             P+G1
    g1 g2               G2             P+G2
    g1 g2 g3            G3             P+G3        <- 현행 배포 = 기준 칸

진단 칸 (--loo, 4칸 추가)
    P+G3 에서 기둥을 하나씩 뺀다 — 기둥 4열 중 무엇이 일하는지.
    ! 주 설계와 달리 다중비교 보정이 없다. 방향 참고용이지 판정용이 아니다.

판정
    Δ = 칸 - 기준(P+G3).  **양수면 그 칸이 나쁘다**(손실이므로).
    paired bootstrap 95% CI + 문턱 0.0005.

--calib — 사후 보정을 각 칸 위에 얹는다
    묻는 것: **표현 7열이 하는 일이 순위 변경인가, 재보정인가.**
    보정기는 정의상 점수의 함수라 순위를 못 바꾼다. 그러므로
        calib(p0_g0) 이 p4_g3 에 닿는다  ->  표현의 값어치가 전부 재보정이다
                                            (= 8,192차 풀링을 버리고 보정기로 대체 가능)
        닿지 않는다                       ->  표현이 순위 정보를 나른다 (대체 불가)
    같이 나오는 것: calib(p4_g3) 이 p4_g3 를 이기면 헤드 출력에 보정 여지가 남아 있다.
    (기록의 보정 기각은 Boost/ATR 기저에서 잰 것이라 이 헤드에는 적용된 적이 없다.)

    보정기 사다리 — 축별로 적용, **승부축은 절편 금지**(반대칭이 깨진다):
        none              기준
        T      (2par)     축별 스칼라  tie: a·z        win: a·z
        platt  (3par)     축별 아핀    tie: a·z + b    win: a·z   (절편은 tie 에만)
        iso    (비모수)   축별 isotonic. 승부축은 (z,y) 에 (−z,1−y) 를 덧붙여 반대칭 강제
    보정기도 **같은 5-fold 로 교차적합**한다 — 같은 행에서 고르고 재면 isotonic 이 외운다
    (기록: 자기적합 0.9717 vs 교차적합 0.9866).

사용
    cd final
    python -u temp/tie_ladder.py 2>&1 | tee temp/log_tie_ladder.txt
    python -u temp/tie_ladder.py --loo --calib 2>&1 | tee temp/log_tie_ladder.txt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

# ══════════════════════════════════════════════ 경로·계약 상수 (fit_heads.py 와 동일)
FINAL = Path(__file__).resolve().parent.parent      # 이 파일은 final/temp/
DATA_DIR = FINAL / 'origin'
MODELS = FINAL / 'models'
ARITH = MODELS / 'ARITH'

SEED = 42
HALF = 5000                     # stage2 우주 경계: [:HALF] valid / [HALF:] holdout
K, INNER = 5, 3                 # 결합기 교차적합 fold / 내부 λ 선택 fold
ABE1_STEP, TB1_STEP = 3109, 5000
LAMS = [1.0, 10.0, 100.0, 1000.0, 10000.0]   # 결합기 λ 격자 (crossfit 이 위로 자동확장)
LAM_CEIL = 1e12
MAXIT = 1000

REF_HOLD = 0.95970              # factorial_heads 의 tr0_p4_r3. 기준 칸이 이걸 내야 한다.
REF_TOL = 5e-4                  # 재구성 오차 허용치 (열은 같은 번들에서 왔으므로 넉넉)

TIE_NAMES = ['s_tb2', 's_tb1', 't_abe2', 't_abe1', 't_ar']
WIN_NAMES = ['m_abe2', 'm_abe1', 'm_ar']
PIL_NAMES = ['p0·tb2', 'p1·abe2', 'p2·abe1', 'p3·tb1']    # _fit_pillar 의 src 순서
GAT_NAMES = ['g1', 'g2', 'g3']


# ══════════════════════════════════════════════ 원본: fit_heads.py 발췌 (그대로)
def rowloss(z, y01):
    return np.logaddexp(0, z) - y01 * z


def ci_str(m, lo, hi):
    return f'{m:+.5f} [{lo:+.5f},{hi:+.5f}] ' + ('유의 ★' if lo > 0 or hi < 0 else '노이즈')


def paired_ci(a, b, n_boot=2000, seed=SEED):
    """a - b 의 부트스트랩 95% CI. 손실이므로 양수면 b 가 낫다."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _fit(X, y01, pen, bias, w0=None):
    """목적함수 = mean(logistic loss) + (1/n)·Σ pen_j·w_j².

    pen_j = 0 이면 그 열은 조건수를 건드리지 않고 정확히 무벌점이 된다.
    반환: (파라미터 벡터, 수렴 성공 여부). bias 면 마지막 원소가 절편이다.
    """
    n, p = X.shape
    pen = np.asarray(pen, np.float64)

    def fg(v):
        w = v[:p]
        z = X @ w
        if bias:
            z = z + v[p]
        val = float(np.mean(np.logaddexp(0, z) - y01 * z) + np.dot(pen * w, w) / n)
        r = (expit(z) - y01) / n
        gw = X.T @ r + 2.0 * pen * w / n
        return val, (np.concatenate([gw, [r.sum()]]) if bias else gw)

    v0 = np.zeros(p + (1 if bias else 0)) if w0 is None else w0
    res = minimize(fg, v0, jac=True, method='L-BFGS-B',
                   options={'maxiter': MAXIT, 'maxcor': 20, 'ftol': 1e-12, 'gtol': 1e-7})
    return res.x, bool(res.success)


def _z(Xtr, Xte, n_even):
    """학습 fold 통계로 정규화. **홀 열은 중심화하지 않는다.**

    홀 열에서 표본평균을 빼면 상수항이 생겨 절편을 몰래 넣는 것과 같고, 그 순간
    반대칭이 깨진다. 그래서 홀 열은 중심화 없이 RMS 로만 나눈다.
    열 배치 규약: [0:n_even) 짝, [n_even:) 홀.
    """
    A, B = Xtr.copy(), Xte.copy()
    if n_even:
        mu = A[:, :n_even].mean(0)
        sd = np.clip(A[:, :n_even].std(0), 1e-9, None)
        A[:, :n_even] = (A[:, :n_even] - mu) / sd
        B[:, :n_even] = (B[:, :n_even] - mu) / sd
    if n_even < A.shape[1]:
        rms = np.clip(np.sqrt((A[:, n_even:] ** 2).mean(0)), 1e-9, None)
        A[:, n_even:] /= rms
        B[:, n_even:] /= rms
    return A, B


def _scores(v, B, n_even, bias):
    p = B.shape[1]
    w = v[:p]
    b = float(v[p]) if bias else 0.0
    ze = B[:, :n_even] @ w[:n_even] if n_even else 0.0
    zo = B[:, n_even:] @ w[n_even:] if n_even < p else 0.0
    return ze + zo + b


def _penvec(p, n_logit, lam):
    """로짓 열은 무벌점(0), 표현 블록만 lam."""
    pen = np.full(p, float(lam))
    if n_logit:
        pen[:n_logit] = 0.0
    return pen


def crossfit(X, y01, fold, bias, lams, rows=None, n_even=0, n_logit=0):
    """바깥 5-fold 교차적합 + 안쪽 3-fold λ 선택. 점수(로그오즈)를 돌려준다.

    λ 는 큰 값부터 훑으며 웜스타트한다. 고른 λ 가 격자 최댓값이면 x10 씩 계속 늘린다
    (최적점이 격자 밖이면 "표현이 해롭다"가 아니라 "규제를 덜 걸었다"가 되므로).
    """
    m = np.ones(len(y01), bool) if rows is None else rows
    z = np.full(len(y01), np.nan)
    picked, n_bad, n_ext = [], 0, 0
    for k in range(K):
        tr, te = (fold != k) & m, (fold == k) & m
        Xtr, ytr = X[tr], y01[tr]
        if len(lams) > 1:
            inner = np.zeros(len(ytr), int)
            for j, (_, iv) in enumerate(StratifiedKFold(INNER, shuffle=True, random_state=SEED)
                                        .split(np.zeros((len(ytr), 1)), ytr)):
                inner[iv] = j
            Zi = [_z(Xtr[inner != j], Xtr[inner == j], n_even) for j in range(INNER)]
            score = {}

            def eval_lam(lam, warm):
                nb, tot = 0, 0.0
                for j in range(INNER):
                    A, B = Zi[j]
                    v, ok = _fit(A, ytr[inner != j], _penvec(A.shape[1], n_logit, lam), bias,
                                 warm[j])
                    warm[j] = v
                    nb += (not ok)
                    tot += rowloss(_scores(v, B, n_even, bias), ytr[inner == j]).sum()
                score[lam] = tot
                return nb

            warm = [None] * INNER
            for lam in sorted(set(lams), reverse=True):
                n_bad += eval_lam(lam, warm)
            e = 0
            while min(score, key=score.get) == max(score) and max(score) < LAM_CEIL:
                warm = [None] * INNER
                n_bad += eval_lam(max(score) * 10, warm)
                e += 1
            n_ext = max(n_ext, e)
            lam = min(score, key=score.get)
        else:
            lam = lams[0]
        picked.append(lam)
        A, B = _z(Xtr, X[te], n_even)
        v, ok = _fit(A, ytr, _penvec(A.shape[1], n_logit, lam), bias)
        n_bad += (not ok)
        z[te] = _scores(v, B, n_even, bias)
    return z, picked, n_bad, n_ext


def combiner(X, y01, fold, rows, bias, n_logit):
    """결합기 1개 적합. 반환 (OOF 점수, 배포 단일적합 dict)."""
    n_even = X.shape[1] if bias else 0
    # 전 열이 무벌점이면(로짓만 있는 칸) λ 가 목적함수에 안 들어간다 — 격자를 훑을 이유가
    # 없고, 전 λ 의 내부CV 가 동률이라 자동확장 조건까지 헛돌게 된다. 한 점으로 고정한다.
    lams = LAMS if n_logit < X.shape[1] else [LAMS[0]]
    z, picked, n_bad, n_ext = crossfit(X, y01, fold, bias, lams,
                                       rows=rows, n_even=n_even, n_logit=n_logit)
    lam = sorted(picked)[len(picked) // 2]        # 배포 단일 적합의 λ = fold 중앙값
    Az, _ = _z(X[rows], X[rows], n_even)
    v, ok = _fit(Az, y01[rows], _penvec(X.shape[1], n_logit, lam), bias)
    return z, dict(v=v, lam=float(lam), picked=[float(p) for p in picked],
                   n_bad=int(n_bad), n_ext=int(n_ext), ok=bool(ok))


# ══════════════════════════════════════════════ 우주 재구성
sd3 = lambda B: (B[:, 0] - B[:, 1], B[:, 2] - np.logaddexp(B[:, 0], B[:, 1]))


def _v3_split():
    """V3(TT_run) 분할 재현 — seed 42 두 번. 라벨 3열만 읽어도 분할이 정확히 같다."""
    full = pd.read_csv(DATA_DIR / 'train_full.csv',
                       usecols=['id', 'winner_model_a', 'winner_model_b'])
    assert len(full) == 57477, f'train_full.csv {len(full)}행 != 57,477'
    lab = np.select([full['winner_model_a'] == 1, full['winner_model_b'] == 1], [0, 1], 2)
    tr51, te = train_test_split(full.assign(label=lab), test_size=0.10, stratify=lab,
                               random_state=42, shuffle=True)
    _, va2 = train_test_split(tr51, test_size=2000, stratify=tr51['label'],
                              random_state=42, shuffle=True)
    return te, va2


def _tta(rel, n_class):
    z = np.load(MODELS / rel)
    L = np.vstack([(z['valid_fwd'] + z['valid_swp']) / 2,
                   (z['hold_fwd'] + z['hold_swp']) / 2]).astype(np.float64)
    assert L.shape[1] == n_class, f'{rel} 열 {L.shape[1]} != {n_class}'
    return L, z


def build(bundle):
    """저장된 열 + 로짓 기저 + 행 우주를 모은다."""
    L_abe2, za = _tta('logits/abe2__checkpoint-3000.npz', 3)
    ids = np.concatenate([za['valid_ids'], za['hold_ids']])
    y = np.concatenate([za['valid_labels'], za['hold_labels']]).astype(np.int64)
    n = len(y)
    assert n == 10748 and len(np.unique(ids)) == n, f'stage2 {n}행'
    hold = np.arange(n) >= HALF
    nz = y != 2

    L_tb2, zt2 = _tta('logits/tb2__checkpoint-4000.npz', 2)
    L_abe1, za1 = _tta(f'logits/abe1__checkpoint-{ABE1_STEP}.npz', 3)
    L_tb1, zt1 = _tta(f'logits/tb1__checkpoint-{TB1_STEP}.npz', 2)
    for nm, zz in (('tb2', zt2), ('abe1', za1), ('tb1', zt1)):
        assert (np.concatenate([zz['valid_ids'], zz['hold_ids']]) == ids).all(), f'{nm} id 정렬'

    zar = np.load(ARITH / 'outputs/arith_experts_raw.npz')
    assert (np.concatenate([zar['ids_valid'], zar['ids_hold']]) == ids).all(), 'arith id 정렬'
    L_ar = np.vstack([zar['raw_valid__xgb'], zar['raw_hold__xgb']]).astype(np.float64)

    m_abe2, t_abe2 = sd3(L_abe2)
    m_abe1, t_abe1 = sd3(L_abe1)
    m_ar, t_ar = sd3(L_ar)
    tie_base = np.column_stack([L_tb2[:, 1] - L_tb2[:, 0], L_tb1[:, 1] - L_tb1[:, 0],
                                t_abe2, t_abe1, t_ar])
    win_base = np.column_stack([m_abe2, m_abe1, m_ar])

    # clean 마스크 — valid 행 중 V3 의 va2 에 있던 2,000행만
    te_v3, va2 = _v3_split()
    cu = np.ones(n, bool)
    cu[:HALF] = pd.Index(va2['id']).get_indexer(ids[:HALF]) >= 0
    assert int(cu.sum()) == 7748, f'clean {int(cu.sum())} != 7748'
    assert (pd.Index(te_v3['id']).get_indexer(ids[HALF:]) >= 0).all(), 'holdout 이 V3 test 밖'

    fold = np.zeros(n, int)
    for k, (_, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=SEED)
                                .split(np.zeros((n, 1)), y)):
        fold[te] = k

    D = np.load(bundle / 'deploy.npz')
    n_pil, n_gat, n_lin = int(D['n_pillar']), int(D['n_gat']), int(D['n_lin'])
    assert (n_pil, n_gat, n_lin) == (4, 3, 12), f'번들 열수 {(n_pil, n_gat, n_lin)} != (4,3,12)'
    PIL = np.column_stack([D[f'col_pillar_{i}'] for i in range(n_pil)])
    GAT = np.column_stack([D[f'col_gat_{i}'] for i in range(n_gat)])
    LIN = np.column_stack([D[f'col_lin_{i}'] for i in range(n_lin)])
    for nm, A in (('기둥', PIL), ('GAT', GAT), ('LIN', LIN)):
        assert A.shape[0] == n and np.isfinite(A).all(), f'{nm} 형상/비유한 {A.shape}'

    print(f'stage2 {n:,}행 (clean {int(cu.sum()):,} / holdout {int(hold.sum()):,}) | '
          f'번들 {bundle}  기둥 {n_pil} · GAT {n_gat} · LIN {n_lin}')
    return dict(ids=ids, y=y, n=n, hold=hold, nz=nz, cu=cu, fold=fold,
                tie_base=tie_base, win_base=win_base, PIL=PIL, GAT=GAT, LIN=LIN)


# ══════════════════════════════════════════════ 칸 평가
def fit_win(U):
    """승부축 결합기 — tie축 열 선택과 무관하므로 한 번만."""
    X = np.hstack([U['win_base'], U['LIN']])
    rows = U['cu'] & U['nz']
    y01 = (U['y'] == 0).astype(np.float64)
    z, dep = combiner(X, y01, U['fold'], rows, bias=False, n_logit=U['win_base'].shape[1])
    ll = rowloss(z[rows], y01[rows])
    print(f"  승부축 결합기 1회 적합 — λ {dep['lam']:g} | 적합행 {int(rows.sum()):,} "
          f"| 미수렴 {dep['n_bad']} | 확장 {dep['n_ext']}")
    return dict(z=z, dep=dep, ll=ll, rows=rows)


def fit_tie(U, use_pil, n_gat, gat=None, gat_names=None):
    """tie축 결합기 — 기둥 블록(전체/부분) + GAT 접두사 n_gat 개.

    gat 를 주면 저장된 GAT 열 대신 그것을 쓴다 (--gat-inputs 의 입력 절단 팔).
    """
    G = U['GAT'] if gat is None else gat
    GN = GAT_NAMES if gat_names is None else list(gat_names)
    parts, names = [U['tie_base']], list(TIE_NAMES)
    if use_pil is not None and len(use_pil):
        parts.append(U['PIL'][:, list(use_pil)])
        names += [PIL_NAMES[i] for i in use_pil]
    if n_gat:
        parts.append(G[:, :n_gat])
        names += list(GN[:n_gat])
    X = np.hstack(parts)
    rows = U['cu']
    y01 = (U['y'] == 2).astype(np.float64)
    z, dep = combiner(X, y01, U['fold'], rows, bias=True, n_logit=U['tie_base'].shape[1])
    ll = rowloss(z[rows], y01[rows])
    return dict(z=z, dep=dep, ll=ll, rows=rows, names=names)


def compose_ll(U, llt, llw, rt, rw):
    """행별 3-class 손실 — 체인 룰. tie 손실 + (비-tie 행이면) 승부 손실. holdout 5,748행."""
    ht, hw = U['hold'][rt], U['hold'][rw]
    lt_h, lw_h = llt[ht], llw[hw]
    nz_h = U['nz'][U['hold']]
    row = lt_h.astype(np.float64).copy()
    assert int(nz_h.sum()) == len(lw_h), f"승부축 정렬 {int(nz_h.sum())} != {len(lw_h)}"
    row[nz_h] += lw_h
    return row, float(lt_h.mean()), float(lw_h.mean())


def compose(U, tie, win):
    return compose_ll(U, tie['ll'], win['ll'], tie['rows'], win['rows'])


# ══════════════════════════════════════════════ 사후 보정 (--calib)
CALIB_KINDS = ('none', 'T', 'platt', 'iso')


def calib_axis(z, y01, rows, fold, kind, bias_ok):
    """축 하나에 보정기를 **같은 5-fold 로 교차적합**해서 보정된 점수를 돌려준다.

    bias_ok=False (승부축) 면 절편 금지 — 점수가 반대칭이어야 하는데 상수항이 그걸 깬다.
    그래서 승부축에서는 T 와 platt 이 같은 모형이 된다 (자유도 1개).
    isotonic 도 승부축에서는 (z,y) 에 (−z,1−y) 를 덧붙여 반대칭을 강제한다 — 그 확대
    자료가 (z,y)->(−z,1−y) 에 불변이라 PAVA 해가 p(−z)=1−p(z) 를 만족한다. 수치 대칭화도
    한 번 더 건다.
    """
    out = np.full(len(z), np.nan)
    for k in range(K):
        tr, te = (fold != k) & rows, (fold == k) & rows
        zt, yt = z[tr], y01[tr]
        if kind == 'none':
            out[te] = z[te]
        elif kind in ('T', 'platt'):
            b = bool(bias_ok and kind == 'platt')
            v, _ = _fit(zt[:, None], yt, np.zeros(1), b)
            out[te] = z[te] * v[0] + (float(v[1]) if b else 0.0)
        elif kind == 'iso':
            if bias_ok:
                xx, yy = zt, yt
            else:
                xx = np.concatenate([zt, -zt]); yy = np.concatenate([yt, 1.0 - yt])
            ir = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(xx, yy)
            p = ir.predict(z[te])
            if not bias_ok:
                p = (p + 1.0 - ir.predict(-z[te])) / 2.0
            p = np.clip(p, 1e-6, 1 - 1e-6)
            out[te] = np.log(p / (1 - p))
        else:
            raise ValueError(f'알 수 없는 보정기 {kind}')
    return out


def calib_cell(U, tie_z, win_z, kind):
    """한 칸의 OOF 점수에 보정기를 얹고 행별 3-class 손실을 돌려준다."""
    yt = (U['y'] == 2).astype(np.float64)
    yw = (U['y'] == 0).astype(np.float64)
    rt, rw = U['cu'], U['cu'] & U['nz']
    zt = calib_axis(tie_z, yt, rt, U['fold'], kind, True)
    zw = calib_axis(win_z, yw, rw, U['fold'], kind, False)
    return compose_ll(U, rowloss(zt[rt], yt[rt]), rowloss(zw[rw], yw[rw]), rt, rw)


# ══════════════════════════════════════════════ GAT 입력 절단 (--gat-inputs)
# 원본: fit_heads.py 셀 51 — 하이퍼파라미터·구조 전부 그대로. 바꾸는 것은 **입력뿐**이다.
LR, EPOCHS, PATIENCE, WD = 3e-4, 3000, 200, 1e-2
H_GAT, RD_CAP, RD_STOP = 128, 8, 0.0003
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _es_split(n_, seed, frac=0.15):
    # 헤드 ES 검증 분할 — train 의 15%
    perm = np.random.default_rng(seed).permutation(n_)
    n_es = max(int(n_ * frac), 200)
    return perm[:n_es], perm[n_es:]


class GATNet(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.W1, self.W2, self.w = nn.Linear(d, h), nn.Linear(d, h), nn.Linear(h, 1)

    def forward(self, x):
        return self.w(torch.tanh(self.W1(x)) * torch.sigmoid(self.W2(x)))


def fit_gat(CUt_, CVt_, y01, s0, even, seed, CUs_, CVs_, hidden=H_GAT,
            es=None, tr_idx=None, net=None, lr=None, sym_fn=None):
    """원본 fit_heads.fit_gat. 두 인자만 추가했다.

    sym_fn=None 이면 원본 대칭화 `f(x⁺) + sgn·f(x⁻)`. 콜러블을 주면 `out()` 을 통째로
    그것으로 갈아끼운다 — 판독을 대칭화 뒤로 옮기는 팔(`temp/readout.py`)이 쓴다.
    서명 `sym_fn(f, Ux, Vx, sgn) -> (n,) 텐서`. 기본 경로는 비트 단위로 그대로다.

    es=None      원본 경로 — TR 의 15% 를 떼어 조기종료 (그 행은 학습에서 빠진다).
    es=dict(CU,CV,y,s0)  외부 집합에서 조기종료. 그 집합의 입력·라벨·앵커를 직접 준다.
        앵커 정규화 상수(rms)는 **TR 의 s0 에서 계산한 것 하나**를 양쪽에 쓴다 —
        train 과 ES 에서 앵커 스케일이 달라지면 a 가 두 의미를 겸하게 된다.
    tr_idx       es 를 줄 때의 학습 행 부분집합. None 이면 TR 전 행.
    net          망 팩토리 (d, h) -> Module. 마지막 층이 `.w` 여야 한다(0 초기화 대상).
                 None 이면 GATNet — 기본 경로는 이 인자들 이전과 비트 단위로 같다.
    lr           학습률 override. None 이면 모듈 상수 LR.
    """
    torch.manual_seed(seed)
    d = CUt_.shape[1] + CVt_.shape[1]
    f = (GATNet if net is None else net)(d, hidden)
    nn.init.zeros_(f.w.weight); nn.init.zeros_(f.w.bias)
    f = f.to(DEVICE)
    anchored = s0 is not None
    a = nn.Parameter(torch.tensor(1.0, device=DEVICE))
    opt = torch.optim.AdamW([{'params': f.parameters(), 'weight_decay': WD},
                             {'params': [a], 'weight_decay': 0.0}],
                            lr=(LR if lr is None else float(lr)))
    rms = float(np.clip(np.sqrt((np.asarray(s0, np.float64) ** 2).mean()), 1e-9, None)) \
        if anchored else 1.0
    s0z = (np.asarray(s0, np.float64) / rms) if anchored else np.zeros(len(y01))
    T = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device=DEVICE)
    U_, V_, S_, Y_ = T(CUt_), T(CVt_), T(s0z), T(y01)
    lossf = nn.BCEWithLogitsLoss()
    sgn = 1 if even else -1

    def out(Ux, Vx):
        if sym_fn is not None:                       # 판독을 대칭화 **뒤로** 옮기는 팔
            return sym_fn(f, Ux, Vx, sgn)
        return (f(torch.cat([Ux, Vx], 1)) + sgn * f(torch.cat([Ux, -Vx], 1))).squeeze(-1)

    if es is None:
        eidx, tidx = _es_split(len(y01), seed)
        Ue, Ve, Se, Ye = U_[eidx], V_[eidx], S_[eidx], Y_[eidx]
        Ut, Vt, St, Yt = U_[tidx], V_[tidx], S_[tidx], Y_[tidx]
    else:
        es_s0z = (np.asarray(es['s0'], np.float64) / rms) if anchored \
            else np.zeros(len(es['y']))
        Ue, Ve, Se, Ye = T(es['CU']), T(es['CV']), T(es_s0z), T(es['y'])
        if tr_idx is None:
            Ut, Vt, St, Yt = U_, V_, S_, Y_
        else:
            ti = torch.as_tensor(np.asarray(tr_idx), device=DEVICE)
            Ut, Vt, St, Yt = U_[ti], V_[ti], S_[ti], Y_[ti]

    best, best_sd, bad, used = np.inf, None, 0, 0
    for ep in range(EPOCHS):
        f.train(); opt.zero_grad()
        lossf(a * St + out(Ut, Vt), Yt).backward()
        opt.step()
        f.eval()
        with torch.no_grad():
            v = float(lossf(a * Se + out(Ue, Ve), Ye))
        if v < best - 1e-6:
            best, bad, used = v, 0, ep + 1
            best_sd = ({k2: t.detach().clone() for k2, t in f.state_dict().items()},
                       float(a.detach()))
        elif (bad := bad + 1) >= PATIENCE:
            break
    f.load_state_dict(best_sd[0])
    with torch.no_grad():
        r_tr = out(U_, V_).cpu().numpy().astype(np.float64)
        r_s2 = out(T(CUs_), T(CVs_)).cpu().numpy().astype(np.float64)
    z_tr = best_sd[1] * s0z + r_tr if anchored else r_tr
    return r_s2, dict(z_tr=z_tr, es_ll=best, es_epoch=used, a=best_sd[1], rms=rms,
                      n_tr=int(Yt.shape[0]), n_es=int(Ye.shape[0]))


def gat_chain(CU_in, CV_in, CU_s2, CV_s2, y01m, tag, es_rows=None, y01_s2=None,
              tr_idx=None):
    """라운드 자기종결 부스팅 — _fit_gat 와 동일. 증분 열만 내보낸다.

    es_rows 를 주면 그 stage2 부분집합에서 조기종료한다. 라운드 k+1 의 ES 앵커는
    그 행들에서의 누적 점수여야 하므로 z_es 를 z_tr 과 같은 갱신식으로 따로 굴린다:
        z_new = a * (z_old / rms) + r_new       (rms 는 TR 기준 하나)
    """
    z_tr, z_es, cols, prev = None, None, [], None
    for rd in range(1, RD_CAP + 1):
        es = None
        if es_rows is not None:
            es = dict(CU=CU_s2[es_rows], CV=CV_s2[es_rows], y=y01_s2[es_rows], s0=z_es)
        r_s2, aux = fit_gat(CU_in, CV_in, y01m, z_tr, True, SEED, CU_s2, CV_s2,
                            es=es, tr_idx=tr_idx)
        cols.append(r_s2)
        gain = None if prev is None else prev - aux['es_ll']
        print(f'    [{tag}] r{rd}: es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]}'
              f' (train {aux["n_tr"]:,} / es {aux["n_es"]:,})'
              + (f' gain {gain:+.5f}' if gain is not None else ''))
        prev = aux['es_ll']
        if es_rows is not None:
            z_es = r_s2[es_rows] if z_es is None else \
                aux['a'] * (z_es / aux['rms']) + r_s2[es_rows]
        z_tr = aux['z_tr']
        if gain is not None and gain < RD_STOP:
            break
    return np.column_stack(cols)


def zstats(X, even):
    # 짝: 평균/표준편차, 홀: 무중심 RMS (반대칭 보존)
    mu = X.mean(0) if even else np.zeros(X.shape[1])
    sc = np.clip(X.std(0) if even else np.sqrt((X ** 2).mean(0)), 1e-9, None)
    return mu, sc


def load_train_logits():
    """train 46,729 의 tie 짝 5열 · win 홀 3열. 모든 파일을 abe2 train npz 의 id 로 대조."""
    za = np.load(MODELS / 'logits/abe2__checkpoint-3000__train.npz')
    ids, y = za['train_ids'], za['train_labels']
    assert len(ids) == 46729 and len(np.unique(ids)) == len(ids), f'train {len(ids)}행'

    def get(rel, n_class):
        z = np.load(MODELS / rel)
        assert (z['train_ids'] == ids).all() and (z['train_labels'] == y).all(), f'{rel} 정렬'
        L = ((z['train_fwd'] + z['train_swp']) / 2).astype(np.float64)
        assert L.shape == (len(ids), n_class) and np.isfinite(L).all(), f'{rel} 형상'
        return L

    L_abe2 = ((za['train_fwd'] + za['train_swp']) / 2).astype(np.float64)
    L_tb2 = get('logits/tb2__checkpoint-4000__train.npz', 2)
    L_abe1 = get(f'logits/abe1__checkpoint-{ABE1_STEP}__train.npz', 3)
    L_tb1 = get(f'logits/tb1__checkpoint-{TB1_STEP}__train.npz', 2)
    zar = np.load(ARITH / 'outputs/arith_experts_raw.npz')
    assert (zar['ids_train'] == ids).all() and (zar['y_train'] == y).all(), 'arith train 정렬'
    L_ar = zar['raw_oof_train__xgb'].astype(np.float64)   # build() 의 raw_valid__xgb 와 같은 계열

    m_abe2, t_abe2 = sd3(L_abe2)
    m_abe1, t_abe1 = sd3(L_abe1)
    m_ar, t_ar = sd3(L_ar)
    tie_tr = np.column_stack([L_tb2[:, 1] - L_tb2[:, 0], L_tb1[:, 1] - L_tb1[:, 0],
                              t_abe2, t_abe1, t_ar])
    win_tr = np.column_stack([m_abe2, m_abe1, m_ar])
    return ids, y.astype(np.int64), tie_tr, win_tr


def load_svd_cache():
    """train 46,729 짜리 SVD-512 캐시. 행 수로 고른다 (TR+dirty 증강본과 섞이지 않게)."""
    for p in sorted((FINAL / 'temp/head_cache').glob('svd512-*.npz')):
        z = np.load(p)
        if z['CUt'].shape == (46729, 512) and z['CUs'].shape == (10748, 512):
            assert int(z['K_TIE']) == 512, f'{p.name} K_TIE {int(z["K_TIE"])}'
            print(f'  SVD 캐시 {p.name}  CUt {z["CUt"].shape} CVt {z["CVt"].shape}')
            return z
    raise FileNotFoundError('temp/head_cache 에 46,729행 svd512 캐시가 없다 — '
                            'fit_heads.py 를 한 번 돌려 캐시를 만들어야 한다')


def run_gat_inputs(U, arms, A=None):
    """GAT 입력을 잘라 다시 학습한다. 구조·하이퍼파라미터는 전부 동일, 입력만 다르다.

        A(logit)  d = 5 + 3     = 8       풀링 없음
        B(svd)    d = 512 + 512 = 1024    로짓 없음
        C(both)   d = 517 + 515 = 1032    현행 — 저장된 열과 대조하는 검증 팔

    ! 주의 — 입력 차원이 바뀌면 nn.Linear 기본 초기화 범위(1/sqrt(d))가 함께 바뀐다.
      A 는 d=8 이라 초기 스케일이 C(d=1032)의 11배다. 같은 lr·같은 epoch 예산에서
      최적화 국면이 달라지므로, A 의 무효 결과를 "로짓 비선형은 값어치 없음"으로
      바로 읽으면 안 된다 (덜 학습된 것일 수 있다).
    """
    if A is None:
        print('\n[GAT 입력 절단] 학습 자산 적재')
        A = gat_assets(U)
    INP = {
        'logit': (A['Lu_tr'], A['Lv_tr'], A['Lu_s2'], A['Lv_s2']),
        'svd':   (A['CUt'], A['CVt'], A['CUs'], A['CVs']),
        'both':  (A['CU_in'], A['CV_in'], A['CU_s2'], A['CV_s2']),
    }
    out = {}
    for arm in arms:
        cu_tr, cv_tr, cu_s2, cv_s2 = (np.ascontiguousarray(x, np.float32) for x in INP[arm])
        t0 = time.time()
        print(f'  [{arm}] d = {cu_tr.shape[1]} + {cv_tr.shape[1]} = '
              f'{cu_tr.shape[1] + cv_tr.shape[1]}')
        G = gat_chain(cu_tr, cv_tr, cu_s2, cv_s2, A['y01m'], arm)
        print(f'    [{arm}] {G.shape[1]}라운드  [{(time.time() - t0) / 60:.1f}분]')
        out[arm] = G
        if arm == 'both':
            d = np.abs(G[:, :U['GAT'].shape[1]] - U['GAT']).max() if G.shape[1] >= \
                U['GAT'].shape[1] else np.inf
            r = np.corrcoef(G[:, 0], U['GAT'][:, 0])[0, 1]
            print(f'    [참고] both 재학습 vs 저장 열: 라운드 {G.shape[1]} vs '
                  f'{U["GAT"].shape[1]} | max|Δ| {d:.3e} | r(g1) {r:.6f}\n'
                  '           (저장 열은 ARITH checkpoint-4000, 여기는 구 npz — '
                  '차이의 원인. 판정은 하류 손실로 한다)')
    return out


def gat_assets(U):
    """GAT 학습 재료 — train 로짓 · SVD 캐시 · 정규화. 풀링 원본(3.1GB)은 안 읽는다."""
    _, y_tr, tie_tr, win_tr = load_train_logits()
    Z = load_svd_cache()
    print(f'  train 로짓 {tie_tr.shape[0]:,}행 (tie {tie_tr.shape[1]}열 / '
          f'win {win_tr.shape[1]}열) | 장치 {DEVICE}')
    mu_u, sc_u = zstats(tie_tr, True)
    _, sc_v = zstats(win_tr, False)
    A = dict(y_tr=y_tr, y01m=(y_tr == 2).astype(np.float64),
             Lu_tr=(tie_tr - mu_u) / sc_u, Lv_tr=win_tr / sc_v,
             Lu_s2=(U['tie_base'] - mu_u) / sc_u, Lv_s2=U['win_base'] / sc_v,
             CUt=Z['CUt'].astype(np.float32), CVt=Z['CVt'].astype(np.float32),
             CUs=Z['CUs'].astype(np.float32), CVs=Z['CVs'].astype(np.float32))
    for k, (a_, b_) in (('CU_in', ('CUt', 'Lu_tr')), ('CV_in', ('CVt', 'Lv_tr')),
                        ('CU_s2', ('CUs', 'Lu_s2')), ('CV_s2', ('CVs', 'Lv_s2'))):
        A[k] = np.ascontiguousarray(np.hstack([A[a_], A[b_]]), np.float32)
    return A


# ── ES 출처 팔 — 조기종료 집합을 오염 밖으로 옮긴다 ──────────────────────────
# 왜: 현행 ES 는 TR 의 15% 인데 TR 은 **네 인코더 전부의 in-sample** 이다. 지난 런에서
#     ES 순위가 전이 순위와 정반대로 나왔다(logit 팔 ES 0.53125/ep2785 인데 전이 노이즈).
#     버려진 3,000행은 ABE1 만 in-sample 이고 abe2/tb2/tb1 에는 out-of-sample 이며
#     (TR ∩ stage2 = 0), cu 밖이라 결합기 적합·holdout 에 전혀 안 들어간다.
# 교란 분리: ES 를 옮기면 그 7,009행이 학습으로 돌아온다 — 변화가 둘이다. 그래서 3팔.
ES_ARMS = ('ctrl', 'es3k', 'es3k_full', 'es2k', 'es7k')
# 누출 등급 — 출력에 그대로 찍는다. 숫자를 정직한 추정치로 읽으면 안 되는 팔이 있다.
ES_LEAK = {
    'ctrl':      '없음        ES = TR 15%(7,009). 네 인코더 전부 in-sample',
    'es3k':      '없음        ES = dirty 3,000. ABE1 만 in-sample (측정: 유의하게 손해)',
    'es3k_full': '없음        위 + TR 전 행 학습',
    'es2k':      '2차         ES = va2 2,000. holdout 미포함이나 결합기 적합 행에 들어 있다',
    'es7k':      '**직접**    ES = clean 7,748. holdout 5,748 을 포함한다 -> 상한이지 추정치 아님',
}


def run_es_arms(U, arms, A=None):
    """조기종료 집합을 바꿔 GAT 를 다시 학습한다. **학습 행은 전부 TR 39,720 으로 고정**
    (ctrl 과 같은 분할)해서 팔 사이 차이가 오직 ES 집합에서만 오게 한다.
    예외는 es3k_full 하나로, 그건 회수한 7,009행의 효과를 따로 보려고 남긴 팔이다.

        ctrl        ES = TR 15%(7,009)     누출 없음. 현행 재현 대조
        es3k        ES = dirty 3,000       누출 없음 (ABE1 in-sample 분포)
        es3k_full   ES = dirty 3,000       + TR 전 행 학습
        es2k        ES = va2 2,000         2차 누출 — holdout 은 안 보지만 결합기 적합 행이다
        es7k        ES = clean 7,748       **직접 누출** — holdout 을 보고 멈춘다

    es7k 의 용도는 오직 하나다 — **"최선의 정지점을 줘도 이 열이 값을 더하는가"** 라는
    상한 질문. 여기서도 못 벌면 그 열은 ES 손잡이로는 살릴 수 없다는 뜻이고, 벌면 es2k 가
    거의 정직한 추정치를 준다.
    ! 팔 사이 es_ll 값은 비교 불가다 (다른 행 집합·다른 tie 기저율). 비교는 홀드아웃으로.
    """
    if A is None:
        print('\n[ES 출처] 학습 자산 적재')
        A = gat_assets(U)
    dirty = ~U['cu']
    va2 = U['cu'] & ~U['hold']
    assert int(dirty.sum()) == 3000, f'dirty {int(dirty.sum())} != 3,000'
    assert int(va2.sum()) == 2000, f'va2 {int(va2.sum())} != 2,000'
    assert int(U['cu'].sum()) == 7748 and not (dirty & U['hold']).any()
    y01_s2 = (U['y'] == 2).astype(np.float64)
    eidx, tidx = _es_split(len(A['y01m']), SEED)
    print(f'  ES 후보 — TR15% {len(eidx):,} / dirty {int(dirty.sum()):,} / '
          f'va2 {int(va2.sum()):,} / clean {int(U["cu"].sum()):,}  '
          f'| 학습 고정 {len(tidx):,} (es3k_full 만 46,729)')

    ROWS = {'es3k': dirty, 'es3k_full': dirty, 'es2k': va2, 'es7k': U['cu']}
    out = {}
    for arm in arms:
        t0 = time.time()
        if arm == 'ctrl':
            kw = dict()
        else:
            kw = dict(es_rows=ROWS[arm], y01_s2=y01_s2)
            if arm != 'es3k_full':
                kw['tr_idx'] = tidx
        print(f'  [{arm}] 누출 {ES_LEAK[arm]}')
        G = gat_chain(A['CU_in'], A['CV_in'], A['CU_s2'], A['CV_s2'], A['y01m'], arm, **kw)
        print(f'    [{arm}] {G.shape[1]}라운드  [{(time.time() - t0) / 60:.1f}분]')
        out[arm] = G
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bundle', default='temp/heads_repro',
                    help='deploy.npz 가 있는 디렉터리 (final/ 기준 상대경로)')
    ap.add_argument('--loo', action='store_true',
                    help='기둥 leave-one-out 4칸 추가 (진단용 — 다중비교 보정 없음)')
    ap.add_argument('--calib', action='store_true',
                    help='각 칸 위에 사후 보정 사다리(none/T/platt/iso)를 얹는다')
    ap.add_argument('--gat-inputs', default='',
                    help="GAT 입력 절단 팔 — 'logit,svd,both' 중 쉼표 목록. "
                         "'both' 는 저장된 열과 대조하는 검증 팔이다 (권장: logit,svd,both)")
    ap.add_argument('--es-src', default='',
                    help="GAT 조기종료 집합 팔 — 'ctrl,es3k,es3k_full' 중 쉼표 목록. "
                         "ctrl 은 현행 재현 대조라 반드시 같이 돌린다")
    ap.add_argument('--out', default='temp/tie_ladder')
    args = ap.parse_args()

    bundle = (FINAL / args.bundle) if not Path(args.bundle).is_absolute() else Path(args.bundle)
    assert (bundle / 'deploy.npz').exists(), f'{bundle}/deploy.npz 없음'

    U = build(bundle)
    win = fit_win(U)

    ALL = (0, 1, 2, 3)
    cells = []
    for n_gat in (0, 1, 2, 3):
        for pil, ptag in ((), 'p0'), (ALL, 'p4'):
            cells.append((f'{ptag}_g{n_gat}', pil, n_gat, True))
    if args.loo:
        for drop in ALL:
            keep = tuple(i for i in ALL if i != drop)
            cells.append((f'loo_{PIL_NAMES[drop].split("·")[0]}', keep, 3, False))

    REF = 'p4_g3'
    res, rows, zs = {}, {}, {}
    print(f'\n{"칸":12s} {"열":>3s}  {"λ":>7s}  {"tie(hold)":>9s} {"win(hold)":>9s} '
          f'{"총 holdout":>10s}')
    print('-' * 78)
    for tag, pil, n_gat, _primary in cells:
        tie = fit_tie(U, pil, n_gat)
        zs[tag] = tie['z']
        row, lt, lw = compose(U, tie, win)
        res[tag] = dict(tot=float(row.mean()), tie=lt, win=lw,
                        lam=tie['dep']['lam'], n_col=len(tie['names']),
                        names=tie['names'], w=[float(x) for x in tie['dep']['v']],
                        n_bad=tie['dep']['n_bad'], n_ext=tie['dep']['n_ext'])
        rows[tag] = row
        print(f"{tag:12s} {len(tie['names']):3d}  {tie['dep']['lam']:7g}  "
              f"{lt:9.5f} {lw:9.5f} {row.mean():10.5f}"
              + ('' if not tie['dep']['n_bad'] else f"  미수렴 {tie['dep']['n_bad']}"))

    # ── 기준 칸 검산 — 여기가 어긋나면 저장된 열이 기준 런의 것이 아니다 ──
    got = res[REF]['tot']
    ok = abs(got - REF_HOLD) < REF_TOL
    print(f'\n[검산] 기준 칸 {REF} = {got:.5f} vs 기록 {REF_HOLD:.5f}  Δ {got - REF_HOLD:+.5f}  '
          + ('OK' if ok else '** 불일치 — 아래 판정 해석 금지'))

    print(f'\nΔ vs 기준 {REF}  (양수 = 그 칸이 나쁘다 / 문턱 0.0005)')
    print('-' * 78)
    for tag in res:
        if tag == REF:
            print(f'{tag:12s} {"기준":>34s}')
            continue
        m, lo, hi = paired_ci(rows[tag], rows[REF])
        print(f'{tag:12s} {res[tag]["tot"]:9.5f}  {ci_str(m, lo, hi)}')

    print('\n[기준 칸 계수] 정규화 좌표계 — 열끼리 크기 비교 가능')
    d = res[REF]
    for nm, w in sorted(zip(d['names'], d['w']), key=lambda x: -abs(x[1])):
        print(f'    {nm:10s} {w:+8.4f}')

    calib = {}
    if args.calib:
        print('\n[보정] 축별 사후 보정 — 보정기도 같은 5-fold 로 교차적합 '
              '(승부축은 절편 금지 → T = platt)')
        print(f'{"칸":12s}' + ''.join(f'{k:>11s}' for k in CALIB_KINDS))
        print('-' * 78)
        for tag in res:
            calib[tag] = {}
            line = f'{tag:12s}'
            for kind in CALIB_KINDS:
                row_c, _, _ = calib_cell(U, zs[tag], win['z'], kind)
                calib[tag][kind] = dict(tot=float(row_c.mean()))
                rows[f'{tag}__{kind}'] = row_c
                line += f'{row_c.mean():11.5f}'
            print(line)

        # 검산 — kind='none' 은 본 사다리를 그대로 재계산한 것이라 비트 단위로 같아야 한다.
        # 어긋나면 보정 경로가 다른 행·다른 점수를 보고 있다는 뜻이다.
        dmax = float(np.abs(rows[f'{REF}__none'] - rows[REF]).max())
        print(f'\n[검산] none 재계산 vs 본 사다리: max|Δ| {dmax:.2e}  '
              + ('OK' if dmax < 1e-12 else '** 불일치 — 아래 해석 금지'))

        ref_row = rows[REF]
        gap0 = res['p0_g0']['tot'] - res[REF]['tot']
        print(f'\n[핵심 대조] Δ vs {REF}(none) = {res[REF]["tot"]:.5f}   '
              f'(양수 = 그쪽이 나쁘다 / 문턱 0.0005)')
        print('-' * 78)
        for tag in ('p0_g0', REF):
            for kind in CALIB_KINDS:
                if tag == REF and kind == 'none':
                    continue
                m, lo, hi = paired_ci(rows[f'{tag}__{kind}'], ref_row)
                print(f'  {tag:8s} + {kind:6s} {calib[tag][kind]["tot"]:9.5f}  '
                      f'{ci_str(m, lo, hi)}')

        best = min(CALIB_KINDS, key=lambda k: calib['p0_g0'][k]['tot'])
        m, lo, hi = paired_ci(rows[f'p0_g0__{best}'], ref_row)
        filled = (1.0 - m / gap0) * 100.0 if abs(gap0) > 1e-12 else float('nan')
        print(f'\n  p0_g0 의 원래 격차 {gap0:+.5f}  ->  보정 최선({best}) {m:+.5f}'
              f'   ({filled:.1f}% 메움)')
        print('  판정 규칙 — 이 CI 가 0 을 포함하면  "표현 7열이 하는 일 = 재보정"'
              ' (풀링 버리고 보정기로 대체 가능)')
        print('              포함하지 않으면        "표현이 순위 정보를 나른다"'
              ' (보정으로 대체 불가)')

    gat_res, Gs, ASSET = {}, {}, None
    arms = [a.strip() for a in args.gat_inputs.split(',') if a.strip()]
    es_arms = [a.strip() for a in args.es_src.split(',') if a.strip()]
    if arms or es_arms:
        print('\n[GAT] 학습 자산 적재 (풀링 원본 미적재)')
        ASSET = gat_assets(U)
    if arms:
        bad = [a for a in arms if a not in ('logit', 'svd', 'both')]
        assert not bad, f'알 수 없는 팔 {bad} — logit/svd/both 만 된다'
        Gs = run_gat_inputs(U, arms, A=ASSET)
        print(f'\n{"칸":12s} {"열":>3s}  {"λ":>7s}  {"tie(hold)":>9s} {"총 holdout":>10s}')
        print('-' * 78)
        for arm in arms:
            G = Gs[arm]
            gn = [f'{arm}{i + 1}' for i in range(G.shape[1])]
            for pil, ptag in ((), 'p0'), (ALL, 'p4'):
                tag = f'{ptag}_{arm}'
                tie = fit_tie(U, pil, G.shape[1], gat=G, gat_names=gn)
                row, lt, _ = compose(U, tie, win)
                gat_res[tag] = dict(tot=float(row.mean()), tie=lt, n_round=int(G.shape[1]),
                                    lam=tie['dep']['lam'], n_col=len(tie['names']),
                                    names=tie['names'], w=[float(x) for x in tie['dep']['v']])
                rows[tag] = row
                print(f'{tag:12s} {len(tie["names"]):3d}  {tie["dep"]["lam"]:7g}  '
                      f'{lt:9.5f} {row.mean():10.5f}')

        print('\n[격자] GAT 입력별 총 holdout')
        print(f'{"GAT 입력":18s}{"기둥 없음":>11s}{"기둥 4열":>11s}')
        print(f'{"없음":18s}{res["p0_g0"]["tot"]:11.5f}{res["p4_g0"]["tot"]:11.5f}')
        for arm in arms:
            print(f'{arm + " (재학습)":18s}'
                  + ''.join(f'{gat_res[f"{p}_{arm}"]["tot"]:11.5f}' for p in ('p0', 'p4')))
        print(f'{"저장열 (현행)":18s}{res["p0_g3"]["tot"]:11.5f}{res["p4_g3"]["tot"]:11.5f}')

        print(f'\nΔ vs 기준 {REF} = {res[REF]["tot"]:.5f}  (양수 = 그 칸이 나쁘다)')
        print('-' * 78)
        for tag in gat_res:
            m, lo, hi = paired_ci(rows[tag], rows[REF])
            print(f'  {tag:12s} {gat_res[tag]["tot"]:9.5f}  {ci_str(m, lo, hi)}')

        # 분해 — 기둥 없는 계열이 GAT 를 고립시킨다
        b0 = res['p0_g0']['tot']
        print(f'\n[분해] 기둥 없는 계열, 기준선 p0_g0 = {b0:.5f} 에서 GAT 가 번 것')
        print('-' * 78)
        for arm in arms:
            t = gat_res[f'p0_{arm}']
            m, lo, hi = paired_ci(rows['p0_g0'], rows[f'p0_{arm}'])
            print(f'  GAT({arm:5s}, {t["n_round"]}라운드)  {b0 - t["tot"]:+.5f}  {ci_str(m, lo, hi)}')
        m, lo, hi = paired_ci(rows['p0_g0'], rows['p0_g3'])
        print(f'  GAT(저장열, 3라운드)   {b0 - res["p0_g3"]["tot"]:+.5f}  {ci_str(m, lo, hi)}')
        print(f'\n  표현 장치 총합 (p0_g0 - p4_g3) = {b0 - res[REF]["tot"]:+.5f}')
        print('  판정 — logit 이 both 에 닿으면 GAT 의 값어치는 로짓의 비선형이고')
        print('         (Kaggle 추론에서 풀링 추출이 통째로 사라진다),')
        print('         svd 만 닿으면 인코더 표현의 새 내용이다.')

    es_res, ESG = {}, {}
    if es_arms:
        bad = [a for a in es_arms if a not in ES_ARMS]
        assert not bad, f'알 수 없는 ES 팔 {bad} — {list(ES_ARMS)} 만 된다'
        ESG = run_es_arms(U, es_arms, A=ASSET)
        nrd = {a: int(G.shape[1]) for a, G in ESG.items()}

        print(f'\n{"칸":20s} {"열":>3s}  {"λ":>7s}  {"tie(hold)":>9s} {"총 holdout":>10s}')
        print('-' * 78)
        for arm in es_arms:
            G = ESG[arm]
            gn = [f'{arm}·g{i + 1}' for i in range(nrd[arm])]
            for k in range(1, nrd[arm] + 1):
                for pil, ptag in ((), 'p0'), (ALL, 'p4'):
                    tag = f'{ptag}_{arm}_g{k}'
                    tie = fit_tie(U, pil, k, gat=G, gat_names=gn)
                    row, lt, _ = compose(U, tie, win)
                    es_res[tag] = dict(tot=float(row.mean()), tie=lt, n_gat=k,
                                       leak=ES_LEAK[arm].split()[0],
                                       lam=tie['dep']['lam'], names=tie['names'],
                                       w=[float(x) for x in tie['dep']['v']])
                    rows[tag] = row
                    print(f'{tag:20s} {len(tie["names"]):3d}  {tie["dep"]["lam"]:7g}  '
                          f'{lt:9.5f} {row.mean():10.5f}')

        ctrl_tag = f'p4_ctrl_g{nrd["ctrl"]}' if 'ctrl' in es_arms else None
        if ctrl_tag:
            m, lo, hi = paired_ci(rows[ctrl_tag], rows[REF])
            print(f'\n[검산] ctrl 재현 vs 저장열 {REF}: {es_res[ctrl_tag]["tot"]:.5f} vs '
                  f'{res[REF]["tot"]:.5f}  {ci_str(m, lo, hi)}')
            print('       벌어지는 만큼은 ARITH 판 차이(구 npz vs checkpoint-4000)다. '
                  '팔 간 비교는 전부 같은 판이라 무관하다.')

        print('\n[격자] ES 팔별 총 holdout   (p0 = 기둥 없음 / p4 = 기둥 4열)')
        print(f'{"":10s}' + ''.join(f'{a:>20s}' for a in es_arms))
        print(f'{"":10s}' + ''.join(f'{"p0":>10s}{"p4":>10s}' for _ in es_arms))
        print(f'{"GAT 없음":10s}'
              + ''.join(f'{res["p0_g0"]["tot"]:10.5f}{res["p4_g0"]["tot"]:10.5f}'
                        for _ in es_arms))
        for k in range(1, max(nrd.values()) + 1):
            line = f'{"g1..g" + str(k):10s}'
            for arm in es_arms:
                line += ((f'{es_res[f"p0_{arm}_g{k}"]["tot"]:10.5f}'
                          f'{es_res[f"p4_{arm}_g{k}"]["tot"]:10.5f}')
                         if k <= nrd[arm] else ' ' * 20)
            print(line)

        print('\n[증분] 각 열이 실제로 값을 더하는가 — paired CI (양수 = 그 열이 이득)')
        print('-' * 78)
        for arm in es_arms:
            nr = nrd[arm]
            print(f'  [{arm}]  누출 {ES_LEAK[arm]}')
            prev, prev_tot = 'p4_g0', res['p4_g0']['tot']
            for k in range(1, nr + 1):
                cur = f'p4_{arm}_g{k}'
                m, lo, hi = paired_ci(rows[prev], rows[cur])
                print(f'    g{k} 추가     {prev_tot:.5f} -> {es_res[cur]["tot"]:.5f}   '
                      f'{ci_str(m, lo, hi)}')
                prev, prev_tot = cur, es_res[cur]['tot']
            a0, a4 = f'p0_{arm}_g{nr}', f'p4_{arm}_g{nr}'
            m, lo, hi = paired_ci(rows[a0], rows[a4])
            print(f'    기둥 4열    {es_res[a0]["tot"]:.5f} -> {es_res[a4]["tot"]:.5f}   '
                  f'{ci_str(m, lo, hi)}')

        if ctrl_tag:
            print(f'\nΔ vs {ctrl_tag} = {es_res[ctrl_tag]["tot"]:.5f}  '
                  '(음수 = 그 칸이 낫다 / 문턱 0.0005)')
            print('-' * 78)
            for arm in es_arms:
                tag = f'p4_{arm}_g{nrd[arm]}'
                if tag == ctrl_tag:
                    continue
                m, lo, hi = paired_ci(rows[tag], rows[ctrl_tag])
                print(f'  {tag:20s} {es_res[tag]["tot"]:9.5f}  {ci_str(m, lo, hi)}  '
                      f'누출 {es_res[tag]["leak"]}')

        print('\n[g 계수] 전 라운드 칸의 정규화 좌표계 계수')
        print('-' * 78)
        for arm in es_arms:
            for ptag in ('p0', 'p4'):
                d = es_res[f'{ptag}_{arm}_g{nrd[arm]}']
                gw = [(nm.split('·')[-1], w) for nm, w in zip(d['names'], d['w'])
                      if nm not in TIE_NAMES and nm not in PIL_NAMES]
                print(f'  {ptag}_{arm:12s} ' + '  '.join(f'{nm} {w:+.4f}' for nm, w in gw)
                      + f'   Σ|w| {sum(abs(w) for _, w in gw):.3f}')

    op = FINAL / args.out
    np.savez_compressed(op.with_suffix('.rowloss.npz'), **rows,
                        **{f'GATCOL_{a}': G for a, G in Gs.items()},
                        **{f'ESCOL_{a}': G for a, G in ESG.items()})
    with open(op.with_suffix('.json'), 'w') as f:
        json.dump({'ref': REF, 'ref_recon': got, 'ref_recorded': REF_HOLD,
                   'ok': bool(ok), 'cells': res, 'calib': calib, 'gat_inputs': gat_res,
                   'es_src': es_res}, f, ensure_ascii=False, indent=1)
    print(f'\n저장: {op}.rowloss.npz / {op}.json')


if __name__ == '__main__':
    main()
