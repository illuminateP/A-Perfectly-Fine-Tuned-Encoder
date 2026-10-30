"""dest_fold.py — 확정 구조에 Arith 폴드 로짓 5개를 양축에 통과시킨다 (단독 파일).

destnation.py 의 복사본이다. 바뀐 것은 **Arith 가 헤드에 내는 열의 수** 뿐이다:
    destnation   t_ar 1 · m_ar 1          (xgb 5-fold 의 마진 공간 평균)
    dest_fold    t_ar_1..5 · m_ar_1..5    (폴드별 — 평균 내지 않는다)
폴드 열 하나하나가 swap-TTA 를 따로 받으므로 패리티 계약(t 짝 / m 홀)은 열 단위로 성립한다.
tie 로짓 5→9, 승부 로짓 3→7. 헤드 판독 [h128 | 로짓] 과 결합기 폭이 따라 넓어지고, 그 외
구조·하이퍼파라미터·위생은 destnation 과 글자 단위로 같다 (결합기·헤드·조립 코드는 복사).

    입력   Q, answer_A, answer_B
      ↓
    전문가 5   ABE1 ABE2 (bi) · TB1 TB2 (cross) · Arith (XGB 5-fold)
      ↓ 로짓 16 = tie 짝 9 [s_tb2 s_tb1 t_abe2 t_abe1 t_ar1..5] · 승부 홀 7 [m_abe2 m_abe1 m_ar1..5]
      ↓ 풀링 4소스 × (u,v) 1024
      ├─ tie 축   SVD-512 [CUt|CVt] → TB 체인 2R · 판독 Linear([u128 | 로짓16]) · 결합기 blk2
      ├─ 승부 축  소스별 raw (u,v) → GATE ×4 1R · 판독 Linear([h128 | 로짓16]) · 결합기 [m7 | W4]
      └─ 조립     p₀ = σ(z_tie), q = σ(z_win)  — 학습 파라미터 없음

λ 격자 — 양축 모두 [0, 1, 10, …, --lam_max] (기본 1e5). destnation 은 승부축에 0 이 없었고
상단이 1e4 였다. 0 은 무벌점 특수해, 상단은 자동 확장(×10, LAM_CEIL 까지)이 어차피 닿는
영역이라 둘 다 선택되지 않으면 결과가 같다 — 명시하는 이유는 내부 CV 가 그 점을 **항상**
평가하게 해서 가치(picked 열)를 눈으로 보기 위해서다. 격자는 json 에 기록된다.

우주 — train_full.csv 를 읽지 않는다. clean 2,000 은 origin/valid.csv 그 자체다.

Arith 폴드 로짓의 출처 — `arith_fold_logits()` 한 곳. 기본은 재학습 피쳐 캐시
(temp/arith_feats_stage1/F.npz — 부스터가 학습한 행렬 그대로) × checkpoint-ab224.
`--old-feats` 면 구 피쳐 parquet × temp/arith_old 부스터 (2026-08-21 프로브 재현).

오염 회계 (foldgate): TR 행에서 5열 중 1열은 out-of-fold, 4열은 in-sample. 평균 열도 같은
4/5 비율을 품고 있었으므로 BASE 와 FOLD5 의 차이는 "합쳐 주느냐 쪼개 주느냐" 뿐이다.

팔 — 기본 FOLD5. `--arms FOLD5,BASE` 로 같은 격자·같은 시드의 BASE(폴드 평균)를 나란히 돌려
시드별 paired Δ 를 낸다. foldfull.json 이 있으면 겹치는 (팔, 시드)에서 비트 대조한다.

실행:  cd final/temp && python dest_fold.py                 (ML2 · GPU · 시드당 약 4분)
       python dest_fold.py --arms FOLD5,BASE --seeds 42,1,2,3,4    (paired · 10런 ≈ 35분)
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

# ══════════════════════════════════════════ 경로·계약 상수
FINAL = Path(__file__).resolve().parent.parent
DATA_DIR = FINAL / 'origin'
MODELS = FINAL / 'models'
ARITH = MODELS / 'ARITH'
sys.path.insert(0, str(ARITH))                 # 배포 패키지 arith (swap 규약) — 로컬 lab import 는 없다
from arith._base import build_swap, catalog_cols   # noqa: E402

SEED = 42
HALF = 5000                     # stage2 우주 경계: [:HALF] valid / [HALF:] holdout
K, INNER = 5, 3                 # 결합기 바깥 교차적합 / 안쪽 λ 선택 fold
ABE1_STEP, TB1_STEP = 3109, 5000
LAMS = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]   # 양축 공통 — 0 = 무벌점 특수해
LAMS0 = list(LAMS)                                          # blk2 λ_L 격자 = 같은 격자
LAM_CEIL = 1e12
MAXIT = 1000
LR, EPOCHS, PATIENCE, WD = 3e-4, 3000, 200, 1e-2
H = 128                                  # 두 축 공통 은닉 폭
TIE_ROUNDS, WIN_ROUNDS = 2, 1            # §3.6 / §3.8
SRCS = ('abe2', 'tb2', 'abe1', 'tb1')
NFOLD = 5                                # Arith xgb 폴드 수 = Arith 열 수
CK = ARITH / 'checkpoint-ab224'
ENC_TIE = ['s_tb2', 's_tb1', 't_abe2', 't_abe1']
ENC_WIN = ['m_abe2', 'm_abe1']
AR_NAMES = {'BASE': (['t_ar'], ['m_ar']),
            'FOLD5': ([f't_ar{k}' for k in range(1, NFOLD + 1)],
                      [f'm_ar{k}' for k in range(1, NFOLD + 1)])}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
R_NZ = 3972 / 5748              # holdout 비-tie 비율 — 축 분해 계수
REF_TOL = 1e-6                  # 비트 재현 문턱 (foldfull 대조)


def set_grid(lam_max):
    """양축 격자 = [0] + 10^0 … 10^log10(lam_max). win_combiner/tie_combiner 는 모듈 전역을 읽는다."""
    global LAMS, LAMS0
    top = int(round(np.log10(lam_max)))
    assert 10.0 ** top == float(lam_max) and top >= 1, f'--lam_max must be a power of ten >= 10, got {lam_max}'
    LAMS = [0.0] + [10.0 ** e for e in range(0, top + 1)]
    LAMS0 = list(LAMS)
    return LAMS


# ══════════════════════════════════════════ 결합기 (층 2)
def rowloss(z, y01):
    return np.logaddexp(0, z) - y01 * z


def _fit(X, y01, pen, bias, w0=None):
    """벌점 로지스틱. pen_j=0 이면 그 열은 조건수를 안 건드리고 정확히 무벌점이다."""
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
    """짝 열만 중심화+표준화. **홀 열은 무중심 RMS** — 중심화는 반대칭을 깬다."""
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


def win_combiner(X, y01, fold, rows, n_logit):
    """승부축 (§3.8) — [m3 무벌점 | W4 λ] · 절편 없음 · 홀 축이라 RMS-only."""
    z = np.full(len(y01), np.nan)
    picked, n_bad = [], 0
    for k in range(K):
        tr, te = (fold != k) & rows, (fold == k) & rows
        Xtr, ytr = X[tr], y01[tr]
        inner = np.zeros(len(ytr), int)
        for j, (_, iv) in enumerate(StratifiedKFold(INNER, shuffle=True, random_state=SEED)
                                    .split(np.zeros((len(ytr), 1)), ytr)):
            inner[iv] = j
        Zi = [_z(Xtr[inner != j], Xtr[inner == j], 0) for j in range(INNER)]
        score, warm = {}, [None] * INNER

        def ev(lam):
            nb, tot = 0, 0.0
            for j in range(INNER):
                A, B = Zi[j]
                pen = np.full(A.shape[1], float(lam))
                pen[:n_logit] = 0.0
                v, ok = _fit(A, ytr[inner != j], pen, False, warm[j])
                warm[j] = v
                nb += (not ok)
                tot += rowloss(_scores(v, B, 0, False), ytr[inner == j]).sum()
            score[lam] = tot
            return nb

        for lam in sorted(set(LAMS), reverse=True):
            n_bad += ev(lam)
        while min(score, key=score.get) == max(score) and max(score) < LAM_CEIL:
            warm = [None] * INNER
            n_bad += ev(max(score) * 10)
        lam = min(score, key=score.get)
        picked.append(lam)
        A, B = _z(Xtr, X[te], 0)
        pen = np.full(X.shape[1], float(lam))
        pen[:n_logit] = 0.0
        v, ok = _fit(A, ytr, pen, False)
        n_bad += (not ok)
        z[te] = _scores(v, B, 0, False)
    lam = sorted(picked)[len(picked) // 2]           # 배포 단일적합 λ = fold 중앙값
    A, _ = _z(X[rows], X[rows], 0)
    pen = np.full(X.shape[1], float(lam))
    pen[:n_logit] = 0.0
    v, ok = _fit(A, y01[rows], pen, False)
    assert ok and n_bad == 0, f'승부 결합기 미수렴 (n_bad {n_bad})'
    return z, dict(lam=float(lam), picked=[float(p) for p in picked],
                   v=[float(x) for x in v])


def tie_combiner(X, y01, fold, rows, nl):
    """tie축 (§3.7 blk2) — pen = [λ_L×로짓 nl | λ_g×g]. 절편 있음(짝 축).

    격자 λ_L ∈ {0}+LAMS × λ_g ∈ LAMS (30점) · 큰 λ 부터 웜스타트 · 자동확장은 축별.
    배포 단일적합 (λ_L, λ_g) = fold 선택의 **성분별 중앙값** (§3.7 계약 3).
    """
    n_even = X.shape[1]
    z = np.full(len(y01), np.nan)
    picked = []
    for k in range(K):
        tr, te = (fold != k) & rows, (fold == k) & rows
        Xtr, ytr = X[tr], y01[tr]
        inner = np.zeros(len(ytr), int)
        for j, (_, iv) in enumerate(StratifiedKFold(INNER, shuffle=True, random_state=SEED)
                                    .split(np.zeros((len(ytr), 1)), ytr)):
            inner[iv] = j
        Zi = [_z(Xtr[inner != j], Xtr[inner == j], n_even) for j in range(INNER)]
        score, warm = {}, [None] * INNER

        def ev(q):
            if q in score:
                return
            tot = 0.0
            for j in range(INNER):
                A, B = Zi[j]
                pen = np.concatenate([np.full(nl, q[0]), np.full(A.shape[1] - nl, q[1])])
                v, _ok = _fit(A, ytr[inner != j], pen, True, warm[j])
                warm[j] = v
                s = _scores(v, B, n_even, True)
                yy = ytr[inner == j]
                tot += float((np.logaddexp(0, s) - yy * s).sum())
            score[q] = tot

        gl, gg = list(LAMS0), list(LAMS)
        for a in sorted(gl, reverse=True):
            for b in sorted(gg, reverse=True):
                ev((a, b))
        while True:
            q = min(score, key=score.get)
            ext = False
            if q[0] == max(gl) and max(gl) < LAM_CEIL:
                gl.append(max(gl) * 10.0)
                for b in gg:
                    ev((gl[-1], b))
                ext = True
            if q[1] == max(gg) and max(gg) < LAM_CEIL:
                gg.append(max(gg) * 10.0)
                for a in gl:
                    ev((a, gg[-1]))
                ext = True
            if not ext:
                break
        picked.append(q)
        A, B = _z(Xtr, X[te], n_even)
        pen = np.concatenate([np.full(nl, q[0]), np.full(X.shape[1] - nl, q[1])])
        v, _ok = _fit(A, ytr, pen, True)
        z[te] = _scores(v, B, n_even, True)
    lam = (float(np.median([q[0] for q in picked])), float(np.median([q[1] for q in picked])))
    A, _ = _z(X[rows], X[rows], n_even)
    pen = np.concatenate([np.full(nl, lam[0]), np.full(X.shape[1] - nl, lam[1])])
    v, ok = _fit(A, y01[rows], pen, True)
    assert ok, 'tie 결합기 배포 적합 미수렴'
    return z, dict(lam=lam, picked=[tuple(map(float, q)) for q in picked],
                   v=[float(x) for x in v])


# ══════════════════════════════════════════ 조립 (§3.5 · 학습 파라미터 없음)
def assemble(z_tie, z_win):
    """로그 공간 체인 룰 — softmax 없음. 열 순서 [P(A), P(B), P(tie)].

    log P(A)   = −softplus(z_tie) − softplus(−z_win)      = log(1−p₀) + log q
    log P(B)   = −softplus(z_tie) − softplus(+z_win)
    log P(tie) = −softplus(−z_tie)
    합=1 은 나눗셈이 아니라 대수적 항등이라 정규화 단계가 없다 (§5 조립 항등, 실측 확정).
    """
    sp = lambda x: np.logaddexp(0.0, x)
    logP = np.column_stack([-sp(z_tie) - sp(-z_win),
                            -sp(z_tie) - sp(z_win),
                            -sp(-z_tie)])
    return logP


def score(U, z_tie, z_win, seg):
    """seg 행의 3-class log loss. tie 행은 z_win 이 없어도 P(tie) 가 결정되므로 축으로 센다.

    (= assemble() 의 손실과 항등: −log P(y) = softplus(±z_tie) [+ softplus(∓z_win)])
    """
    y, nz = U['y'], U['nz']
    yT, yA = (y == 2).astype(np.float64), (y == 0).astype(np.float64)
    lt = rowloss(z_tie[seg], yT[seg])
    lw = rowloss(z_win[seg & nz], yA[seg & nz])
    row = lt.astype(np.float64).copy()
    row[nz[seg]] += lw
    return row, float(lt.mean()), float(lw.mean())


# ══════════════════════════════════════════ 자산
sd3 = lambda B: (B[:, 0] - B[:, 1], B[:, 2] - np.logaddexp(B[:, 0], B[:, 1]))

def _tta(rel, n_class):
    """뷰평균 — (fwd + swp)/2. 이 접기로 아래 패리티가 정확해진다."""
    z = np.load(MODELS / rel)
    L = np.vstack([(z['valid_fwd'] + z['valid_swp']) / 2,
                   (z['hold_fwd'] + z['hold_swp']) / 2]).astype(np.float64)
    assert L.shape[1] == n_class, f'{rel} 열 {L.shape[1]} != {n_class}'
    return L, z



FEAT_CACHE = FINAL / 'temp/arith_feats_stage1/F.npz'      # arith_train 이 재학습 때 쓴 피쳐 그대로
OLD_PARQUET = FINAL / 'temp/arith_old/features_57k.parquet'  # 구 계약(256토큰) 보관본 — --old-feats 전용


def arith_fold_logits(old=False):
    """Arith 폴드별 raw 마진 — id 색인 + (NFOLD, n, 3). **피쳐 출처는 여기 한 곳.**

    기본: 재학습 캐시 temp/arith_feats_stage1/F.npz (arith_train.feature_matrix 가 저장; 부스터가
          학습한 바로 그 행렬) × models/ARITH/checkpoint-ab224.  stage1 57,477행 전부를 덮는다.
    old : 구 피쳐 parquet × temp/arith_old/checkpoint-ab224 — 2026-08-21 까지의 프로브 재현용.
    폴드마다 계약과 같은 swap-TTA 를 걸어 마진 공간에서 접으므로, 5열의 평균은 배포 npz 의
    `margins` 와 같다 (배치 구성에 따른 마지막 비트 드리프트 제외).
    """
    import xgboost as xgb
    if old:
        ck = FINAL / 'temp/arith_old' / CK.name
        F = pd.read_parquet(OLD_PARQUET)
        ids, have, Xall = F['id'].to_numpy(), list(F.columns), None
        src = 'old parquet(256tok) × arith_old/checkpoint-ab224'
    else:
        ck = CK
        assert FEAT_CACHE.exists(), (f'{FEAT_CACHE} 가 없다 — temp/arith/retrain_ab224.py --fresh 로 '
                                     f'재추출·재학습을 먼저 돌려라 (구 계약으로 보려면 --old-feats)')
        z = np.load(FEAT_CACHE, allow_pickle=False)
        ids, have, Xall = z['ids'], [str(c) for c in z['cols']], z['X']
        src = 'arith_feats_stage1/F.npz × checkpoint-ab224'
    meta = json.load(open(ck / 'meta.json'))
    cols = list(meta['cols'])
    if Xall is None:
        X = np.ascontiguousarray(F[cols].to_numpy(np.float32))
    else:
        X = np.ascontiguousarray(Xall[:, [have.index(c) for c in cols]].astype(np.float32))
    assert np.isfinite(X).all(), '비유한 피쳐값'
    _, swap_series = catalog_cols(DATA_DIR / 'feature_catalog.csv')
    swap_X, _ = build_swap(cols, swap_series)
    dX, dXs = xgb.DMatrix(X), xgb.DMatrix(swap_X(X))
    M = np.zeros((NFOLD, len(X), 3), np.float64)
    for k in range(1, NFOLD + 1):
        m = xgb.XGBClassifier()
        m.load_model(str(ck / f'xgb_fold{k}.json'))
        bst, bi = m.get_booster(), getattr(m, 'best_iteration', None)
        kw = dict(output_margin=True)
        if bi is not None:
            kw['iteration_range'] = (0, bi + 1)
        M[k - 1] = (bst.predict(dX, **kw) + bst.predict(dXs, **kw)[:, [1, 0, 2]]) / 2
    emb = meta.get('source', {}).get('embedder', {'max_seq_len': '?(구 meta — 256 추정)'})
    print(f'  arith 폴드 마진 {M.shape}  ({src} · 임베더 {emb})')
    return pd.Index(ids), M


def arith_cols(M, arm):
    """(NFOLD, n, 3) → (t 열, m 열). BASE = 마진 평균 1열 · FOLD5 = 폴드별 5열."""
    if arm == 'BASE':
        m, t = sd3(M.mean(0))
        return t[:, None], m[:, None]
    m = np.stack([sd3(M[k])[0] for k in range(NFOLD)], 1)
    t = np.stack([sd3(M[k])[1] for k in range(NFOLD)], 1)
    return t, m


def build():
    """stage2 10,748 — 인코더 로짓 기저 + 행 우주. Arith 열은 팔별로 뒤에 붙인다."""
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

    m_abe2, t_abe2 = sd3(L_abe2)
    m_abe1, t_abe1 = sd3(L_abe1)
    enc_tie = np.column_stack([L_tb2[:, 1] - L_tb2[:, 0], L_tb1[:, 1] - L_tb1[:, 0],
                               t_abe2, t_abe1])
    enc_win = np.column_stack([m_abe2, m_abe1])

    # clean = 결합기 적합 우주. valid 5,000 중 origin/valid.csv 의 2,000 (어떤 전문가도 못 본 행)
    cv = pd.read_csv(DATA_DIR / 'valid.csv', usecols=['id'])
    assert len(cv) == 2000, f'origin/valid.csv {len(cv)}행 != clean 2,000'
    cu = np.ones(n, bool)
    cu[:HALF] = pd.Index(cv['id']).get_indexer(ids[:HALF]) >= 0
    assert int(cu.sum()) == 7748, f'clean {int(cu.sum())} != 7748'
    assert cu[hold].all(), 'hold ⊄ cu — 회계 전제 붕괴'

    fold = np.zeros(n, int)
    for k, (_, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=SEED)
                                .split(np.zeros((n, 1)), y)):
        fold[te] = k

    nh, nw = int(hold.sum()), int((hold & nz).sum())
    assert (nh, nw) == (5748, 3972), f'holdout 우주 {(nh, nw)}'
    print(f'stage2 {n:,}행 — clean {int(cu.sum()):,} (tie 적합) / '
          f'clean∩비-tie {int((cu & nz).sum()):,} (승부 적합) | '
          f'holdout {nh:,} 중 비-tie {nw:,} (비율 {nw / nh:.3f})')
    return dict(ids=ids, y=y, n=n, hold=hold, nz=nz, cu=cu, fold=fold,
                enc_tie=enc_tie, enc_win=enc_win)


def load_train_logits():
    """TR 46,729 의 인코더 로짓 — tie 짝 4열 · 승부 홀 2열. 전 파일을 abe2 train npz 의 id 로 대조."""
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
    m_abe2, t_abe2 = sd3(L_abe2)
    m_abe1, t_abe1 = sd3(L_abe1)
    enc_tie = np.column_stack([L_tb2[:, 1] - L_tb2[:, 0], L_tb1[:, 1] - L_tb1[:, 0],
                               t_abe2, t_abe1])
    enc_win = np.column_stack([m_abe2, m_abe1])
    return ids, y.astype(np.int64), enc_tie, enc_win


def arith_sources(U, tr_ids, arms, old=False):
    """팔별 (t_s2, m_s2, t_tr, m_tr). stage2·TR 모두 같은 폴드 마진에서 — id 로 색인."""
    fid, M = arith_fold_logits(old=old)
    i_s2, i_tr = fid.get_indexer(U['ids']), fid.get_indexer(tr_ids)
    assert (i_s2 >= 0).all() and (i_tr >= 0).all(), 'arith id 커버리지'
    out = {}
    for arm in arms:
        t2, m2 = arith_cols(M[:, i_s2], arm)
        tt, mt = arith_cols(M[:, i_tr], arm)
        assert all(np.isfinite(x).all() for x in (t2, m2, tt, mt)), f'{arm}: 비유한 arith 열'
        out[arm] = (t2, m2, tt, mt)
    if 'FOLD5' in out:
        R = np.corrcoef(out['FOLD5'][1][U['hold']].T)
        print(f'  arith 폴드 m 쌍상관 (holdout) 평균 {R[np.triu_indices(NFOLD, 1)].mean():.4f}')
    return out


def load_svd_cache():
    """TR 46,729 짜리 SVD-512 화이트닝 캐시 (짝·홀 각각). 행 수로 고른다."""
    for p in sorted((FINAL / 'temp/head_cache').glob('svd512-*.npz')):
        z = np.load(p)
        if z['CUt'].shape == (46729, 512) and z['CUs'].shape == (10748, 512):
            assert int(z['K_TIE']) == 512, f'{p.name} K_TIE {int(z["K_TIE"])}'
            print(f'  SVD 캐시 {p.name}  CUt {z["CUt"].shape} CVt {z["CVt"].shape}')
            return z
    raise FileNotFoundError('temp/head_cache 에 46,729행 svd512 캐시가 없다')


def zstats(X, even):
    mu = X.mean(0) if even else np.zeros(X.shape[1])
    sc = np.clip(X.std(0) if even else np.sqrt((X ** 2).mean(0)), 1e-9, None)
    return mu, sc



def head_assets(U, Z, TR, src):
    """헤드 학습 재료 — TR 로짓(정규화)·SVD. 정규화 통계는 **TR 전용**. Arith 열은 팔의 것."""
    tr_ids, y_tr, enc_tie_tr, enc_win_tr = TR
    t2, m2, tt, mt = src
    tie_tr, win_tr = np.hstack([enc_tie_tr, tt]), np.hstack([enc_win_tr, mt])
    tie_s2, win_s2 = np.hstack([U['enc_tie'], t2]), np.hstack([U['enc_win'], m2])
    print(f'  train 로짓 {tie_tr.shape[0]:,}행 (tie {tie_tr.shape[1]}열 / '
          f'win {win_tr.shape[1]}열) | 장치 {DEVICE}')
    mu_u, sc_u = zstats(tie_tr, True)                 # 짝: z-score
    _, sc_v = zstats(win_tr, False)                   # 홀: RMS, 무중심
    A = dict(y_tr=y_tr, y01m=(y_tr == 2).astype(np.float64),
             Lu_tr=(tie_tr - mu_u) / sc_u, Lv_tr=win_tr / sc_v,
             Lu_s2=(tie_s2 - mu_u) / sc_u, Lv_s2=win_s2 / sc_v,
             CUt=Z['CUt'].astype(np.float32), CVt=Z['CVt'].astype(np.float32),
             CUs=Z['CUs'].astype(np.float32), CVs=Z['CVs'].astype(np.float32))
    return A, tie_s2, win_s2


def raw_uv(Z):
    """승부축 공급 — 소스별 raw (u,v). joint 캐시의 NU/NV 통계로 **좌표 정규화만**.

    기저 변환(회전·절단·화이트닝)을 하지 않는 것이 §3.8 의 확정이다 — 승부 신호는
    고랭크·좌표 정렬이라 어떤 회전도 손해다.
    """
    spec = [('abe2__checkpoint-3000', ('ea',), ('eb',)),
            ('tb2__checkpoint-4000', ('a_f', 'b_s'), ('b_f', 'a_s')),
            (f'abe1__checkpoint-{ABE1_STEP}', ('ea',), ('eb',)),
            (f'tb1__checkpoint-{TB1_STEP}', ('x_f',), ('x_s',))]
    out = {}
    for i, (tag, ka, kb) in enumerate(spec):
        zt = np.load(MODELS / f'pooled/{tag}__train.npz')
        zs = np.load(MODELS / f'pooled/{tag}.npz')
        g = lambda z, pre, ks: sum(z[f'{pre}_{k}'].astype(np.float32) for k in ks) / len(ks)
        c = lambda z, ks: sum(np.vstack([z[f'valid_{k}'], z[f'hold_{k}']]).astype(np.float32)
                              for k in ks) / len(ks)
        At, Bt, As, Bs = g(zt, 'train', ka), g(zt, 'train', kb), c(zs, ka), c(zs, kb)
        sl = slice(i * 1024, (i + 1) * 1024)
        mu, sc = Z['NU_mu'][sl].astype(np.float32), Z['NU_sc'][sl].astype(np.float32)
        vsc = Z['NV_sc'][sl].astype(np.float32)
        out[SRCS[i]] = dict(
            u_tr=((At + Bt) / 2 - mu) / sc, v_tr=((At - Bt) / 2) / vsc,
            u_s2=((As + Bs) / 2 - mu) / sc, v_s2=((As - Bs) / 2) / vsc)
        del At, Bt, As, Bs
        gc.collect()
    print(f'  raw 소스별 적재 — 4소스 × (u,v) 1,024  [TR {out["abe2"]["u_tr"].shape[0]:,}행]')
    return out


# ══════════════════════════════════════════ 헤드 (층 1)
def _es_split(n_, seed, frac=0.15):
    perm = np.random.default_rng(seed).permutation(n_)
    n_es = max(int(n_ * frac), 200)
    return perm[:n_es], perm[n_es:]


class HeadLin(nn.Module):
    """판독 — 순수 선형. 대칭화 뒤에 놓는다 (readout.sym_post)."""

    def __init__(self, d_in):
        super().__init__()
        self.w = nn.Linear(d_in, 1)

    def forward(self, u):
        return self.w(u)


class BodyTT(nn.Module):
    """r1 — tanh(W1x) ⊙ tanh(W2x). 사영 두 개를 각각 포화시켜 곱한다(합성 아님)."""

    def __init__(self, d, h):
        super().__init__()
        self.W1, self.W2 = nn.Linear(d, h), nn.Linear(d, h)
        self.out_dim = h

    def forward(self, x):
        return torch.tanh(self.W1(x)) * torch.tanh(self.W2(x))


class BodyBL(nn.Module):
    """r2 — (W1x) ⊙ (W2x). 순수 쌍선형(무계). 앵커가 본체를 담아 증분이 작다."""

    def __init__(self, d, h):
        super().__init__()
        self.W1, self.W2 = nn.Linear(d, h), nn.Linear(d, h)
        self.out_dim = h

    def forward(self, x):
        return self.W1(x) * self.W2(x)


TIE_BODIES = {'TANHTANH': BodyTT, 'BILIN': BodyBL}


class TieNet(nn.Module):
    """tie 헤드 — 몸통은 풀링만, 로짓 8 은 판독 직결. `.w` 는 프로퍼티(이중 등록 방지)."""

    def __init__(self, d, h, body_kind, skip_idx, body_idx):
        super().__init__()
        self.body_idx, self.skip_idx = body_idx, skip_idx
        self.body = TIE_BODIES[body_kind](len(body_idx), h)
        self.head = HeadLin(self.body.out_dim + len(skip_idx))

    @property
    def w(self):
        return self.head.w


class WinBody(nn.Module):
    """승부 GATE — σ(G·u) ⊙ tanh(V·v). 게이트 가지는 짝(u), 값 가지는 홀(v·무절편)."""

    def __init__(self, d, h, d_u):
        super().__init__()
        self.d_u, self.out_dim = d_u, h
        self.G = nn.Linear(d_u, h)
        self.V = nn.Linear(d - d_u, h, bias=False)

    def forward(self, x):
        return torch.sigmoid(self.G(x[:, :self.d_u])) * torch.tanh(self.V(x[:, self.d_u:]))


class WinNet(nn.Module):
    """승부 헤드 — 판독 절편은 생성 시점부터 0 동결 (홀 축 무절편 계약)."""

    def __init__(self, d, h, skip_idx, body_idx, d_u):
        super().__init__()
        self.body_idx, self.skip_idx = body_idx, skip_idx
        self.body = WinBody(len(body_idx), h, d_u)
        self.head = HeadLin(self.body.out_dim + len(skip_idx))
        nn.init.zeros_(self.head.w.bias)
        self.head.w.bias.requires_grad_(False)

    @property
    def w(self):
        return self.head.w


def sym_post(f, Ux, Vx, sgn):
    """대칭화 뒤 판독 — u = g(x⁺) + sgn·g(x⁻), sk = L(x⁺) + sgn·L(x⁻).

    sgn=+1(tie, 짝)이면 홀 로짓 3칸이, sgn=−1(승부, 홀)이면 짝 로짓 5칸이 **정확히 0** 이
    된다. 그 칸의 가중치는 입력이 0 이라 학습 내내 0 에 머문다 — 8칸 유지가 재현에 안전.
    """
    xp = torch.cat([Ux, Vx], 1)
    xm = torch.cat([Ux, -Vx], 1)
    u = f.body(xp[:, f.body_idx]) + sgn * f.body(xm[:, f.body_idx])
    u = torch.cat([u, xp[:, f.skip_idx] + sgn * xm[:, f.skip_idx]], 1)
    return f.head(u).squeeze(-1)


def fit_head(CUt_, CVt_, y01, s0, even, seed, CUs_, CVs_, net, hidden=H):
    """헤드 1개 학습 — TR 46,729, ES = TR 15%, AdamW, 풀배치, best state 복원.

    s0 를 주면 앵커 부스팅: 목적 z = a·s0z + f(x) 이고 **내보내는 열은 f 출력만**.
    판독 w 는 0 초기화라 첫 순전파가 정확히 0 이다 (앵커 계약).
    """
    torch.manual_seed(seed)
    d = CUt_.shape[1] + CVt_.shape[1]
    f = net(d, hidden)
    nn.init.zeros_(f.w.weight)
    nn.init.zeros_(f.w.bias)
    f = f.to(DEVICE)
    anchored = s0 is not None
    a = nn.Parameter(torch.tensor(1.0, device=DEVICE))
    opt = torch.optim.AdamW([{'params': f.parameters(), 'weight_decay': WD},
                             {'params': [a], 'weight_decay': 0.0}], lr=LR)
    rms = float(np.clip(np.sqrt((np.asarray(s0, np.float64) ** 2).mean()), 1e-9, None)) \
        if anchored else 1.0
    s0z = (np.asarray(s0, np.float64) / rms) if anchored else np.zeros(len(y01))
    T = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device=DEVICE)
    U_, V_, S_, Y_ = T(CUt_), T(CVt_), T(s0z), T(y01)
    lossf = nn.BCEWithLogitsLoss()
    sgn = 1 if even else -1
    out = lambda Ux, Vx: sym_post(f, Ux, Vx, sgn)

    eidx, tidx = _es_split(len(y01), seed)
    Ue, Ve, Se, Ye = U_[eidx], V_[eidx], S_[eidx], Y_[eidx]
    Ut, Vt, St, Yt = U_[tidx], V_[tidx], S_[tidx], Y_[tidx]

    best, best_sd, bad, used = np.inf, None, 0, 0
    for ep in range(EPOCHS):
        f.train()
        opt.zero_grad()
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
    return r_s2, dict(z_tr=z_tr, es_ll=best, es_epoch=used, state=best_sd[0])


def tie_head(A, seed):
    """§3.6 — TANHTANH → BILIN(TR 잔차앵커) · h128 · 몸통 풀링 1,024 · 로짓 8 판독 직결."""
    Kd = A['CUt'].shape[1]
    nu, nv = A['Lu_tr'].shape[1], A['Lv_tr'].shape[1]
    sk = list(range(Kd, Kd + nu)) + list(range(2 * Kd + nu, 2 * Kd + nu + nv))
    pool = list(range(0, Kd)) + list(range(Kd + nu, 2 * Kd + nu))
    assert len(pool) == 2 * Kd and not set(pool) & set(sk), '풀링/로짓 열 분리'
    ct = np.ascontiguousarray(np.hstack([A['CUt'], A['Lu_tr']]), np.float32)
    cv = np.ascontiguousarray(np.hstack([A['CVt'], A['Lv_tr']]), np.float32)
    cs = np.ascontiguousarray(np.hstack([A['CUs'], A['Lu_s2']]), np.float32)
    vs = np.ascontiguousarray(np.hstack([A['CVs'], A['Lv_s2']]), np.float32)
    z, cols, states = None, [], []
    for rd, unit in enumerate(('TANHTANH', 'BILIN')[:TIE_ROUNDS], 1):
        net = lambda d, h, _u=unit: TieNet(d, h, _u, sk, pool)
        r, aux = fit_head(ct, cv, A['y01m'], z, True, seed, cs, vs, net)
        cols.append(r)
        states.append(aux['state'])
        z = aux['z_tr']
        print(f'      tie r{rd} {unit:9s} es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]}')
    return np.column_stack(cols), states


def win_head(A, RAW, seed):
    """§3.8 — GATE · 소스별 raw · h128 · SK 판독 · 1R. 열 순서 = 소스 순서."""
    nu, nv = A['Lu_tr'].shape[1], A['Lv_tr'].shape[1]
    m = A['y_tr'] != 2                       # 승부 헤드는 비-tie 행에서만 학습한다
    y01w = (A['y_tr'] == 0).astype(np.float64)[m]
    cols, states = [], []
    for si, s in enumerate(SRCS):
        R = RAW[s]
        du, dv = R['u_tr'].shape[1], R['v_tr'].shape[1]
        sk = list(range(du, du + nu)) + list(range(du + nu + dv, du + nu + dv + nv))
        pool = list(range(0, du)) + list(range(du + nu, du + nu + dv))
        cu_t = np.ascontiguousarray(np.hstack([R['u_tr'][m], A['Lu_tr'][m]]), np.float32)
        cv_t = np.ascontiguousarray(np.hstack([R['v_tr'][m], A['Lv_tr'][m]]), np.float32)
        cu_s = np.ascontiguousarray(np.hstack([R['u_s2'], A['Lu_s2']]), np.float32)
        cv_s = np.ascontiguousarray(np.hstack([R['v_s2'], A['Lv_s2']]), np.float32)
        net = lambda d, h, _s=sk, _p=pool, _du=du: WinNet(d, h, _s, _p, _du)
        r_s2, aux = fit_head(cu_t, cv_t, y01w, None, False, seed + si, cu_s, cv_s, net)
        cols.append(r_s2)
        states.append(aux['state'])
        print(f'      win r1 {s:5s} es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]}')
        del cu_t, cv_t, cu_s, cv_s
    gc.collect()
    return np.column_stack(cols), states


# ══════════════════════════════════════════ 배선 가드
@torch.no_grad()
def wiring_checks(nu, nv, Kd=512, du=1024):
    """§3.6 체크리스트 ①②③ 를 로짓 폭 (nu, nv) 에 일반화 — 0초기 출력 0 · 파리티.

    (5, 3) 이면 destnation 의 파라미터 수 262,537 / 262,409 가 그대로 나와야 한다.
    """
    g = torch.Generator().manual_seed(0)
    sk = list(range(Kd, Kd + nu)) + list(range(2 * Kd + nu, 2 * Kd + nu + nv))
    pool = list(range(0, Kd)) + list(range(Kd + nu, 2 * Kd + nu))
    tie = TieNet(2 * Kd + nu + nv, H, 'TANHTANH', sk, pool)
    n_tie = sum(p.numel() for p in tie.parameters())
    Ux, Vx = torch.randn(64, Kd + nu, generator=g), torch.randn(64, Kd + nv, generator=g)
    nn.init.zeros_(tie.w.weight)
    nn.init.zeros_(tie.w.bias)
    z0 = float(sym_post(tie, Ux, Vx, +1).abs().max())
    nn.init.normal_(tie.w.weight)
    ev = float((sym_post(tie, Ux, Vx, +1) - sym_post(tie, Ux, -Vx, +1)).abs().max())

    skw = list(range(du, du + nu)) + list(range(2 * du + nu, 2 * du + nu + nv))
    poolw = list(range(0, du)) + list(range(du + nu, 2 * du + nu))
    win = WinNet(2 * du + nu + nv, H, skw, poolw, du)
    n_win = sum(p.numel() for p in win.parameters())
    Uw, Vw = torch.randn(64, du + nu, generator=g), torch.randn(64, du + nv, generator=g)
    nn.init.normal_(win.w.weight)
    od = float((sym_post(win, Uw, Vw, -1) + sym_post(win, Uw, -Vw, -1)).abs().max())

    print(f'[배선 nu={nu} nv={nv}] 층1 파라미터 tie {n_tie:,}×{TIE_ROUNDS} · win {n_win:,}×4 | '
          f'0초기 출력 {z0:.1e} · tie 짝 위반 {ev:.1e} · win 홀 위반 {od:.1e}')
    if (nu, nv) == (5, 3):
        assert n_tie == 262537 and n_win == 262409, '파라미터 수 계약 위반 (destnation 과 다름)'
    assert z0 == 0.0 and ev < 1e-4 and od < 1e-4, '파리티/0초기 계약 위반'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seeds', default='42,1,2,3,4')
    ap.add_argument('--arms', default='FOLD5', help='FOLD5 | FOLD5,BASE (paired)')
    ap.add_argument('--lam_max', type=float, default=1e5, help='grid top, power of ten (1e4 = foldfull)')
    ap.add_argument('--old-feats', action='store_true', help='구 피쳐 parquet + arith_old 부스터 (2026-08-21 프로브 재현)')
    ap.add_argument('--out', default='temp/dest_fold')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s]
    arms = [a for a in args.arms.split(',') if a]
    grid = set_grid(args.lam_max)
    print(f'[격자] 양축 λ ∈ {[f"{g:g}" for g in grid]}  (자동 확장은 그 위로 ×10, 상한 {LAM_CEIL:g})')
    t_all = time.time()

    print('[적재]')
    U = build()
    y, fold, cu, nz, hold = U['y'], U['fold'], U['cu'], U['nz'], U['hold']
    yT, yA = (y == 2).astype(np.float64), (y == 0).astype(np.float64)
    mw = cu & nz
    TR = load_train_logits()
    SRC = arith_sources(U, TR[0], arms, old=args.old_feats)
    Z = load_svd_cache()
    RAW = raw_uv(Z)
    wiring_checks(5, 3)
    wiring_checks(4 + NFOLD, 2 + NFOLD)

    # foldfull 대조 — 같은 코드 경로. 격자가 1e4 로 같으면 비트 단위로, 1e5 면 picked 가
    # 바뀌지 않는 한 같아야 한다 (바깥 적합은 cold start 라 내부 CV 선택만 결과를 정한다)
    ff = FINAL / 'temp/foldfull.json'                       # 구 피쳐 런 — --old-feats 일 때만 비교가 성립
    FF = json.load(open(ff))['cells'] if (ff.exists() and args.old_feats) else {}

    res, rows_, store = {}, {}, {}
    for arm in arms:
        A, tie_s2, win_s2 = head_assets(U, Z, TR, SRC[arm])
        nl, nw_ = tie_s2.shape[1], win_s2.shape[1]
        for sd in seeds:
            tag = f'{arm}-s{sd}'
            print(f'\n{"=" * 78}\n[{tag}]  tie 로짓 {nl} · 승부 로짓 {nw_}\n{"=" * 78}')
            t0 = time.time()
            Gt, st_t = tie_head(A, sd)
            W, st_w = win_head(A, RAW, sd)

            z_tie, dep_t = tie_combiner(np.hstack([tie_s2, Gt]), yT, fold, cu, nl)
            z_win, dep_w = win_combiner(np.hstack([win_s2, W]), yA, fold, mw, nw_)

            row, tie_ll, win_ll = score(U, z_tie, z_win, hold)
            lp = assemble(z_tie[hold & nz], z_win[hold & nz])         # 조립 산출물 (배포 형식)
            assert abs(float(np.abs(np.exp(lp).sum(1) - 1).max())) < 1e-9, '조립 확률합 위반'

            res[tag] = dict(arm=arm, seed=sd, tot=float(row.mean()), tie=tie_ll, win=win_ll,
                            tie_lam=dep_t['lam'], tie_picked=dep_t['picked'],
                            win_lam=dep_w['lam'], win_picked=dep_w['picked'],
                            tie_w=dep_t['v'], win_w=dep_w['v'],
                            sec=round(time.time() - t0, 1))
            rows_[f'row_{tag}'] = row
            rows_[f'tie_{tag}'] = rowloss(z_tie[hold], yT[hold])
            rows_[f'win_{tag}'] = rowloss(z_win[hold & nz], yA[hold & nz])
            store[f'Gt_{tag}'] = Gt.astype(np.float32)
            store[f'W_{tag}'] = W.astype(np.float32)
            store[f'z_tie_{tag}'], store[f'z_win_{tag}'] = z_tie, z_win
            line = (f'  tie {tie_ll:.5f} (λ {dep_t["lam"]}) · win {win_ll:.5f} (λ {dep_w["lam"]:g}) '
                    f'→ **{row.mean():.5f}**  [{res[tag]["sec"]:.0f}s]')
            if tag in FF:
                d = max(abs(tie_ll - FF[tag]['tie']), abs(win_ll - FF[tag]['win']))
                line += f'   | foldfull 대조 max|Δ| {d:.1e} ' + ('비트 일치' if d < REF_TOL else '*** 불일치')
            print(line)
            torch.save({'tie': st_t, 'win': st_w, 'arm': arm, 'seed': sd,
                        'tie_names': ENC_TIE + AR_NAMES[arm][0],
                        'win_names': ENC_WIN + AR_NAMES[arm][1]},
                       FINAL / f'{args.out}_heads_{tag}.pt')

    op = FINAL / args.out
    np.savez_compressed(op.with_suffix('.rowloss.npz'), **rows_, **store)
    with open(op.with_suffix('.json'), 'w') as f:
        json.dump({'cells': res, 'seeds': seeds, 'arms': arms, 'lams': LAMS,
                   'names': {a: dict(tie=ENC_TIE + AR_NAMES[a][0] + [f'g{i}' for i in range(1, TIE_ROUNDS + 1)],
                                     win=ENC_WIN + AR_NAMES[a][1] + list(SRCS)) for a in arms}},
                  f, ensure_ascii=False, indent=1)
    print(f'\n저장: {op}.rowloss.npz / {op}.json / {args.out}_heads_*.pt')

    # ── 시드 평균 (노트북 보고 형식)
    for arm in arms:
        g = lambda k: np.array([res[f'{arm}-s{sd}'][k] for sd in seeds])
        print(f'\n{"=" * 78}\n[{arm}] {len(seeds)}시드 · holdout 5,748\n{"=" * 78}')
        print(f'  {"시드":6s}{"tie ll":>10s}{"win ll":>10s}{"총 3-class":>12s}   λ(tie) / λ(win)')
        for sd in seeds:
            c = res[f'{arm}-s{sd}']
            print(f'  {sd:<6d}{c["tie"]:10.5f}{c["win"]:10.5f}{c["tot"]:12.5f}   '
                  f'{tuple(c["tie_lam"])} / {c["win_lam"]:g}')
        sdv = g('tot').std(ddof=1) if len(seeds) > 1 else 0.0
        print(f'  {"평균":6s}{g("tie").mean():10.5f}{g("win").mean():10.5f}'
              f'{g("tot").mean():12.5f}   (시드 sd {sdv:.5f})')
        print(f'  축 분해  {g("tie").mean():.5f} + {R_NZ:.3f} × {g("win").mean():.5f} = '
              f'{g("tie").mean() + R_NZ * g("win").mean():.5f}  (총점과 항등)')

    # ── paired — 두 팔을 같이 돌렸을 때만
    if 'FOLD5' in arms and 'BASE' in arms:
        print(f'\n{"=" * 78}\n[paired] FOLD5 − BASE · {len(seeds)}시드 · 행별 부트스트랩 95% CI\n{"=" * 78}')
        print(f'  {"시드":6s}{"Δtie":>10s}{"Δwin":>10s}{"Δtot":>10s}')
        for sd in seeds:
            b, f5 = res[f'BASE-s{sd}'], res[f'FOLD5-s{sd}']
            print(f'  {sd:<6d}{f5["tie"] - b["tie"]:+10.5f}{f5["win"] - b["win"]:+10.5f}'
                  f'{f5["tot"] - b["tot"]:+10.5f}')
        rng = np.random.default_rng(SEED)
        for ax, key in (('tie', 'tie'), ('win', 'win'), ('tot', 'row')):
            a = np.mean([rows_[f'{key}_FOLD5-s{s}'] for s in seeds], 0)
            b = np.mean([rows_[f'{key}_BASE-s{s}'] for s in seeds], 0)
            d = a - b
            bs = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(1)
            lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
            per = np.array([res[f'FOLD5-s{s}'][ax] - res[f'BASE-s{s}'][ax] for s in seeds])
            print(f'  {ax:4s} Δ {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}] '
                  f'{"유의 ★" if (lo > 0 or hi < 0) else "노이즈":8s} 시드 {int((per < 0).sum())}/{len(seeds)}')
        print('  ! tot = tie + 0.691 × win — 리더보드 단위는 tot 행')
    print(f'\n총 {((time.time() - t_all) / 60):.1f}분')


if __name__ == '__main__':
    main()
