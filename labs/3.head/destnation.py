"""destnation.py — 확정 구조 조립·실측 (단독 파일, 로컬 import 없음).

구조 탐색은 끝났다. 이 파일은 **확정 다이어그램 그대로** 조립해 점수를 내고 배포용 열·계수를
저장한다. 실험 팔·확인 프로브는 없다 (전부 §3.6/§3.7/§3.8/§5 에서 닫혔다).

    입력   Q, answer_A, answer_B
      ↓
    전문가 5   ABE1 ABE2 (bi) · TB1 TB2 (cross) · Arith (XGB)
      ↓ 로짓 8 = tie 짝 5 [s_tb2 s_tb1 t_abe2 t_abe1 t_ar] · 승부 홀 3 [m_abe2 m_abe1 m_ar]
      ↓ 풀링 4소스 × (u,v) 1024
      │
      ├─ tie 축 (짝)   SVD-512 화이트닝 [CUt|CVt] 1,024 → TB 체인 2R
      │                  r1 tanh(W1x)⊙tanh(W2x) → r2 (W1x)⊙(W2x) 잔차앵커 · h128
      │                판독 Linear([u 128 | 로짓 8] → 1)   ※ 대칭화에서 홀 3칸 소멸
      │                결합기 [로짓5 | g1 g2] 7열 + 절편 · blk2 릿지 [λ_L×5 | λ_g×2]  → z_tie
      │
      ├─ 승부 축 (홀)  소스별 raw (u,v) 1024 — 기저 변환 없음 → GATE ×4소스 1R
      │                  σ(G·u) ⊙ tanh(V·v) · h128
      │                판독 Linear([h 128 | 로짓 8] → 1) · 절편 0 동결  ※ 짝 5칸 소멸
      │                결합기 [m3 무벌점 | W4] 7열 · 절편 없음 (LIN12 대체)      → z_win
      │
      └─ 조립 (학습 파라미터 없음 · softmax 없음)
             p₀ = σ(z_tie),  q = σ(z_win)
             P(A) = (1−p₀)·q,  P(B) = (1−p₀)·(1−q),  P(tie) = p₀
             log loss = tie ll + (비-tie 비율) × win ll   ← 정확히 분해된다

계약 (진실의 원천 plan.md)
    §3.6 tie 헤드 · §3.7 층2 결합기(blk2) · §3.8 승부축 · §3.5 출력 · §5 닫힌 항목
    Arith = tie_ladder 기준 `ARITH/outputs/arith_experts_raw.npz` (캠페인 전체가 이 위에서
    측정됐고, arith 변경 민감도는 §3.6 "층1 오염 민감도" −0.00007(3/3) 로 닫혔다)
    위생: 헤드는 TR 46,729 에서만 학습(ES 도 TR 15%) · stage2 는 결합기 5-fold 교차적합과
    평가에만 · holdout 5,748 은 학습·선택에 일절 안 들어간다

배선 가드 (실험이 아니라 오류 검출 — 어긋나면 즉시 멈춘다)
    파라미터 262,537(tie)·262,409(win) · 0초기 첫 순전파 0 · 파리티(tie 짝 / win 홀) ·
    자산 id 정렬 · 행 우주 46,729 / 10,748 / clean 7,748 / holdout 5,748 / 비-tie 3,972

실행:  cd final && python temp/destnation.py          (ML2 · GPU · 예상 11~13분/3시드)
       시드 추가:  --seeds 42,1,2,5,6                  (시드당 약 +3.3분)
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold, train_test_split

# ══════════════════════════════════════════ 경로·계약 상수
FINAL = Path(__file__).resolve().parent.parent
DATA_DIR = FINAL / 'origin'
MODELS = FINAL / 'models'
ARITH = MODELS / 'ARITH'

SEED = 42
HALF = 5000                     # stage2 우주 경계: [:HALF] valid / [HALF:] holdout
K, INNER = 5, 3                 # 결합기 바깥 교차적합 / 안쪽 λ 선택 fold
ABE1_STEP, TB1_STEP = 3109, 5000
LAMS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
LAMS0 = [0.0] + LAMS                     # blk2 의 λ_L 격자 (0 = 무벌점 = 구 base 특수해)
LAM_CEIL = 1e12
MAXIT = 1000
LR, EPOCHS, PATIENCE, WD = 3e-4, 3000, 200, 1e-2
H = 128                                  # 두 축 공통 은닉 폭
TIE_ROUNDS, WIN_ROUNDS = 2, 1            # §3.6 / §3.8
SRCS = ('abe2', 'tb2', 'abe1', 'tb1')
TIE_NAMES = ['s_tb2', 's_tb1', 't_abe2', 't_abe1', 't_ar']
WIN_NAMES = ['m_abe2', 'm_abe1', 'm_ar']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
R_NZ = 3972 / 5748              # holdout 비-tie 비율 — 축 분해 계수

# ── 캠페인 실측 기준 (시드별). 같은 시드·같은 구조라 **비트 단위로 같아야** 한다.
#    win = SK-RAW 대체 g1  (42,1,2: winlab3·winlab4 일치 / 5~9: winconf 사전등록 확인)
#    tie = blk2 결합기      (42,1,2: penlab / 5~9: penlab_conf)
#    이 대조가 어긋나면 단독 파일 이식이 틀린 것이다 (성능 판정이 아니라 배선 검증).
#    grp 'sel' = 구조 선별에 쓴 시드 (절대값에 선택 낙관이 실린다)
#        'conf' = 사전등록 **확인** 시드 (두 축 모두 새 시드로 다시 잰 집합 — 낙관 없음)
REF = {
    42: dict(win=0.5639401303471346, tie=0.5682889411230686, grp='sel',
             src='winlab3/4·penlab'),
    1:  dict(win=0.5639007077122791, tie=0.5677446373991767, grp='sel',
             src='winlab3/4·penlab'),
    2:  dict(win=0.5641047007249240, tie=0.5689767003823945, grp='sel',
             src='winlab3/4·penlab'),
    5:  dict(win=0.5635333636732063, tie=0.5693148842376126, grp='conf',
             src='winconf·penlab_conf'),
    6:  dict(win=0.5630398169232196, tie=0.5682832975932272, grp='conf',
             src='winconf·penlab_conf'),
    7:  dict(win=0.5638791093807394, tie=0.5677760181436105, grp='conf',
             src='winconf·penlab_conf'),
    8:  dict(win=0.5636327356142526, tie=0.5694795504676662, grp='conf',
             src='winconf·penlab_conf'),
    9:  dict(win=0.5641406974293696, tie=0.5682156383113173, grp='conf',
             src='winconf·penlab_conf'),
}
GRP_NAME = {'sel': '선별 42,1,2', 'conf': '확인 5~9'}
REF_TOL = 1e-6                  # 비트 재현 문턱. 넘으면 이식 오류로 본다


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


def _v3_split():
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
    """뷰평균 — (fwd + swp)/2. 이 접기로 아래 패리티가 정확해진다."""
    z = np.load(MODELS / rel)
    L = np.vstack([(z['valid_fwd'] + z['valid_swp']) / 2,
                   (z['hold_fwd'] + z['hold_swp']) / 2]).astype(np.float64)
    assert L.shape[1] == n_class, f'{rel} 열 {L.shape[1]} != {n_class}'
    return L, z


def build():
    """stage2 10,748 — 로짓 기저 + 행 우주."""
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

    te_v3, va2 = _v3_split()
    cu = np.ones(n, bool)                       # clean = 결합기 적합 우주
    cu[:HALF] = pd.Index(va2['id']).get_indexer(ids[:HALF]) >= 0
    assert int(cu.sum()) == 7748, f'clean {int(cu.sum())} != 7748'
    assert (pd.Index(te_v3['id']).get_indexer(ids[HALF:]) >= 0).all(), 'holdout 이 V3 test 밖'
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
                tie_base=tie_base, win_base=win_base)


def load_train_logits():
    """TR 46,729 의 tie 짝 5열 · 승부 홀 3열. 전 파일을 abe2 train npz 의 id 로 대조."""
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
    L_ar = zar['raw_oof_train__xgb'].astype(np.float64)

    m_abe2, t_abe2 = sd3(L_abe2)
    m_abe1, t_abe1 = sd3(L_abe1)
    m_ar, t_ar = sd3(L_ar)
    tie_tr = np.column_stack([L_tb2[:, 1] - L_tb2[:, 0], L_tb1[:, 1] - L_tb1[:, 0],
                              t_abe2, t_abe1, t_ar])
    win_tr = np.column_stack([m_abe2, m_abe1, m_ar])
    return ids, y.astype(np.int64), tie_tr, win_tr


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


def head_assets(U, Z):
    """헤드 학습 재료 — TR 로짓(정규화)·SVD. 정규화 통계는 **TR 전용**."""
    _, y_tr, tie_tr, win_tr = load_train_logits()
    print(f'  train 로짓 {tie_tr.shape[0]:,}행 (tie {tie_tr.shape[1]}열 / '
          f'win {win_tr.shape[1]}열) | 장치 {DEVICE}')
    mu_u, sc_u = zstats(tie_tr, True)                 # 짝: z-score
    _, sc_v = zstats(win_tr, False)                   # 홀: RMS, 무중심
    return dict(y_tr=y_tr, y01m=(y_tr == 2).astype(np.float64),
                Lu_tr=(tie_tr - mu_u) / sc_u, Lv_tr=win_tr / sc_v,
                Lu_s2=(U['tie_base'] - mu_u) / sc_u, Lv_s2=U['win_base'] / sc_v,
                CUt=Z['CUt'].astype(np.float32), CVt=Z['CVt'].astype(np.float32),
                CUs=Z['CUs'].astype(np.float32), CVs=Z['CVs'].astype(np.float32))


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
def wiring_checks():
    """§3.6 체크리스트 ①②③ — 파라미터 수 · 0초기 출력 0 · 파리티. 실험이 아니라 오류 검출."""
    g = torch.Generator().manual_seed(0)
    sk = list(range(512, 517)) + list(range(1029, 1032))
    pool = list(range(0, 512)) + list(range(517, 1029))
    tie = TieNet(1032, H, 'TANHTANH', sk, pool)
    n_tie = sum(p.numel() for p in tie.parameters())
    Ux, Vx = torch.randn(64, 517, generator=g), torch.randn(64, 515, generator=g)
    nn.init.zeros_(tie.w.weight)
    nn.init.zeros_(tie.w.bias)
    z0 = float(sym_post(tie, Ux, Vx, +1).abs().max())
    nn.init.normal_(tie.w.weight)
    ev = float((sym_post(tie, Ux, Vx, +1) - sym_post(tie, Ux, -Vx, +1)).abs().max())

    skw = list(range(1024, 1029)) + list(range(2053, 2056))
    poolw = list(range(0, 1024)) + list(range(1029, 2053))
    win = WinNet(2056, H, skw, poolw, 1024)
    n_win = sum(p.numel() for p in win.parameters())
    Uw, Vw = torch.randn(64, 1029, generator=g), torch.randn(64, 1027, generator=g)
    nn.init.normal_(win.w.weight)
    od = float((sym_post(win, Uw, Vw, -1) + sym_post(win, Uw, -Vw, -1)).abs().max())

    print(f'[배선] 층1 파라미터 tie {n_tie:,}×{TIE_ROUNDS} = {n_tie * TIE_ROUNDS:,} · '
          f'win {n_win:,}×4 = {n_win * 4:,}\n'
          f'       0초기 출력 {z0:.1e} · tie 짝 위반 {ev:.1e} · win 홀 위반 {od:.1e}')
    assert n_tie == 262537 and n_win == 262409, '파라미터 수 계약 위반'
    assert z0 == 0.0 and ev < 1e-4 and od < 1e-4, '파리티/0초기 계약 위반'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    # 기본 = 42(노트북 배포 시드) + 기존 lab 이 확정 구조를 잰 전 시드.
    #   42,1,2 winlab3/4·penlab (선별)  |  5~9 winconf·penlab_conf (사전등록 확인)
    # 이 8개가 두 축 모두 기준값이 있는 전부라, 대조가 빠짐없이 걸린다.
    ap.add_argument('--seeds', default='42,1,2,5,6,7,8,9')
    ap.add_argument('--out', default='temp/destnation')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s]
    t_all = time.time()

    print('[적재]')
    U = build()
    y, fold, cu, nz, hold = U['y'], U['fold'], U['cu'], U['nz'], U['hold']
    yT, yA = (y == 2).astype(np.float64), (y == 0).astype(np.float64)
    mw = cu & nz
    Z = load_svd_cache()
    A = head_assets(U, Z)
    RAW = raw_uv(Z)
    wiring_checks()

    res, rows_, store = {}, {}, {}
    for sd in seeds:
        print(f'\n{"=" * 78}\n[시드 {sd}]\n{"=" * 78}')
        t0 = time.time()
        Gt, st_t = tie_head(A, sd)
        W, st_w = win_head(A, RAW, sd)

        z_tie, dep_t = tie_combiner(np.hstack([U['tie_base'], Gt]), yT, fold, cu, 5)
        z_win, dep_w = win_combiner(np.hstack([U['win_base'], W]), yA, fold, mw, 3)

        row, tie_ll, win_ll = score(U, z_tie, z_win, hold)
        lp = assemble(z_tie[hold & nz], z_win[hold & nz])         # 조립 산출물 (배포 형식)
        assert abs(float(np.abs(np.exp(lp).sum(1) - 1).max())) < 1e-9, '조립 확률합 위반'

        res[f's{sd}'] = dict(tot=float(row.mean()), tie=tie_ll, win=win_ll,
                             tie_lam=dep_t['lam'], tie_picked=dep_t['picked'],
                             win_lam=dep_w['lam'], win_picked=dep_w['picked'],
                             tie_w=dep_t['v'], win_w=dep_w['v'],
                             sec=round(time.time() - t0, 1))
        rows_[f'row_s{sd}'] = row
        store[f'Gt_s{sd}'] = Gt.astype(np.float32)
        store[f'W_s{sd}'] = W.astype(np.float32)
        store[f'z_tie_s{sd}'], store[f'z_win_s{sd}'] = z_tie, z_win
        print(f'  tie {tie_ll:.5f} (λ {dep_t["lam"]}) · win {win_ll:.5f} (λ {dep_w["lam"]:g}) '
              f'→ **{row.mean():.5f}**  [{res[f"s{sd}"]["sec"]:.0f}s]')
        torch.save({'tie': st_t, 'win': st_w}, FINAL / f'{args.out}_heads_s{sd}.pt')

    op = FINAL / args.out
    np.savez_compressed(op.with_suffix('.rowloss.npz'), **rows_, **store)
    with open(op.with_suffix('.json'), 'w') as f:
        json.dump({'cells': res, 'seeds': seeds,
                   'tie_names': TIE_NAMES + [f'g{i}' for i in range(1, TIE_ROUNDS + 1)],
                   'win_names': WIN_NAMES + list(SRCS)}, f, ensure_ascii=False, indent=1)
    print(f'\n저장: {op}.rowloss.npz / {op}.json / {args.out}_heads_s*.pt')

    g = lambda k: np.array([res[f's{sd}'][k] for sd in seeds])
    print(f'\n{"=" * 78}\n[확정 구조 실측] {len(seeds)}시드 · holdout 5,748\n{"=" * 78}')
    print(f'  {"시드":6s}{"tie ll":>10s}{"win ll":>10s}{"총 3-class":>12s}   λ(tie) / λ(win)')
    for sd in seeds:
        c = res[f's{sd}']
        print(f'  {sd:<6d}{c["tie"]:10.5f}{c["win"]:10.5f}{c["tot"]:12.5f}   '
              f'{tuple(c["tie_lam"])} / {c["win_lam"]:g}')
    sdv = g('tot').std(ddof=1) if len(seeds) > 1 else 0.0
    print(f'  {"평균":6s}{g("tie").mean():10.5f}{g("win").mean():10.5f}'
          f'{g("tot").mean():12.5f}   (시드 sd {sdv:.5f})')
    for gk, gn in GRP_NAME.items():                  # 선별 / 확인 집합을 갈라서도 본다
        ss = [s for s in seeds if REF.get(s, {}).get('grp') == gk]
        if len(ss) < 2:
            continue
        sub = lambda k: np.array([res[f's{s}'][k] for s in ss])
        print(f'  └ {gn:11s}{sub("tie").mean():8.5f}{sub("win").mean():10.5f}'
              f'{sub("tot").mean():12.5f}   (n={len(ss)} · sd {sub("tot").std(ddof=1):.5f})')
    if 42 in seeds:
        print(f'  └ {"배포 시드 42":11s}{res["s42"]["tie"]:8.5f}{res["s42"]["win"]:10.5f}'
              f'{res["s42"]["tot"]:12.5f}   ← 노트북이 고정으로 쓸 시드')
    print(f'\n  축 분해  {g("tie").mean():.5f} + {R_NZ:.3f} × {g("win").mean():.5f} = '
          f'{g("tie").mean() + R_NZ * g("win").mean():.5f}  (총점과 항등)')

    # ── 기준 대조 — 캠페인 실측(winconf/winlab3·4 · penlab/penlab_conf)과 시드별 paired
    have = [sd for sd in seeds if sd in REF]
    if have:
        print(f'\n{"=" * 78}\n[기준 대조] 캠페인 실측 vs 이 런 — 같은 시드·같은 구조라 '
              f'Δ 는 0 이어야 한다\n{"=" * 78}')
        print(f'  {"시드":5s}{"승부 기준":>10s}{"실측":>10s}{"Δ":>10s} │'
              f'{"tie 기준":>10s}{"실측":>10s}{"Δ":>10s} │{"합성 총점":>11s}{"실측":>10s}'
              f'{"Δ":>10s}   출처')
        worst, acc = 0.0, []
        for sd in have:
            r, c = REF[sd], res[f's{sd}']
            dw, dt = c['win'] - r['win'], c['tie'] - r['tie']
            rt = r['tie'] + R_NZ * r['win']
            dd = c['tot'] - rt
            worst = max(worst, abs(dw), abs(dt), abs(dd))
            acc.append((r['win'], c['win'], r['tie'], c['tie'], rt, c['tot']))
            print(f'  {sd:<5d}{r["win"]:10.5f}{c["win"]:10.5f}{dw:+10.6f} │'
                  f'{r["tie"]:10.5f}{c["tie"]:10.5f}{dt:+10.6f} │'
                  f'{rt:11.5f}{c["tot"]:10.5f}{dd:+10.6f}   {r["src"]}')
        a = np.array(acc)
        print(f'  {"평균":5s}{a[:,0].mean():10.5f}{a[:,1].mean():10.5f}'
              f'{(a[:,1]-a[:,0]).mean():+10.6f} │{a[:,2].mean():10.5f}{a[:,3].mean():10.5f}'
              f'{(a[:,3]-a[:,2]).mean():+10.6f} │{a[:,4].mean():11.5f}{a[:,5].mean():10.5f}'
              f'{(a[:,5]-a[:,4]).mean():+10.6f}   n={len(acc)}')
        miss = [sd for sd in seeds if sd not in REF]
        v = ('**비트 재현** — 단독 파일 이식이 캠페인과 동일'
             if worst < REF_TOL else
             f'*** 이식 불일치 (max|Δ| {worst:.2e} > {REF_TOL:.0e}) — 배선을 의심하라')
        print(f'\n  max|Δ| {worst:.2e}  →  {v}')
        if miss:
            print(f'  (기준 없는 시드 {miss} — 캠페인이 안 돌린 시드다)')
        print('  ! 합성 총점 = tie 기준 + 0.691 × 승부 기준. 두 축이 각각 다른 런에서 잰 값이라 '
              '\n    이 열이 맞으면 **조합이 산술대로 성립**한다는 뜻이다 (이 런의 존재 이유)')
    print(f'\n총 {((time.time() - t_all) / 60):.1f}분')


if __name__ == '__main__':
    main()
