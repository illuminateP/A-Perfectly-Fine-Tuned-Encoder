"""comblab.py — 층 2 결합기의 **함수족** 축을 연다.

§3.7·§3.8 은 "릿지 로지스틱"을 고정해 두고 **벌점 구조** 축만 전수했다(base/unif/blk2/
ard/ardg/eqL). 족 자체를 바꾸는 축은 한 번도 안 열렸다. 여기서 연다.

사전등록 (실행 전 고정 — 실행 후 이 블록을 고치지 않는다)
────────────────────────────────────────────────────────────
질문   결합기를 릿지 로지스틱 이외의 족으로 바꾸면 이기는가.
       (a) 라쏘 — 수축이 아니라 **삭제**(정확한 0)
       (b) 비선형 — GBDT / MLP 가 결합기를 대체
       (c) 층 2 행이 모자라면 TR 46,729 에서 비선형을 배워 온다 (TR ∩ stage2 = ∅)

팔 — 두 축 **각각 독립**. 한 축을 바꿀 때 반대 축은 BASE 로 고정한다.
  BASE    릿지 · tie blk2 [λ_L×로짓5 | λ_g×g2] · win [로짓3 무벌점 | λ×W4]   ← 검산 팔
  L1      라쏘 역할보존 — BASE 와 **같은 블록 구조**, 벌점만 L1
  L1A     라쏘 전열 — 단일 λ·L1 을 로짓 포함 전 열에. 로짓도 삭제 후보
  GBDT    XGBoost 가 결합기를 대체 (7열 → 마진)
  MLP     2층 FFN 이 결합기를 대체 (7 → h → 1)
  GBDT_T  GBDT + TR 사전학습 마진 1열 (8열)
  MLP_T   MLP  + TR 사전학습 마진 1열 (8열)
  RG_T    BASE 릿지 + TR 사전학습 마진 1열 — 층 2 예산을 **1 파라미터**만 쓰는 최소 개입

조립부(§3.5)는 손대지 않는다. 모든 팔은 축 로짓 하나를 내고 `D.assemble()` 이 동일하게
조립한다. 확률합 1 을 매 팔마다 assert 한다.

패리티 계약 — 이걸 어기면 비교가 성립하지 않는다
  tie 축 입력 7열은 전부 **짝**(스왑 불변)이라 제약이 없다. 어떤 족이든 합법.
  승부 축 입력 7열은 전부 **홀**(스왑 시 x→−x, y→1−y)이라 f(−x) = −f(x) 여야 한다.
    · 선형 무절편 팔(BASE·L1·L1A·RG_T)은 **구조적으로** 홀이다 → 손대지 않고 assert 로 확인
    · 비선형 팔은 홀이 아니므로 Sym 래퍼로 강제한다
        학습행 증강 (x,y) + (−x,1−y)   ·   예측 접기 z = [f(x) − f(−x)] / 2
      증강은 홀 축 비선형 팔에 **공짜 데이터 2배**이기도 하다(가설 (c)의 일부).

규율 — BASE 와 완전히 동일
  적합 우주   tie clean 7,748 · 승부 clean∩비-tie 5,354
  바깥 5-fold 교차적합으로 OOF z 생산 (holdout 5,748 은 자기를 뺀 fold 가 채점)
  안쪽 3-fold 로 하이퍼파라미터 선택 (BASE 가 λ 를 고르는 그 자리)
  표준화     tie 중심화+표준화 · 승부 RMS-only(무중심) — 학습행 통계만 사용
  격자       L1 은 릿지와 **같은** LAMS/LAMS0 (10배 간격)

판정
  시드 42·1·2 평균 · holdout 5,748 · 3-class ll (= tie ll + 0.691 × 승부 ll)
  채택 = Δ ≤ −0.0005  AND  paired bootstrap 95% CI 가 0 을 배제  AND  부호 3/3
  그 외는 전부 기각. 부분 충족을 "유망"으로 쓰지 않는다.

검산 (통과 못 하면 판정 무효)
  · BASE 재계산이 destnation.rowloss.npz 의 z_tie/z_win 과 **비트 일치**
  · BASE 총점이 destnation.json 의 cells/s{sd}/tot 과 일치
  · 선형 승부 팔의 홀성 |f(−x) + f(x)| < 1e-9
  · 모든 팔의 조립 확률합 |Σp − 1| < 1e-9

실행
  cd final && python temp/comblab.py                       # 전 팔, 시드 42,1,2
  cd final && python temp/comblab.py --arms L1,L1A         # 일부만
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # destnation 을 이 폴더에서 찾는다

import numpy as np
import torch
import xgboost as xgb
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold, train_test_split

import destnation as D

warnings.filterwarnings('ignore', category=UserWarning)

NBOOT, THR = 10_000, 5e-4
XGB_GRID = [(2, 5), (2, 50), (3, 5), (3, 50)]        # (max_depth, min_child_weight)
MLP_GRID = [(8, 1e-3), (8, 1e-1), (32, 1e-3), (32, 1e-1)]   # (hidden, weight_decay)
XGB_ROUNDS, XGB_ES = 800, 50
MLP_EPOCHS, MLP_ES, MLP_LR = 1500, 100, 1e-2
ARMS = ['BASE', 'L1', 'L1A', 'GBDT', 'MLP', 'GBDT_T', 'MLP_T', 'RG_T']
NEEDS_TR = ('GBDT_T', 'MLP_T', 'RG_T')
COLS = {'tie': D.TIE_NAMES + ['g1', 'g2'], 'win': D.WIN_NAMES + [f'W4_s{i}' for i in range(4)]}


# ══════════════════════════════════════════ 라쏘 (FISTA · 정확한 0 을 만든다)
def fista(X, y01, pen, bias, w0=None, maxit=3000, tol=1e-11):
    """L1 벌점 로지스틱.  목적 = mean(logaddexp(0,z) − y·z) + Σ pen_j |w_j| / n

    릿지 `D._fit` 과 **같은 1/n 규약**이라 λ 격자가 그대로 비교된다. 절편은 무벌점.
    근위 경사(FISTA) + 적응 재시작 — 좌표를 정확히 0 으로 보낸다(수축이 아니라 삭제).
    w0 = 웜스타트(큰 λ 해에서 내려오는 호모토피). 릿지 결합기의 warm 규약과 같다.
    """
    n, p = X.shape
    Xb = np.hstack([X, np.ones((n, 1))]) if bias else X
    pf = np.concatenate([np.asarray(pen, np.float64), [0.0]]) if bias else np.asarray(pen, np.float64)
    lip = 0.25 * float(np.linalg.eigvalsh(Xb.T @ Xb).max()) / n
    step = 1.0 / max(lip, 1e-12)
    thr = step * pf / n

    def obj(v):
        z = Xb @ v
        return float(np.mean(np.logaddexp(0, z) - y01 * z) + np.dot(pf, np.abs(v)) / n)

    w = np.zeros(Xb.shape[1]) if w0 is None else np.asarray(w0, np.float64).copy()
    u, t, prev, flat = w.copy(), 1.0, obj(w), 0
    for _ in range(maxit):
        g = Xb.T @ (expit(Xb @ u) - y01) / n
        v = u - step * g
        w_new = np.sign(v) * np.maximum(np.abs(v) - thr, 0.0)
        cur = obj(w_new)
        if cur > prev:                                    # 적응 재시작 — 단조성 회복
            u, t = w.copy(), 1.0
            g = Xb.T @ (expit(Xb @ u) - y01) / n
            v = u - step * g
            w_new = np.sign(v) * np.maximum(np.abs(v) - thr, 0.0)
            cur = obj(w_new)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        u = w_new + ((t - 1.0) / t_new) * (w_new - w)
        w, t = w_new, t_new
        flat = flat + 1 if prev - cur < tol * max(1.0, abs(cur)) else 0
        prev = cur
        if flat >= 5:
            return w, True
    return w, False


# ══════════════════════════════════════════ 추정기 — 공통 인터페이스
#   fit(Xtr, ytr, Xva, yva, hp, seed, steps) -> (model, steps_used)
#       Xva 가 있으면 조기중단으로 steps 를 **찾고**, steps 가 오면 그만큼만 학습한다.
#   pred(model, X) -> z (로짓)
class Lasso:
    """L1. mode 'blk' = [λ_L×nl | λ_g×나머지] · 'all' = 단일 λ 전 열."""

    def __init__(self, nl, bias, mode):
        self.nl, self.bias, self.mode = nl, bias, mode
        self.warm = {}

    def grid(self):
        """큰 λ → 작은 λ 순. 웜스타트 호모토피가 성립하는 유일한 순서다."""
        g = ([(a, b) for a in D.LAMS0 for b in D.LAMS] if self.mode == 'blk'
             else [(a,) for a in D.LAMS])
        return sorted(g, reverse=True)

    def _pen(self, hp, p):
        if self.mode == 'blk':
            return np.concatenate([np.full(self.nl, hp[0]), np.full(p - self.nl, hp[1])])
        return np.full(p, hp[0])

    def fit(self, Xtr, ytr, Xva, yva, hp, seed, steps=None):
        v, ok = fista(Xtr, ytr, self._pen(hp, Xtr.shape[1]), self.bias,
                      w0=self.warm.get(seed))
        assert ok, f'FISTA 미수렴 hp={hp}'
        self.warm[seed] = v
        return (v, self.bias), None

    def pred(self, m, X):
        v, bias = m
        return X @ v[:X.shape[1]] + (float(v[-1]) if bias else 0.0)


class XGBEst:
    def __init__(self, base_score=None):
        self.bs = base_score            # None = 학습행에서 산출 (누수 없음)

    def grid(self):
        return XGB_GRID

    def fit(self, Xtr, ytr, Xva, yva, hp, seed, steps=None):
        md, mcw = hp
        bs = self.bs if self.bs is not None else float(np.clip(ytr.mean(), 1e-3, 1 - 1e-3))
        par = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist',
                   max_depth=md, min_child_weight=mcw, eta=0.05, subsample=0.8,
                   colsample_bytree=0.8, base_score=bs, seed=int(seed), nthread=8)
        dtr = xgb.DMatrix(Xtr, label=ytr)
        if steps is None:
            dva = xgb.DMatrix(Xva, label=yva)
            b = xgb.train(par, dtr, XGB_ROUNDS, evals=[(dva, 'va')],
                          early_stopping_rounds=XGB_ES, verbose_eval=False)
            return b, int(b.best_iteration) + 1
        b = xgb.train(par, dtr, max(int(steps), 1), verbose_eval=False)
        return b, int(steps)

    def pred(self, m, X):
        return m.predict(xgb.DMatrix(X), output_margin=True).astype(np.float64)


class MLPEst:
    def grid(self):
        return MLP_GRID

    def fit(self, Xtr, ytr, Xva, yva, hp, seed, steps=None):
        h, wd = hp
        torch.manual_seed(int(seed))
        net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], h), torch.nn.GELU(),
                                  torch.nn.Linear(h, 1))
        opt = torch.optim.Adam(net.parameters(), lr=MLP_LR, weight_decay=wd)
        lo = torch.nn.BCEWithLogitsLoss()
        xt = torch.tensor(Xtr, dtype=torch.float32)
        yt = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
        if steps is not None:
            for _ in range(max(int(steps), 1)):
                opt.zero_grad(); lo(net(xt), yt).backward(); opt.step()
            return net, int(steps)
        xv = torch.tensor(Xva, dtype=torch.float32)
        yv = torch.tensor(yva, dtype=torch.float32).unsqueeze(1)
        best, bep, bstate, bad = np.inf, 1, None, 0
        for ep in range(1, MLP_EPOCHS + 1):
            opt.zero_grad(); lo(net(xt), yt).backward(); opt.step()
            with torch.no_grad():
                v = float(lo(net(xv), yv))
            if v < best - 1e-7:
                best, bep, bad = v, ep, 0
                bstate = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= MLP_ES:
                    break
        net.load_state_dict(bstate)
        return net, bep

    def pred(self, m, X):
        with torch.no_grad():
            return m(torch.tensor(X, dtype=torch.float32)).squeeze(-1).numpy().astype(np.float64)


class Sym:
    """홀 축 래퍼 — 학습행 증강 (x,y)+(−x,1−y) · 예측 접기 [f(x)−f(−x)]/2.

    조기중단용 val 도 함께 증강한다(대칭 학습 중 비대칭 val 손실은 척도가 어긋난다).
    hp 선택은 **접은 예측**으로 채점하므로 선택 기준은 진짜 홀 함수의 손실이다.
    """

    def __init__(self, base):
        self.b = base

    def grid(self):
        return self.b.grid()

    def fit(self, Xtr, ytr, Xva, yva, hp, seed, steps=None):
        X2, y2 = np.vstack([Xtr, -Xtr]), np.concatenate([ytr, 1.0 - ytr])
        if Xva is not None:
            Xva, yva = np.vstack([Xva, -Xva]), np.concatenate([yva, 1.0 - yva])
        return self.b.fit(X2, y2, Xva, yva, hp, seed, steps)

    def pred(self, m, X):
        return (self.b.pred(m, X) - self.b.pred(m, -X)) / 2.0


# ══════════════════════════════════════════ 교차적합 하네스 (BASE 와 같은 규율)
def crossfit(X, y01, fold, rows, n_even, est, seed):
    """바깥 5-fold OOF · 안쪽 3-fold 로 hp 선택 · 바깥 재적합은 안쪽 steps 의 중앙값."""
    z = np.full(len(y01), np.nan)
    picks, steps_log = [], []
    grid = est.grid()
    for k in range(D.K):
        tr, te = (fold != k) & rows, (fold == k) & rows
        Xtr, ytr = X[tr], y01[tr]
        inner = np.zeros(len(ytr), int)
        for j, (_, iv) in enumerate(StratifiedKFold(D.INNER, shuffle=True, random_state=D.SEED)
                                    .split(np.zeros((len(ytr), 1)), ytr)):
            inner[iv] = j
        Zi = [D._z(Xtr[inner != j], Xtr[inner == j], n_even) for j in range(D.INNER)]
        if hasattr(est, 'warm'):
            est.warm = {}                       # fold 마다 호모토피를 새로 시작한다
        best, bhp, bsteps = None, None, None
        for hp in grid:
            tot, st = 0.0, []
            for j in range(D.INNER):
                A, B = Zi[j]
                m, s = est.fit(A, ytr[inner != j], B, ytr[inner == j], hp,
                               seed * 1000 + k * 10 + j)
                tot += float(D.rowloss(est.pred(m, B), ytr[inner == j]).sum())
                st.append(s)
            if best is None or tot < best:
                best, bhp, bsteps = tot, hp, st
        picks.append(bhp)
        steps = None if bsteps[0] is None else int(np.median(bsteps))
        steps_log.append(steps)
        A, B = D._z(Xtr, X[te], n_even)
        m, _ = est.fit(A, ytr, None, None, bhp, seed * 1000 + k, steps)
        z[te] = est.pred(m, B)
    assert np.isfinite(z[rows]).all(), 'OOF 결측'
    return z, picks, steps_log


def deploy_l1(X, y01, rows, n_even, est, picks):
    """L1 배포 단일적합 — fold 선택의 성분별 중앙값. 어떤 열이 **삭제**됐는지 보고용."""
    lam = tuple(float(np.median([p[i] for p in picks])) for i in range(len(picks[0])))
    A, _ = D._z(X[rows], X[rows], n_even)
    v, ok = fista(A, y01[rows], est._pen(lam, A.shape[1]), est.bias)
    assert ok, 'L1 배포 적합 미수렴'
    return lam, v


# ══════════════════════════════════════════ TR 사전학습 (가설 c)
def tr_margin(kind, axis, U):
    """TR 46,729 의 **로짓만**으로 비선형을 학습해 stage2 마진 1열을 만든다.

    TR ∩ stage2 = ∅ (46,729 + 10,748 = 57,477 = train_full) 이라 이 열은 정직하다.
    헤드 열(g·W4)은 TR 에서 in-sample 이라 쓰지 않는다 — 로짓 블록만으로 배운다.
    정규화 통계는 **TR 전용**(head_assets 와 같은 규약)이라 stage2 를 보지 않는다.
    """
    _, y_tr, tie_tr, win_tr = D.load_train_logits()
    if axis == 'tie':
        Xt, s2 = tie_tr, U['tie_base']
        mu, sc = D.zstats(tie_tr, True)
        yt, keep, odd = (y_tr == 2).astype(np.float64), np.ones(len(y_tr), bool), False
    else:
        Xt, s2 = win_tr, U['win_base']
        mu, sc = D.zstats(win_tr, False)
        yt, keep, odd = (y_tr == 0).astype(np.float64), y_tr != 2, True
    Xt, s2 = (Xt - mu) / sc, (s2 - mu) / sc
    Xt, yt = Xt[keep], yt[keep]

    est = XGBEst(0.5 if odd else None) if kind == 'GBDT' else MLPEst()
    if odd:
        est = Sym(est)
    itr, iva = train_test_split(np.arange(len(yt)), test_size=0.15, random_state=D.SEED,
                                stratify=yt)
    best, bhp, bst = None, None, None
    for hp in est.grid():
        m, s = est.fit(Xt[itr], yt[itr], Xt[iva], yt[iva], hp, D.SEED)
        v = float(D.rowloss(est.pred(m, Xt[iva]), yt[iva]).mean())
        if best is None or v < best:
            best, bhp, bst = v, hp, s
    m, _ = est.fit(Xt, yt, None, None, bhp, D.SEED, bst)
    col = est.pred(m, s2)
    print(f'    TR {kind}/{axis}: hp={bhp} steps={bst} es_ll={best:.5f} '
          f'| stage2 마진 sd {col.std():.4f}')
    return col.astype(np.float64)


# ══════════════════════════════════════════ 팔 실행
def run_arm(arm, axis, U, Xb, Gt, yv, rows, seed, trc):
    """축 하나에 팔 하나를 적용해 OOF z 를 낸다. 반대 축은 호출자가 BASE 로 고정한다."""
    nl = Xb.shape[1]
    X = np.hstack([Xb, Gt])
    if arm in NEEDS_TR:
        X = np.hstack([X, trc[:, None]])
    tie = axis == 'tie'
    n_even, bias = (X.shape[1], True) if tie else (0, False)

    if arm in ('BASE', 'RG_T'):
        f = D.tie_combiner if tie else D.win_combiner
        z, dep = f(X, yv, U['fold'], rows, nl)
        return z, dict(lam=dep['lam'], v=[round(x, 5) for x in dep['v']])

    if arm in ('L1', 'L1A'):
        est = Lasso(nl, bias, 'blk' if arm == 'L1' else 'all')
        z, picks, _ = crossfit(X, yv, U['fold'], rows, n_even, est, seed)
        lam, v = deploy_l1(X, yv, rows, n_even, est, picks)
        names = COLS[axis] + (['tr'] if arm in NEEDS_TR else [])
        dead = [names[i] for i in range(X.shape[1]) if v[i] == 0.0]
        return z, dict(lam=lam, picked=[list(map(float, p)) for p in picks],
                       dead=dead, n_dead=len(dead), v=[round(float(x), 5) for x in v])

    base = XGBEst(None if tie else 0.5) if arm.startswith('GBDT') else MLPEst()
    est = base if tie else Sym(base)
    z, picks, steps = crossfit(X, yv, U['fold'], rows, n_even, est, seed)
    return z, dict(picked=[list(p) for p in picks], steps=steps)


def oddness(X, z_fn):
    """선형 승부 팔의 홀성 검산 — 구조적으로 홀이어야 한다."""
    return float(np.abs(z_fn(X) + z_fn(-X)).max())


# ══════════════════════════════════════════ 판정
def boot(d, rng):
    idx = rng.integers(0, len(d), size=(NBOOT, len(d)))
    b = d[idx].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seeds', default='42,1,2')
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--npz', default='temp/destnation.rowloss.npz')
    ap.add_argument('--ref', default='temp/destnation.json')
    ap.add_argument('--out', default='temp/comblab')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s]
    want = set(args.arms.split(',')) | {'BASE'}      # BASE 는 검산 팔이라 항상 돈다
    arms = [a for a in ARMS if a in want]
    t_all = time.time()

    print('[적재]')
    U = D.build()
    y, fold, cu, nz, hold = U['y'], U['fold'], U['cu'], U['nz'], U['hold']
    yT, yA = (y == 2).astype(np.float64), (y == 0).astype(np.float64)
    mw = cu & nz
    Z = np.load(D.FINAL / args.npz)
    REF = json.load(open(D.FINAL / args.ref))['cells']
    print(f'  헤드 열 적재 — 시드 {seeds} · GPU 재학습 없음')

    trc = {}
    if any(a in NEEDS_TR for a in arms):
        print('[TR 사전학습] — TR 46,729 로짓만 · stage2 와 교집합 0행')
        for ax in ('tie', 'win'):
            if {'GBDT_T', 'RG_T'} & set(arms):
                trc[('GBDT', ax)] = tr_margin('GBDT', ax, U)
            if 'MLP_T' in arms:
                trc[('MLP', ax)] = tr_margin('MLP', ax, U)

    res, rows_ = {}, {}
    for sd in seeds:
        print(f'\n{"=" * 78}\n[시드 {sd}]\n{"=" * 78}')
        Gt, W = Z[f'Gt_s{sd}'].astype(np.float64), Z[f'W_s{sd}'].astype(np.float64)
        z0t, z0w = Z[f'z_tie_s{sd}'], Z[f'z_win_s{sd}']

        for arm in arms:
            for axis in ('tie', 'win'):
                t0 = time.time()
                tie = axis == 'tie'
                Xb, G = (U['tie_base'], Gt) if tie else (U['win_base'], W)
                yv, rows = (yT, cu) if tie else (yA, mw)
                key = ('MLP' if arm == 'MLP_T' else 'GBDT', axis)
                z, dep = run_arm(arm, axis, U, Xb, G, yv, rows, sd, trc.get(key))

                if arm == 'BASE':                                   # ── 검산
                    ref = z0t if tie else z0w
                    assert np.allclose(z[rows], ref[rows], atol=1e-12, rtol=0), \
                        f'BASE {axis} 가 destnation 과 불일치 — 하네스 이식 오류'

                zt, zw = (z, z0w) if tie else (z0t, z)
                row, tll, wll = D.score(U, zt, zw, hold)
                lp = D.assemble(zt[hold & nz], zw[hold & nz])
                assert float(np.abs(np.exp(lp).sum(1) - 1).max()) < 1e-9, f'{arm}/{axis} 확률합'
                res[(arm, axis, sd)] = dict(tie=tll, win=wll, tot=float(row.mean()),
                                            sec=round(time.time() - t0, 1), **dep)
                rows_[(arm, axis, sd)] = row
                print(f'  {arm:<7}{axis:<4} tie {tll:.5f} win {wll:.5f} tot {row.mean():.5f}'
                      f'  ({time.time() - t0:.0f}s)'
                      + (f'  삭제 {dep["n_dead"]}열 {dep["dead"]}' if 'dead' in dep else ''))

        b = res[('BASE', 'tie', sd)]
        assert abs(b['tot'] - REF[f's{sd}']['tot']) < 1e-9, \
            f'BASE 총점 {b["tot"]:.6f} != 기록 {REF[f"s{sd}"]["tot"]:.6f}'

        # 선형 승부 팔의 홀성 — 구조 주장의 실측 확인
        for arm in [a for a in arms if a in ('L1', 'L1A')]:
            Xw = np.hstack([U['win_base'], W])
            A, _ = D._z(Xw[mw], Xw[mw], 0)
            est = Lasso(3, False, 'blk' if arm == 'L1' else 'all')
            lam, v = deploy_l1(Xw, yA, mw, 0, est,
                               [tuple(p) for p in res[(arm, 'win', sd)]['picked']])
            o = oddness(A, lambda M: est.pred((v, False), M))
            assert o < 1e-9, f'{arm} 승부 홀성 위반 {o:.2e}'

    # ── 집계
    rng = np.random.default_rng(D.SEED)
    print(f'\n{"=" * 78}\n[판정] 시드 {seeds} 평균 · holdout 5,748 · 문턱 {THR}\n{"=" * 78}')
    print(f'{"팔":<8}{"축":<5}{"Δ축 ll":>10}{"Δ총점":>10}{"95% CI":>22}{"부호":>7}  비고')
    summary = {}
    for arm in arms:
        if arm == 'BASE':
            m = np.mean([res[('BASE', 'tie', s)]['tot'] for s in seeds])
            print(f'{"BASE":<8}{"—":<5}{"":>10}{m:>10.5f}{"(기준)":>22}{"":>7}  검산 통과')
            continue
        for axis in ('tie', 'win'):
            rb = np.mean([rows_[('BASE', axis, s)] for s in seeds], 0)
            ra = np.mean([rows_[(arm, axis, s)] for s in seeds], 0)
            d_tot = float(ra.mean() - rb.mean())
            d_ax = float(np.mean([res[(arm, axis, s)][axis] - res[('BASE', axis, s)][axis]
                                  for s in seeds]))
            lo, hi = boot(ra - rb, rng)
            sg = sum(res[(arm, axis, s)][axis] < res[('BASE', axis, s)][axis] for s in seeds)
            ok = d_tot <= -THR and hi < 0 and sg == len(seeds)
            note = ''
            dead = [res[(arm, axis, s)].get('n_dead') for s in seeds]
            if dead[0] is not None:
                note = f'삭제 {dead}열'
            summary[f'{arm}/{axis}'] = dict(d_tot=d_tot, d_ax=d_ax, ci=[lo, hi],
                                            sign=f'{sg}/{len(seeds)}', adopt=bool(ok))
            print(f'{arm:<8}{axis:<5}{d_ax:>+10.5f}{d_tot:>+10.5f}'
                  f'{f"[{lo:+.5f},{hi:+.5f}]":>22}{f"{sg}/{len(seeds)}":>7}'
                  f'  {"★ 채택" if ok else "기각"} {note}')

    out = {f'{a}|{x}|{s}': v for (a, x, s), v in res.items()}
    json.dump(dict(cells=out, summary=summary, seeds=seeds, arms=arms),
              open(D.FINAL / f'{args.out}.json', 'w'), ensure_ascii=False, indent=1)
    print(f'\n판정 요약: 채택 {sum(v["adopt"] for v in summary.values())}개 / {len(summary)}개')
    print(f'저장 {args.out}.json  ·  총 {((time.time() - t_all) / 60):.1f}분')


if __name__ == '__main__':
    main()
