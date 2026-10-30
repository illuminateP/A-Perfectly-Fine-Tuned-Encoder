"""mlphead.py — 인코더 헤드를 떼고 **비선형 swap-등변 헤드**를 풀링 표현 위에 직접 얹는다.

문제 의식
  10시간씩 태운 인코더가 만든 고품질 표현을 (m, t) 스칼라 2개로 압축한 뒤,
  그걸 등가중 평균해서 6 파라미터 선형 결합기에 넘긴다. 압축과 평균이 학습 **이전**에
  일어나므로 결합기가 되돌릴 수 없다. 실측 근거: ledger.py [3] 에서 체크포인트를
  개별 열로 펼치자 tie축 0.5740 -> 0.5725 (+0.0014 유의).

  다만 상한도 분명히 해둔다. 보정기 8계열·꼬리 샤프닝은 전부 기각됐고 낙관 상한이 +0.0001 이다.
  즉 **확률을 그냥 뾰족하게 깎아서는 못 얻는다** — 고확신 구간은 이미 정확도 ≈ 확신이다.
  0.98 급 확신은 그 구간의 **판별력**이 올라야 나온다. 뾰족함은 판별력의 결과지 원인이 아니다.
  그래서 이 파일이 시험하는 건 보정이 아니라 "표현 + 비선형이 판별력을 올리나" 다.

헤드 구조 — 등변을 학습이 아니라 **구조**로 넣는다
      p = net(cat[U,  V]);   q = net(cat[U, −V])      # 같은 가중치, 홀 블록 부호만 반전
      d  = p₀ − q₀      (반대칭화 → 승부축)
      nu = p₁ + q₁      (대칭화   → tie축)
      logits = [d/2, −d/2, nu + logaddexp(d/2, −d/2)]
  swap 을 넣으면 V→−V 이므로 p 와 q 가 자리를 바꾸고, d→−d / nu→nu 가 **정확히** 성립한다.
  결과: (1) TTA 가 항등식이 되어 필요 없다 (2) 응답 순서를 바꿔도 답이 같다
        (3) nu 가 V 의 임의의 짝함수(|V|, V⊙V ...)를, d 가 U⊙V 게이트를 자동으로 만든다.

  **hidden=0 이면 net 이 Linear(du+dv, 2) 한 장이고, 위 구성은 패리티 제약 선형 모형과
  정확히 같아진다** (d = 2·W_v·V, nu = 2·W_u·U + 2b). 선형과 MLP 가 한 코드 경로라
  "비선형이 문제였나" 가 순수 비교로 나온다.

팔
    1  로짓 5   / 선형   현행 결합기 (Davidson κ=1 과 등가)
    2  표현     / 선형   표현이 로짓보다 나은가
    3  표현     / MLP    **비선형이 문제였나**
    4  표현+로짓/ MLP    둘 다 쓰면

차원 축소 (stage2 단계에서만 필요)
  블록당 8,445 차원을 8,600 행으로 받으면 분산이 감당이 안 된다. 블록별로 **fold 안에서**
  SVD 로 k 차원 축소한다. 짝 블록은 중심화 PCA, **홀 블록은 무중심 SVD** 다 —
  홀을 중심화하면 상수항이 생겨 반대칭이 깨진다. 선형 사영이므로 패리티는 보존된다.
  train 46,729 행으로 가면 축소 없이(--reduce 0) 돌릴 수 있다.

규율
  · stage2 10,748 을 기존과 같은 5-fold 로 교차적합. 축소·표준화·조기중단 모두 fold 안에서만.
  · 각 학습 fold 에서 15% 를 조기중단 전용으로 뗀다. 평가 fold 는 어디에도 안 쓴다.
  · 하이퍼파라미터는 고정이다 (탐색하면 중첩이 필요한데 그만한 행이 없다). 값은 출력에 찍는다.
  · 시작 시 등변성을 난수로 검산한다 — 여기가 깨지면 뒤가 전부 무의미하다.

선행
  python Boost/extract_logits.py --pooled --only abe,tb3 --steps abe:3000,tb3:4000

사용
  python Boost/mlphead.py                      # 팔 4개, 기본 설정
  python Boost/mlphead.py --reduce 32          # 축소 차원
  python Boost/mlphead.py --hidden 512 --epochs 400
  python Boost/mlphead.py --arms 1,3           # 일부만
  python Boost/mlphead.py --diag               # 진단만 (팔별 AUC·온도 + 로짓 복원 R²)

진단 (--diag) 이 답하는 것
  A. 표현 팔이 진 게 판별력 부족인가 보정 붕괴인가 — AUC 는 순위만 보므로 보정과 무관하다.
  B. 표현으로 인코더 **자기 로짓**을 선형 복원할 수 있나 — 축소가 범인인지 선형성이 범인인지.
     둘 다 R² 가 높으면 mlphead 열세는 순수 분산(행 부족)이고 train 규모로 풀린다.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BOOST = Path('./Boost')
sys.path.insert(0, str(BOOST))
import reprhead as RH  # noqa: E402

SEED, K = 42, 5
EPS = 1e-15
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
EV_BLOCKS = ['abe_u', 'tb_u', 'tbq_u', 'tbd_u', 'ari']      # 짝
OD_BLOCKS = ['abe_v', 'tb_v', 'tbq_v', 'tbd_v', 'ari']      # 홀


# ══════════════════════════════════════════════════════ 헤드
class EquiHead(nn.Module):
    """swap 등변이 구조로 보장되는 헤드. hidden=0 이면 패리티 제약 선형 모형과 동일하다."""

    def __init__(self, du, dv, hidden=256, drop=0.1):
        super().__init__()
        d = du + dv
        if hidden <= 0:
            self.net = nn.Linear(d, 2)
        else:
            self.net = nn.Sequential(
                nn.Linear(d, hidden), nn.GELU(), nn.Dropout(drop),
                nn.Linear(hidden, max(hidden // 2, 8)), nn.GELU(), nn.Dropout(drop),
                nn.Linear(max(hidden // 2, 8), 2))

    def forward(self, U, V):
        p = self.net(torch.cat([U, V], 1))
        q = self.net(torch.cat([U, -V], 1))
        return p[:, 0] - q[:, 0], p[:, 1] + q[:, 1]         # (d 홀, nu 짝)


def logits3(d, nu):
    """축 분리 합성. p_tie = σ(nu), p_A/(p_A+p_B) = σ(d) 가 정확히 성립한다."""
    return torch.stack([d / 2, -d / 2, nu + torch.logaddexp(d / 2, -d / 2)], 1)


def equivariance_check(du=7, dv=5, hidden=16, n=64, tol=1e-5):
    """난수로 등변성 검산. d(U,−V) = −d(U,V), nu(U,−V) = nu(U,V) 여야 한다."""
    torch.manual_seed(0)
    m = EquiHead(du, dv, hidden).eval()
    U, V = torch.randn(n, du), torch.randn(n, dv)
    with torch.no_grad():
        d1, n1 = m(U, V)
        d2, n2 = m(U, -V)
    e_d = float((d1 + d2).abs().max())
    e_n = float((n1 - n2).abs().max())
    assert e_d < tol and e_n < tol, f'등변성 위반 d {e_d:.2e} nu {e_n:.2e}'
    return e_d, e_n


# ══════════════════════════════════════════════════════ 블록 축소 (fold 안에서)
def reduce_fit(Xtr, k, even):
    """짝은 중심화 PCA, **홀은 무중심 SVD**. 홀을 중심화하면 상수항이 생겨 반대칭이 깨진다.
    반환한 (mu, W, scale) 로 test 행도 같은 사영을 받는다. 선형이므로 패리티가 보존된다."""
    from sklearn.utils.extmath import randomized_svd
    A = Xtr.astype(np.float64)
    mu = A.mean(0) if even else np.zeros(A.shape[1])
    C = A - mu
    kk = min(k, C.shape[1], C.shape[0] - 1) if k > 0 else C.shape[1]
    if kk >= C.shape[1]:
        W = np.eye(C.shape[1])
    else:
        _, _, Vt = randomized_svd(C, n_components=kk, random_state=SEED)
        W = Vt.T
    Z = C @ W
    scale = np.clip(Z.std(0) if even else np.sqrt((Z ** 2).mean(0)), 1e-9, None)
    return mu, W, scale


def reduce_apply(X, p):
    mu, W, scale = p
    return ((X.astype(np.float64) - mu) @ W) / scale


def make_uv(blocks_ev, blocks_od, names_ev, names_od, tr, te, k, cache=None, tag=0):
    """학습 fold(tr) 로 축소를 적합하고 tr / te 두 덩이에 적용.

    반환은 (U_tr, U_te), (V_tr, V_te) 이고 **행 순서는 tr / te 가 준 순서 그대로**다.
    cache 를 주면 (fold, 블록, 짝여부, k) 키로 사영을 재사용한다 — 팔 2·3·4 가 같은
    블록·같은 fold 를 쓰므로 SVD 를 팔마다 다시 돌 이유가 없다."""
    Us, Vs = [], []
    for names, src, even, out in ((names_ev, blocks_ev, True, Us),
                                  (names_od, blocks_od, False, Vs)):
        for nm in names:
            X = src[nm]
            if X.shape[1] == 0:
                continue
            key = (tag, nm, even, k)
            if cache is None or key not in cache:
                p = reduce_fit(X[tr], k, even)
                if cache is not None:
                    cache[key] = p
            else:
                p = cache[key]
            out.append((reduce_apply(X[tr], p), reduce_apply(X[te], p)))
    U = (np.hstack([a for a, _ in Us]), np.hstack([b for _, b in Us]))
    V = (np.hstack([a for a, _ in Vs]), np.hstack([b for _, b in Vs]))
    return U, V


# ══════════════════════════════════════════════════════ 학습
def train_head(Utr, Vtr, ytr, Ues, Ves, yes_, hidden, drop, wd, lr, epochs, patience, seed):
    """전배치 AdamW + 조기중단. 캐시된 벡터 위 연산이라 전배치가 빠르고 재현성도 좋다."""
    torch.manual_seed(seed)
    m = EquiHead(Utr.shape[1], Vtr.shape[1], hidden, drop).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=DEV)
    Ut, Vt, yt = T(Utr), T(Vtr), T(ytr, torch.long)
    Ue, Ve, ye = T(Ues), T(Ves), T(yes_, torch.long)
    lossf = nn.CrossEntropyLoss()
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        m.train()
        opt.zero_grad()
        lossf(logits3(*m(Ut, Vt)), yt).backward()
        opt.step()
        m.eval()
        with torch.no_grad():
            v = float(lossf(logits3(*m(Ue, Ve)), ye))
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {kk: vv.detach().clone() for kk, vv in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        m.load_state_dict(best_state)
    return m.eval(), best, ep + 1


@torch.no_grad()
def predict(m, U, V):
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    d, nu = m(T(U), T(V))
    return torch.softmax(logits3(d, nu), 1).cpu().numpy().astype(np.float64)


# ══════════════════════════════════════════════════════ 평가
def per_row(P, y):
    P = np.clip(P, EPS, 1.0)
    P = P / P.sum(1, keepdims=True)
    return -np.log(P[np.arange(len(y)), y])


def axis_report(P, y):
    P = np.clip(P, EPS, 1.0)
    P = P / P.sum(1, keepdims=True)
    total = float(per_row(P, y).mean())
    it = y == 2
    t = np.clip(P[:, 2], EPS, 1 - EPS)
    tie = float(-np.where(it, np.log(t), np.log(1 - t)).mean())
    nz = ~it
    rA = np.clip(P[nz, 0] / (P[nz, 0] + P[nz, 1]), EPS, 1 - EPS)
    win = float(-np.where(y[nz] == 0, np.log(rA), np.log(1 - rA)).mean())
    return total, tie, win


def conf_profile(P):
    """확신도 분포 — '확률이 안 뾰족하다' 를 숫자로 본다."""
    mx = P.max(1)
    return [float((mx > t).mean()) for t in (0.6, 0.7, 0.8, 0.9, 0.95)], float(mx.mean())


# ══════════════════════════════════════════════════════ 진단 (--diag)
def _auc(v, yb):
    import pandas as pd
    r = pd.Series(v).rank().to_numpy()
    n1 = int(yb.sum())
    return float((r[yb].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yb) - n1)))


def to_axes(P):
    """P -> (d, nu). 무손실 역변환이다 — 3-class 확률의 자유도가 2 이기 때문."""
    P = np.clip(P, EPS, 1.0)
    P = P / P.sum(1, keepdims=True)
    return np.log(P[:, 0] / P[:, 1]), np.log(P[:, 2] / (1 - P[:, 2]))


def cf_temp(v, yb, fold, m=None):
    """교차적합 온도. 스케일만 다시 맞췄을 때 얼마가 회수되나 = 과신분이 얼마나 되나.
    같은 데이터로 맞추고 같은 데이터로 보고하면 낙관 편향이라 fold 밖에서만 맞춘다."""
    from scipy.optimize import minimize_scalar
    n = len(v)
    m = np.ones(n, bool) if m is None else m
    out = np.full(n, np.nan)
    Ts = []
    for k in range(K):
        tr, te = (fold != k) & m, (fold == k) & m

        def f(a, x=v[tr], t=yb[tr]):
            z = a * x
            return float(np.mean(np.logaddexp(0, z) - t * z))
        r = minimize_scalar(f, bounds=(0.02, 5.0), method='bounded')
        Ts.append(1.0 / r.x)
        z = r.x * v[te]
        out[te] = np.logaddexp(0, z) - yb[te] * z
    return float(np.nanmean(out[m])), float(np.median(Ts))


def oof_r2(blocks, tgt, fold, even, red, lams=(1.0, 10.0, 100.0, 1e3, 1e4)):
    """표현으로 대상(로짓)을 선형 복원했을 때의 out-of-fold R². 최소제곱이라 닫힌형이다."""
    n = len(tgt)
    pred = np.zeros(n)
    for k in range(K):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        A, B = [], []
        for X in blocks:
            if red > 0:
                p = reduce_fit(X[tr], red, even)
                A.append(reduce_apply(X[tr], p))
                B.append(reduce_apply(X[te], p))
            else:
                mu = X[tr].mean(0) if even else 0.0
                sc = np.clip(X[tr].std(0) if even else np.sqrt((X[tr] ** 2).mean(0)), 1e-9, None)
                A.append((X[tr] - mu) / sc)
                B.append((X[te] - mu) / sc)
        A, B = np.hstack(A), np.hstack(B)
        if even:                                            # 짝 채널만 절편을 갖는다
            A = np.hstack([A, np.ones((len(A), 1))])
            B = np.hstack([B, np.ones((len(B), 1))])
        dg = np.ones(A.shape[1])
        if even:
            dg[-1] = 0.0                                    # 절편은 벌점 대상이 아니다
        inner = np.arange(len(A)) % 3
        best, bl = lams[0], np.inf
        for lam in lams:
            tot = 0.0
            for j in range(3):
                it, iv = inner != j, inner == j
                w = np.linalg.solve(A[it].T @ A[it] + lam * np.diag(dg), A[it].T @ tgt[tr][it])
                tot += float(((A[iv] @ w - tgt[tr][iv]) ** 2).sum())
            if tot < bl:
                bl, best = tot, lam
        w = np.linalg.solve(A.T @ A + best * np.diag(dg), A.T @ tgt[tr])
        pred[te] = B @ w
    ss = float(((pred - tgt) ** 2).sum())
    st = float(((tgt - tgt.mean()) ** 2).sum())
    return 1 - ss / st, float(np.corrcoef(pred, tgt)[0, 1])


def diagnose(R, fold, red):
    """[진단 A] 팔별 AUC·온도 — 열세가 판별력 부족인가 보정 붕괴인가.
       [진단 B] 표현으로 인코더 자기 로짓을 복원 — 축소·선형성 중 무엇이 병목인가."""
    y, EV, OD = R['y'], R['EV'], R['OD']
    tie, nz = y == 2, y != 2

    p = BOOST / 'mlphead_oof.npz'
    if p.exists():
        RH.head('진단 A', 'AUC 는 순위만 본다 — 보정과 무관하게 정보량을 잰다')
        print('  AUC 유지 + ll 나쁨 -> 보정 붕괴(과신). 온도로 상당 부분 회수된다.')
        print('  AUC 도 나쁨        -> 판별력 손실. 온도로 회수되지 않는다.\n')
        z = np.load(p, allow_pickle=True)
        rows = []
        for kk in [c for c in z.files if c.startswith('P__')]:
            d, nu = to_axes(z[kk])
            t_ll, t_T = cf_temp(nu, tie.astype(float), fold)
            w_ll, w_T = cf_temp(d, (y == 0).astype(float), fold, nz)
            rows.append([kk[3:], f'{_auc(nu, tie):.4f}', f'{t_ll:.4f}', f'{t_T:.2f}',
                         f'{_auc(d[nz], y[nz] == 0):.4f}', f'{w_ll:.4f}', f'{w_T:.2f}'])
        RH.table(['팔', 'tie AUC', 'tie ll(온도후)', 'T', '승부 AUC', '승부 ll(온도후)', 'T'],
                 rows, ['l', 'r', 'r', 'r', 'r', 'r', 'r'])
        print('  기준 — 인코더 단독 tie AUC: TB3 0.6563 / ABE 0.6421 / Arith 0.6500')
        print('         인코더 단독 승부 AUC: ABE 0.7645 / Arith 0.7367')
        print('  T > 1 이면 과신이다. 온도 후에도 기준 팔보다 나쁘면 보정이 아니라 판별력 문제다.')

    RH.head('진단 B', '표현으로 **인코더 자기 로짓**을 선형 복원할 수 있나')
    print('  두 인코더 헤드는 MLP 다 (GELU + tanh(a)tanh(b) / |a−b| 입력).')
    print('  따라서 로짓은 풀링 벡터의 선형 함수가 아니고, 선형 사영으로는 원리적 한계가 있다.')
    print('  전차원과 축소판을 나란히 놓으면 축소·선형성 중 어느 쪽이 병목인지 갈린다.\n')
    CASES = [('m3  (ABE 승부)', [OD['abe_v']], False, OD['logit'][:, 0]),
             ('t3  (ABE tie)', [EV['abe_u'], EV['abs_abe']], True, EV['logit'][:, 1]),
             ('s_tie (TB tie)', [EV['tb_u'], EV['abs_tb'], EV['tbq_u']], True,
              EV['logit'][:, 0])]
    rows = []
    for nm, blocks, even, tgt in CASES:
        d_full = sum(b.shape[1] for b in blocks)
        r_full, _ = oof_r2(blocks, tgt, fold, even, 0)
        r_red, c_red = oof_r2(blocks, tgt, fold, even, red)
        rows.append([nm, f'{d_full:,}', f'{r_full:.4f}', f'{r_red:.4f}', f'{c_red:.4f}'])
    RH.table(['복원 대상', '전차원', '전차원 R²', f'PCA-{red} R²', f'PCA-{red} 상관'],
             rows, ['l', 'r', 'r', 'r', 'r'])
    print('  전차원 높고 축소판 낮음 -> 축소가 범인. 차원을 키우거나 지도 사영으로 바꾼다.')
    print('  둘 다 높음              -> 표현은 멀쩡하다. mlphead 열세는 분산 = 행 부족이다.')
    print('                             train 46,729 행이면 풀린다 — 추출에 GPU 를 쓸 근거가 된다.')
    print('  둘 다 낮음              -> 선형으로는 원 헤드를 못 흉내 낸다. 행을 늘려도 같은 병목이')
    print('                             남으므로, 비선형 사영이 먼저다.')


# ══════════════════════════════════════════════════════ 본체
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--abe-step', type=int, default=3000)
    ap.add_argument('--tb-step', type=int, default=4000)
    ap.add_argument('--reduce', type=int, default=64, help='블록당 축소 차원 (0=축소 없음)')
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--drop', type=float, default=0.1)
    ap.add_argument('--wd', type=float, default=1e-2)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--epochs', type=int, default=600)
    ap.add_argument('--patience', type=int, default=60)
    ap.add_argument('--es-frac', type=float, default=0.15)
    ap.add_argument('--arms', default='1,2,3,4')
    ap.add_argument('--diag', action='store_true',
                    help='진단만 실행 — 팔별 AUC·온도(mlphead_oof.npz 필요) + 로짓 복원 R²')
    args = ap.parse_args()
    # 파이프(| tee)로 넘기면 stdout 이 블록 버퍼링돼 한참 아무것도 안 보인다.
    sys.stdout.reconfigure(line_buffering=True)

    if args.diag:
        R = RH.load_blocks(args.abe_step, args.tb_step)
        from sklearn.model_selection import StratifiedKFold
        y = R['y']
        fold = np.zeros(len(y), int)
        for k, (_, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=SEED)
                                    .split(np.zeros((len(y), 1)), y)):
            fold[te] = k
        diagnose(R, fold, args.reduce)
        return

    e_d, e_n = equivariance_check()
    print(f'[검산] 등변성  max|d(U,V)+d(U,−V)| {e_d:.2e} | max|nu 차| {e_n:.2e}  OK')
    print('        구조로 보장되므로 TTA 가 항등식이다 — 추가 인코더 forward 가 0 이다.\n')

    R = RH.load_blocks(args.abe_step, args.tb_step)
    y, EV, OD, is_hold = R['y'], R['EV'], R['OD'], R['is_hold']
    N = len(y)
    from sklearn.model_selection import StratifiedKFold
    fold = np.zeros(N, int)
    for k, (_, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=SEED)
                                .split(np.zeros((N, 1)), y)):
        fold[te] = k

    n_ev = sum(EV[b].shape[1] for b in EV_BLOCKS)
    n_od = sum(OD[b].shape[1] for b in OD_BLOCKS)
    print(f'stage2 {N:,}행 | ABE ckpt-{args.abe_step} / TB3 ckpt-{args.tb_step} | device {DEV}')
    print(f'원 차원  짝 {n_ev:,} + 홀 {n_od:,} = {n_ev + n_od:,}  '
          f'-> 축소 {args.reduce} (블록당) | 학습행 {int(N * 0.8 * (1 - args.es_frac)):,}/fold')
    print(f'하이퍼파라미터  hidden {args.hidden} drop {args.drop} wd {args.wd} lr {args.lr} '
          f'epochs≤{args.epochs} patience {args.patience}  (탐색 안 함 — 중첩할 행이 부족하다)\n')

    ARMS = {
        '1': ('로짓 5 / 선형', ['logit'], ['logit'], 0, 0),
        '2': ('표현 / 선형', EV_BLOCKS, OD_BLOCKS, 0, args.reduce),
        '3': ('표현 / MLP', EV_BLOCKS, OD_BLOCKS, args.hidden, args.reduce),
        '4': ('표현+로짓 / MLP', ['logit'] + EV_BLOCKS, ['logit'] + OD_BLOCKS,
              args.hidden, args.reduce),
    }
    want = [a.strip() for a in args.arms.split(',') if a.strip()]
    # fold 별 조기중단 분할은 팔마다 같아야 비교가 성립한다 — 미리 고정해 둔다
    rng = np.random.default_rng(SEED)
    SPLIT = {}
    for k in range(K):
        tr_all = np.where(fold != k)[0]
        pos = rng.permutation(len(tr_all))
        n_es = max(int(len(tr_all) * args.es_frac), 200)
        SPLIT[k] = (tr_all, np.where(fold == k)[0], pos[n_es:], pos[:n_es])
    cache, out, store = {}, [], {}

    for key in want:
        nm, ev_names, od_names, hid, red = ARMS[key]
        t0 = time.time()
        P = np.zeros((N, 3))
        eps_used, ncol = [], 0
        for k in range(K):
            tr_all, te, tr_pos, es_pos = SPLIT[k]
            # 축소는 **학습 fold(조기중단 몫 포함)** 로만 적합한다. 반환 행 순서는 tr_all / te 순서다.
            (Ua, Ute), (Va, Vte) = make_uv(EV, OD, ev_names, od_names, tr_all, te, red,
                                           cache, tag=k)
            # 위치 인덱스로 자른다 — 값 마스크(np.isin)를 쓰면 순서가 tr_all 순으로 돌아가
            # y[tr] 과 행이 어긋난다.
            Utr, Vtr = Ua[tr_pos], Va[tr_pos]
            Ues, Ves = Ua[es_pos], Va[es_pos]
            ytr, yes_ = y[tr_all[tr_pos]], y[tr_all[es_pos]]
            assert len(Utr) == len(ytr) and len(Ues) == len(yes_), '행 정렬 불일치'
            m, bl, ne = train_head(Utr, Vtr, ytr, Ues, Ves, yes_,
                                   hid, args.drop, args.wd, args.lr, args.epochs,
                                   args.patience, SEED + k)
            P[te] = predict(m, Ute, Vte)
            eps_used.append(ne)
            ncol = Ua.shape[1] + Va.shape[1]
        tot, tie, win = axis_report(P, y)
        th, mean_mx = conf_profile(P)
        store[key] = (nm, P, per_row(P, y))
        out.append([nm, f'{ncol:,}', f'{tot:.4f}', f'{tie:.4f}',
                    f'{win:.4f}', f'{mean_mx:.3f}', f'{th[3]:.2%}',
                    f'{int(np.mean(eps_used))}', f'{time.time() - t0:.0f}s'])

    RH.head('1', '팔별 결과 — 같은 fold 라 기존 결합기 숫자와 직접 비교된다')
    RH.table(['팔', '입력 차원', '총 ll', 'tie축', '승부축(비-tie 행당)', '평균 최대확률',
              'conf>0.9', '평균 epoch', '시간'], out,
             ['l', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r'])
    print('  참고 — Davidson κ=1 (로짓 6 파라미터) 총 0.9705 / tie 0.5740 / 승부 0.5739')
    print('        기존 기록: 부스터 0.9719~0.9735, M1d 등가중+온도2 0.9716')

    if len(store) > 1:
        RH.head('2', '짝비교 — 행별 손실 차이의 부트스트랩 95% CI')
        base_k = want[0]
        rows = []
        for k in want[1:]:
            rows.append([f'{store[k][0]}  vs  {store[base_k][0]}',
                         RH.ci_str(*RH.paired_ci(store[base_k][2], store[k][2]))])
        RH.table(['비교', 'Δ 총 ll (양수 = 뒤가 이득)'], rows, ['l', 'l'])
        hm = is_hold
        rows = [[f'{store[k][0]}  vs  {store[base_k][0]}',
                 RH.ci_str(*RH.paired_ci(store[base_k][2][hm], store[k][2][hm]))]
                for k in want[1:]]
        print()
        RH.table(['비교 (holdout 5,748 만)', 'Δ 총 ll'], rows, ['l', 'l'])
        print('  stage2 valid 5,000 은 두 인코더의 체크포인트 선택 집합이라 표현이 낙관적이다.')

    RH.head('3', '확신도 분포 — "확률이 안 뾰족하다" 를 숫자로')
    rows = []
    for k in want:
        nm, P, _ = store[k]
        th, mx = conf_profile(P)
        rows.append([nm, f'{mx:.3f}'] + [f'{t:.2%}' for t in th])
    RH.table(['팔', '평균 최대확률', '>0.6', '>0.7', '>0.8', '>0.9', '>0.95'], rows,
             ['l', 'r', 'r', 'r', 'r', 'r', 'r'])
    print('  * 뾰족함은 판별력의 **결과**다. 보정만으로 올리는 건 이미 기각됐다(낙관 상한 +0.0001).')
    print('    확신 구간이 넓어지면서 총 ll 도 같이 내려가야 진짜 이득이다 —')
    print('    확신만 넓어지고 ll 이 그대로면 그냥 과신이 된 것이다.')

    np.savez_compressed(BOOST / 'mlphead_oof.npz', ids=R['ids'], y=y,
                        **{f'P__{store[k][0]}': store[k][1] for k in want})
    print(f'\nsaved: {BOOST / "mlphead_oof.npz"}')


if __name__ == '__main__':
    main()
