#!/usr/bin/env python
"""probe_g.py — g1·g2·g3 의 정체 조사. **판정이 아니라 서술**이다. 자립 모듈.

무엇을 묻나
    g 는 저장된 열로만 보면 "행마다 숫자 하나"다. 그 숫자가 무엇의 함수인지를 네 각도에서
    본다. 새로 학습하는 것은 없고, 저장 자산만 읽는다.

    [1] 선형성 분해   g 를 입력 1,032 로 선형 회귀하면 얼마나 설명되나.
        구조적 예측 — 출력이 f(x⁺)+f(x⁻) 라 **홀에 대해 짝함수**이므로 홀 입력의
        *선형* 항은 원리적으로 기여가 0 이어야 한다. 홀만의 R² 가 0 근처면 그 구조가
        확인되고, [짝 + 홀의 제곱항] 이 크게 올라가면 g 가 나르는 것이 "격차의 크기"다.

    [2] 민감도       입력 좌표별 기울기(RMS). 블록별(CUt/Lu/CVt/Lv)과, CUt 를 W 로
        원래 4,096 열로 되돌려 **소스별**로 집계한다.

    [3] 의미론       g 를 Arith 255 **명명 피쳐**로 회귀. R² 와 상위 기여 피쳐 이름.
        대조군으로 로짓(t_abe2·s_tb2)도 같은 회귀를 건다 — g 가 로짓보다 Arith 어휘로
        덜 설명되면 그만큼 어휘 밖 내용이다.
        ! R² 가 낮아도 "설명 불가" 가 아니라 "Arith 어휘로는 설명 안 됨" 이다. 상한이다.

    [4] 어디서 이기나  holdout 행별 손실 개선(p4_g0 − p4_g3)을 층화한다 —
        |격차| 5분위 / 전문가 불일치도 5분위 / 실제 라벨.

읽는 법
    R² 는 전부 5-fold OOF 다. λ 를 같은 OOF 에서 고르므로 미세하게 낙관적이고,
    **설명력의 상한** 성격으로 읽어야 한다.

사용
    cd final
    python -u temp/probe_g.py 2>&1 | tee temp/log_probe_g.txt
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

FINAL = Path(__file__).resolve().parent.parent
DATA_DIR = FINAL / 'origin'
MODELS = FINAL / 'models'
ARITH = MODELS / 'ARITH'
SEED, HALF, K = 42, 5000, 5
ABE1_STEP, TB1_STEP = 3109, 5000
H_GAT = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EV = ['abe2', 'tb2', 'abe1', 'tb1']          # EVK/ODK 의 hstack 순서
LAMS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)

sd3 = lambda B: (B[:, 0] - B[:, 1], B[:, 2] - np.logaddexp(B[:, 0], B[:, 1]))


class GATNet(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.W1, self.W2, self.w = nn.Linear(d, h), nn.Linear(d, h), nn.Linear(h, 1)

    def forward(self, x):
        return self.w(torch.tanh(self.W1(x)) * torch.sigmoid(self.W2(x)))


# ══════════════════════════════════════════════ 적재
def _tta(rel, n_class):
    z = np.load(MODELS / rel)
    L = np.vstack([(z['valid_fwd'] + z['valid_swp']) / 2,
                   (z['hold_fwd'] + z['hold_swp']) / 2]).astype(np.float64)
    assert L.shape[1] == n_class, f'{rel} 열 {L.shape[1]} != {n_class}'
    return L, z


def load_universe():
    L_abe2, za = _tta('logits/abe2__checkpoint-3000.npz', 3)
    ids = np.concatenate([za['valid_ids'], za['hold_ids']])
    y = np.concatenate([za['valid_labels'], za['hold_labels']]).astype(np.int64)
    n = len(y)
    assert n == 10748
    L_tb2, _ = _tta('logits/tb2__checkpoint-4000.npz', 2)
    L_abe1, _ = _tta(f'logits/abe1__checkpoint-{ABE1_STEP}.npz', 3)
    L_tb1, _ = _tta(f'logits/tb1__checkpoint-{TB1_STEP}.npz', 2)
    zar = np.load(ARITH / 'outputs/arith_experts_raw.npz')
    assert (np.concatenate([zar['ids_valid'], zar['ids_hold']]) == ids).all()
    L_ar = np.vstack([zar['raw_valid__xgb'], zar['raw_hold__xgb']]).astype(np.float64)

    m_abe2, t_abe2 = sd3(L_abe2)
    m_abe1, t_abe1 = sd3(L_abe1)
    m_ar, t_ar = sd3(L_ar)
    tie_base = np.column_stack([L_tb2[:, 1] - L_tb2[:, 0], L_tb1[:, 1] - L_tb1[:, 0],
                                t_abe2, t_abe1, t_ar])
    win_base = np.column_stack([m_abe2, m_abe1, m_ar])
    hold = np.arange(n) >= HALF
    fold = np.zeros(n, int)
    for k, (_, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=SEED)
                                .split(np.zeros((n, 1)), y)):
        fold[te] = k
    return dict(ids=ids, y=y, n=n, hold=hold, fold=fold,
                tie_base=tie_base, win_base=win_base)


def load_train_logit_stats():
    """GAT 입력 정규화 상수(mu_u/sc_u/sc_v)는 TR 통계다 — 그대로 다시 만든다."""
    za = np.load(MODELS / 'logits/abe2__checkpoint-3000__train.npz')
    tids = za['train_ids']

    def get(rel, nc):
        z = np.load(MODELS / rel)
        assert (z['train_ids'] == tids).all(), f'{rel} 정렬'
        L = ((z['train_fwd'] + z['train_swp']) / 2).astype(np.float64)
        assert L.shape == (len(tids), nc)
        return L

    L_abe2 = ((za['train_fwd'] + za['train_swp']) / 2).astype(np.float64)
    L_tb2 = get('logits/tb2__checkpoint-4000__train.npz', 2)
    L_abe1 = get(f'logits/abe1__checkpoint-{ABE1_STEP}__train.npz', 3)
    L_tb1 = get(f'logits/tb1__checkpoint-{TB1_STEP}__train.npz', 2)
    zar = np.load(ARITH / 'outputs/arith_experts_raw.npz')
    assert (zar['ids_train'] == tids).all()
    L_ar = zar['raw_oof_train__xgb'].astype(np.float64)
    m_a2, t_a2 = sd3(L_abe2); m_a1, t_a1 = sd3(L_abe1); m_ar, t_ar = sd3(L_ar)
    tie_tr = np.column_stack([L_tb2[:, 1] - L_tb2[:, 0], L_tb1[:, 1] - L_tb1[:, 0],
                              t_a2, t_a1, t_ar])
    win_tr = np.column_stack([m_a2, m_a1, m_ar])
    mu_u, sc_u = tie_tr.mean(0), np.clip(tie_tr.std(0), 1e-9, None)
    sc_v = np.clip(np.sqrt((win_tr ** 2).mean(0)), 1e-9, None)
    return mu_u, sc_u, sc_v


def load_svd_cache():
    for p in sorted((FINAL / 'temp/head_cache').glob('svd512-*.npz')):
        z = np.load(p)
        if z['CUt'].shape == (46729, 512) and z['CUs'].shape == (10748, 512):
            print(f'  SVD 캐시 {p.name}')
            return z
    raise FileNotFoundError('46,729행 svd512 캐시 없음')


# ══════════════════════════════════════════════ 릿지 OOF R²
def oof_r2(X, y, fold, lams=LAMS):
    """열 표준화 후 5-fold OOF R². λ 는 OOF 최선을 고른다(약한 낙관 — 설명력 상한).

    그람행렬 AᵀA 는 λ 와 무관하므로 fold 당 한 번만 만들고 λ 는 대각만 바꿔 푼다.
    (λ 루프 안에서 만들면 |lams| 배 낭비다 — X 가 10,748×1,032 라 실측으로 유의미하다.)
    반환 (OOF R², 고른 λ, 전 데이터 적합 계수 — 표준화 좌표계, 최선 λ 의 OOF 예측).
    """
    X = np.asarray(X, np.float64)
    mu, sd = X.mean(0), np.clip(X.std(0), 1e-12, None)
    Xz = (X - mu) / sd
    yc = np.asarray(y, np.float64) - float(np.mean(y))
    p = Xz.shape[1]
    I = np.eye(p)
    pre = []
    for k in range(K):
        tr, te = fold != k, fold == k
        A, yt = Xz[tr], yc[tr]
        m = float(yt.mean())
        pre.append((te, A.T @ A, A.T @ (yt - m), m))
    best_r2, best_lam, best_pred = -np.inf, lams[0], None
    for lam in lams:
        pred = np.zeros(len(yc))
        for te, G, b, m in pre:
            pred[te] = Xz[te] @ np.linalg.solve(G + lam * I, b) + m
        r2 = 1.0 - float(((yc - pred) ** 2).sum() / (yc ** 2).sum())
        if r2 > best_r2:
            best_r2, best_lam, best_pred = r2, lam, pred
    w = np.linalg.solve(Xz.T @ Xz + best_lam * I, Xz.T @ yc)
    return best_r2, best_lam, w, best_pred + float(np.mean(y))


# ══════════════════════════════════════════════ 조사
def probe(U, args):
    Z = load_svd_cache()
    mu_u, sc_u, sc_v = load_train_logit_stats()
    CUs = Z['CUs'].astype(np.float64)
    CVs = Z['CVs'].astype(np.float64)
    Lu = (U['tie_base'] - mu_u) / sc_u
    Lv = U['win_base'] / sc_v
    EVEN = np.hstack([CUs, Lu])            # 517
    ODD = np.hstack([CVs, Lv])             # 515
    X = np.hstack([EVEN, ODD])             # 1,032
    D = np.load(FINAL / 'temp/heads_repro/deploy.npz')
    G = np.column_stack([D[f'col_gat_{i}'] for i in range(int(D['n_gat']))])
    print(f'  입력 {X.shape} | g {G.shape} | 짝 {EVEN.shape[1]} 홀 {ODD.shape[1]}')

    # ── [1] 선형성 분해 ──────────────────────────────────────────────
    print('\n[1] 선형성 분해 — 5-fold OOF R² (λ 는 OOF 최선 = 설명력 상한)')
    print(f'{"":6s}{"짝만 선형":>12s}{"홀만 선형":>12s}{"전체 선형":>12s}'
          f'{"짝+홀제곱":>12s}{"남는 비선형":>12s}')
    QUAD = np.hstack([EVEN, ODD ** 2])
    for j in range(G.shape[1]):
        g = G[:, j]
        r_e = oof_r2(EVEN, g, U['fold'])[0]
        r_o = oof_r2(ODD, g, U['fold'])[0]
        r_a = oof_r2(X, g, U['fold'])[0]
        r_q = oof_r2(QUAD, g, U['fold'])[0]
        print(f'g{j + 1:<5d}{r_e:12.4f}{r_o:12.4f}{r_a:12.4f}{r_q:12.4f}'
              f'{1 - max(r_a, r_q):12.4f}')
    print('  구조 예측: 홀만 선형 R² ≈ 0 이어야 한다 (출력이 홀에 대해 짝함수).')
    print('  짝+홀제곱 이 전체선형보다 크게 오르면 g 가 나르는 것이 "격차의 크기" 다.')

    # ── [2] 민감도 ───────────────────────────────────────────────────
    print('\n[2] 민감도 — 입력 좌표별 기울기 RMS (행 평균)')
    sds = torch.load(FINAL / 'temp/heads_repro/tie_gat.pt',
                     map_location='cpu', weights_only=False)['gat_sds']
    nU = EVEN.shape[1]
    xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    Wu, ucs = Z['U_W'], Z['U_sc']          # (4096,512), (512,)
    print(f'{"":6s}{"CUt 512":>11s}{"Lu 5":>11s}{"CVt 512":>11s}{"Lv 3":>11s}'
          f'   |  소스별(짝, W 로 환원)')
    for j, sd in enumerate(sds):
        f = GATNet(X.shape[1], H_GAT)
        f.load_state_dict(sd)
        f = f.to(DEVICE).eval()
        xr = xt.clone().requires_grad_(True)
        xp = xr
        xn = torch.cat([xr[:, :nU], -xr[:, nU:]], 1)
        (f(xp) + f(xn)).sum().backward()
        gr = xr.grad.detach().cpu().numpy()
        rms = np.sqrt((gr ** 2).mean(0))
        blocks = [rms[:512], rms[512:nU], rms[nU:nU + 512], rms[nU + 512:]]
        # CUt 기울기를 원래 4,096 열로 환원: dg/dcol = Σ_k dg/dCUt_k · W[col,k]/sc_k
        back = np.abs(Wu @ (rms[:512] / np.clip(ucs, 1e-9, None)))
        per = {s: float(np.sqrt((back[1024 * i:1024 * (i + 1)] ** 2).mean()))
               for i, s in enumerate(EV)}
        tot = sum(per.values()) or 1.0
        print(f'g{j + 1:<5d}' + ''.join(f'{float(np.sqrt((b ** 2).mean())):11.5f}'
                                        for b in blocks)
              + '   |  ' + '  '.join(f'{s} {per[s] / tot * 100:4.1f}%' for s in EV))

    # ── [3] 의미론 ───────────────────────────────────────────────────
    print('\n[3] 의미론 — Arith 255 명명 피쳐로의 설명력 (5-fold OOF R²)')
    df = pd.read_parquet(ARITH / 'features_57k.parquet')
    idx = pd.Index(df['id']).get_indexer(U['ids'])
    assert (idx >= 0).all(), 'stage2 id 가 parquet 에 없다'

    # ! parquet 에는 `label` 과 `is_tie` 가 같이 들어 있다. 그대로 회귀하면 정답을 읽어
    #   R² 가 무의미해지고 상위 피쳐 목록도 라벨이 먹는다. 카탈로그의 피쳐만 화이트리스트로
    #   쓴다 — 이러면 label/is_tie/role 이 자동으로 빠지고 이름·의미도 같이 온다.
    cat = pd.read_csv(DATA_DIR / 'feature_catalog.csv')
    white = [c for c in cat['feature'].astype(str) if c in df.columns]
    banned = [c for c in ('label', 'is_tie') if c in df.columns]
    assert not (set(white) & set(banned)), f'화이트리스트에 라벨 열이 있다: {banned}'
    feat = df[white].iloc[idx].select_dtypes(include=[np.number]).astype(np.float64)
    F = np.nan_to_num(feat.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    names = list(feat.columns)
    mean_map = {str(r['feature']): str(r['의미']) for _, r in cat.iterrows()}
    use_map = {str(r['feature']): str(r['사용']) for _, r in cat.iterrows()}
    print(f'  피쳐 {F.shape} (카탈로그 {len(cat)} 중 매칭 {len(white)}) | '
          f'제외한 라벨 열 {banned}')

    def meaning(nm):
        return mean_map.get(nm, '')

    targets = [(f'g{j + 1}', G[:, j]) for j in range(G.shape[1])] + \
              [('t_abe2', U['tie_base'][:, 2]), ('s_tb2', U['tie_base'][:, 0])]
    raw_r2 = {}
    for nm, tgt in targets:
        r2, lam, w, _ = oof_r2(F, tgt, U['fold'])
        raw_r2[nm] = r2
        top = np.argsort(-np.abs(w))[:args.topk]
        print(f'  {nm:8s} R² {r2:6.4f} (λ {lam:g})   상위 {args.topk}:')
        for t in top:
            u = use_map.get(names[t], '')
            print(f'      {names[t]:24s} {w[t]:+8.4f}  [{u:3s}] {meaning(names[t])[:44]}')

    # ── [3'] 로짓단 통제 ─────────────────────────────────────────────
    # 왜: t_ar·m_ar 이 이미 결합기 base 와 GAT 입력에 들어가 있다. 그래서 [3] 의 R² 는
    #     "g 가 Arith 어휘로 서술 가능한 비율" 일 뿐, 그게 **새 정보인지**는 말하지 않는다.
    #     base 5열로 먼저 회귀해 잔차를 만들고, 그 잔차를 Arith 어휘로 설명해 본다.
    #       잔차 R² ≈ 0  -> g 의 Arith-설명가능분은 이미 로짓단에 있다 (중복)
    #       잔차 R² 큼   -> Arith 가 255 피쳐를 갖고도 자기 GBDT 로짓에 못 담은 것을 g 가 담는다
    print("\n[3'] 로짓단 통제 — base 5열 잔차를 Arith 어휘로 설명")
    B = U['tie_base']
    print(f'  {"대상":8s}{"base5 R²":>10s}{"잔차의 Arith R²":>16s}'
          f'{"통제전 Arith R²":>16s}{"로짓 밖 설명분":>15s}')
    print('    (마지막 열 = 잔차R² x 잔차분산/원분산 — 통제전과 같은 척도. '
          '통제전과 비슷하면 중복 아님)')
    for nm, tgt in targets:
        rb, _, _, pb = oof_r2(B, tgt, U['fold'])
        res = np.asarray(tgt, np.float64) - pb
        rr, lam, w, _ = oof_r2(F, res, U['fold'])
        new = rr * float(np.var(res) / np.var(tgt))   # 원 분산 기준으로 환산
        print(f'  {nm:8s}{rb:10.4f}{rr:16.4f}{raw_r2[nm]:16.4f}{new:15.4f}')
        if nm.startswith('g'):
            top = np.argsort(-np.abs(w))[:args.topk]
            for t in top:
                print(f'      {names[t]:24s} {w[t]:+8.4f}   {meaning(names[t])[:44]}')

    # ── [4] 어디서 이기나 ────────────────────────────────────────────
    print('\n[4] 어디서 이기나 — holdout 행별 손실 개선 (p4_g0 − p4_g3, 양수 = GAT 이득)')
    rl = np.load(FINAL / args.rowloss)
    need = ['p4_g0', 'p4_g3']
    miss = [k for k in need if k not in rl.files]
    assert not miss, f'{args.rowloss} 에 {miss} 없음 — tie_ladder 를 먼저 돌려야 한다'
    d = rl['p4_g0'] - rl['p4_g3']
    h = U['hold']
    strat = {
        '|격차| (|m_abe2|)': np.abs(U['win_base'][h, 0]),
        '전문가 불일치 (tie로짓 std)': U['tie_base'][h].std(1),
        '전문가 합의 tie 크기 (t_abe2)': U['tie_base'][h, 2],
    }
    for nm, v in strat.items():
        q = np.quantile(v, [0.2, 0.4, 0.6, 0.8])
        b = np.digitize(v, q)
        print(f'  {nm}')
        for i in range(5):
            m = b == i
            print(f'    5분위 {i + 1}  n {int(m.sum()):5,}  개선 {d[m].mean():+.5f}'
                  f'   (구간 중앙 {np.median(v[m]):+.3f})')
    yy = U['y'][h]
    for lab, nm in ((2, 'tie'), (0, 'A승'), (1, 'B승')):
        m = yy == lab
        print(f'  라벨 {nm:4s}  n {int(m.sum()):5,}  개선 {d[m].mean():+.5f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rowloss', default='temp/tie_ladder_es7k.rowloss.npz')
    ap.add_argument('--topk', type=int, default=8)
    args = ap.parse_args()
    print('[적재]')
    U = load_universe()
    print(f'  stage2 {U["n"]:,}행 (holdout {int(U["hold"].sum()):,}) | 장치 {DEVICE}')
    probe(U, args)


if __name__ == '__main__':
    main()
