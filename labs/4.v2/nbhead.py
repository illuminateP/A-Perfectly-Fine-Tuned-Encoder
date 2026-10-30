"""nbhead.py — 배포 노트북(a-perfectly-fine-tuned-encoder.ipynb)의 통합 헤드 코드를 글자 그대로 담은 모듈.

머리말(여기부터 SENTINEL 줄까지)은 노트북 셀 13·14·22·29·32·36·40·95 가 하던 준비를 번들 경로 하나로 대신한다.
  use(bundle) — <bundle>/origin/split_ids.csv + 대회 train.csv 로 train/valid/holdout/stage2 프레임을 만들고(셀 29·32·36),
                <bundle>/outputs/logits 에서 전문가별 npz 이름(체크포인트 번호)을 찾아 NPZ·ARITH_NPZ 를 채운다(셀 95).
SENTINEL 아래는 노트북 셀 원문(build · 헤드 모듈 · Layer1 · Layer2 · assemble/score · wiring_checks)이다.
`python lab.py sync` 가 노트북에서 다시 떠 오고 `python lab.py check` 가 글자 단위로 대조한다 — 손으로 고치지 말고,
변형 실험은 실험 스크립트에서 속성·함수를 덮어써서 한다(README 참고).
"""
import os
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')   # cell 13 (torch 2.9 의 새 이름 — 구 이름은 경고만 낸다)
import json, gc, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.utils.extmath import randomized_svd
from sklearn.model_selection import StratifiedKFold

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        # cell 14
SEED = 42                                                                     # cell 22 — lab.set_seed() 가 바꾼다

# 대회 원본 train.csv (Kaggle 에서는 COMP_DIR / 'train.csv') / the competition train.csv
COMP_CSV = Path(os.environ.get('APFTE_COMP_CSV', Path(__file__).resolve().parents[2] / 'hidden' / 'train.csv'))

# cell 40 — build() 가 읽는 풀링 키만 옮겼다 / only the pooled keys build() reads
SPECS = {'abe1': dict(pool_a=('ea',), pool_b=('eb',)),
         'abe2': dict(pool_a=('ea',), pool_b=('eb',)),
         'tb1':  dict(pool_a=('x_f',), pool_b=('x_s',)),
         'tb2':  dict(pool_a=('a_f', 'b_s'), pool_b=('b_f', 'a_s'))}

BUNDLE = OUTPUT_DIR = NPZ = ARITH_NPZ = None
train_df = valid_df = holdout_df = stage2_df = test_df = None
_COMP = None


def use(bundle):
    """번들 하나를 고른다 — 프레임(셀 29·32·36)과 npz 이름(셀 95)을 이 모듈의 전역에 채운다."""
    global BUNDLE, OUTPUT_DIR, NPZ, ARITH_NPZ, train_df, valid_df, holdout_df, stage2_df, _COMP
    BUNDLE = Path(bundle).resolve()
    OUTPUT_DIR = BUNDLE / 'outputs'
    if _COMP is None:
        _COMP = pd.read_csv(COMP_CSV)
    split = pd.read_csv(BUNDLE / 'origin' / 'split_ids.csv')
    assert split['id'].isin(_COMP['id']).all(), 'split ids are missing from the competition train.csv!!'
    d = split.merge(_COMP, on='id', how='left')                                  # cell 29 — 표 순서 유지
    fr = {s: d[d['split'] == s].drop(columns='split').reset_index(drop=True) for s in ('train', 'valid', 'holdout')}
    for nm, f in fr.items():                                                      # cell 32 · 34
        f['label'] = np.select([f['winner_model_a'] == 1, f['winner_model_b'] == 1, f['winner_tie'] == 1],
                               [0, 1, 2], default=-1)
        assert (f['label'] != -1).all() and f['id'].is_unique, f'{nm}: invalid label or duplicated id!!'
    train_df, valid_df, holdout_df = fr['train'], fr['valid'], fr['holdout']
    s2 = set(valid_df['id']) | set(holdout_df['id'])
    assert not (set(train_df['id']) & s2), 'stage1 and stage2 overlap!!'
    assert not (set(valid_df['id']) & set(holdout_df['id'])), 'valid and holdout overlap!!'
    stage2_df = pd.concat([valid_df, holdout_df], ignore_index=True)             # cell 36

    def stem(pattern):                                                            # cell 95 — 이름은 번들에서 찾는다
        hit = sorted((OUTPUT_DIR / 'logits').glob(pattern))
        assert len(hit) == 1, f'{pattern}: expected exactly one npz, found {[h.name for h in hit]}!!'
        return hit[0].name[:-len('__stage1.npz')]

    NPZ = {}
    for tag in SPECS:
        st = stem(f'{tag}__checkpoint-*__stage1.npz')
        NPZ[tag] = {f'{k}_{s}': f'{k}/{st}__{s}.npz' for k in ('logits', 'pooled') for s in ('stage1', 'stage2')}
    st_ar = stem('arith__*__stage1.npz')
    ARITH_NPZ = {f'logits_{s}': f'logits/{st_ar}__{s}.npz' for s in ('stage1', 'stage2')}
    miss = [r for t in NPZ.values() for r in t.values() if not (OUTPUT_DIR / r).exists()] + \
           [r for r in ARITH_NPZ.values() if not (OUTPUT_DIR / r).exists()]
    assert not miss, f'missing npz: {miss}!!'
    names = ' · '.join(NPZ[t]['logits_stage1'][len('logits/'):-len('__stage1.npz')] for t in SPECS)
    print(f' [use] {BUNDLE}\n       train {len(train_df):,} · valid {len(valid_df):,} · holdout {len(holdout_df):,}'
          f' · {names} · {st_ar}', flush=True)


# ======== 아래는 노트북 셀 원문 — `python lab.py sync` 가 다시 떠 온다 / verbatim notebook cells below ========


# ---- notebook cell 105 · def build(stage): · sha256 38d5801b0e3e1c76 ----
# Parity basis / 패리티 기저
SRCS = ('abe2', 'tb2', 'abe1', 'tb1')       
EVK = ('abe2_u', 'tb2_u', 'abe1_u', 'tb1_u')  # even pooled blocks u=(A+B)/2 / 짝 풀링 블록 
ODK = ('abe2_v', 'tb2_v', 'abe1_v', 'tb1_v')  # odd pooled blocks v=(A-B)/2 / 홀 풀링 블록
  
POOLING_DIM = 1024


# Prepare material for unified head / 헤드 학습을 위한 재료 구성
def build(stage):
    if stage == 'stage1':
        df, pre = train_df, ('train',)
    elif stage == 'stage2':
        df, pre = stage2_df, ('valid', 'hold')
    elif stage == 'test':
        df, pre = test_df, ('test',)
    else:
        raise ValueError(f"*** Unknown stage {stage!r} (expected 'stage1', 'stage2' or 'test') ***")
    key, pool_key, arith_rel = f'logits_{stage}', f'pooled_{stage}', ARITH_NPZ[f'logits_{stage}']
    ids = df['id'].to_numpy()
    y = df['label'].to_numpy().astype(np.int64) if 'label' in df.columns else None   # test 는 라벨 없음 / no label on test

    # Swap-TTA mean (fwd+swp)/2, blocks stacked in order / 스왑 TTA 평균 (fwd+swp)/2 를 블록 순서대로 쌓는다.
    def uv(get, tag):
        g = lambda ks: sum(get(k) for k in ks) / len(ks)
        A_, B_ = g(SPECS[tag]['pool_a']), g(SPECS[tag]['pool_b'])
        return (A_ + B_) / 2, (A_ - B_) / 2

    def tta(rel):
        z = np.load(OUTPUT_DIR / rel)
        assert (np.concatenate([z[f'{p}_ids'] for p in pre]) == ids).all(), f'{rel}: ids disagree!!'
        return np.vstack([(z[f'{p}_fwd'] + z[f'{p}_swp']) / 2 for p in pre]).astype(np.float64)

    # 3-class 로짓 -> (홀 m, 짝 t). TB 는 2-class 라 두 로짓의 차 자체가 짝이라 사용 X
    # 3-class logits -> (odd m, even t). TB is 2-class, so its own difference is already even.
    mt = lambda L: (L[:, 0] - L[:, 1], L[:, 2] - np.logaddexp(L[:, 0], L[:, 1]))
    L_tb2, L_tb1 = tta(NPZ['tb2'][key]), tta(NPZ['tb1'][key])
    m_abe2, t_abe2 = mt(tta(NPZ['abe2'][key]))
    m_abe1, t_abe1 = mt(tta(NPZ['abe1'][key]))
    m_ar, t_ar = mt(np.load(OUTPUT_DIR / arith_rel)['margins'].astype(np.float64))

    pool = {}
    for tag in SRCS:
        z = np.load(OUTPUT_DIR / NPZ[tag][pool_key])
        assert (np.concatenate([z[f'{p}_ids'] for p in pre]) == ids).all(), f'{tag} {pool_key}: ids disagree!!'
        pool[f'{tag}_u'], pool[f'{tag}_v'] = uv(
            lambda k: np.vstack([z[f'{p}_{k}'] for p in pre]).astype(np.float64), tag)
    for k, v in pool.items():
        assert v.shape[1] == POOLING_DIM, f'{pool_key} {k}: width {v.shape[1]} != {POOLING_DIM}!!'
        assert np.isfinite(v).all(), f'{pool_key} {k}: non-finite values!!'

    S = dict(ids=ids, y=y, nz=None if y is None else y != 2, pool=pool,
             tie_base=np.column_stack([L_tb2[:, 1] - L_tb2[:, 0], L_tb1[:, 1] - L_tb1[:, 0],
                                       t_abe2, t_abe1, t_ar]),
             win_base=np.column_stack([m_abe2, m_abe1, m_ar]))

    if stage == 'stage2':
        S['hold'] = np.arange(len(y)) >= len(valid_df)   # valid 뒤가 holdout / holdout follows valid
        nh, nw = int(S['hold'].sum()), int((S['hold'] & S['nz']).sum())
        assert (nh, nw) == (5748, 3972), f'holdout universe {(nh, nw)} != (5748, 3972)!!'
    print(f' [Build] Load {stage} {len(ids):,} rows — logit basis + pooling')
    return S


# ---- notebook cell 107 · class HeadLin(nn.Module): · sha256 3446c3942a20a98b ----
# shared hidden width / 두 축 공통 은닉 폭
H = 128                                               

class HeadLin(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.w = nn.Linear(d_in, 1)

    def forward(self, u):
        return self.w(u)


# r1 — tanh(W1x) ⊙ tanh(W2x). 
# Two independent projections, each saturated, multiplied elementwise.
class BodyTT(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.W1, self.W2 = nn.Linear(d, h), nn.Linear(d, h)
        self.out_dim = h

    def forward(self, x):
        return torch.tanh(self.W1(x)) * torch.tanh(self.W2(x))


# r2 — (W1x) ⊙ (W2x) Pure bilinear (unbounded).
class BodyBL(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.W1, self.W2 = nn.Linear(d, h), nn.Linear(d, h)
        self.out_dim = h

    def forward(self, x):
        return self.W1(x) * self.W2(x)


TIE_BODIES = {'TANHTANH': BodyTT, 'BILIN': BodyBL}

# Tie head — body sees pooling only; the 8 logits skip straight into the readout.
# tie 헤드 — 풀링만 받아 새로운 로짓을 생성하고, 8개의 로짓은 흘려보낸다.
class TieNet(nn.Module):
    def __init__(self, d, h, body_kind, skip_idx, body_idx):
        super().__init__()
        self.body_idx, self.skip_idx = body_idx, skip_idx
        self.body = TIE_BODIES[body_kind](len(body_idx), h)
        self.head = HeadLin(self.body.out_dim + len(skip_idx))

    @property
    def w(self):
        return self.head.w


# Win-axis GATE — gate branch reads even u, value branch reads odd v (no bias).
# 승부 GATE — σ(G·u) ⊙ tanh(V·v). 게이트 가지는 짝(u), 값 가지는 홀(v·무절편).
class WinBody(nn.Module):
    def __init__(self, d, h, d_u):
        super().__init__()
        self.d_u, self.out_dim = d_u, h
        self.G = nn.Linear(d_u, h)
        self.V = nn.Linear(d - d_u, h, bias=False)

    def forward(self, x):
        return torch.sigmoid(self.G(x[:, :self.d_u])) * torch.tanh(self.V(x[:, self.d_u:]))


# Win head — the readout bias is frozen at 0 from construction (odd-axis contract).
# 승부 헤드 — 판독 절편 X (홀 축은 bias X)
class WinNet(nn.Module):
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


# Symmetrize then read out — the wrong-parity logit slots become exactly zero.
# 대칭화 뒤 판독 — u = g(x⁺) + sgn·g(x⁻), sk = L(x⁺) + sgn·L(x⁻).
def sym_post(f, Ux, Vx, sgn):
    xp = torch.cat([Ux, Vx], 1)
    xm = torch.cat([Ux, -Vx], 1)
    u = f.body(xp[:, f.body_idx]) + sgn * f.body(xm[:, f.body_idx])
    u = torch.cat([u, xp[:, f.skip_idx] + sgn * xm[:, f.skip_idx]], 1)
    return f.head(u).squeeze(-1)


# Parity-aware normalization statistics, shared by both layers: even columns are centred and standardised, odd columns get an uncentered RMS. 
# Subtracting a mean on the odd axis would break antisymmetry, therefore mu is pinned at zero.
# 패리티 정규화 통계 — 두 층이 함께 쓴다. 짝은 중심화+표준화, 홀은 무중심 RMS.
# 홀에서 평균을 빼면 반대칭이 깨지므로 절편은 쓰지 않는다.
def zstats(X, even):
    mu = X.mean(0) if even else np.zeros(X.shape[1])
    sc = np.clip(X.std(0) if even else np.sqrt((X ** 2).mean(0)), 1e-9, None)
    return mu, sc


# Computed in log space so a large logit cannot overflow to INF.
# 높은 로짓에서 값이 INF 로 튀지 않도록 logspace 에서 계산한다.
def rowloss(z, y01):
    return np.logaddexp(0, z) - y01 * z


# ---- notebook cell 108 · class Layer1: · sha256 d7e29c49e2b9d28e ----
# Layer 1, fitted on stage1. build unified logits with each experts' logits & pooled vectors. submitted to each aixs(tie, win)
# Layer 1, stage1 자산으로 적합하며, 각 전문가별 로짓과 풀링을 받아 무승부축과 승부축에 제출하는 새로운 로짓을 산출한다.
class Layer1:  
    """
    Call chain
    fit()       -> _assets()        -> _svd() -> zstats · randomized_svd
                                    -> _blk()
                -> _tie_head()      -> _split() -> _fit_head() x2 -> TieNet · sym_post
                -> _win_head()      -> _split() -> _fit_head() x4 -> WinNet · sym_post
    
    transform() -> _transform_tie() -> _split()           -> _fwd_chunk() x2 -> TieNet · sym_post
                -> _transform_win() -> _split() · _blk()  -> _fwd_chunk() x4 -> WinNet · sym_post
    """
    
    # 배포 번들 키 / Deploy bundle key
    KEYS = ('NU_mu', 'NU_sc', 'NV_mu', 'NV_sc',               # 풀링 정규화 / pooling normalization
            'U_mu', 'U_W', 'U_sc', 'V_mu', 'V_W', 'V_sc',     # SVD 기저 + 화이트닝 / SVD basis + whitening
            'LU_mu', 'LU_sc', 'LV_mu', 'LV_sc')               # 로짓 정규화 / logit normalization

    # Hparams for Layer 1
    LR = 3e-4
    EPOCHS = 3000
    PATIENCE = 200
    WD = 1e-2  
    K_TIE = 512                        # tie 축 SVD 랭크 / tie-axis SVD rank
    UNITS = ('TANHTANH', 'BILIN')      # tie 라운드별 몸통 / tie body per round    

    
    def __init__(self, Z, states):
        self.Z, self.states = Z, states           

    
    # ────────── Shared primitives / 공용 연산자 ──────────
    # The columns source i occupies in the pooled hstack / 소스 i 가 풀링 hstack 에서 차지하는 열
    @staticmethod
    def _blk(i):
        return slice(i * POOLING_DIM, (i + 1) * POOLING_DIM)

        
    # Column layout of one input row: [pool u | logit u | pool v | logit v].
    # 입력 한 행의 열 배치 : [풀링 u | 로짓 u | 풀링 v | 로짓 v]. 
    @staticmethod
    def _split(d, nu, nv):
        skip = list(range(d, d + nu)) + list(range(2 * d + nu, 2 * d + nu + nv))
        body = list(range(0, d)) + list(range(d + nu, 2 * d + nu))
        return skip, body


    # ────────── layer-1 training primitives / 층 1 학습 연산자  ──────────
    # Layer 1 training methods, fit() -> _tie_head() / _win_head() -> _fit_head() 
    @classmethod
    def fit(cls):
        Z, A = cls._assets()
        states = {'tie': cls._tie_head(A, SEED), 'win': cls._win_head(A, SEED)}
        print(f' [Layer 1] train completed')
        return cls(Z, states)

        
    @staticmethod
    def _assets():
        TR = build('stage1')

        def axis(keys, even):                     
            X = np.hstack([TR['pool'][k] for k in keys])
            mu, sc = zstats(X, even)
            C, svd = Layer1._svd((X - mu) / sc, even)
            return C, mu, sc, svd

        CU, NU_mu, NU_sc, SU = axis(EVK, True)
        CV, NV_mu, NV_sc, SV = axis(ODK, False)
        LU_mu, LU_sc = zstats(TR['tie_base'], True)
        LV_mu, LV_sc = zstats(TR['win_base'], False)
        Z = dict(NU_mu=NU_mu, NU_sc=NU_sc, NV_mu=NV_mu, NV_sc=NV_sc,
                 U_mu=SU['mu'], U_W=SU['W'], U_sc=SU['sc'], V_mu=SV['mu'], V_W=SV['W'], V_sc=SV['sc'],
                 LU_mu=LU_mu, LU_sc=LU_sc, LV_mu=LV_mu, LV_sc=LV_sc)
        #  Win-head feed: per-source (u, v), pooling-normalized only / 승부 헤드 공급 — 소스별 (u,v) 를 풀링 정규화만
        raw = {}
        for i, s in enumerate(SRCS):
            sl = Layer1._blk(i)
            raw[s] = dict(u=((TR['pool'][f'{s}_u'] - NU_mu[sl]) / NU_sc[sl]).astype(np.float32),
                          v=((TR['pool'][f'{s}_v'] - NV_mu[sl]) / NV_sc[sl]).astype(np.float32))
        A = dict(y=TR['y'], CU=CU, CV=CV, raw=raw,
                 Lu=(TR['tie_base'] - LU_mu) / LU_sc, Lv=(TR['win_base'] - LV_mu) / LV_sc)
        return Z, A

        
    # (Un)centred SVD plus per-component whitening by the TR RMS
    # (무)중심 SVD + 성분별 TR-RMS 화이트닝
    @staticmethod
    def _svd(X, even):
        mu, _ = zstats(X, even)              # centre only; zero on the odd axis / 중심만 — 홀은 0 
        Xc = X - mu
        _, _, Vt = randomized_svd(Xc, n_components=Layer1.K_TIE, random_state=SEED)
        C = Xc @ Vt.T
        _, sc = zstats(C, False)             # per-component RMS whitening / 성분별 무중심 RMS = 화이트닝 
        return (C / sc).astype(np.float32), dict(mu=mu, W=Vt.T, sc=sc)


    # Tie head: TANHTANH then BILIN with the TR residual anchor in between.
    # The body reads the SVD coordinates and the 8 logits skip into the readout.
    # tie 헤드 — TANHTANH → BILIN, 라운드 사이 TR 잔차앵커. 몸통은 SVD 좌표, 판독은 로짓 8 직결.
    @staticmethod
    def _tie_head(A, seed):
        skip, body = Layer1._split(A['CU'].shape[1], A['Lu'].shape[1], A['Lv'].shape[1])
        cu = np.ascontiguousarray(np.hstack([A['CU'], A['Lu']]), np.float32)
        cv = np.ascontiguousarray(np.hstack([A['CV'], A['Lv']]), np.float32)
        y01 = (A['y'] == 2).astype(np.float64)
        z, states = None, []
        for rd, unit in enumerate(Layer1.UNITS, 1):
            net = lambda d, _u=unit: TieNet(d, H, _u, skip, body)
            aux = Layer1._fit_head(cu, cv, y01, z, True, seed, net)
            states.append(aux['state'])
            z = aux['z_tr']
            print(f' [tie_head] tie r{rd} {unit:9s} es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]}')
        return states


    # Win head: one GATE round per source
    # 승부 헤드 — 소스별 GATE 1라운드 
    @staticmethod
    def _win_head(A, seed):
        skip, body = Layer1._split(POOLING_DIM, A['Lu'].shape[1], A['Lv'].shape[1])
        m = A['y'] != 2
        y01 = (A['y'] == 0).astype(np.float64)[m]
        states = []
        for i, s in enumerate(SRCS):
            cu = np.ascontiguousarray(np.hstack([A['raw'][s]['u'][m], A['Lu'][m]]), np.float32)
            cv = np.ascontiguousarray(np.hstack([A['raw'][s]['v'][m], A['Lv'][m]]), np.float32)
            net = lambda d: WinNet(d, H, skip, body, POOLING_DIM)
            aux = Layer1._fit_head(cu, cv, y01, None, False, seed, net)
            states.append(aux['state'])
            print(f' [win_head] win r1 {s:5s} es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]}')
        return states


    # Trains one head with full-batch AdamW and early stopping. With s0 the fit is anchor-boosted
    # 헤드 하나를 풀배치로(AdamW + ES) 학습한다. s0 를 주면 앵커 부스팅: 목적함수는 z = a·s0z + f(x).
    @staticmethod
    def _fit_head(U, V, y01, s0, even, seed, net):
        torch.manual_seed(seed)
        f = net(U.shape[1] + V.shape[1])
        nn.init.zeros_(f.w.weight)
        nn.init.zeros_(f.w.bias)
        f = f.to(DEVICE)
        anchored = s0 is not None
        groups = [{'params': f.parameters(), 'weight_decay': Layer1.WD}]
        if anchored:                              # 앵커가 없으면 계수 a 도 없다 / no anchor, no scale parameter
            s0 = np.asarray(s0, np.float64)
            s0z = s0 / np.clip(np.sqrt((s0 ** 2).mean()), 1e-9, None)
            a = nn.Parameter(torch.tensor(1.0, device=DEVICE))
            groups.append({'params': [a], 'weight_decay': 0.0})   # 앵커 계수는 무벌점 / unpenalized
        opt = torch.optim.AdamW(groups, lr=Layer1.LR)
        T = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device=DEVICE)
        U_, V_, Y_ = T(U), T(V), T(y01)
        lossf = nn.BCEWithLogitsLoss()
        out = lambda Ux, Vx: sym_post(f, Ux, Vx, 1 if even else -1)
        # 분기는 루프 밖에서 한 번만 정한다 / the branch is settled once, outside the loop
        fwd = (lambda Ux, Vx, Sx: a * Sx + out(Ux, Vx)) if anchored else (lambda Ux, Vx, Sx: out(Ux, Vx))

        perm = np.random.default_rng(seed).permutation(len(y01))     # ES 검증 분할 = TR 의 15% / ES split
        n_es = max(int(len(y01) * 0.15), 200)
        eidx, tidx = perm[:n_es], perm[n_es:]
        Ue, Ve, Ye = U_[eidx], V_[eidx], Y_[eidx]
        Ut, Vt, Yt = U_[tidx], V_[tidx], Y_[tidx]
        Se, St = (T(s0z)[eidx], T(s0z)[tidx]) if anchored else (None, None)

        best, best_sd, best_a, bad, used = np.inf, None, None, 0, 0
        for ep in range(Layer1.EPOCHS):
            f.train()
            opt.zero_grad()
            lossf(fwd(Ut, Vt, St), Yt).backward()
            opt.step()
            f.eval()
            with torch.no_grad():
                v = float(lossf(fwd(Ue, Ve, Se), Ye))
            if v < best - 1e-6:
                best, bad, used = v, 0, ep + 1
                best_sd = {k2: t.detach().clone() for k2, t in f.state_dict().items()}
                if anchored:
                    best_a = float(a.detach())
            elif (bad := bad + 1) >= Layer1.PATIENCE:
                break
        f.load_state_dict(best_sd)
        with torch.no_grad():
            r_tr = out(U_, V_).cpu().numpy().astype(np.float64)
        return dict(state=best_sd, es_ll=best, es_epoch=used,
                    z_tr=best_a * s0z + r_tr if anchored else r_tr)


    # ────────── Inference methods  ──────────
    # Layer 1 inference methods : transform -> _transform_tie() / _transform_win()
    def transform(self, tie_base, win_base, pool):
        Z = self.Z
        Lu, Lv = (tie_base - Z['LU_mu']) / Z['LU_sc'], (win_base - Z['LV_mu']) / Z['LV_sc']
        U = (np.hstack([pool[k] for k in EVK]) - Z['NU_mu']) / Z['NU_sc']
        V = (np.hstack([pool[k] for k in ODK]) - Z['NV_mu']) / Z['NV_sc']
        CU = (((U - Z['U_mu']) @ Z['U_W']) / Z['U_sc']).astype(np.float32)
        CV = (((V - Z['V_mu']) @ Z['V_W']) / Z['V_sc']).astype(np.float32)
        g, w = self._transform_tie(CU, Lu, CV, Lv), self._transform_win(U, V, Lu, Lv)
        assert np.isfinite(g).all() and np.isfinite(w).all(), 'layer-1 columns have non-finite values!!'
        return g, w

        
    # forward with the stored tie states -> g1·g2 / 저장 가중치 순전파 — 라운드별 TieNet 으로 g1·g2
    @torch.no_grad()
    def _transform_tie(self, CU, Lu, CV, Lv):
        skip, body = Layer1._split(CU.shape[1], Lu.shape[1], Lv.shape[1])
        cu = np.ascontiguousarray(np.hstack([CU, Lu]), np.float32)
        cv = np.ascontiguousarray(np.hstack([CV, Lv]), np.float32)
        cols = []
        for st, unit in zip(self.states['tie'], Layer1.UNITS):
            f = TieNet(cu.shape[1] + cv.shape[1], H, unit, skip, body)
            f.load_state_dict(st)
            cols.append(Layer1._fwd_chunk(f.to(DEVICE).eval(), cu, cv, +1))
        return np.column_stack(cols)

    
    # Forward with the stored win states -> W4 / 저장 가중치 순전파 — 소스별 WinNet 으로 W4
    @torch.no_grad()
    def _transform_win(self, U, V, Lu, Lv):
        skip, body = Layer1._split(POOLING_DIM, Lu.shape[1], Lv.shape[1])
        cols = []
        for i, st in enumerate(self.states['win']):
            sl = Layer1._blk(i)
            cu = np.ascontiguousarray(np.hstack([U[:, sl], Lu]), np.float32)
            cv = np.ascontiguousarray(np.hstack([V[:, sl], Lv]), np.float32)
            f = WinNet(cu.shape[1] + cv.shape[1], H, skip, body, POOLING_DIM)
            f.load_state_dict(st)
            cols.append(Layer1._fwd_chunk(f.to(DEVICE).eval(), cu, cv, -1))
        return np.column_stack(cols)


    # Chunked so the whole inference frame never sits on the GPU at once
    @staticmethod
    @torch.no_grad()
    def _fwd_chunk(f, cs, vs, sgn, bs=16384):
        out = []
        for st in range(0, len(cs), bs):
            Ux = torch.tensor(cs[st:st + bs], dtype=torch.float32, device=DEVICE)
            Vx = torch.tensor(vs[st:st + bs], dtype=torch.float32, device=DEVICE)
            out.append(sym_post(f, Ux, Vx, sgn).cpu().numpy().astype(np.float64))
        return np.concatenate(out)


    # ────────── Bundle I/O / 번들 입출력 ──────────
    # Restore the coordinate system and every head's weights from the one bundle file
    # 번들 파일 하나에서 좌표계와 라운드/소스별 가중치를 복원한다
    @classmethod
    def restore(cls, D0):
        assert int(D0['seed']) == SEED and int(D0['K_TIE']) == cls.K_TIE and int(D0['h']) == H, \
            'deploy bundle contract mismatch!!'
        assert D0['U_W'].shape[1] == D0['V_W'].shape[1] == cls.K_TIE, 'bundle SVD rank != K_TIE!!'
        assert tuple(D0['units']) == cls.UNITS, 'tie-round unit contract mismatch!!'
        sd = lambda p: {k[len(p):]: torch.from_numpy(D0[k]) for k in D0.files if k.startswith(p)}
        states = {'tie': [sd(f'tie{r}__') for r in range(len(cls.UNITS))],
                  'win': [sd(f'win{i}__') for i in range(len(SRCS))]}
        assert all(states['tie']) and all(states['win']), 'bundle is missing head weights!!'
        return cls({k: D0[k] for k in cls.KEYS}, states)

    # This layer's share of the deploy bundle: the coordinates plus every head's weights, flattened
    # 배포 번들에 들어갈 몫 — 좌표계 + 모든 헤드의 가중치(평탄화: tie0__body.W1.weight …)
    def bundle(self):
        out = {k: self.Z[k] for k in Layer1.KEYS}
        for ax, sds in self.states.items():
            for i, s in enumerate(sds):
                out.update({f'{ax}{i}__{k}': t.detach().cpu().numpy() for k, t in s.items()})
        return out


# ---- notebook cell 110 · class Layer2: · sha256 c68fedc7d8627a66 ----
# Layer 2, fitted on stage2. two-block ridge-logistic combiner reducing each axis to one score.
# Layer 2, stage2 에서 cross-fit 하는 두 묶음 릿지 로지스틱 결합기. 축마다 점수 하나로 줄인다.
class Layer2:
    """
    Call chain
    fit()       -> _outer_fold()
                -> _combiner() x2   -> _z()       -> zstats
                                    -> _fit_one() -> minimize (L-BFGS-B)
                                    -> _score()   -> rowloss
    
    transform() -> _score() x2
    """
    
    #  Deploy bundle key, per axis / 축별 배포 묶음 키
    KEYS = ('lam', 'v', 'mu', 'sc')

    # Hparams for Layer 2
    K = 5                                             # Outer cross-fit folds / 바깥 cross-fit fold  
    INNER_K = 3                                       # Inner folds for the ridge λ / 안쪽 λ 선택 fold  
    LAMS = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]   # λ grid; 0 is exactly unpenalized / λ 격자, 0 = 무벌점  
    LAM_CEIL = 1e12                                   # Grid-expansion cap / 자동 확장 상한 
    MAXIT = 1000                                      # Optimizer iteration cap / L-BFGS-B 반복 상한 

    
    def __init__(self, dep_t, dep_w, z_tie=None, z_win=None):
        self.dep_t, self.dep_w = dep_t, dep_w     #  per-axis λ, coefficients, stats
        self.z_tie, self.z_win = z_tie, z_win     

    
    # ────────── Shared primitives / 공용 연산자  ──────────
    @staticmethod
    def _score(v, B, bias):
        return B @ v[:B.shape[1]] + (v[B.shape[1]] if bias else 0.0)


    # ────────── penalized-logistic primitives / 벌점 로지스틱 연산자   ──────────
    # Layer 2 training methods, fit() -> _outer_fold() / _combiner() -> _z() / _fit_one() / _score()
    @classmethod
    def fit(cls, S2, g_cols, w_cols):
        y = S2['y']
        fold = cls._outer_fold(y)
        z_tie, dep_t = cls._combiner(np.hstack([S2['tie_base'], g_cols]), (y == 2).astype(np.float64),
                                     fold, np.ones(len(y), bool), S2['tie_base'].shape[1], even=True)
        z_win, dep_w = cls._combiner(np.hstack([S2['win_base'], w_cols]), (y == 0).astype(np.float64),
                                     fold, S2['nz'], S2['win_base'].shape[1], even=False)
        print(f' [Layer 2] fitted — tie λ_L,λ_g {dep_t["lam"]} · win λ_L,λ_g {dep_w["lam"]}')
        return cls(dep_t, dep_w, z_tie, z_win)

        
    # Outer fold stratified on the 3-class stage2 label; both axes share it.
    # stage2 3-class 라벨로 층화한 바깥 fold, 두 축이 같은 fold 를 쓴다.
    @staticmethod
    def _outer_fold(y):
        fold = np.zeros(len(y), int)
        for k, (_, te) in enumerate(StratifiedKFold(Layer2.K, shuffle=True, random_state=SEED)
                                    .split(np.zeros((len(y), 1)), y)):
            fold[te] = k
        return fold


    # Two-block ridge combiner, shared by both axes: pen = [λ_L on the nl logits | λ_g on the head columns].
    # Only the even (tie) axis carries a bias — a bias on the odd axis would break antisymmetry.
    # Deploy λ is the componentwise median of the outer-fold picks.
    # 두 묶음 릿지 결합기 — 양축이 같은 코드를 쓴다. pen = [λ_L × 로짓 nl | λ_g × 헤드 열].
    # 짝축(tie)만 절편을 갖는다 — 홀축의 절편은 반대칭을 깬다. 배포 λ = 바깥 fold 선택의 성분별 중앙값.
    @staticmethod
    def _combiner(X, y01, fold, rows, nl, even):
        z = np.full(len(y01), np.nan)
        picked = []
        for k in range(Layer2.K):
            tr, te = (fold != k) & rows, (fold == k) & rows
            Xtr, ytr = X[tr], y01[tr]
            inner = np.zeros(len(ytr), int)
            for j, (_, iv) in enumerate(StratifiedKFold(Layer2.INNER_K, shuffle=True, random_state=SEED)
                                        .split(np.zeros((len(ytr), 1)), ytr)):
                inner[iv] = j
            Zi = [Layer2._z(Xtr[inner != j], Xtr[inner == j], even) for j in range(Layer2.INNER_K)]
            score, warm = {}, [None] * Layer2.INNER_K

            def ev(q):
                if q in score:
                    return
                tot = 0.0
                for j in range(Layer2.INNER_K):
                    A, B = Zi[j]
                    pen = np.concatenate([np.full(nl, q[0]), np.full(A.shape[1] - nl, q[1])])
                    v, _ok = Layer2._fit_one(A, ytr[inner != j], pen, even, warm[j])
                    warm[j] = v
                    tot += float(rowloss(Layer2._score(v, B, even), ytr[inner == j]).sum())
                score[q] = tot

            gl, gg = list(Layer2.LAMS), list(Layer2.LAMS)
            for a in sorted(gl, reverse=True):                # Warm-start from the largest λ
                for b in sorted(gg, reverse=True):
                    ev((a, b))
            while True:                                       # Extend the axis that hit its top
                q = min(score, key=score.get)
                ext = False
                if q[0] == max(gl) and max(gl) < Layer2.LAM_CEIL:
                    gl.append(max(gl) * 10.0)
                    for b in gg:
                        ev((gl[-1], b))
                    ext = True
                if q[1] == max(gg) and max(gg) < Layer2.LAM_CEIL:
                    gg.append(max(gg) * 10.0)
                    for a in gl:
                        ev((a, gg[-1]))
                    ext = True
                if not ext:
                    break
            picked.append(q)
            A, B = Layer2._z(Xtr, X[te], even)
            pen = np.concatenate([np.full(nl, q[0]), np.full(X.shape[1] - nl, q[1])])
            v, _ok = Layer2._fit_one(A, ytr, pen, even)
            z[te] = Layer2._score(v, B, even)
        lam = (float(np.median([q[0] for q in picked])), float(np.median([q[1] for q in picked])))
        mu, sc = zstats(X[rows], even)
        pen = np.concatenate([np.full(nl, lam[0]), np.full(X.shape[1] - nl, lam[1])])
        v, ok = Layer2._fit_one((X[rows] - mu) / sc, y01[rows], pen, even)
        assert ok, f'{"tie" if even else "win"} combiner deploy fit did not converge!!'
        return z, dict(lam=lam, v=v, mu=mu, sc=sc, picked=[tuple(map(float, q)) for q in picked])

        
    # Normalize train and held-out rows with train statistics / 학습 행의 통계로 학습·검증 행을 함께 정규화한다 
    @staticmethod
    def _z(Xtr, Xte, even):
        mu, sc = zstats(Xtr, even)
        return (Xtr - mu) / sc, (Xte - mu) / sc

    
    # One penalized-logistic fit, L-BFGS-B in float64 / 벌점 로지스틱(어느 전문가 출력을 얼마나 줄일지) 구현  
    @staticmethod
    def _fit_one(X, y01, pen, bias, w0=None):
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
                       options={'maxiter': Layer2.MAXIT, 'maxcor': 20, 'ftol': 1e-12, 'gtol': 1e-7})
        return res.x, bool(res.success)


    # ────────── Inference methods  ──────────
    # Deploy-coefficient z for fresh rows; a width mismatch stops here (stale bundle, changed experts).
    # 배포 계수로 새 행의 z — 열 폭이 계수와 다르면 여기서 멈춘다 (옛 번들 · 바뀐 전문가 구성).
    def transform(self, tie_base, g_cols, win_base, w_cols):
        t, w = self.dep_t, self.dep_w
        Xt, Xw = np.hstack([tie_base, g_cols]), np.hstack([win_base, w_cols])
        assert Xt.shape[1] == len(t['mu']) and Xw.shape[1] == len(w['mu']), 'combiner width != columns!!'
        return (Layer2._score(t['v'], (Xt - t['mu']) / t['sc'], True),
                Layer2._score(w['v'], (Xw - w['mu']) / w['sc'], False))


    # ────────── Bundle I/O / 번들 입출력 ──────────
    # Never refitted on labels: the frozen coefficients are read back as they are. The cross-fit z is not
    # in the bundle, so z_tie/z_win stay None.
    # 라벨로 다시 맞추지 않는다 — 동결한 계수를 그대로 읽는다. cross-fit z 는 번들에 없어 None 으로 남는다.
    @classmethod
    def restore(cls, D0):
        def dep(p):
            d = {k: D0[f'{p}_{k}'] for k in cls.KEYS}
            d['lam'] = tuple(map(float, d['lam']))
            return d
        return cls(dep('t'), dep('w'))


    # This layer's share of the deploy bundle: λ, coefficients and normalization statistics
    # 배포 번들에 들어갈 항목 — 축별 λ · 계수 · 정규화 통계
    def bundle(self):
        return {f'{p}_{k}': np.asarray(d[k]) for p, d in (('t', self.dep_t), ('w', self.dep_w)) for k in Layer2.KEYS}


# ---- notebook cell 112 · def assemble(z_tie, z_win): · sha256 7c6ea2443374bb1c ----
"""
log-space chain rule to compute probabilities. no softmax needed.

let z_tie = logit(P(tie)), z_win = logit(P(winner_A))

then

log P(A)   = −softplus(z_tie) − softplus(−z_win)      = log(1−p₀) + log q

log P(B)   = −softplus(z_tie) − softplus(+z_win)

log P(tie) = −softplus(−z_tie)
"""
def assemble(z_tie, z_win):
    sp = lambda x: np.logaddexp(0.0, x)
    logP = np.column_stack([-sp(z_tie) - sp(-z_win),
                            -sp(z_tie) - sp(z_win),
                            -sp(-z_tie)])
    return logP


def score(y, nz, z_tie, z_win, seg):
    yT, yA = (y == 2).astype(np.float64), (y == 0).astype(np.float64)
    lt = rowloss(z_tie[seg], yT[seg])
    lw = rowloss(z_win[seg & nz], yA[seg & nz])
    row = lt.astype(np.float64).copy()
    row[nz[seg]] += lw
    return row, float(lt.mean()), float(lw.mean())


# ---- notebook cell 114 · def wiring_checks(L1, L2): · sha256 af53b6fd60fce6ff ----
@torch.no_grad()
def wiring_checks(L1, L2):
    g = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    nu, nv = 5, 3                             # 짝 로짓 5 · 홀 로짓 3 / even logits, odd logits

    # ① Column split — body and readout are disjoint and together cover the whole input
    # ① 열 분할 — 몸통과 판독이 겹치지 않고, 둘이 입력 전체를 덮는다
    for d in (Layer1.K_TIE, POOLING_DIM):
        skip, body = Layer1._split(d, nu, nv)
        assert len(body) == 2 * d and len(skip) == nu + nv, '[wiring] split width contract violated!!'
        assert not set(body) & set(skip), '[wiring] pooled/logit column split overlaps!!'
        assert sorted(body + skip) == list(range(2 * d + nu + nv)), '[wiring] split does not cover the input!!'

    # ② Pooling slices — the block names derive from SRCS and the four slices tile the hstack exactly
    # ② 풀링 슬라이스 — 블록 이름이 SRCS 에서 유도되고, 넷이 hstack 을 빈틈없이 덮는다
    assert EVK == tuple(f'{s}_u' for s in SRCS) and ODK == tuple(f'{s}_v' for s in SRCS), \
        '[wiring] even/odd block names are not derived from SRCS!!'
    wide = len(SRCS) * POOLING_DIM
    cov = [i for j in range(len(SRCS)) for i in range(*Layer1._blk(j).indices(wide))]
    assert cov == list(range(wide)), '[wiring] pooling slices do not tile the per-axis hstack!!'

    # ③ Normalization parity — the odd axis never centres and preserves a sign flip
    # ③ 정규화 패리티 — 홀 축은 중심을 옮기지 않고 부호 반전을 그대로 보존한다
    X = rng.normal(size=(256, 16))
    mu_o, sc_o = zstats(X, False)
    assert not mu_o.any(), '[wiring] odd-axis zstats must not centre!!'
    assert np.abs((-X) / zstats(-X, False)[1] + X / sc_o).max() < 1e-12, \
        '[wiring] odd-axis normalization breaks antisymmetry!!'
    mu_e, sc_e = zstats(X, True)
    assert np.abs(((X - mu_e) / sc_e).mean(0)).max() < 1e-12, '[wiring] even-axis zstats must centre!!'

    # ④ Layer-1 torch modules — parameter counts, zero-init output, parity (tie even / win odd)
    # ④ 층 1 토치 모듈 — 파라미터 수 · 0초기 출력 0 · 파리티(tie 짝 / win 홀)
    k = Layer1.K_TIE
    skip, body = Layer1._split(k, nu, nv)
    tie = TieNet(2 * k + nu + nv, H, Layer1.UNITS[0], skip, body)
    n_tie = sum(q.numel() for q in tie.parameters())
    Ux, Vx = torch.randn(64, k + nu, generator=g), torch.randn(64, k + nv, generator=g)
    nn.init.zeros_(tie.w.weight)
    nn.init.zeros_(tie.w.bias)
    z0 = float(sym_post(tie, Ux, Vx, +1).abs().max())
    nn.init.normal_(tie.w.weight)
    ev = float((sym_post(tie, Ux, Vx, +1) - sym_post(tie, Ux, -Vx, +1)).abs().max())

    p = POOLING_DIM                           # win body sees one pooled block / 승부 몸통은 소스별 풀링 블록 
    skip, body = Layer1._split(p, nu, nv)
    win = WinNet(2 * p + nu + nv, H, skip, body, p)
    n_win = sum(q.numel() for q in win.parameters())
    Uw, Vw = torch.randn(64, p + nu, generator=g), torch.randn(64, p + nv, generator=g)
    nn.init.normal_(win.w.weight)
    od = float((sym_post(win, Uw, Vw, -1) + sym_post(win, Uw, -Vw, -1)).abs().max())
    assert n_tie == 262537 and n_win == 262409, '[wiring] parameter-count contract violated!!'
    assert z0 == 0.0 and ev < 1e-4 and od < 1e-4, '[wiring] parity/zero-init contract violated!!'

    # ⑤ Layer-2 primitives — bias on the even axis only, parity-aware normalization, optimizer converges
    # ⑤ 층 2 연산자 — 절편은 짝축에만, 정규화는 패리티를 따르고, 최적화가 수렴한다
    B = rng.normal(size=(64, 4))
    v = np.arange(5, dtype=np.float64)
    assert np.abs(Layer2._score(v, B, True) - (B @ v[:4] + v[4])).max() < 1e-12, '[wiring] even-axis score drops the bias!!'
    assert np.abs(Layer2._score(v, B, False) - B @ v[:4]).max() < 1e-12, '[wiring] odd-axis score must carry no bias!!'
    za_, zb_ = Layer2._z(B, -B, False)
    assert np.abs(za_ + zb_).max() < 1e-12, '[wiring] odd-axis combiner normalization breaks antisymmetry!!'
    Xs = rng.normal(size=(300, 3))
    _, ok = Layer2._fit_one(Xs, (Xs[:, 0] > 0).astype(np.float64), np.zeros(3), True)
    assert ok, '[wiring] penalized-logistic optimizer did not converge on the guard problem!!'

    # ⑥ Assembly identities — the three probabilities sum to 1, the axis split reproduces the total
    # ⑥ 조립 항등 — 3-class 확률합 1, 축 분해가 총손실과 일치
    zt, zw = rng.normal(scale=3.0, size=512), rng.normal(scale=3.0, size=512)
    ps = float(np.abs(np.exp(assemble(zt, zw)).sum(1) - 1).max())
    ys = np.resize(np.array([0, 1, 2]), 512)
    nz = ys != 2
    row, tie_ll, win_ll = score(ys, nz, zt, zw, np.ones(512, bool))
    dec = abs(tie_ll + nz.mean() * win_ll - float(row.mean()))
    assert ps < 1e-12, '[wiring] assembly probabilities do not sum to 1!!'
    assert dec < 1e-12, '[wiring] axis decomposition identity violated!!'

    # ⑦ Fitted layer 1 — the bundle keys are complete, the SVD ranks agree, one state per round/source
    # ⑦ 적합된 층 1 — 번들 키가 빠짐없고, 두 축의 SVD 랭크가 같고, 라운드/소스마다 상태가 하나씩
    assert set(L1.Z) == set(Layer1.KEYS), '[wiring] layer-1 Z keys != Layer1.KEYS!!'
    assert L1.Z['U_W'].shape[1] == L1.Z['V_W'].shape[1] == Layer1.K_TIE, '[wiring] SVD rank disagrees with K_TIE!!'
    assert len(L1.states['tie']) == len(Layer1.UNITS), '[wiring] tie states != tie rounds!!'
    assert len(L1.states['win']) == len(SRCS), '[wiring] win states != sources!!'
    assert all(np.isfinite(L1.Z[k2]).all() for k2 in Layer1.KEYS), '[wiring] layer-1 Z has non-finite values!!'

    # ⑧ Fitted layer 2 — width = logit basis + layer-1 columns, bias on tie only, odd axis uncentred
    # ⑧ 적합된 층 2 — 폭 = 로짓 기저 + 층 1 열, 절편은 tie 에만, 홀 축은 무중심
    nt, nw = nu + len(Layer1.UNITS), nv + len(SRCS)
    assert len(L2.dep_t['mu']) == nt and len(L2.dep_w['mu']) == nw, '[wiring] combiner width != logits + layer-1 columns!!'
    assert len(L2.dep_t['v']) == nt + 1, '[wiring] tie combiner is missing its bias coefficient!!'
    assert len(L2.dep_w['v']) == nw, '[wiring] win combiner must carry no bias coefficient!!'
    assert not np.asarray(L2.dep_w['mu']).any(), '[wiring] win combiner centred the odd axis!!'
    if L2.z_tie is not None:                  # cross-fit z 는 학습 런에만 있다 / training run only
        assert np.isfinite(L2.z_tie).all(), '[wiring] layer-2 tie cross-fit z has non-finite values!!'

    print(f'*** Wiring check passed ***')
