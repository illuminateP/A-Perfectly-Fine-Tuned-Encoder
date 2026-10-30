"""승부축 실험실 v3 — GATE·S·h128 기준점 위에서 정제·구조 팔을 잰다. tie축은 저장 열로 고정.

    [로짓 축 — v2 에서 닫힘]  SK(판독 [h|로짓]) 승. PURE·ANC 팔은 대조용으로 유지.
        SK    판독 [h | 로짓]  · r1 무앵커        tie §3.6 확정형의 거울 — 현행 승자
        PURE  판독 [h]        · r1 무앵커        로짓은 결합기만 (v2: +0.00037, SK 열세)
        ANC   판독 [h]        · r1 = s0 앵커      s0 = TR 비-tie [m3] 적합 (v2: +0.00083)

    [정제 축]  소스를 헤드에 주기 전에 어떻게 다듬나 (로짓 = SK 고정)
        RAW      그냥 (좌표 z-score/RMS 만) — 현행 승자
        PSVD     소스별 SVD-512 + 화이트닝 (v2 사망 +0.0008 — 절단+회전 동시범)
        SHRINK   자기 기저 릿지 수축 e/(e+λ_s) (v2 사망 +0.0020)
        ARD      v 좌표를 적응릿지 관련도 r=|w|/RMS(|w|) 로 재척도 (run_ard 규칙:
                 pen_j ← λ·(1/(w²+ε))/평균, 5회). λ 는 SHRINK_LAM 이월. 재정규화 없음
                 (하면 상쇄). 라벨을 전처리에 쓰는 팔 (TR 만) — v2 유일 생존 (−0.00033, 3/3)
        ARDCV    ARD 의 λ 를 win 전용 CV 로 재선택 (10배 격자 {1,10,10²,10³} · 2-fold ·
                 목적 = 적응릿지 최종 반복의 fold-밖 ll). 이월값이 [31.6,316] 이라 양쪽을 덮는다
        ARDCV05  ARDCV 관련도의 α=0.5 — v ← v ⊙ r^α (약한 재척도)
        ARDCV2   ARDCV 관련도의 α=2   —              (강한 재척도)
        WHITE    소스별 **전랭크** 회전+화이트닝 (절단 없음) — PSVD 패배에서 '절단 죄'와
                 '회전·등분산 죄'를 분리. RAW 가 이미 대각 화이트닝이라 순수 추가분은 회전

    [구조 축]  GATE 코어에 용량을 얹는다 (정제 = RAW · 로짓 = SK 고정).
               두 팔 다 추가 출력층 0 초기화 — 초기 함수가 GATE 와 비트 동일
        FFN      코어 뒤 h→4h→h 잔차 MLP (gelu) — tie 에선 FFN 판독 2/81 로 사망 전례
        INMLP    코어 앞 가지별 1024→4096→1024 잔차 MLP (gelu) — 입력 특징을 학습으로 추출

    [유닛 축]  타입드 게이트(짝가지⊙홀가지)의 활성 조합 — v1 사영형 6종과 별개 (v4)
        TGATE tanh⊙tanh · LGATE id⊙tanh · GLIN σ⊙id · BLIN id⊙id   (기준 GATE = σ⊙tanh.
        게이트 가지는 짝이라 활성 무엇이든 파리티 무해 · 값 가지는 tanh/id 둘 다 홀)
    [폭 축]   h ∈ {64, 256, 512} — 판독 재료 [h|로짓8]의 h 가 128 이 꼭짓점인지 (v4)
    [판독 축]  ROFFN — 판독 [h|로짓] 뒤 무절편 tanh MLP(d→4d→1) 잔차 (홀 보존·0초기화) (v4)
    [대칭화 축] AUG — 대칭화 없이 스왑 증강(행 2배·라벨 반전)으로 학습. 짝 채널 개방:
        Lu 5칸 생존·판독 절편 자유. 배포 열 2벌 — AUG(원시 f(x)) · AUGTTA(홀 사영
        (f(x)−f(swap x))/2). 판독: AUGTTA−SK = 제약 시점 효과(학습 중 강제 vs 배포 사영)
        · AUG−AUGTTA = 배포된 짝 채널 신호(위치 편향 등). 1라운드 전용 (v4)

    [배치]  같은 열로 결합기만 두 번 — 학습 비용 0
        대체  [m3 | W]           헤드가 LIN 을 대신한다 (v2 승자 배치 — LIN12 제거)
        얹기  [m3 | LIN12 | W]   헤드가 LIN 밖을 보는지 — tie 기둥 제거 논리의 거울

열 순서는 **라운드-우선** [r1×4소스 | r2×4소스] — g1 접두사가 "전 소스의 r1" 이 되게.
(v1 은 소스-우선이라 S 공급의 g1 칸이 "소스 2개의 전 라운드"로 잘못 잘렸다. g2 무사.)

행 우주: 두 축은 완전 분리 — tie 성분은 전 팔에 행별 동일 상수라 paired Δ 에서 상쇄.
판정은 승부 전용 ll (holdout∩비-tie), 총점은 축척 비교용 (p0_g3 = 0.96044 기준).
파리티 계약: 홀 경로 절편 금지 — 판독 절편은 생성 시점부터 0 동결.
"""
import argparse
import gc
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.extmath import randomized_svd

import tie_ladder as TL
import readout as RO

FINAL = TL.FINAL
TB_TIE_OFFSET = -0.00064          # 확정 TB tie(§3.6) 가 저장 GAT tie 대비 총점에서 버는 상수
SRCS = ('abe2', 'tb2', 'abe1', 'tb1')                    # EVK/ODK 순서 (1024 씩)
SHRINK_LAM = {'abe2': 100.0, 'tb2': 316.228, 'abe1': 100.0, 'tb1': 31.6228}
ARCH = {'GATE': ('GATE', 'GATE'), 'TANH': ('TANH', 'TANH'),
        'TB': ('TANHTANH', 'BILIN'), 'GAT': ('GAT', 'GAT'),
        'TGATE': ('TGATE', 'TGATE'), 'LGATE': ('LGATE', 'LGATE'),
        'GLIN': ('GLIN', 'GLIN'), 'BLIN': ('BLIN', 'BLIN')}
# 타입드 유닛: (게이트 활성, 값 활성) — 게이트 가지(짝)는 활성 자유, 값 가지(홀)는 홀 활성만
TYPED = {'GATE': ('sig', 'tanh'), 'TGATE': ('tanh', 'tanh'), 'LGATE': ('id', 'tanh'),
         'GLIN': ('sig', 'id'), 'BLIN': ('id', 'id')}
_ACT = {'sig': torch.sigmoid, 'tanh': torch.tanh, 'id': lambda t: t}
# 팔 사전: lg = 로짓 취급 · sup = 공급(정제) · net = 몸통 구조 · grp = 요약 축
#   선택 필드: arch(유닛 override) · h(폭 override) · head(판독) · aug(스왑 증강) · virtual
ARMS_DEF = {
    'SK-RAW':     dict(lg='SK',   sup='RAW',     net='GATE'),
    'PURE-RAW':   dict(lg='PURE', sup='RAW',     net='GATE',  grp='로짓'),
    'ANC-RAW':    dict(lg='ANC',  sup='RAW',     net='GATE',  grp='로짓'),
    'SK-PSVD':    dict(lg='SK',   sup='PSVD',    net='GATE',  grp='정제'),
    'SK-SHRINK':  dict(lg='SK',   sup='SHRINK',  net='GATE',  grp='정제'),
    'SK-ARD':     dict(lg='SK',   sup='ARD',     net='GATE',  grp='정제'),
    'SK-ARDCV':   dict(lg='SK',   sup='ARDCV',   net='GATE',  grp='정제'),
    'SK-ARDCV05': dict(lg='SK',   sup='ARDCV05', net='GATE',  grp='정제'),
    'SK-ARDCV2':  dict(lg='SK',   sup='ARDCV2',  net='GATE',  grp='정제'),
    'SK-WHITE':   dict(lg='SK',   sup='WHITE',   net='GATE',  grp='정제'),
    'SK-FFN':     dict(lg='SK',   sup='RAW',     net='FFN',   grp='구조'),
    'SK-INMLP':   dict(lg='SK',   sup='RAW',     net='INMLP', grp='구조'),
    'SK-TGATE':   dict(lg='SK',   sup='RAW',     net='GATE',  arch='TGATE', grp='유닛'),
    'SK-LGATE':   dict(lg='SK',   sup='RAW',     net='GATE',  arch='LGATE', grp='유닛'),
    'SK-GLIN':    dict(lg='SK',   sup='RAW',     net='GATE',  arch='GLIN',  grp='유닛'),
    'SK-BLIN':    dict(lg='SK',   sup='RAW',     net='GATE',  arch='BLIN',  grp='유닛'),
    'SK-H64':     dict(lg='SK',   sup='RAW',     net='GATE',  h=64,  grp='폭'),
    'SK-H256':    dict(lg='SK',   sup='RAW',     net='GATE',  h=256, grp='폭'),
    'SK-H512':    dict(lg='SK',   sup='RAW',     net='GATE',  h=512, grp='폭'),
    'SK-ROFFN':   dict(lg='SK',   sup='RAW',     net='GATE',  head='OFFN', grp='판독'),
    'SK-AUG':     dict(lg='SK',   sup='RAW',     net='GATE',  aug=True, grp='대칭화'),
    'SK-AUGTTA':  dict(grp='대칭화', virtual=True),   # SK-AUG 학습에서 파생 (홀 사영 배포)
}
ARMS_ALL = [a for a in ARMS_DEF if not ARMS_DEF[a].get('virtual')]


# ══════════════════════════════════════════════ 망 (v1 그대로)
class ParityBody(nn.Module):
    """(짝 xu, 홀 xv) 를 나눠 쓰는 구조적 홀 몸통. GATE: σ(짝게이트)⊙tanh(홀값·무절편).

    kind 는 GATE 코어에 얹는 용량 증설 (추가 출력층 0 초기화 → 초기 함수 == GATE):
        FFN    코어 뒤 h→4h→h 잔차 MLP (gelu)
        INMLP  코어 앞 가지별 d→4d→d 잔차 MLP (gelu) — 입력 특징을 학습으로 추출.
               비홀 성분은 sym_post 의 g(x⁺)−g(x⁻) 가 사영으로 제거한다
    """

    def __init__(self, d, h, d_u, kind='GATE', gact='sig', vact='tanh'):
        super().__init__()
        self.d_u, self.kind, self.out_dim = d_u, kind, h
        self.gact, self.vact = gact, vact
        d_v = d - d_u
        self.G = nn.Linear(d_u, h)
        self.V = nn.Linear(d_v, h, bias=False)
        if kind == 'FFN':
            self.f1, self.f2 = nn.Linear(h, 4 * h), nn.Linear(4 * h, h)
            nn.init.zeros_(self.f2.weight)
            nn.init.zeros_(self.f2.bias)
        elif kind == 'INMLP':
            self.eu1, self.eu2 = nn.Linear(d_u, 4 * d_u), nn.Linear(4 * d_u, d_u)
            self.ev1, self.ev2 = nn.Linear(d_v, 4 * d_v), nn.Linear(4 * d_v, d_v)
            for lin in (self.eu2, self.ev2):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        elif kind != 'GATE':
            raise ValueError(kind)

    def forward(self, x):
        xu, xv = x[:, :self.d_u], x[:, self.d_u:]
        if self.kind == 'INMLP':
            xu = xu + self.eu2(F.gelu(self.eu1(xu)))
            xv = xv + self.ev2(F.gelu(self.ev1(xv)))
        u = _ACT[self.gact](self.G(xu)) * _ACT[self.vact](self.V(xv))
        if self.kind == 'FFN':
            u = u + self.f2(F.gelu(self.f1(u)))
        return u


class HeadOddFFN(nn.Module):
    """홀 보존 판독 MLP — 선형부 + 무절편 tanh MLP(d→4d→1) 잔차. 홀 입력에 대해 홀
    (tanh 홀·전 층 무절편). c2 는 0 초기화 → fit_gat 의 .w 0 초기화와 합쳐 초기 판독 0."""

    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, 1)                    # fit_gat 이 zeros_ 하는 선형부
        self.c1 = nn.Linear(dim, 4 * dim, bias=False)
        self.c2 = nn.Linear(4 * dim, 1, bias=False)
        nn.init.zeros_(self.c2.weight)

    def forward(self, x):
        return self.w(x) + self.c2(torch.tanh(self.c1(x)))


class WinNet(nn.Module):
    """몸통 + 판독. 판독 절편은 생성 시점부터 0 동결 (홀 축 무절편 계약).

    free_bias=True (AUG 팔) 는 동결을 푼다 — 대칭화 없는 학습에서 절편은 짝 채널
    (위치 편향 상수)의 일부다."""

    def __init__(self, d, h, unit, skip_idx, body_idx, d_u, netk='GATE',
                 headk='LIN', free_bias=False):
        super().__init__()
        self.body_idx, self.skip_idx = body_idx, skip_idx
        d_body = d if body_idx is None else len(body_idx)
        if unit in TYPED:
            ga, va = TYPED[unit]
            self.body = ParityBody(d_body, h, d_u, netk, ga, va)
        else:
            assert netk == 'GATE', f'netk={netk} 는 GATE 코어 전용'
            self.body = RO.BODIES[unit](d_body, h)
        n_sk = 0 if skip_idx is None else len(skip_idx)
        self.head = (RO.HeadLin if headk == 'LIN' else HeadOddFFN)(
            self.body.out_dim + n_sk)
        nn.init.zeros_(self.head.w.bias)
        if not free_bias:
            self.head.w.bias.requires_grad_(False)

    @property
    def w(self):
        return self.head.w

    def forward(self, x):
        u = self.body(x if self.body_idx is None else x[:, self.body_idx])
        if self.skip_idx is not None:
            u = torch.cat([u, x[:, self.skip_idx]], 1)
        return self.head(u)


def make_net(unit, skip_idx, body_idx, d_u, netk='GATE', headk='LIN', free_bias=False):
    def _f(d, h, _u=unit, _s=skip_idx, _b=body_idx, _du=d_u, _n=netk,
           _hk=headk, _fb=free_bias):
        return WinNet(d, h, _u, _s, _b, _du, _n, _hk, _fb)
    return _f


def nosym(f, Ux, Vx, sgn):
    """대칭화 없음 — 원시 f(x) 판독 (AUG 팔 전용, fit_gat 의 sym_fn 자리)."""
    return f(torch.cat([Ux, Vx], 1)).squeeze(-1)


# ══════════════════════════════════════════════ 자산 · 전처리
def raw_uv(U):
    """raw 소스별 (u,v) — TR/stage2, joint 캐시의 NU/NV 슬라이스로 정규화 (좌표 동일)."""
    M, Z = TL.MODELS, TL.load_svd_cache()
    spec = [('abe2__checkpoint-3000', ('ea',), ('eb',)),
            ('tb2__checkpoint-4000', ('a_f', 'b_s'), ('b_f', 'a_s')),
            (f'abe1__checkpoint-{TL.ABE1_STEP}', ('ea',), ('eb',)),
            (f'tb1__checkpoint-{TL.TB1_STEP}', ('x_f',), ('x_s',))]
    out = {}
    for i, (tag, ka, kb) in enumerate(spec):
        zt, zs = np.load(M / f'pooled/{tag}__train.npz'), np.load(M / f'pooled/{tag}.npz')
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
    print(f'  raw 소스별 적재 — 4소스 × (u,v) 1024  [TR {out["abe2"]["u_tr"].shape[0]:,}행]')
    return out


def t_psvd(RAW, K=512):
    """소스별 SVD-K + 화이트닝 (u: TR 중심화 / v: 무중심). d_u 가 512 로 준다."""
    out = {}
    for s, R in RAW.items():
        t0, d = time.time(), {}
        for p, center in (('u', True), ('v', False)):
            tr, s2 = R[f'{p}_tr'], R[f'{p}_s2']
            mu = tr.mean(0) if center else np.zeros(tr.shape[1], np.float32)
            _, _, Vt = randomized_svd((tr - mu).astype(np.float64), n_components=K,
                                      random_state=42)
            W = Vt.T.astype(np.float32)
            ct, cs = (tr - mu) @ W, (s2 - mu) @ W
            sc = np.clip(np.sqrt((ct.astype(np.float64) ** 2).mean(0)), 1e-9,
                         None).astype(np.float32)
            d[f'{p}_tr'], d[f'{p}_s2'] = ct / sc, cs / sc
        out[s] = d
        print(f'    [psvd·{s}] SVD-{K} ×(u,v) [{time.time() - t0:.0f}s]')
    return out


def t_shrink(RAW):
    """v 만 소스 자기 기저에서 릿지 수축 x' = Q·diag(e/(e+λ_s))·Qᵀx. 절단·재정규화 없음."""
    out = {}
    for s, R in RAW.items():
        t0 = time.time()
        X = R['v_tr'].astype(np.float64)
        e, Q = np.linalg.eigh((X.T @ X) / len(X))
        d = (e / (e + SHRINK_LAM[s])).astype(np.float32)
        T = (Q.astype(np.float32) * d) @ Q.T.astype(np.float32)
        out[s] = dict(u_tr=R['u_tr'], u_s2=R['u_s2'],
                      v_tr=R['v_tr'] @ T, v_s2=R['v_s2'] @ T)
        print(f'    [shrink·{s}] λ {SHRINK_LAM[s]:g} · 수축 중앙 {np.median(d):.3f} '
              f'· 상위 {d[-1]:.3f} [{time.time() - t0:.0f}s]')
    return out


def t_white(RAW):
    """소스별 **전랭크** 회전+화이트닝 (K 절단 없음). u: TR 중심화 / v: 무중심 (파리티).

    PSVD(SVD-512+화이트닝)의 패배에서 '절단 죄'와 '회전·등분산 죄'를 분리한다 —
    RAW 가 이미 좌표(대각) 화이트닝이므로 이 팔의 순수 추가분은 기저 회전이다.
    """
    out = {}
    for s, R in RAW.items():
        t0, d, amp = time.time(), {}, {}
        for p, center in (('u', True), ('v', False)):
            tr, s2 = R[f'{p}_tr'], R[f'{p}_s2']
            mu = tr.mean(0) if center else np.zeros(tr.shape[1], np.float32)
            X = (tr - mu).astype(np.float64)
            e, Q = np.linalg.eigh((X.T @ X) / len(X))
            e = np.clip(e, 1e-10, None)
            W = (Q / np.sqrt(e)).astype(np.float32)
            d[f'{p}_tr'], d[f'{p}_s2'] = (tr - mu) @ W, (s2 - mu) @ W
            amp[p] = float(np.sqrt(e[-1] / e[0]))
        out[s] = d
        print(f'    [white·{s}] 전랭크 {R["v_tr"].shape[1]} · 꼬리 증폭 u {amp["u"]:.0f}× '
              f'v {amp["v"]:.0f}× [{time.time() - t0:.0f}s]')
    return out


def ard_probe(RAW, A, lam_map, iters=5, eps=1e-8, tag='ard'):
    """적응릿지 관련도 프로브 — r_s = |w|/RMS(|w|) 만 반환 (변환은 ard_apply).

    설계 X = [Lv 3 무벌점 | v_s 1024 λ] · TR 비-tie · pen_j ← λ·(1/(w²+ε))/평균
    (run_ard 규칙, 5회). λ 는 lam_map (이월 SHRINK_LAM 또는 ard_cv_lam 선택값).
    """
    m = A['y_tr'] != 2
    y = (A['y_tr'] == 0).astype(np.float64)[m]
    Lv = A['Lv_tr'][m].astype(np.float64)
    out = {}
    nl = Lv.shape[1]
    for s, R in RAW.items():
        t0 = time.time()
        dv = R['v_tr'].shape[1]
        X = np.hstack([Lv, R['v_tr'][m].astype(np.float64)])
        lam, v, w = lam_map[s], None, None
        pen = np.concatenate([np.zeros(nl), np.full(dv, lam)])
        for _ in range(iters):
            v, _ = TL._fit(X, y, pen, False, v)
            w = v[nl:nl + dv]
            inv = 1.0 / (w ** 2 + eps)
            pen = np.concatenate([np.zeros(nl), lam * inv / inv.mean()])
        r = np.abs(w) / np.sqrt((w ** 2).mean())
        a = np.sort(np.abs(w))[::-1]
        n90 = int(np.searchsorted(np.cumsum(a) / max(a.sum(), 1e-12), 0.90) + 1)
        out[s] = r.astype(np.float32)
        print(f'    [{tag}·{s}] λ {lam:g} · |w| 90% 를 좌표 {n90}/{dv} 개가 설명 '
              f'[{time.time() - t0:.0f}s]')
    return out


def ard_apply(RAW, R_map, alpha=1.0):
    """v ← v ⊙ r^α. 재정규화 없음 (하면 재척도가 정확히 상쇄된다). u 는 참조 공유."""
    out = {}
    for s, R in RAW.items():
        r = R_map[s] if alpha == 1.0 else \
            (R_map[s].astype(np.float64) ** alpha).astype(np.float32)
        out[s] = dict(u_tr=R['u_tr'], u_s2=R['u_s2'],
                      v_tr=R['v_tr'] * r, v_s2=R['v_s2'] * r)
    return out


def ard_cv_lam(RAW, A, grid=(1.0, 10.0, 100.0, 1000.0), folds=2, iters=5, eps=1e-8):
    """ARD 의 λ 를 win 전용 CV 로 재선택 — 목적은 적응릿지 **최종 반복** 적합의 fold-밖 ll.

    §3.7 교훈대로 격자는 10배 간격(촘촘한 눈금은 선택 자체가 과적합). 이월값
    SHRINK_LAM 이 [31.6, 316] 이라 {1,10,10²,10³} 이 양쪽 한 자릿수씩을 덮는다.
    """
    m = A['y_tr'] != 2
    y = (A['y_tr'] == 0).astype(np.float64)[m]
    Lv = A['Lv_tr'][m].astype(np.float64)
    nl = Lv.shape[1]
    fold = np.random.default_rng(42).permutation(len(y)) % folds
    out = {}
    for s, R in RAW.items():
        t0 = time.time()
        dv = R['v_tr'].shape[1]
        X = np.hstack([Lv, R['v_tr'][m].astype(np.float64)])

        def ll_at(lam):
            tot = 0.0
            for f in range(folds):
                tr, va = fold != f, fold == f
                Xt, yt = X[tr], y[tr]
                v = None
                pen = np.concatenate([np.zeros(nl), np.full(dv, lam)])
                for _ in range(iters):
                    v, _ = TL._fit(Xt, yt, pen, False, v)
                    w = v[nl:nl + dv]
                    inv = 1.0 / (w ** 2 + eps)
                    pen = np.concatenate([np.zeros(nl), lam * inv / inv.mean()])
                tot += float(TL.rowloss(X[va] @ v, y[va]).mean()) / folds
            return tot

        grid_s = [float(g) for g in grid]
        lls = [ll_at(g) for g in grid_s]
        # §3.7 계약 미러 — 고른 값이 격자 최댓값이면 그 축만 ×10 (최대 +3자릿수)
        while int(np.argmin(lls)) == len(grid_s) - 1 and len(grid_s) < len(grid) + 3:
            grid_s.append(grid_s[-1] * 10.0)
            lls.append(ll_at(grid_s[-1]))
        out[s] = grid_s[int(np.argmin(lls))]
        print(f'    [ardcv·{s}] λ* {out[s]:g} ← '
              + ' '.join(f'{g:g}:{l:.5f}' for g, l in zip(grid_s, lls))
              + f' (이월 {SHRINK_LAM[s]:g}) [{time.time() - t0:.0f}s]')
    return out


def win_s0(A):
    """ANC 앵커 — TR 비-tie 에서 [m3] 만 무벌점·무절편 적합한 점수 (위생: TR 전용)."""
    m = A['y_tr'] != 2
    y = (A['y_tr'] == 0).astype(np.float64)[m]
    X = A['Lv_tr'][m].astype(np.float64)
    v, ok = TL._fit(X, y, np.zeros(X.shape[1]), False)
    z0 = X @ v[:X.shape[1]]
    print(f'  ANC 앵커 s0 — [m3] TR 적합 w {np.round(v[:3], 3)} '
          f'{"수렴" if ok else "** 미수렴"}')
    return z0


def fixed_tie(U):
    """tie축 고정 — 학습 없음. 저장 배포 GAT 3열로 결합기 1회 (축척 = p0_g3)."""
    X = np.hstack([U['tie_base'], U['GAT']])
    rows = U['cu']
    y01 = (U['y'] == 2).astype(np.float64)
    zt, dep = TL.combiner(X, y01, U['fold'], rows, bias=True,
                          n_logit=U['tie_base'].shape[1])
    ll = TL.rowloss(zt[rows], y01[rows])
    print(f'  tie 고정 (저장 GAT · 학습 없음) — λ {dep["lam"]:g} '
          f'· tie(hold) {ll[U["hold"][rows]].mean():.5f}')
    return dict(ll=ll, rows=rows)


# ══════════════════════════════════════════════ 체인 · 평가
def chain_src(A, SUP, arch, seed, tag, h, logit, s0=None, rounds=2, netk='GATE',
              headk='LIN'):
    """소스별 체인 — **라운드-우선** 열 [r1×4소스 | r2×4소스]. 앵커는 소스 안에서 굴린다."""
    n_lu, n_lv = A['Lu_tr'].shape[1], A['Lv_tr'].shape[1]
    m = A['y_tr'] != 2
    y01w = (A['y_tr'] == 0).astype(np.float64)[m]
    zs = {s: (s0 if logit == 'ANC' else None) for s in SRCS}
    cols = []
    for rd in range(1, rounds + 1):
        unit = ARCH[arch][rd - 1]
        for si, s in enumerate(SRCS):
            R = SUP[s]
            du, dv = R['u_tr'].shape[1], R['v_tr'].shape[1]
            if logit == 'SK':
                sk = list(range(du, du + n_lu)) + \
                    list(range(du + n_lu + dv, du + n_lu + dv + n_lv))
                pool = list(range(0, du)) + list(range(du + n_lu, du + n_lu + dv))
                cu_t = np.ascontiguousarray(np.hstack([R['u_tr'][m], A['Lu_tr'][m]]),
                                            np.float32)
                cv_t = np.ascontiguousarray(np.hstack([R['v_tr'][m], A['Lv_tr'][m]]),
                                            np.float32)
                cu_s = np.ascontiguousarray(np.hstack([R['u_s2'], A['Lu_s2']]), np.float32)
                cv_s = np.ascontiguousarray(np.hstack([R['v_s2'], A['Lv_s2']]), np.float32)
            else:                                   # PURE·ANC — 헤드에 로짓 0
                sk, pool = None, None
                cu_t, cv_t = R['u_tr'][m], R['v_tr'][m]
                cu_s, cv_s = R['u_s2'], R['v_s2']
            net = make_net(unit, sk, pool, du, netk, headk)
            r_s2, aux = TL.fit_gat(cu_t, cv_t, y01w, zs[s], False, seed + si,
                                   cu_s, cv_s, hidden=h, net=net, sym_fn=RO.sym_post)
            cols.append(r_s2)
            print(f'      [{tag}·{s}] r{rd}: es_ll {aux["es_ll"]:.5f} '
                  f'ep{aux["es_epoch"]} | {unit}'
                  + (' | s0앵커' if rd == 1 and logit == 'ANC' else ''))
            zs[s] = aux['z_tr']
            del cu_t, cv_t, cu_s, cv_s
    gc.collect()
    return np.column_stack(cols)


def chain_aug(A, SUP, seed, tag, h):
    """스왑 증강 팔 (1라운드 전용) — 대칭화 없이 원시 f(x) 를 학습, 짝 채널 개방.

    학습행 = 원본 ∪ 스왑쌍 (짝가지 불변·홀가지 반부호·라벨 반전). ES 는 fit_gat 의
    내부 분할과 같은 규칙(_es_split, seed+si)으로 뗀 **원본 15% 행**에서 원시 판독으로.
    배포 열 2벌: raw = f(x) (짝 채널 포함) · tta = (f(x)−f(swap x))/2 (홀 사영) —
    stage2 를 [원본; 스왑] 으로 쌓아 한 번의 예측에서 둘 다 얻는다.
    """
    n_lu, n_lv = A['Lu_tr'].shape[1], A['Lv_tr'].shape[1]
    m = A['y_tr'] != 2
    y = (A['y_tr'] == 0).astype(np.float64)[m]
    raw_cols, tta_cols = [], []
    for si, s in enumerate(SRCS):
        R = SUP[s]
        du, dv = R['u_tr'].shape[1], R['v_tr'].shape[1]
        sk = list(range(du, du + n_lu)) + \
            list(range(du + n_lu + dv, du + n_lu + dv + n_lv))
        pool = list(range(0, du)) + list(range(du + n_lu, du + n_lu + dv))
        cu_t = np.ascontiguousarray(np.hstack([R['u_tr'][m], A['Lu_tr'][m]]), np.float32)
        cv_t = np.ascontiguousarray(np.hstack([R['v_tr'][m], A['Lv_tr'][m]]), np.float32)
        cu_s = np.ascontiguousarray(np.hstack([R['u_s2'], A['Lu_s2']]), np.float32)
        cv_s = np.ascontiguousarray(np.hstack([R['v_s2'], A['Lv_s2']]), np.float32)
        n = len(y)
        eidx, tidx = TL._es_split(n, seed + si)
        net = make_net('GATE', sk, pool, du, free_bias=True)
        r2, aux = TL.fit_gat(
            np.vstack([cu_t, cu_t]), np.vstack([cv_t, -cv_t]),
            np.concatenate([y, 1.0 - y]), None, False, seed + si,
            np.vstack([cu_s, cu_s]), np.vstack([cv_s, -cv_s]),
            hidden=h, net=net, sym_fn=nosym,
            es=dict(CU=cu_t[eidx], CV=cv_t[eidx], y=y[eidx], s0=None),
            tr_idx=np.concatenate([tidx, tidx + n]))
        m2 = len(r2) // 2
        raw_cols.append(r2[:m2])
        tta_cols.append((r2[:m2] - r2[m2:]) / 2.0)
        print(f'      [{tag}·{s}] r1: es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]} '
              f'| AUG 학습 {2 * len(tidx):,}행 · 절편 자유 · Lu 생존')
        del cu_t, cv_t, cu_s, cv_s
    gc.collect()
    return np.column_stack(raw_cols), np.column_stack(tta_cols)


def eval_win(U, tie, W_cols, with_lin=False):
    parts = [U['win_base']]
    if with_lin:
        parts.append(U['LIN'])
    if W_cols is not None:
        parts.append(W_cols)
    X = np.hstack(parts)
    rows = U['cu'] & U['nz']
    y01 = (U['y'] == 0).astype(np.float64)
    z, dep = TL.combiner(X, y01, U['fold'], rows, bias=False,
                         n_logit=U['win_base'].shape[1])
    llw = TL.rowloss(z[rows], y01[rows])
    row, lt, lw = TL.compose_ll(U, tie['ll'], llw, tie['rows'], rows)
    return row, llw[U['hold'][rows]], dict(tot=float(row.mean()), win=lw,
                                           n_col=X.shape[1], lam=dep['lam'])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bundle', default='temp/heads_repro')
    ap.add_argument('--arch', default='GATE')
    ap.add_argument('--arms', default=','.join(ARMS_ALL),
                    help='로짓-정제 팔: ' + ' '.join(ARMS_ALL))
    ap.add_argument('--seeds', default='42,1,2')
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--h', type=int, default=128, help='소스당 은닉 폭')
    ap.add_argument('--out', default='temp/winlab3')
    args = ap.parse_args()

    arch = args.arch
    arms = [a for a in args.arms.split(',') if a]
    seeds = [int(s) for s in args.seeds.split(',') if s]
    assert arch in ARCH, f'모르는 아키텍처 {arch}'
    assert set(arms) <= set(ARMS_ALL), f'모르는 팔 {set(arms) - set(ARMS_ALL)}'

    print('[적재]')
    U = TL.build(FINAL / args.bundle)
    A = TL.gat_assets(U)
    tie = fixed_tie(U)
    n_h, n_w = int(U['hold'].sum()), int((U['hold'] & U['nz']).sum())
    print(f'  행: holdout {n_h:,} 중 비-tie {n_w:,} → Δ총점 = {n_w / n_h:.3f} × Δ승부')
    RAW = raw_uv(U)
    s0 = win_s0(A) if any(ARMS_DEF[a]['lg'] == 'ANC' for a in arms) else None

    SUP = {'RAW': RAW}
    need = {ARMS_DEF[a]['sup'] for a in arms}
    lam_cv = None
    if 'PSVD' in need:
        print('  [전처리 PSVD]')
        SUP['PSVD'] = t_psvd(RAW)
    if 'SHRINK' in need:
        print('  [전처리 SHRINK]')
        SUP['SHRINK'] = t_shrink(RAW)
    if 'WHITE' in need:
        print('  [전처리 WHITE — 전랭크 회전+화이트닝]')
        SUP['WHITE'] = t_white(RAW)
    if 'ARD' in need:
        print('  [전처리 ARD — λ 이월(SHRINK_LAM)]')
        SUP['ARD'] = ard_apply(RAW, ard_probe(RAW, A, SHRINK_LAM))
    if need & {'ARDCV', 'ARDCV05', 'ARDCV2'}:
        print('  [전처리 ARDCV — λ win 전용 CV 재선택]')
        lam_cv = ard_cv_lam(RAW, A)
        r_cv = ard_probe(RAW, A, lam_cv, tag='ardcv')
        for nm, al in (('ARDCV', 1.0), ('ARDCV05', 0.5), ('ARDCV2', 2.0)):
            if nm in need:
                SUP[nm] = ard_apply(RAW, r_cv, al)

    res, rowc, roww, saved_cols = {}, {}, {}, {}

    def run(tag, W, with_lin, note=''):
        row, rw, d = eval_win(U, tie, W, with_lin=with_lin)
        res[tag], rowc[tag], roww[tag] = d, row, rw
        print(f'  [{tag}] win {d["win"]:.5f} · tot {d["tot"]:.5f}  '
              f'(열 {d["n_col"]} · λ {d["lam"]:g}){note}')
        return d

    print(f'\n{"="*78}\n[LIN 기준선] 결정론\n{"="*78}')
    run('LIN-S', U['LIN'], False, '  ← 현행 소스별 3R')
    run('logit', None, False, '  ← 로짓만 [m3]')

    print(f'\n{"="*78}\n[십자 격자] {arch} · S · h{args.h} · {len(arms)}팔 × {len(seeds)}시드 '
          f'· 배치 {{대체, 얹기}}\n{"="*78}')
    for a in arms:
        c = ARMS_DEF[a]
        print(f'  {a:11s} 판독 {"[h|로짓]" if c["lg"] == "SK" else "[h]":9s} '
              f'정제 {c["sup"]:8s} 망 {c["net"]:6s} 유닛 {c.get("arch", arch):6s} '
              f'h {c.get("h", args.h):<4d}'
              + (' 판독MLP' if c.get('head') == 'OFFN' else '')
              + (' r1=s0앵커' if c['lg'] == 'ANC' else '')
              + (' 증강(비대칭·절편자유)' if c.get('aug') else ''))

    for a in arms:
        c = ARMS_DEF[a]
        ah = c.get('h', args.h)
        for sd in seeds:
            tag = f'{a}-s{sd}'
            print(f'  [{tag}]')
            t0 = time.time()
            try:
                if c.get('aug'):
                    assert args.rounds == 1, 'AUG 팔은 1라운드 전용'
                    Wr, Wt = chain_aug(A, SUP[c['sup']], sd, tag, ah)
                else:
                    W = chain_src(A, SUP[c['sup']], c.get('arch', arch), sd, tag,
                                  ah, c['lg'], s0=s0, rounds=args.rounds,
                                  netk=c['net'], headk=c.get('head', 'LIN'))
            except RuntimeError as e:                # OOM 등 — 팔만 버리고 런은 지킨다
                print(f'    !! {tag} 실패 ({str(e).splitlines()[0][:80]}) — 건너뜀')
                torch.cuda.empty_cache()
                continue
            secs = time.time() - t0
            if c.get('aug'):                         # 한 학습에서 raw·tta 두 벌
                for t2, a2, W2 in ((tag, a, Wr), (f'SK-AUGTTA-s{sd}', 'SK-AUGTTA', Wt)):
                    saved_cols[f'cols__{t2}'] = W2.astype(np.float32)
                    d = run(f'{t2}-g1', W2, False)
                    d.update(arm=a2, seed=sd, n_gat=1, sec=round(secs, 1))
                    d = run(f'{t2}+L-g1', W2, True)
                    d.update(arm=a2, seed=sd, n_gat=1, overlay=True)
                continue
            saved_cols[f'cols__{tag}'] = W.astype(np.float32)
            for k in range(1, args.rounds + 1):
                d = run(f'{tag}-g{k}', W[:, :k * 4], False)
                d.update(arm=a, seed=sd, n_gat=k, sec=round(secs, 1))
            d = run(f'{tag}+L-g{args.rounds}', W, True)
            d.update(arm=a, seed=sd, n_gat=args.rounds, overlay=True)

    op = FINAL / args.out
    np.savez_compressed(op.with_suffix('.rowloss.npz'), **rowc,
                        **{f'WIN__{k}': v for k, v in roww.items()}, **saved_cols)
    with open(op.with_suffix('.json'), 'w') as f:
        json.dump({'cells': res, 'tb_tie_offset': TB_TIE_OFFSET,
                   'ard_lam_cv': lam_cv}, f, ensure_ascii=False, indent=1)
    print(f'\n저장: {op}.rowloss.npz / {op}.json (헤드 열 포함 — 재학습 없이 재배치 가능)')

    R = args.rounds
    arms = [a for a in arms                          # 실패 팔은 요약에서 제외
            if all(f'{a}-s{s}-g{R}' in res for s in seeds)]
    if 'SK-AUG' in arms and all(f'SK-AUGTTA-s{s}-g{R}' in res for s in seeds):
        arms.insert(arms.index('SK-AUG') + 1, 'SK-AUGTTA')   # 파생 팔을 요약에 편입
    wm = lambda a, suf: np.mean([res[f'{a}-s{s}{suf}']['win'] for s in seeds])
    wv = lambda a, suf: np.array([res[f'{a}-s{s}{suf}']['win'] for s in seeds])
    wr = lambda a, suf: np.mean([roww[f'{a}-s{s}{suf}'] for s in seeds], axis=0)
    g2 = f'-g{R}'

    print(f'\n{"="*78}\n[요약] 승부 전용 ll · {len(seeds)}시드 평균 · '
          f'현행 LIN-S {res["LIN-S"]["win"]:.5f}\n{"="*78}')
    print(f'  {"팔":12s}{"g1":>9s}{"g2":>9s}{"얹기":>9s}   LIN-S 대비 (g2 paired)')
    for a in arms:
        m, lo, hi = TL.paired_ci(wr(a, g2), roww['LIN-S'])
        d = wv(a, g2) - res['LIN-S']['win']
        print(f'  {a:12s}{wm(a, "-g1"):9.5f}{wm(a, g2):9.5f}{wm(a, "+L" + g2):9.5f}   '
              f'{TL.ci_str(m, lo, hi)}  시드 {int((d < 0).sum())}/{len(d)}')

    if 'SK-RAW' in arms:
        def _pline(a, base='SK-RAW', label=None):
            m, lo, hi = TL.paired_ci(wr(a, g2), wr(base, g2))
            d = wv(a, g2) - wv(base, g2)
            print(f'  {label or (a + " − " + base):24s} {TL.ci_str(m, lo, hi)}  '
                  f'시드 {d.mean():+.5f} {int((d < 0).sum())}/{len(d)}')

        for gname in ('로짓', '정제', '유닛', '폭', '판독', '대칭화', '구조'):
            ga = [x for x in arms if x != 'SK-RAW' and ARMS_DEF[x].get('grp') == gname]
            if not ga:
                continue
            print(f'\n[{gname} 축] SK-RAW 기준 · 같은 시드 paired (g{R})')
            for a in ga:
                _pline(a)
            if gname == '대칭화' and {'SK-AUG', 'SK-AUGTTA'} <= set(ga):
                _pline('SK-AUG', 'SK-AUGTTA', 'AUG − AUGTTA (짝 채널)')

    print(f'\n[얹기 − 대체] 같은 열 · 결합기만 (g{R}) — 0 이면 헤드가 LIN 을 포함')
    for a in arms:
        m, lo, hi = TL.paired_ci(wr(a, '+L' + g2), wr(a, g2))
        d = wv(a, '+L' + g2) - wv(a, g2)
        print(f'  {a:12s} {TL.ci_str(m, lo, hi)}  시드 {d.mean():+.5f} '
              f'{int((d < 0).sum())}/{len(d)} 얹기 유리')
    print('\n  ! r2 기여는 g2−g1 로 읽는다 (v1 의 S-g1 은 열 순서 버그로 무효였다 — '
          '여기선 라운드-우선이라 유효)')
    print(f'  ! 총점 축척 p0_g3 · 확정 TB tie 환산 = 전 칸 {TB_TIE_OFFSET:+.5f} · 순위 불변')


if __name__ == '__main__':
    main()
