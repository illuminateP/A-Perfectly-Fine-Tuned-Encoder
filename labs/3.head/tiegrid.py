#!/usr/bin/env python
"""tiegrid.py — tie축 전용 격자. **기둥 없음** (plan.md §3.5 결정 반영).

무엇을 재나 — 축 다섯
    K       SVD 랭크 {64,128,256,512}. 512 캐시의 **앞 K열을 자른다** — SVD 는 상위
            특이벡터 순서라 K=128 은 512 의 앞 128열과 정확히 같다. 재계산·raw 적재 0.
            (1024·FULL 은 hgv2 가 이미 기각: +0.0011 / +0.0007 vs 512 의 +0.0023)
    lr      {1e-4, 3e-4, 1e-3}. headgrid 계보에서 한 번도 안 흔든 축.
    head    GAT      TR 에서 게이트망 3라운드 -> 3열                (현행)
            MLP      TR 에서 GELU 1은닉층 3라운드 -> 3열            (게이트가 필요한가)
            RIDGE_TR TR 에서 릿지 부스팅 3라운드 -> 3열             (비선형이 필요한가)
            RIDGE_S2 결합기가 [로짓5 | CUt_K] 를 직접 받음          (중간층·TR 자체가 필요한가)
    seed    {42,1,2,3,4}. 선발전 실측 시드 표준편차가 0.0001~0.0005 라 기대는 작다.
    extra   결합기에 덧붙이는 짝 블록 (stage2 만 필요, GAT 입력에는 안 넣는다)
            norm12    소스별 ||A−B|| · ||A+B|| · cos(A,B)  = 12열
                      **원 풀링에서** 잰다 (SVD 뒤 재구성이 아니다). 선형 회전이
                      원리적으로 못 만드는 2차 스칼라 — SVD 가 못 보는 것이 아니라
                      SVD 출력의 선형 결합으로 못 만드는 것이다.
            arith_ev  Arith 카탈로그의 swap='불변' 178 + '교환' 쌍의 (a+b)/2
                      전부 짝 (실측 187열). 추론 비용 0 (이미 GBDT 먹이려고 뽑는 값).

패리티
    tie축은 짝이다. 그래서 릿지 계열의 입력은 **짝 재료만**(CUt_K + 짝로짓 5) 쓴다 —
    홀의 선형항은 반대칭을 깨고, 실측으로도 기여가 0 이다(probe_g [1] 홀만 R² = −0.001).
    GAT/MLP 는 대칭화 f(x⁺)+f(x⁻) 가 짝을 강제하므로 홀 블록도 받는다(현행 그대로).

행 계약
    전 칸 기둥 없음. 승부축(LIN 12열)은 tie축 선택과 독립이라 **한 번만** 적합하고 공유한다.
    기준 칸 = 저장된 `p0_g3` (0.96042). Δ 는 그 행벡터에 paired bootstrap.
    ! 이 하네스는 구 arith npz 를 읽는다 — 배포 계약(checkpoint-4000) 대비 −0.00010.
      절대값이 아니라 **칸 간 Δ** 로 읽어야 한다.

사용
    cd final
    python -u temp/tiegrid.py 2>&1 | tee temp/log_tiegrid.txt
    python -u temp/tiegrid.py --stage 1          # 1단계만
    python -u temp/tiegrid.py --ks 256,512 --lrs 3e-4
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

import tie_ladder as TL          # 검증된 원시함수 재사용 (rowloss/_fit/crossfit/build/…)

FINAL = TL.FINAL
KS = (64, 128, 256, 512)
LRS = (1e-4, 3e-4, 1e-3)
SEEDS = (1, 2, 3, 4)             # 42 는 1단계에서 이미 돈다
EXTRAS = ('norm12', 'arith_ev', 'both')
HEADS = ('GAT', 'RIDGE_TR', 'RIDGE_S2', 'MLP')
ROUNDS = 3                        # 라운드 상한. 자기종결(TL.RD_STOP)은 그대로
# λ 격자 1e1…1e5 (반데케이드 9칸). fit_heads 는 1e8 까지 15칸을 쓰지만 그건 1,025열짜리
# 기둥 기준이고, 여기 릿지는 K+5 열(≤517)이라 최적 λ 가 그만큼 아래다 — 기준선 런의 기둥
# 실측 선택이 1000·1000·1000·31.6 으로 전부 이 범위 안이다. 15칸이면 RIDGE_TR 한 팔이
# 20~36분으로 불어난다(46,729행 x 518열 x 3라운드).
LAMS_TR = [float(10 ** (e / 2)) for e in range(2, 11)]   # 1e1…1e5
REF_TAG = 'p0_g3'
REF_VAL = 0.96042


class MLPNet(nn.Module):
    """headgrid.py:97 의 MLP 와 같은 형태. 마지막 층 이름을 `.w` 로 맞춰 0 초기화 계약을 탄다."""

    def __init__(self, d, h):
        super().__init__()
        self.h1 = nn.Linear(d, h)
        self.w = nn.Linear(h, 1)

    def forward(self, x):
        return self.w(torch.nn.functional.gelu(self.h1(x)))


# ── 유닛 분해 (--stage 4) ────────────────────────────────────────────────────
# 왜: 옛 유닛 스윕(K=256,h=64)이 **값 가지가 tanh 인지**로 갈렸다 —
#       tanh·gauss +0.0020 ★ · tanh·sig +0.0018 ★   vs   gauss·sig +0.0006 · gelu·sig +0.0003
#     게이트 가지를 바꾸면 살아남고 값 가지를 바꾸면 죽는다. 그런데
#     "tanh 만으로 되는가 / 곱이 필요한가" 는 한 번도 안 갈랐다 (MLP 는 GELU 였다).
class TanhOnly(nn.Module):
    """w·tanh(W1x) — 게이트 제거. 포화 비선형만 남긴다."""

    def __init__(self, d, h):
        super().__init__()
        self.W1 = nn.Linear(d, h)
        self.w = nn.Linear(h, 1)

    def forward(self, x):
        return self.w(torch.tanh(self.W1(x)))


class Bilinear(nn.Module):
    """w·(W1x ⊙ W2x) — 포화 제거. 독립 사영 두 개의 곱만 남긴다.

    ! 옛 격자 주석의 '쌍선형 실측 0' 은 **고정 랜덤특징 + LIN** 이었다 (log_resfit3/4).
      여기는 W1·W2 가 학습된다 — 다른 모형이다.
    """

    def __init__(self, d, h):
        super().__init__()
        self.W1, self.W2 = nn.Linear(d, h), nn.Linear(d, h)
        self.w = nn.Linear(h, 1)

    def forward(self, x):
        return self.w(self.W1(x) * self.W2(x))


class TanhTanh(nn.Module):
    """w·tanh(W1x) ⊙ tanh(W2x) — 곱의 양쪽을 다 포화시킨다."""

    def __init__(self, d, h):
        super().__init__()
        self.W1, self.W2 = nn.Linear(d, h), nn.Linear(d, h)
        self.w = nn.Linear(h, 1)

    def forward(self, x):
        return self.w(torch.tanh(self.W1(x)) * torch.tanh(self.W2(x)))


class GLU(nn.Module):
    """w·(W1x ⊙ σ(W2x)) — 게이트는 남기고 포화를 뺀다 (Dauphin et al. 2017 의 GLU).

    BILIN 과의 차이가 **σ 하나** 다. GAT↔TANHLIN 과 함께 σ 의 효과를 두 경로에서
    독립으로 분리한다.
    """

    def __init__(self, d, h):
        super().__init__()
        self.W1, self.W2 = nn.Linear(d, h), nn.Linear(d, h)
        self.w = nn.Linear(h, 1)

    def forward(self, x):
        return self.w(self.W1(x) * torch.sigmoid(self.W2(x)))


class TanhLin(nn.Module):
    """w·(tanh(W1x) ⊙ W2x) — 포화는 남기고 게이트를 뺀다. GAT 에서 σ 만 뺀 팔."""

    def __init__(self, d, h):
        super().__init__()
        self.W1, self.W2 = nn.Linear(d, h), nn.Linear(d, h)
        self.w = nn.Linear(h, 1)

    def forward(self, x):
        return self.w(torch.tanh(self.W1(x)) * self.W2(x))


class SqOnly(nn.Module):
    """w·(W1x)² — 사영 **하나**의 제곱. 곱이 두 독립 방향이어야 하나를 본다.

    ! 함수족은 BILIN 과 같다 — xᵀMx 는 M 의 대칭부만 보고, Σw(W1x)² 는 모든 대칭행렬을
      친다. 그러니 이 팔이 재는 것은 표현력이 아니라 **매개화·최적화 경로** 다.
    ! 폭을 2배로 잡아 파라미터를 곱 유닛(2dh)과 맞춘다.
    """

    def __init__(self, d, h):
        super().__init__()
        self.W1 = nn.Linear(d, 2 * h)
        self.w = nn.Linear(2 * h, 1)

    def forward(self, x):
        return self.w(self.W1(x) ** 2)


class AbsOnly(nn.Module):
    """w·|W1x| — 곱이 아닌 **짝(even) 활성**. 죽은 MLP·TANH 는 둘 다 단조였다.

    조건이 '곱'인지 '짝'인지 가른다. 짝이면 되는 것이라면 사영 하나로 끝나므로
    이 자리에서 가장 싼 답이다. 폭 2배로 파라미터를 맞춘다.
    """

    def __init__(self, d, h):
        super().__init__()
        self.W1 = nn.Linear(d, 2 * h)
        self.w = nn.Linear(2 * h, 1)

    def forward(self, x):
        return self.w(torch.abs(self.W1(x)))


class TanhSq(nn.Module):
    """w·tanh(W1x)² — **유계** 짝 활성. SQ·ABS 를 포화시킨 판.

    대칭화 f(x⁺)+f(x⁻) 뒤 SQ 는 2(p²+q²), ABS 는 2·max(|p|,|q|) 로 둘 다 무계다.
    지금 최선 유닛이 유계(TANHTANH)라 '큰 |A−B| 를 포화시키는' 칸이 비어 있다.
    폭 2배로 파라미터를 곱 유닛과 맞춘다.
    """

    def __init__(self, d, h):
        super().__init__()
        self.W1 = nn.Linear(d, 2 * h)
        self.w = nn.Linear(2 * h, 1)

    def forward(self, x):
        return self.w(torch.tanh(self.W1(x)) ** 2)


NETS = {'GAT': None, 'MLP': MLPNet, 'TANH': TanhOnly,
        'BILIN': Bilinear, 'TANHTANH': TanhTanh,
        'GLU': GLU, 'TANHLIN': TanhLin, 'SQ': SqOnly, 'ABS': AbsOnly,
        'TANHSQ': TanhSq}

# 이질 부스팅 팔 — 이름 -> 라운드별 유닛. --mix 로 채운다.
MIXES = {}


# ── 유닛 뒤에 학습 가능한 어텐션 (--attn) ────────────────────────────────────
# 어떤 어텐션인가가 갈린다:
#   무의미  out = w(A·u),  A 는 h×h 학습 행렬
#           뒤가 선형 w 라 w(A u) = (Aᵀw)·u 로 접힌다 — 평범한 선형 판독기와 같은 함수족.
#   유의미  a = softmax_h(u) · h,  out = w(a ⊙ Vx)
#           softmax 가 h 축에서 **경쟁**을 만든다(Σa 고정). 현행 GAT 는 여기서 softmax 를
#           뺀 것이라, 이 팔이 Ilse et al.(2018) 게이티드 어텐션의 완성형이다.
#
# ! 스케일 공정성 — softmax 만 쓰면 a_k ~ 1/h 라 출력이 GAT 대비 ~30배 작아지고, 0 초기화된
#   w 가 lr 1e-4 · 수백 epoch 안에 그 손실을 못 메운다. 아키텍처가 아니라 스케일 때문에
#   지는 교란이 생긴다. 그래서 softmax 에 h 를 곱해 평균 가중을 1 로 맞춘다.
def _make_attn(score, two):
    """score(a, b) 를 어텐션 점수로 쓰는 헤드를 만든다. two=False 면 b 는 None."""

    class _Attn(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.W1 = nn.Linear(d, h)
            self.W2 = nn.Linear(d, h) if two else None
            self.V = nn.Linear(d, h)          # 값 사영
            self.w = nn.Linear(h, 1)          # 0 초기화 대상 (부스팅 계약)

        def forward(self, x):
            s = score(self.W1(x), self.W2(x) if two else None)
            a = torch.softmax(s, dim=-1) * s.shape[-1]
            return self.w(a * self.V(x))

    return _Attn


_G = torch.nn.functional.gelu
NETS.update({
    'MLP_A':      _make_attn(lambda a, b: _G(a), False),
    'TANH_A':     _make_attn(lambda a, b: torch.tanh(a), False),
    'BILIN_A':    _make_attn(lambda a, b: a * b, True),
    'GAT_A':      _make_attn(lambda a, b: torch.tanh(a) * torch.sigmoid(b), True),
    'TANHTANH_A': _make_attn(lambda a, b: torch.tanh(a) * torch.tanh(b), True),
})


# ══════════════════════════════════════════════ 릿지 (fit_heads.fit_beta 등가)
def ridge_beta(s0, P, y01, tag, free=0, lams=LAMS_TR, verbose=False):
    """짝 재료 P 에 릿지. [s0(무벌점, 선택)] + P[:, :free](무벌점) + 나머지(λ).

    λ 는 층화 3-fold 내부 CV, 큰 값부터 웜스타트. 정규화는 전 행 통계(라벨 미접촉)로
    한 번만 — fit_heads.fit_beta 와 같은 계약이라 호출부가 같은 통계를 stage2 에 쓴다.
    반환 (β — 정규화 좌표계, dict(mu, sc, lam, z_tr)).
    """
    n, p = P.shape
    mu = P.mean(0)
    sc = np.clip(P.std(0), 1e-9, None)
    Pz = (P - mu) / sc
    off = 0 if s0 is None else 1
    if off:
        s0z = s0 / np.clip(np.sqrt((s0 ** 2).mean()), 1e-9, None)
        X = np.hstack([s0z[:, None], Pz])
    else:
        s0z = None
        X = Pz
    inner = np.zeros(n, int)
    for j, (_, iv) in enumerate(StratifiedKFold(3, shuffle=True, random_state=TL.SEED)
                                .split(np.zeros((n, 1)), y01)):
        inner[iv] = j

    def penvec(lam):
        pen = np.full(X.shape[1], float(lam))
        pen[:off + int(free)] = 0.0
        return pen

    score, warm = {}, {}
    for lam in sorted(lams, reverse=True):
        tot = 0.0
        for j in range(3):
            tr, te = inner != j, inner == j
            v, _ = TL._fit(X[tr], y01[tr], penvec(lam), True, warm.get(j))
            warm[j] = v
            z = X[te] @ v[:X.shape[1]] + v[X.shape[1]]
            tot += float((np.logaddexp(0, z) - y01[te] * z).sum())
        score[lam] = tot
    lam = min(score, key=score.get)
    v, ok = TL._fit(X, y01, penvec(lam), True, warm.get(0))
    z1 = X @ v[:X.shape[1]] + v[X.shape[1]]
    if verbose:
        edge = ' ** 격자 끝' if lam in (min(lams), max(lams)) else ''
        print(f'      [{tag}] λ {lam:g}{edge}' + ('' if ok else ' ** 미수렴'))
    return v[off:off + p], dict(mu=mu, sc=sc, lam=float(lam), z_tr=z1)


def ridge_chain(Ptr, Ps2, y01m, tag, free, rounds=ROUNDS):
    """릿지 부스팅 — GAT 체인과 같은 계약. 증분 열만 내보낸다."""
    z, cols, lams_used = None, [], []
    for rd in range(1, rounds + 1):
        beta, st = ridge_beta(z, Ptr, y01m, f'{tag}·r{rd}', free=free, verbose=True)
        cols.append(((Ps2 - st['mu']) / st['sc']) @ beta)
        lams_used.append(st['lam'])
        z = st['z_tr']
    return np.column_stack(cols), lams_used


def net_chain(CU_in, CV_in, CU_s2, CV_s2, y01m, tag, net, lr, seed, rounds=ROUNDS,
              stop=True, nets=None, unames=None, ens=False, seed_step=100):
    """GAT/MLP 부스팅 — TL.gat_chain 과 같되 net·lr·seed·정지규칙을 명시로 받는다.

    stop=False 면 자기종결(ES 이득 < RD_STOP)을 끄고 rounds 만큼 다 돈다.
    ES 가 전이와 무관하다는 것이 실측됐으므로(아키텍처 간 스피어만 ρ = −0.20),
    '몇 라운드가 맞는가' 를 ES 대신 홀드아웃 접두사 사다리로 정하기 위한 스위치다.

    nets  라운드별 망 팩토리 목록. None 이면 net 을 rounds 번 쓴다(기존 경로와 비트 동일).
          라운드마다 함수족을 바꾸는 **이질 부스팅** 을 위한 것 — r2 가 r1 의 잔차를
          r1 과 다른 표현력으로 긁으면 같은 유닛을 반복할 때보다 더 볼 수 있는가.
    ens   True 면 앵커를 끊는다(z=None 고정) — 잔차가 아니라 **독립 학습** 이다.
          시드만 seed_step 씩 달라진 라운드1 헤드 R 개를 쌓는다. 부스팅 3열과
          열 수·파라미터 수가 같은 **교체 대조** 이고, 라운드 이득이 잔차적합에서
          오는지 시드 평균화에서 오는지를 가른다. 정지규칙은 뜻이 없어 강제로 끈다.
    """
    if ens:
        stop = False
    z, cols, prev, eps = None, [], None, []
    for rd in range(1, rounds + 1):
        nd = net if nets is None else nets[rd - 1]
        sd = seed + seed_step * (rd - 1) if ens else seed
        r_s2, aux = TL.fit_gat(CU_in, CV_in, y01m, None if ens else z, True, sd,
                               CU_s2, CV_s2, net=nd, lr=lr)
        cols.append(r_s2)
        gain = None if prev is None else prev - aux['es_ll']
        eps.append(aux['es_epoch'])
        mk = (f' | {unames[rd - 1]}' if unames else '') + (f' | 독립 seed {sd}' if ens else '')
        print(f'      [{tag}] r{rd}: es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]}{mk}'
              + (f' gain {gain:+.5f}' if gain is not None else ''))
        prev, z = aux['es_ll'], aux['z_tr']
        if stop and gain is not None and gain < TL.RD_STOP:
            break
    return np.column_stack(cols), eps


# ══════════════════════════════════════════════ 추가 입력 블록 (전부 짝)
def load_norm12(ids):
    """원 stage2 풀링에서 소스별 ||A−B|| · ||A+B|| · cos(A,B). 전부 swap 불변(짝).

    SVD 뒤 재구성이 아니라 **원 좌표계**에서 잰다 — 정규화(열별 z-score/RMS)는 1,024개
    좌표를 제각각 다른 상수로 나누므로 그 공간의 각도는 원 임베딩 기하가 아니다.
    stage2 풀링만 읽으므로 0.53GB(디스크) 로 끝난다. TR 2.3GB 는 건드리지 않는다.
    """
    M = TL.MODELS

    def get(rel):
        z = np.load(M / rel)
        assert (np.concatenate([z['valid_ids'], z['hold_ids']]) == ids).all(), f'{rel} id'
        return z

    cat = lambda z, k: np.vstack([z[f'valid_{k}'], z[f'hold_{k}']]).astype(np.float64)
    za = get('pooled/abe2__checkpoint-3000.npz')
    zt = get('pooled/tb2__checkpoint-4000.npz')
    zq = get(f'pooled/abe1__checkpoint-{TL.ABE1_STEP}.npz')
    z1 = get(f'pooled/tb1__checkpoint-{TL.TB1_STEP}.npz')
    AB = {
        'abe2': (cat(za, 'ea'), cat(za, 'eb')),
        'tb2': ((cat(zt, 'a_f') + cat(zt, 'b_s')) / 2, (cat(zt, 'b_f') + cat(zt, 'a_s')) / 2),
        'abe1': (cat(zq, 'ea'), cat(zq, 'eb')),
        'tb1': (cat(z1, 'x_f'), cat(z1, 'x_s')),      # 브랜치 없음 — "순서 바꿔도 표현 유지되나"
    }
    cols, names = [], []
    for s, (A, B) in AB.items():
        nA = np.sqrt((A ** 2).sum(1)); nB = np.sqrt((B ** 2).sum(1))
        cols += [np.sqrt(((A - B) ** 2).sum(1)),
                 np.sqrt(((A + B) ** 2).sum(1)),
                 (A * B).sum(1) / np.clip(nA * nB, 1e-9, None)]
        names += [f'{s}·dist', f'{s}·sum', f'{s}·cos']
    X = np.column_stack(cols)
    assert X.shape == (len(ids), 12) and np.isfinite(X).all(), f'norm12 {X.shape}'
    return X, names


def load_arith_even(ids):
    """Arith 카탈로그에서 **짝** 피쳐만. 불변 178 + 교환쌍의 (a+b)/2 (실측 187열).

    ! parquet 에는 label·is_tie 가 들어 있다. 카탈로그를 화이트리스트로 써서 배제하고
      assert 로 재확인한다. '부호반전'(58)은 홀이라 tie축에 넣으면 안 된다. '역수'(1)도 뺀다.
    """
    df = pd.read_parquet(TL.ARITH / 'features_57k.parquet')
    idx = pd.Index(df['id']).get_indexer(ids)
    assert (idx >= 0).all(), 'stage2 id 가 parquet 에 없다'
    cat = pd.read_csv(TL.DATA_DIR / 'feature_catalog.csv')
    cat = cat[cat['사용'] == 1]
    sw = {str(r['feature']): str(r['swap']) for _, r in cat.iterrows()}
    inv = [f for f, s in sw.items() if s == '불변' and f in df.columns]
    pairs = set()
    for f, s in sw.items():
        if s.startswith('교환:'):
            g = s.split(':', 1)[1]
            if f in df.columns and g in df.columns:
                pairs.add(tuple(sorted((f, g))))
    banned = {'label', 'is_tie'}
    assert not (banned & set(inv)), '화이트리스트에 라벨 열'
    sub = df.iloc[idx]
    cols = [sub[f].to_numpy(np.float64) for f in inv]
    names = list(inv)
    for a, b in sorted(pairs):
        assert not ({a, b} & banned)
        cols.append((sub[a].to_numpy(np.float64) + sub[b].to_numpy(np.float64)) / 2)
        names.append(f'sym({a},{b})')
    X = np.nan_to_num(np.column_stack(cols), nan=0.0, posinf=0.0, neginf=0.0)
    print(f'  arith 짝 블록 {X.shape}  (불변 {len(inv)} + 교환쌍 {len(pairs)})')
    return X, names


def build_extras(U, want):
    out = {}
    if want & {'norm12', 'both'}:
        out['norm12'] = load_norm12(U['ids'])
    if want & {'arith_ev', 'both'}:
        out['arith_ev'] = load_arith_even(U['ids'])
    if 'both' in want:
        a, an = out['norm12']
        b, bn = out['arith_ev']
        out['both'] = (np.hstack([a, b]), an + bn)
    return out


# ══════════════════════════════════════════════ 칸 평가
def _logit_residual(A):
    """TR 에서 짝 로짓 5열만으로 적합한 뒤의 잔차. stage2 미접촉."""
    y, X = A['y01m'], A['Lu_tr']
    v, ok = TL._fit(X, y, np.zeros(X.shape[1]), True)
    r = y - expit(X @ v[:X.shape[1]] + v[X.shape[1]])
    return r - r.mean(), ok


def _raw_even(ids_s2):
    """raw **짝** 풀링 — TR 46,729 · stage2 10,748, EVK 순서로 hstack (n, 4096).

    메모리: float32 로 TR 765MB + stage2 176MB. 홀은 안 읽는다 (아래 pls_assets 주석 참고).
    """
    M = TL.MODELS
    spec = [('abe2__checkpoint-3000', ('ea', 'eb')),
            ('tb2__checkpoint-4000', ('a_f', 'b_s', 'b_f', 'a_s')),
            (f'abe1__checkpoint-{TL.ABE1_STEP}', ('ea', 'eb')),
            (f'tb1__checkpoint-{TL.TB1_STEP}', ('x_f', 'x_s'))]
    Tr, S2 = [], []
    for tag, ks in spec:
        zt = np.load(M / f'pooled/{tag}__train.npz')
        zs = np.load(M / f'pooled/{tag}.npz')
        assert (np.concatenate([zs['valid_ids'], zs['hold_ids']]) == ids_s2).all(), f'{tag} id'
        g = lambda z, pre, k: z[f'{pre}_{k}'].astype(np.float32)
        c = lambda z, k: np.vstack([z[f'valid_{k}'], z[f'hold_{k}']]).astype(np.float32)
        Tr.append(sum(g(zt, 'train', k) for k in ks) / len(ks))    # u = 뷰 평균 (짝)
        S2.append(sum(c(zs, k) for k in ks) / len(ks))
    return np.hstack(Tr), np.hstack(S2)


def pls_assets(A, U, K, verbose=True):
    """raw 짝 4,096 에서 **잔차 기준으로** K 방향을 뽑는다 (직교 PLS). TR 전용.

    현행 SVD 는 분산 상위 K 다. 여기는 같은 4,096 공간에서 "로짓이 못 맞춘 것" 과
    공분산이 큰 방향을 순차로 뽑고, 뽑을 때마다 잔차를 깎고 방향을 직교화한다.
    정규화는 현행과 **똑같은 상수**(캐시의 NU_mu/NU_sc/U_mu)를 써서 기저만 갈아끼운다.

    왜 짝만 — tie축은 짝이고, 홀 성분은 잔차 r(짝)과 **선형 공분산이 0** 이라 PLS 기준이
    무의미하다. 홀은 현행 분산순 SVD 를 그대로 쓴다.
    ! raw 는 TR 만 PLS 에 쓴다. stage2 는 뽑힌 방향에 **사영만** 당한다 (위생 유지).
    ! 배포 비용 0 — 결과가 4096xK 행렬 하나라 현행 W 와 형태가 같다.
    """
    import gc
    Z = TL.load_svd_cache()
    t0 = time.time()
    Xt, Xs = _raw_even(U['ids'])
    print(f'  raw 짝 풀링 적재 {Xt.shape} / {Xs.shape}  [{time.time() - t0:.0f}s]')
    mu, sc = Z['NU_mu'].astype(np.float32), Z['NU_sc'].astype(np.float32)
    cen = Z['U_mu'].astype(np.float32)
    Xt = (Xt - mu) / sc - cen
    Xs = (Xs - mu) / sc - cen
    r, ok = _logit_residual(A)
    rr = r.astype(np.float64).copy()
    p = Xt.shape[1]
    W = np.zeros((p, K))
    t0 = time.time()
    kk = 0
    for k in range(K):
        w = (Xt.T @ rr.astype(np.float32)).astype(np.float64)
        if k:
            w -= W[:, :k] @ (W[:, :k].T @ w)          # 그람-슈미트
        nw = float(np.linalg.norm(w))
        if nw < 1e-9:
            break
        w /= nw
        W[:, k] = w
        t = Xt @ w.astype(np.float32)
        tt = float(t @ t)
        rr -= (float(t @ rr.astype(np.float32)) / max(tt, 1e-12)) * t.astype(np.float64)
        kk = k + 1
    W = W[:, :kk]
    Ct, Cs = Xt @ W.astype(np.float32), Xs @ W.astype(np.float32)
    wsc = np.clip(np.sqrt((Ct.astype(np.float64) ** 2).mean(0)), 1e-9, None)
    A['PLS_t'] = (Ct / wsc).astype(np.float32)
    A['PLS_s'] = (Cs / wsc).astype(np.float32)
    A['PLS_W'] = W
    if verbose:
        print(f'  PLS {kk}성분 추출 [{time.time() - t0:.0f}s] (로짓적합 '
              f'{"수렴" if ok else "미수렴"}) | 잔차 설명 잔량 '
              f'{float((rr ** 2).sum() / (r ** 2).sum()):.4f}')
    del Xt, Xs, Ct, Cs
    gc.collect()
    return A


def residual_order(A, verbose=True):
    """성분을 **잔차 기준**으로 재정렬한다. TR 46,729행만 쓴다 — stage2 는 안 본다.

        (1) 짝 로짓 5열로 TR 무벌점 적합  ->  p_hat
        (2) 잔차 r = y01 - p_hat            ("로짓이 설명 못 한 것")
        (3) 짝 성분  |cov(CUt_k , r)|       내림차순
            홀 성분  |cov(CVt_k^2 , r)|     <- 홀은 그대로면 짝인 r 과 공분산이 0 이다.
                                               제곱해야 짝이 되어 tie 잔차와 겹칠 수 있다.

    왜: 현행 SVD 는 **분산** 순이고 라벨을 한 번도 안 본다. 그런데 K 사다리가
        256(분산 95.0%) 0.96185 -> 512(97.3%) 0.96037 로, **분산 2.3% 를 더 넣어
        0.00148 을 벌었다.** 라벨 관련 신호가 저분산 방향에 흩어져 있다는 뜻이다.

    ! 선택 용량은 성분당 스칼라 하나(512개 순위)뿐이다. 헤드 264k 파라미터에 비하면
      미미하고, 결합기 lambda 의 방어(무한대면 로짓 기준선으로 수렴)도 그대로 걸린다.
    """
    y = A['y01m']
    X = A['Lu_tr']
    v, ok = TL._fit(X, y, np.zeros(X.shape[1]), True)
    r = y - expit(X @ v[:X.shape[1]] + v[X.shape[1]])
    rc = r - r.mean()
    cu = np.abs((A['CUt'].astype(np.float64) * rc[:, None]).mean(0))
    cv = np.abs(((A['CVt'].astype(np.float64) ** 2) * rc[:, None]).mean(0))
    ou, ov = np.argsort(-cu), np.argsort(-cv)
    if verbose:
        print(f'  잔차 순위 (TR 전용 · 로짓적합 {"수렴" if ok else "미수렴"}) '
              f'| 짝 상위10 성분 {ou[:10]}')
        for k in (64, 128, 256):
            print(f'    분산순 상위{k} 과 잔차순 상위{k} 의 교집합: '
                  f'{len(set(ou[:k]) & set(range(k))):3d}/{k} (짝) · '
                  f'{len(set(ov[:k]) & set(range(k))):3d}/{k} (홀)')
    return ou, ov


def chain_basis(A, Kd, lr, seed, tag, rounds=ROUNDS):
    """**기저의 부스팅** — 라운드마다 성분을 새로 고른다.

        r1  분산순 상위 K
        rk  직전 누적 점수의 잔차와 공분산이 큰, **아직 안 쓴** 성분 K개
            (짝은 |cov(c, r)|, 홀은 |cov(c^2, r)| — 홀은 그대로면 짝인 r 과 0 이다)

    현행과의 차이: 현행은 3라운드가 **같은 512** 를 본다. 여기는 라운드마다 다른 K 를 본다.
    잔차가 '훈련된 헤드가 못 맞춘 것' 이라 로짓 잔차 정적 기준(sel='resid')보다 적응적이다.

    ! 캐시가 512 성분뿐이라 rounds x K <= 512 여야 한다. 그 밖의 방향을 뽑으려면
      raw 풀링(짝 1.53GB + 홀 1.53GB) 적재가 필요하다 — 이 함수 범위 밖이다.
    ! TR 만 쓴다. stage2 는 고른 열을 적용만 받는다.
    """
    assert rounds * Kd <= A['CUt'].shape[1], \
        f'rounds x K = {rounds * Kd} > 성분 {A["CUt"].shape[1]} — 캐시 밖은 못 뽑는다'
    y = A['y01m']
    CUt64, CVt64 = A['CUt'].astype(np.float64), A['CVt'].astype(np.float64)
    used_u, used_v = np.zeros(CUt64.shape[1], bool), np.zeros(CVt64.shape[1], bool)
    z, cols, prev, picks = None, [], None, []
    for rd in range(1, rounds + 1):
        if rd == 1:
            iu = iv = np.arange(Kd)
        else:
            rc = (y - expit(z))
            rc = rc - rc.mean()
            cu = np.abs((CUt64 * rc[:, None]).mean(0)); cu[used_u] = -1.0
            cv = np.abs(((CVt64 ** 2) * rc[:, None]).mean(0)); cv[used_v] = -1.0
            iu, iv = np.sort(np.argsort(-cu)[:Kd]), np.sort(np.argsort(-cv)[:Kd])
        used_u[iu] = True; used_v[iv] = True
        picks.append((int(iu[0]), int(iu[-1])))
        cu_t = np.ascontiguousarray(np.hstack([A['CUt'][:, iu], A['Lu_tr']]), np.float32)
        cv_t = np.ascontiguousarray(np.hstack([A['CVt'][:, iv], A['Lv_tr']]), np.float32)
        cu_s = np.ascontiguousarray(np.hstack([A['CUs'][:, iu], A['Lu_s2']]), np.float32)
        cv_s = np.ascontiguousarray(np.hstack([A['CVs'][:, iv], A['Lv_s2']]), np.float32)
        r_s2, aux = TL.fit_gat(cu_t, cv_t, y, z, True, seed, cu_s, cv_s, net=None, lr=lr)
        cols.append(r_s2)
        gain = None if prev is None else prev - aux['es_ll']
        print(f'      [{tag}] r{rd}: es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]}'
              f' | 성분 {iu[0]}..{iu[-1]} ({"분산순" if rd == 1 else "잔차순"})'
              + (f' gain {gain:+.5f}' if gain is not None else ''))
        prev, z = aux['es_ll'], aux['z_tr']
    return np.column_stack(cols), picks


def get_chain(A, head, Kd, lr, seed, tag, n_logit, chains, rounds=ROUNDS, stop=True,
              sel='var'):
    """(head,K,lr,seed) 별 3열 체인. 캐시한다 — extra 축은 키에 없으므로 **같은 열**이
    재사용되고, 그래서 extra 비교에서는 시드 잡음이 정확히 상쇄된다."""
    key = (head, Kd, lr, seed, rounds, stop, sel)
    if key not in chains:
        t0 = time.time()
        # 열 선택 — 잔차순이면 상위 K 를 고른 뒤 **오름차순으로 되돌린다**.
        # 순서까지 바꾸면 고정 초기화와의 짝이 달라져 K=512 에서 분산순과 비트 동일하지
        # 않게 되고, 선택 효과와 난수 재배치가 섞인다.
        iu = np.sort(A['ord_u'][:Kd]) if sel == 'resid' else np.arange(Kd)
        iv = np.sort(A['ord_v'][:Kd]) if sel == 'resid' else np.arange(Kd)
        if sel == 'pls':
            cu_t = np.ascontiguousarray(np.hstack([A['PLS_t'][:, :Kd], A['Lu_tr']]), np.float32)
            cv_t = np.ascontiguousarray(np.hstack([A['CVt'][:, :Kd], A['Lv_tr']]), np.float32)
            cu_s = np.ascontiguousarray(np.hstack([A['PLS_s'][:, :Kd], A['Lu_s2']]), np.float32)
            cv_s = np.ascontiguousarray(np.hstack([A['CVs'][:, :Kd], A['Lv_s2']]), np.float32)
            G, aux = net_chain(cu_t, cv_t, cu_s, cv_s, A['y01m'], tag,
                               NETS[head], lr, seed, rounds=rounds, stop=stop)
            chains[key] = (G, aux, time.time() - t0)
            return chains[key]
        if sel == 'chain':
            G, aux = chain_basis(A, Kd, lr, seed, tag, rounds=rounds)
            chains[key] = (G, aux, time.time() - t0)
            return chains[key]
        if head == 'RIDGE_TR':
            Ptr = np.hstack([A['Lu_tr'], A['CUt'][:, iu]])        # 짝 재료만
            Ps2 = np.hstack([A['Lu_s2'], A['CUs'][:, iu]])
            G, aux = ridge_chain(Ptr, Ps2, A['y01m'], tag, free=n_logit, rounds=rounds)
        else:
            cu_t = np.ascontiguousarray(np.hstack([A['CUt'][:, iu], A['Lu_tr']]), np.float32)
            cv_t = np.ascontiguousarray(np.hstack([A['CVt'][:, iv], A['Lv_tr']]), np.float32)
            cu_s = np.ascontiguousarray(np.hstack([A['CUs'][:, iu], A['Lu_s2']]), np.float32)
            cv_s = np.ascontiguousarray(np.hstack([A['CVs'][:, iv], A['Lv_s2']]), np.float32)
            if head in MIXES:
                un = MIXES[head]
                assert len(un) == rounds, f'{head} 는 {len(un)}라운드인데 rounds={rounds}'
                G, aux = net_chain(cu_t, cv_t, cu_s, cv_s, A['y01m'], tag, None, lr, seed,
                                   rounds=rounds, stop=stop,
                                   nets=[NETS[u] for u in un], unames=un)
            elif head == 'ENS':
                G, aux = net_chain(cu_t, cv_t, cu_s, cv_s, A['y01m'], tag, NETS['GAT'],
                                   lr, seed, rounds=rounds, stop=stop, ens=True)
            else:
                G, aux = net_chain(cu_t, cv_t, cu_s, cv_s, A['y01m'], tag,
                                   NETS[head], lr, seed, rounds=rounds, stop=stop)
        chains[key] = (G, aux, time.time() - t0)
    return chains[key]


def compose_all(U, llt, llw, rt, rw):
    """clean 7,748행 전부의 행별 3-class 손실 — holdout 5,748 의 상위집합.

    va2 2,000 은 네 인코더 모두에게 **학습 노출이 없고**(ABE1 은 선택 노출만) 결합기
    점수도 5-fold OOF 라 평가에 넣어도 된다. 평가 행이 35% 늘어 CI 가 1.16배 좁아진다.
    옛 headgrid 도 '전체 OOF' 와 'holdout' 을 따로 보고했다 — 같은 이유다.
    """
    nz_c = U['nz'][rt]
    row = np.asarray(llt, np.float64).copy()
    assert int(nz_c.sum()) == len(llw) == int(rw.sum()), '승부축 정렬'
    row[nz_c] += llw
    return row


def eval_cell(U, A, win, spec, EX, chains):
    """한 칸 = (head, K, lr, seed, extra). 반환 (holdout 행손실, clean 행손실, 진단)."""
    head, Kd, lr, seed, extra = spec['head'], spec['K'], spec['lr'], spec['seed'], spec['extra']
    parts, names = [U['tie_base']], list(TL.TIE_NAMES)
    n_logit = U['tie_base'].shape[1]
    diag = {}

    if head == 'RIDGE_S2':
        iu = np.sort(A['ord_u'][:Kd]) if spec.get('sel') == 'resid' else np.arange(Kd)
        parts.append(A['CUs'][:, iu])                       # 결합기가 직접 받는다 (TR 미사용)
        names += [f'c{i}' for i in range(Kd)]
    elif head == 'GAT+RIDGE':
        # 상보성 — 릿지가 GAT 밖의 것을 보나. 6열이 3열을 이기면 상보, 아니면 GAT 가 포함.
        G1, a1, s1 = get_chain(A, 'GAT', Kd, lr, seed, f'{spec["tag"]}·gat', n_logit, chains)
        G2, a2, s2 = get_chain(A, 'RIDGE_TR', Kd, TL.LR, TL.SEED,
                               f'{spec["tag"]}·ridge', n_logit, chains)
        G = np.hstack([G1, G2])
        diag.update(n_round=int(G.shape[1]), aux={'gat': a1, 'ridge': a2},
                    chain_sec=round(s1 + s2, 1))
        parts.append(G)
        names += [f'g{i + 1}' for i in range(G1.shape[1])] \
            + [f'r{i + 1}' for i in range(G2.shape[1])]
    else:
        G, aux, secs = get_chain(A, head, Kd, lr, seed, spec['tag'], n_logit, chains,
                                 rounds=spec.get('rounds', ROUNDS),
                                 stop=spec.get('stop', True),
                                 sel=spec.get('sel', 'var'))
        ng = spec.get('n_gat')
        if ng:                                   # 접두사 절단 — 체인이 접두사 일관이라 정확하다
            G = G[:, :ng]
        diag.update(n_round=int(G.shape[1]), aux=aux, chain_sec=round(secs, 1))
        parts.append(G)
        names += [f'g{i + 1}' for i in range(G.shape[1])]

    if extra != 'none':
        Xe, ne = EX[extra]
        parts.append(Xe)
        names += ne

    X = np.hstack(parts)
    rows = U['cu']
    y01 = (U['y'] == 2).astype(np.float64)
    z, dep = TL.combiner(X, y01, U['fold'], rows, bias=True, n_logit=n_logit)
    ll = TL.rowloss(z[rows], y01[rows])
    row, lt, _ = TL.compose_ll(U, ll, win['ll'], rows, win['rows'])
    row_all = compose_all(U, ll, win['ll'], rows, win['rows'])
    diag.update(tot=float(row.mean()), tot_all=float(row_all.mean()), tie=lt,
                n_col=X.shape[1], lam=dep['lam'], n_bad=dep['n_bad'], n_ext=dep['n_ext'])
    # 표현 열 계수만 (로짓·extra 제외) — 라운드 구조 확인용
    v = np.asarray(dep['v'], np.float64)
    diag['w_g'] = [float(v[n_logit + i]) for i in range(len(names) - n_logit)][:8]
    return row, row_all, diag


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bundle', default='temp/heads_repro')
    ap.add_argument('--ref', default='temp/tie_ladder_es7k.rowloss.npz')
    ap.add_argument('--ks', default=','.join(str(k) for k in KS))
    ap.add_argument('--lrs', default=','.join(f'{x:g}' for x in LRS))
    ap.add_argument('--heads', default=','.join(HEADS))
    ap.add_argument('--stage', default='1,2,3,4')
    ap.add_argument('--fix-best', default='',
                    help="1단계를 생략하고 승자를 고정 — 'head,K,lr,seed' (예 GAT,512,1e-4,42)")
    ap.add_argument('--sweep', default='',
                    help="유닛 목록을 시드 전수로 반복 측정 — 'MLP,TANH,BILIN,GAT,TANHTANH'")
    ap.add_argument('--sweep-seeds', default='42,1,2,3,4')
    ap.add_argument('--mix', default='',
                    help="이질 부스팅 팔 정의 — '이름=유닛1/유닛2/유닛3' 쉼표 목록. "
                         "정의한 이름을 --sweep 에 넣어야 실제로 돈다. "
                         "'ENS' 는 정의 없이 --sweep 에 바로 넣는다(앵커 없는 독립 3헤드)")
    ap.add_argument('--attn', default='',
                    help="유닛 뒤에 어텐션을 붙인 팔 — 'MLP,TANH,BILIN,GAT,TANHTANH' (단일 시드)")
    ap.add_argument('--attn-seed', type=int, default=42)
    ap.add_argument('--rounds', type=int, default=ROUNDS,
                    help='부스팅 라운드 상한. --prefix 와 같이 쓰면 자기종결을 끈다')
    ap.add_argument('--sels', default='var',
                    help="성분 선택 기준 — 'var'(분산순, 현행) / 'resid'(로짓잔차 재정렬) / "
                         "'chain'(라운드마다 기저 재선택) 쉼표 목록")
    ap.add_argument('--pls-k', type=int, default=512,
                    help='PLS 로 뽑을 성분 수 (K 이상이면 앞 K 만 쓴다)')
    ap.add_argument('--prefix', action='store_true',
                    help='--sweep 을 접두사 사다리로 — 체인 1회 학습 후 g1..gR 을 다 잰다')
    ap.add_argument('--out', default='temp/tiegrid')
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(',') if x]
    lrs = [float(x) for x in args.lrs.split(',') if x]
    heads = [h for h in args.heads.split(',') if h]
    for spec in (x for x in args.mix.split(',') if x):
        nm, _, us = spec.partition('=')
        un = [u for u in us.split('/') if u]
        assert nm and un, f'--mix 형식은 이름=유닛/유닛/유닛 — {spec}'
        assert set(un) <= set(NETS), f'{spec} 에 모르는 유닛 {set(un) - set(NETS)}'
        assert nm not in NETS and nm != 'ENS', f'--mix 이름 {nm} 이 유닛 이름과 겹친다'
        MIXES[nm] = un
    for u in (x for x in args.sweep.split(',') if x):
        assert u in NETS or u in MIXES or u == 'ENS', \
            f'--sweep 의 {u} 는 유닛도 --mix 정의도 ENS 도 아니다'
    stages = {int(s) for s in args.stage.split(',') if s}
    assert max(ks) <= 512, 'K>512 는 512 캐시에서 못 자른다 (hgv2 가 이미 기각)'

    print('[적재]')
    U = TL.build(FINAL / args.bundle)
    win = TL.fit_win(U)
    A = TL.gat_assets(U)
    A['ord_u'], A['ord_v'] = residual_order(A)
    rl = np.load(FINAL / args.ref)
    assert REF_TAG in rl.files, f'{args.ref} 에 {REF_TAG} 없음'
    ref_row = rl[REF_TAG]
    print(f'  기준 {REF_TAG} = {ref_row.mean():.5f} (기록 {REF_VAL:.5f}, '
          f'Δ {ref_row.mean() - REF_VAL:+.5f}) | 기둥 없음 · 승부축 공유')

    EX = build_extras(U, set(EXTRAS)) if 3 in stages else {}
    res, rows, rows_all, chains = {}, {}, {}, {}

    def run(tag, head, Kd, lr, seed, extra, **kw):
        spec = dict(tag=tag, head=head, K=Kd, lr=lr, seed=seed, extra=extra, **kw)
        print(f'  [{tag}]')
        row, row_all, d = eval_cell(U, A, win, spec, EX, chains)
        m, lo, hi = TL.paired_ci(row, ref_row)
        d.update(spec={k: v for k, v in spec.items() if k != 'tag'},
                 d_mean=m, d_lo=lo, d_hi=hi)
        res[tag], rows[tag], rows_all[tag] = d, row, row_all
        print(f'    -> holdout {d["tot"]:.5f} · clean {d["tot_all"]:.5f}  '
              f'Δ {TL.ci_str(m, lo, hi)}  (열 {d["n_col"]} · λ {d["lam"]:g})')
        return d

    if 1 in stages:
        print(f'\n{"="*78}\n[1단계] K x lr x head — 기둥 없음, seed 42, extra 없음\n{"="*78}')
        for Kd in ks:
            for h in heads:
                if h == 'MLP':
                    continue                       # MLP 는 2단계에서 승자 구성 1칸만
                for lr in (lrs if h in ('GAT',) else [None]):
                    tag = f'{h}-K{Kd}' + (f'-lr{lr:g}' if lr else '')
                    run(tag, h, Kd, lr or TL.LR, TL.SEED, 'none')

    best, b = None, None
    if args.fix_best:
        h_, k_, lr_, sd_ = args.fix_best.split(',')
        b = dict(head=h_, K=int(k_), lr=float(lr_), seed=int(sd_), extra='none')
        best = f'{h_}-K{int(k_)}-lr{float(lr_):g}'
        print(f'\n[승자 고정] {best}  (--fix-best — 1단계 생략. Δ 기준이 같아 이전 런과 비교 가능)')
    elif res:
        best = min(res, key=lambda t: res[t]['tot'])
        b = res[best]['spec']
        print(f'\n[1단계 승자] {best}  {res[best]["tot"]:.5f}')

    if 2 in stages and best:
        print(f'\n{"="*78}\n[2단계] 승자 구성에 MLP 1칸 + 시드 4개\n{"="*78}')
        if b['head'] in ('GAT', 'MLP'):
            run(f'MLP-K{b["K"]}-lr{b["lr"]:g}', 'MLP', b['K'], b['lr'], TL.SEED, 'none')
            for sd in SEEDS:
                run(f'{best}-s{sd}', b['head'], b['K'], b['lr'], sd, 'none')
        else:
            print('  승자가 릿지 계열 — 시드/MLP 축이 의미 없다 (결정론). 생략')

    if 3 in stages and best:
        print(f'\n{"="*78}\n[3단계] 승자 구성에 추가 입력 블록\n{"="*78}')
        for e in EXTRAS:
            run(f'{best}+{e}', b['head'], b['K'], b['lr'], b['seed'], e)

    if 4 in stages and best:
        print(f'\n{"="*78}\n[4단계] g 의 비선형이 무엇으로 만들어지나 — 유닛 분해 + 상보성\n'
              f'{"="*78}')
        print('  값 가지 tanh 가 핵심이라는 옛 단서(tanh·* +0.0018~0.0020 vs '
              'gelu/gauss·sig +0.0003~0.0006)를\n'
              '  "게이트가 필요한가 / 곱이 필요한가" 로 가른다.')
        for h in ('TANH', 'BILIN', 'TANHTANH'):
            run(f'{h}-K{b["K"]}-lr{b["lr"]:g}', h, b['K'], b['lr'], b['seed'], 'none')
        run(f'GAT+RIDGE-K{b["K"]}', 'GAT+RIDGE', b['K'], b['lr'], b['seed'], 'none')

    if args.sweep and b:
        units = [u for u in args.sweep.split(',') if u]
        sds = [int(x) for x in args.sweep_seeds.split(',') if x]
        print(f'\n{"="*78}\n[시드 전수] {len(units)}유닛 x {len(sds)}시드 = '
              f'{len(units) * len(sds)} 체인  (구성 K{b["K"]} · lr {b["lr"]:g})\n{"="*78}')
        if 'ENS' in units:
            print('  ! ENS 는 증설이 아니라 **교체** 다 — 부스팅 3열과 열 수·파라미터 수가 같고')
            print('    앵커(직전 라운드 점수)만 끊었다. 라운드 이득이 잔차적합에서 오는지')
            print('    시드 평균화에서 오는지를 가르는 유일한 대조다.')
        else:
            print('  ! 앙상블은 하지 않는다 — 파라미터가 늘어 잡음과 구별이 안 된다(사용자 판정).')
            print('    여기서 재는 것은 아키텍처별 **전이의 시드평균과 분산** 이다.')
        if MIXES:
            print('  이질 부스팅 팔 — 라운드마다 함수족을 바꾼다:')
            for nm, un in MIXES.items():
                print(f'      {nm:10s} r1 {un[0]:9s} r2 {un[1]:9s} r3 {un[2] if len(un) > 2 else "-":9s}')
        R = args.rounds
        if args.prefix:
            print(f'  접두사 사다리 — 체인당 {R}라운드, **자기종결 끔**. 체인은 접두사 일관이라')
            print(f'  1회 학습으로 g1..g{R} {R}칸이 결합기 재적합만으로 나온다.')
            print('  왜: 지금 정지 규칙이 ES 이득 < 3e-4 인데 그 ES 가 전이와 무관하다'
                  '(아키텍처 간 ρ = −0.20).')
            print('      "몇 라운드가 맞는가" 를 ES 대신 홀드아웃으로 정한다.')
        sels = [x for x in args.sels.split(',') if x]
        assert set(sels) <= {'var', 'resid', 'chain', 'pls'}, f'알 수 없는 선택기준 {sels}'
        if 'pls' in sels:
            print('\n[PLS] raw 짝 풀링 적재 + 잔차 기준 기저 추출 (TR 전용)')
            pls_assets(A, U, max(args.pls_k, b['K']))
        arms = [(u, sl) for u in units for sl in sels]
        for u, sl in arms:
            for sd in sds:
                if args.prefix:
                    for k in range(1, R + 1):
                        run(f'{u}-{sl}-s{sd}-g{k}', u, b['K'], b['lr'], sd, 'none',
                            rounds=R, stop=False, n_gat=k, sel=sl)
                else:
                    run(f'{u}-{sl}-s{sd}', u, b['K'], b['lr'], sd, 'none', sel=sl)

        if args.prefix:
            print(f'\n{"="*78}\n[접두사 사다리] 유닛별 · 라운드별 {len(sds)}시드 평균\n{"="*78}')
            for u, sl in arms:
                print(f'  [{u}·{sl}]  {"접두사":10s}{"tie축":>10s}{"holdout":>10s}{"sd":>9s}'
                      f'{"clean":>10s}   직전 대비 (시드평균 paired)')
                prev = None
                for k in range(1, R + 1):
                    tg = [f'{u}-{sl}-s{sd}-g{k}' for sd in sds]
                    ti = [res[t]['tie'] for t in tg]
                    th = [res[t]['tot'] for t in tg]
                    ta = [res[t]['tot_all'] for t in tg]
                    cur = np.mean([rows[t] for t in tg], axis=0)
                    line = (f'{"":8s}{"g1..g" + str(k):10s}{np.mean(ti):10.5f}'
                            f'{np.mean(th):10.5f}{np.std(th, ddof=1):9.5f}{np.mean(ta):10.5f}')
                    if prev is None:
                        print(line + '   —')
                    else:
                        m, lo, hi = TL.paired_ci(prev, cur)
                        print(line + '   ' + TL.ci_str(m, lo, hi))
                    prev = cur
                bst = min(range(1, R + 1),
                          key=lambda k, _u=u, _s=sl: np.mean(
                              [res[f'{_u}-{_s}-s{sd}-g{k}']['tot'] for sd in sds]))
                print(f'        -> 시드평균 최선 접두사 g1..g{bst}  '
                      f'(현행 계약은 자기종결 3라운드)')
            print('  ! 직전 대비 Δ 는 같은 체인의 접두사끼리라 g1..g(k-1) 실현이 양쪽에 동일하다'
                  ' — 시드 잡음이 부분 상쇄된다.')
            print('  ! R 칸 중 최선을 고르는 선택 낙관이 있다. 채택하려면 그 접두사를'
                  ' 다시 확인해야 한다.')

            if len(units) >= 2:
                sl0 = sels[0]
                base = 'GAT' if 'GAT' in units else units[0]
                print(f'\n{"="*78}\n[팔 대조] 기준 {base}-{sl0} · {len(sds)}시드 평균 '
                      f'(K{b["K"]})\n{"="*78}')
                print('  r1 이 같은 유닛인 팔은 같은 시드에서 g1 실현이 **비트 동일** 하다')
                print('  — 그 팔의 Δ 는 라운드 2·3 만의 차이다.')
                print(f'{"팔":11s}{"g1":>10s}{"g2":>10s}{"g3":>10s}   '
                      f'g2 Δ 기준대비{"":>16s}g3 Δ 기준대비')
                for u in units:
                    mu = [np.mean([res[f'{u}-{sl0}-s{sd}-g{k}']['tot'] for sd in sds])
                          for k in (1, 2, 3)]
                    line = f'{u:11s}' + ''.join(f'{v:10.5f}' for v in mu)
                    if u == base:
                        print(line + '   기준')
                        continue
                    seg = []
                    for k in (2, 3):
                        a = np.mean([rows[f'{base}-{sl0}-s{sd}-g{k}'] for sd in sds], axis=0)
                        c = np.mean([rows[f'{u}-{sl0}-s{sd}-g{k}'] for sd in sds], axis=0)
                        m, lo, hi = TL.paired_ci(c, a)
                        seg.append(TL.ci_str(m, lo, hi))
                    print(line + '   ' + f'{seg[0]:28s}{seg[1]}')
                print('  음수 = 그 팔이 기준보다 낫다.')
                if 'ENS' in units:
                    print('  ! ENS 의 g2·g3 는 부스팅이 아니라 독립 헤드 2·3개다 — 같은 자리에서')
                    print('    앵커만 끊은 교체 대조. 여기서 지면 라운드 이득은 잔차적합의 것이고,')
                    print('    비기면 시드 평균화로 설명된다.')
            if len(sels) >= 2 and R >= 3:
                print(f'\n{"="*78}\n[2x2] K={b["K"]} · {len(sds)}시드 평균 (tie축 / 총)'
                      f'\n{"="*78}')
                print(f'{"":16s}{"g 1라운드":>20s}{"g 3라운드":>20s}')
                for sl in sels:
                    cells = []
                    for k in (1, 3):
                        tg = [f'{units[0]}-{sl}-s{sd}-g{k}' for sd in sds]
                        cells.append((np.mean([res[t]['tie'] for t in tg]),
                                      np.mean([res[t]['tot'] for t in tg])))
                    nm = {'var': 'SVD 분산순', 'resid': 'SVD 잔차재정렬',
                          'chain': 'SVD 라운드별재선택', 'pls': 'PLS(raw 4096)'}[sl]
                    print(f'{nm:16s}' + ''.join(f'{a:11.5f}/{c:8.5f}' for a, c in cells))
                if 'var' in sels:
                    for sl in [x for x in sels if x != 'var']:
                        for k in (1, 3):
                            a = np.mean([rows[f'{units[0]}-var-s{sd}-g{k}'] for sd in sds], axis=0)
                            c = np.mean([rows[f'{units[0]}-{sl}-s{sd}-g{k}'] for sd in sds], axis=0)
                            m, lo, hi = TL.paired_ci(c, a)
                            print(f'  g{k}  {sl} − var:  {TL.ci_str(m, lo, hi)}')
                print('  ! 잔차 순위는 TR 46,729행만으로 정했다 — stage2 는 열 선택을'
                      ' 적용만 받는다 (위생 유지).')
        else:
            print(f'\n{"="*78}\n[시드 요약] 유닛별 {len(sds)}시드\n{"="*78}')
            print(f'{"유닛":11s}{"holdout 평균":>13s}{"sd":>9s}{"clean 평균":>12s}{"sd":>9s}'
                  f'   GAT 대비 (시드평균 paired)')
            base = 'GAT' if 'GAT' in units else units[0]
            bh = np.mean([rows[f'{base}-{sels[0]}-s{s}'] for s in sds], axis=0)
            for u in units:
                th = [res[f'{u}-{sels[0]}-s{s}']['tot'] for s in sds]
                ta = [res[f'{u}-{sels[0]}-s{s}']['tot_all'] for s in sds]
                line = (f'{u:11s}{np.mean(th):13.5f}{np.std(th, ddof=1):9.5f}'
                        f'{np.mean(ta):12.5f}{np.std(ta, ddof=1):9.5f}')
                if u == base:
                    print(line + '   기준')
                else:
                    ah = np.mean([rows[f'{u}-{sels[0]}-s{s}'] for s in sds], axis=0)
                    m, lo, hi = TL.paired_ci(ah, bh)
                    print(line + '   ' + TL.ci_str(m, lo, hi))
            print('  시드평균 paired = 시드별 행손실을 먼저 평균한 뒤 행 부트스트랩.')
            print('  (시드는 고정으로 취급 — 아키텍처 간 "평균 전이" 비교다. 앙상블 점수가 아니다.)')

    if args.attn and b:
        au = [u for u in args.attn.split(',') if u]
        print(f'\n{"="*78}\n[어텐션] 유닛 뒤 softmax 어텐션 — seed {args.attn_seed} 단일\n'
              f'{"="*78}')
        print('  a = softmax_h(유닛출력) x h,  out = w(a ⊙ Vx).  현행 유닛은 여기서 softmax 를')
        print('  뺀 것 — 즉 h 축의 경쟁(Σa 고정)이 값을 더하는지만 본다.')
        print('  ! 순수 선형 혼합 A·u 는 뒤가 선형이라 w(Au)=(Aᵀw)·u 로 접혀 무의미하므로 뺐다.')
        for u in au:
            run(f'{u}+ATTN-s{args.attn_seed}', f'{u}_A', b['K'], b['lr'],
                args.attn_seed, 'none')
        print('\n  [어텐션 대조] 같은 시드의 어텐션 없는 팔과 짝지어')
        for u in au:
            t0, t1 = f'{u}-s{args.attn_seed}', f'{u}+ATTN-s{args.attn_seed}'
            if t0 in rows:
                m, lo, hi = TL.paired_ci(rows[t1], rows[t0])
                print(f'    {u:10s} {res[t0]["tot"]:.5f} -> {res[t1]["tot"]:.5f}   '
                      f'{TL.ci_str(m, lo, hi)}')
            else:
                print(f'    {u:10s} 대조 칸 없음 (--sweep 에 {u} 를 넣어야 짝이 생긴다)')
        print('  ! 시드 1개다 — 시드 sd 가 0.0008 급이므로 0.0016 미만 차이는 판정 불가.')

    print(f'\n{"="*78}\n[순위] Δ vs {REF_TAG} ({ref_row.mean():.5f}) '
          f'— 음수 = 그 칸이 낫다 / 문턱 0.0005\n{"="*78}')
    print(f'{"칸":26s}{"총 holdout":>11s}{"열":>5s}{"λ":>8s}  Δ')
    for t in sorted(res, key=lambda x: res[x]['tot']):
        d = res[t]
        print(f'{t:26s}{d["tot"]:11.5f}{d["n_col"]:5d}{d["lam"]:8g}  '
              f'{TL.ci_str(d["d_mean"], d["d_lo"], d["d_hi"])}')

    op = FINAL / args.out
    np.savez_compressed(op.with_suffix('.rowloss.npz'), **rows, ref=ref_row,
                        **{f'ALL__{k}': v for k, v in rows_all.items()})
    with open(op.with_suffix('.json'), 'w') as f:
        json.dump({'ref_tag': REF_TAG, 'ref': float(ref_row.mean()),
                   'cells': {k: {kk: vv for kk, vv in v.items() if kk != 'aux'}
                             for k, v in res.items()}}, f, ensure_ascii=False, indent=1)
    print(f'\n저장: {op}.rowloss.npz / {op}.json')


if __name__ == '__main__':
    main()
