"""newhead.py — 융합 헤드 **본학습**: train.csv 로 훈련, stage2 로 평가 (v2).

프로토콜 (stage2 예지 v1 에서 전환 — 사용자 결정)
  · 훈련   train.csv 46,729행. --folds 5 앙상블: fold 마다 4/5 로 그래디언트,
           (d, nu) 를 fold 평균 (로짓공간 평균 관례와 동일)
  · ES     valid.csv 5,000 (§1.6 허용 노출 — 인코더들과 같은 규율)
  · 평가   holdout.csv 5,748 (무접촉) — **판정은 여기서만**. 전체 stage2 수치는 참고용
  · 선행   python Boost/extract_logits.py --pooled --split train --only abe,tb3 \\
               --steps abe:3000,tb3:4000          (~2시간, 저장 ~1.5GB)

v1 예지(stage2 8,600행, log_newhead.txt)에서 배운 것
  · 팔 0 이 0.9814 로 미수렴 — wd 가 스킵 계수를 눌렀고 epoch 상한에 닿았다. 수정:
    (1) 편향·LN·1차원 파라미터 무감쇠, (2) 기준선은 torch 가 아니라 **LBFGS 로 stage2
    교차적합** (Davidson κ=1 과 동일 모형 — 0.9705 가 나와야 배선이 맞다)
  · 로짓 차단 대결 순위: concat(0.9857) > 합의도(1.009) > 어텐션(1.007, α 균등 붕괴)
    > concat-풀(1.027). 전부 기준선에 졌지만 행 부족 서명(α 붕괴, 로짓 재발견 ρ 0.5)이
    뚜렷하므로 **아무 팔도 탈락시키지 않고** 46,729행에서 재대결한다 (사용자 결정)
  · 용량 가설 추가 (사용자 결정): v1 의 실패가 FFN 이 너무 작아서일 수 있다.
    원 헤드 규모(단일 은닉 1024)와 BERT-style FFN(1024→4096→1024) 팔을 추가한다.

로짓 규율
  train 행의 ABE/TB 로짓은 in-sample 이라 애초에 쓸 수 없다. 따라서 **모든 융합 팔이
  구조적으로 순수**다 — 토큰에 로짓 없음, 스킵 없음. 헤드는 풀러에서 직접 뽑는다.
  팔 0 은 stage2 로짓 교차적합(LBFGS) 기준선이고 holdout 예측이 out-of-fold 라 비교가 공정하다.

팔 (내용 x 구조)
   0  로짓 기준선 (stage2 5-fold LBFGS = Davidson κ=1)
   1  concat 직결(→128→64→2)   원시     2  〃                        공학
   3  전문가별 사영→어텐션      원시     4  〃                        공학
   5  전체 concat→attnpool      원시     6  〃                        공학
   7  합의도 저계수 쌍곱        원시     8  〃                        공학
   9  원헤드 규모(→1024→2)      원시    10  〃                        공학
  11  BERT-FFN(→1024→4096→1024→2) 원시  12  〃                        공학
  9·10: ABE/TB 가 로짓을 뽑을 때 쓰던 규모(단일 은닉 1024)의 근사. 공학 내용이면 입력형까지
        원 헤드와 같아 "원 헤드를 두 전문가 합동으로 재훈련" 에 가장 가깝다.
  11·12: 사용자 지정 BERT-style — 사영(→1024) 뒤 1024→4096→1024 잔차 FFN 블록.

공통 뼈대 (v1 과 동일)
  · swap 등변은 프레임 평균으로 구조 보장: 홀 성분 부호를 뒤집어 두 번 태우고
        d = p0 − q0 (홀, 승부축),  nu = p1 + q1 (짝, tie축)
        logits = [d/2, −d/2, nu + logaddexp(d/2, −d/2)]
  · 융합부 마지막 층 0 초기화, 표준화는 그래디언트 fold 통계로만
    (짝: 중심화+표준화 / 홀: **중심화 없이 RMS 만** — 중심화는 상수항이라 반대칭을 깬다)
  · 토큰 구축이 train/stage2 **공용 함수**라 두 분할의 정의가 어긋날 수 없다

사용
  python Boost/newhead.py                          # 팔 13개 전부
  python Boost/newhead.py --arms 0,1,9,11          # 용량 사다리만
  python Boost/newhead.py --folds 1                # 앙상블 없이 전 train 단일 적합
  python Boost/newhead.py --selfcheck              # 합성 소형 데이터로 전 배선 검사 (초 단위)
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

# torch 가 HIP/CUDA 를 초기화하기 전에 걸어야 효력이 있다 — 단편화로 인한 OOM 을 줄인다
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BOOST = Path('./Boost')
sys.path.insert(0, str(BOOST))
import reprhead as RH  # noqa: E402  (경로·패리티 분해·출력 유틸·CI 재사용)

SEED, K2 = 42, 5                     # K2 = stage2 기준선의 fold 수
EPS = 1e-15
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
EXPERTS = ('ABE', 'TB3', 'Arith')
NM = {
    '0': '0(로짓 기준선 — stage2 LBFGS, Davidson κ=1)',
    '1': '1(concat 직결 + 원시)', '2': '2(concat 직결 + 공학)',
    '3': '3(전문가별 사영→어텐션 + 원시)', '4': '4(전문가별 사영→어텐션 + 공학)',
    '5': '5(전체 concat→attnpool + 원시)', '6': '6(전체 concat→attnpool + 공학)',
    '7': '7(합의도 저계수 쌍곱 + 원시)', '8': '8(합의도 저계수 쌍곱 + 공학)',
    '9': '9(원헤드 규모 →1024→2 + 원시)', '10': '10(원헤드 규모 →1024→2 + 공학)',
    '11': '11(BERT-FFN →1024→4096→1024→2 + 원시)', '12': '12(BERT-FFN + 공학)',
}
ARM_CFG = {'1': ('raw', 'concat'), '2': ('eng', 'concat'),
           '3': ('raw', 'attn'), '4': ('eng', 'attn'),
           '5': ('raw', 'cpool'), '6': ('eng', 'cpool'),
           '7': ('raw', 'agree'), '8': ('eng', 'agree'),
           '9': ('raw', 'bighead'), '10': ('eng', 'bighead'),
           '11': ('raw', 'bertffn'), '12': ('eng', 'bertffn')}


# ══════════════════════════════════════════════════════ 토큰 (train/stage2 공용)
def build_tokens(split, abe_step, tb_step):
    """풀링 npz + Arith 피쳐 -> 전문가별 (짝 E, 홀 O). 두 분할이 같은 코드를 타므로
    열 정의·순서가 어긋날 수 없다. 로짓 열은 없다 (본학습 규율)."""
    suf, parts = ('__train', ('train',)) if split == 'train' else ('', ('valid', 'hold'))
    pa = RH.POOL / f'abe__checkpoint-{abe_step}{suf}.npz'
    pt = RH.POOL / f'tb3__checkpoint-{tb_step}{suf}.npz'
    for pth in (pa, pt):
        assert pth.exists(), (
            f'{pth} 없음 — 먼저 실행:\n  python Boost/extract_logits.py --pooled '
            f'--split {split} --only abe,tb3 --steps abe:{abe_step},tb3:{tb_step}')
    za, zt = np.load(pa), np.load(pt)
    cat_ = lambda z, k: np.vstack([z[f'{p}_{k}'] for p in parts]).astype(np.float64)
    ids = np.concatenate([za[f'{p}_ids'] for p in parts])
    y = np.concatenate([za[f'{p}_labels'] for p in parts]).astype(np.int64)
    assert (np.concatenate([zt[f'{p}_ids'] for p in parts]) == ids).all(), 'ABE/TB3 id 불일치'

    ea, eb = cat_(za, 'ea'), cat_(za, 'eb')
    qf, af, bf = cat_(zt, 'q_f'), cat_(zt, 'a_f'), cat_(zt, 'b_f')
    qs, as_, bs = cat_(zt, 'q_s'), cat_(zt, 'a_s'), cat_(zt, 'b_s')
    # 패리티 분해 — swp 뷰의 a 구간에는 B 가 들어 있다 (실응답 기준 정렬)
    A_rep, B_rep = (af + bs) / 2, (bf + as_) / 2
    D_a, D_b = (af - bs) / 2, (bf - as_) / 2
    abe_u, abe_v = (ea + eb) / 2, (ea - eb) / 2
    tb_u, tb_v = (A_rep + B_rep) / 2, (A_rep - B_rep) / 2
    tbq_u, tbq_v = (qf + qs) / 2, (qf - qs) / 2
    tbd_u, tbd_v = (D_a - D_b) / 2, (D_a + D_b) / 2

    cat = pd.read_csv(RH.ARITH / 'feature_catalog.csv')
    feat = pd.read_parquet(RH.ARITH / 'features_57k.parquet')
    pos = pd.Index(feat['id']).get_indexer(ids)
    assert (pos >= 0).all(), 'features_57k 에 없는 id'
    ariE, _, ariO, _ = RH.parity_split_arith(feat, pos, cat)

    # 공학 = 원 헤드 입력형. 가중치 없는 결정적 변환뿐이라 누수가 없다.
    # 뷰별 피쳐는 swap 시 뷰가 통째로 교환되므로 (f+s)/2 짝, (f−s)/2 홀 — 일반 규칙.
    tt_abe = np.tanh(ea) * np.tanh(eb)
    abf, abs2 = np.abs(af - bf), np.abs(as_ - bs)
    ttf, tts = np.tanh(af) * np.tanh(bf), np.tanh(as_) * np.tanh(bs)

    pack = lambda Es, Os: (np.hstack(Es).astype(np.float32), np.hstack(Os).astype(np.float32))
    raw = {'ABE': pack([abe_u], [abe_v]),
           'TB3': pack([tb_u, tbq_u, tbd_u], [tb_v, tbq_v, tbd_v]),
           'Arith': pack([ariE], [ariO])}
    eng = {'ABE': pack([abe_u, tt_abe], [abe_v]),
           'TB3': pack([tb_u, tbq_u, tbd_u, (abf + abs2) / 2, (ttf + tts) / 2],
                       [tb_v, tbq_v, tbd_v, (abf - abs2) / 2, (ttf - tts) / 2]),
           'Arith': raw['Arith']}
    return dict(ids=ids, y=y, raw=raw, eng=eng,
                n_first=len(za[f'{parts[0]}_ids']))


# ══════════════════════════════════════════════════════ 헤드
class FuseHead(nn.Module):
    """swap 등변 융합 헤드 (스킵 없음 — 본학습 규율).
    arch: concat | attn | cpool | agree | bighead | bertffn.
    등변은 프레임 평균으로 정확히 보장된다. 마지막 층은 0 초기화."""

    def __init__(self, dims, arch, d=64, heads=4, layers=1, hidden=128, big=1024,
                 ffn_mult=4, drop=0.1, rank=4):
        super().__init__()
        self.arch = arch
        D = sum(de + do for de, do in dims)
        if arch == 'concat':
            h2 = max(hidden // 2, 8)
            self.net = nn.Sequential(nn.Linear(D, hidden), nn.GELU(), nn.Dropout(drop),
                                     nn.Linear(hidden, h2), nn.GELU(), nn.Dropout(drop),
                                     nn.Linear(h2, 2))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        elif arch == 'bighead':
            # 원 헤드 규모의 근사 — ABE/TB 헤드가 정확히 이 형(단일 은닉 1024)이다
            self.net = nn.Sequential(nn.Linear(D, big), nn.GELU(), nn.Dropout(drop),
                                     nn.Linear(big, 2))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        elif arch == 'bertffn':
            # 사용자 지정: 사영(→big) 뒤 big→ffn_mult*big→big 잔차 FFN (BERT 블록형)
            self.inp = nn.Linear(D, big)
            self.ln1, self.ln2 = nn.LayerNorm(big), nn.LayerNorm(big)
            self.ff = nn.Sequential(nn.Linear(big, ffn_mult * big), nn.GELU(),
                                    nn.Dropout(drop), nn.Linear(ffn_mult * big, big))
            self.out = nn.Linear(big, 2)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        elif arch == 'attn':
            self.projs = nn.ModuleList([nn.Linear(de + do, d) for de, do in dims])
            self.ln0 = nn.LayerNorm(d)
            self.emb = nn.Parameter(torch.randn(len(dims), d) * 0.02)
            self.mha = nn.ModuleList([nn.MultiheadAttention(d, heads, dropout=drop,
                                                            batch_first=True)
                                      for _ in range(layers)])
            self.lnA = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
            self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                                    nn.Dropout(drop), nn.Linear(2 * d, d))
                                      for _ in range(layers)])
            self.lnB = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
            self.rq = nn.Parameter(torch.randn(d) * 0.02)          # readout attnpool
            self.out = nn.Linear(d, 2)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        elif arch == 'cpool':
            # 전체 concat 을 d 조각으로 잘라 attnpool 로 d 까지 축소. 조각내기는 파라미터 0.
            self.d = d
            self.T = (D + d - 1) // d
            self.padn = self.T * d - D
            self.ln0 = nn.LayerNorm(d)
            self.score = nn.Linear(d, 1)
            self.out = nn.Linear(d, 2)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            frac = torch.zeros(self.T, len(dims))                  # 조각 -> 전문가 귀속
            c = 0
            for i, (de, do) in enumerate(dims):
                for col in range(c, c + de + do):
                    frac[col // d, i] += 1.0
                c += de + do
            self.register_buffer('frac', frac / frac.sum(1, keepdim=True).clamp(min=1e-9))
        elif arch == 'agree':
            # 전문가별 판단을 직접 뽑고 곱으로 합의도를 만든다. 패리티가 자동으로 맞는다:
            #   â_i⊙â_j (홀x홀=짝)->tie축 "방향이 같으면 승부 확실",  â_i⊙b̂_j (홀x짝=홀)->승부축
            self.rank = rank
            self.pa = nn.ModuleList([nn.Linear(do, rank, bias=False) for _, do in dims])
            self.pb = nn.ModuleList([nn.Linear(de, rank) for de, _ in dims])
            Z = 2 * len(dims) * rank
            iu, iv = torch.triu_indices(Z, Z)
            self.register_buffer('iu', iu)
            self.register_buffer('iv', iv)
            self.out = nn.Linear(Z + iu.numel(), 2)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        else:
            raise ValueError(arch)

    def _once(self, Es, Os, s):
        if self.arch == 'agree':
            a = [p(s * O) for p, O in zip(self.pa, Os)]
            b = [p(E) for p, E in zip(self.pb, Es)]
            z = torch.cat(a + b, 1)
            return self.out(torch.cat([z, z[:, self.iu] * z[:, self.iv]], 1)), z
        xs = [torch.cat([E, s * O], 1) for E, O in zip(Es, Os)]
        if self.arch in ('concat', 'bighead'):
            return self.net(torch.cat(xs, 1)), None
        if self.arch == 'bertffn':
            h = self.inp(torch.cat(xs, 1))
            h = self.ln2(h + self.ff(self.ln1(h)))
            return self.out(h), None
        if self.arch == 'cpool':
            x = torch.cat(xs, 1)
            if self.padn:
                x = torch.cat([x, x.new_zeros(x.shape[0], self.padn)], 1)
            t = self.ln0(x.view(x.shape[0], self.T, self.d))
            alpha = torch.softmax(self.score(t).squeeze(-1), dim=1)
            # bmm 은 (B,T,d) 곱을 실체화하지 않고 바로 축약한다 — 브로드캐스트 곱 후 sum 을
            # 쓰면 입력과 같은 크기(eng 전배치에서 2GB)의 중간 텐서가 하나 더 생겨 OOM 이 난다.
            # 수학적으로 동일하므로 이미 돌린 팔의 수치와 그대로 비교된다.
            g = torch.bmm(alpha.unsqueeze(1), t).squeeze(1)
            return self.out(g), alpha @ self.frac
        t = torch.stack([self.ln0(p(x)) for p, x in zip(self.projs, xs)], 1) + self.emb
        for mha, la, ffn, lb in zip(self.mha, self.lnA, self.ffn, self.lnB):
            a, _ = mha(t, t, t, need_weights=False)
            t = la(t + a)
            t = lb(t + ffn(t))
        alpha = torch.softmax(t @ self.rq / math.sqrt(t.shape[-1]), dim=1)
        g = (alpha.unsqueeze(-1) * t).sum(1)
        return self.out(g), alpha

    def forward(self, Es, Os):
        p, a1 = self._once(Es, Os, 1.0)
        q, a2 = self._once(Es, Os, -1.0)
        d = p[:, 0] - q[:, 0]                    # 반대칭화 -> 홀 -> 승부축
        nu = p[:, 1] + q[:, 1]                   # 대칭화   -> 짝 -> tie축
        aux = a1 if self.arch == 'agree' else (None if a1 is None else (a1 + a2) / 2)
        return d, nu, aux


def logits3(d, nu):
    return torch.stack([d / 2, -d / 2, nu + torch.logaddexp(d / 2, -d / 2)], 1)


def compose(d, nu):
    z = np.stack([d / 2, -d / 2, nu + np.logaddexp(d / 2, -d / 2)], 1)
    z -= z.max(1, keepdims=True)
    P = np.exp(z)
    return P / P.sum(1, keepdims=True)


def equivariance_check():
    """전 구조 난수 검산 — swap(홀 부호반전) 시 d 반전·nu 불변이 정확히 성립해야 한다."""
    torch.manual_seed(0)
    dims = [(7, 5), (6, 4), (5, 3)]
    Es = [torch.randn(24, de) for de, _ in dims]
    Os = [torch.randn(24, do) for _, do in dims]
    worst = 0.0
    for arch in ('concat', 'attn', 'cpool', 'agree', 'bighead', 'bertffn'):
        m = FuseHead(dims, arch, d=16, heads=2, hidden=24, big=32, rank=3)
        for last in (getattr(m, 'out', None), getattr(m, 'net', [None])[-1]
                     if hasattr(m, 'net') else None):
            if isinstance(last, nn.Linear):
                nn.init.normal_(last.weight, std=0.1)      # 0 초기화면 검산이 자명해진다
        m.eval()
        with torch.no_grad():
            d1, n1, _ = m(Es, Os)
            d2, n2, _ = m(Es, [-o for o in Os])
        worst = max(worst, float((d1 + d2).abs().max()), float((n1 - n2).abs().max()))
    assert worst < 1e-4, f'등변성 위반 {worst:.2e}'
    return worst


# ══════════════════════════════════════════════════════ 학습
@torch.no_grad()
def fwd_chunked(m, Es, Os, chunk):
    """평가용 전방 계산을 행 단위로 쪼갠다. no_grad 라도 cpool 은 (B,T,d) 텐서를 잡으므로
    37k 행을 통째로 태우면 메모리가 튄다."""
    n = Es[0].shape[0]
    if chunk <= 0 or chunk >= n:
        d, nu, aux = m(Es, Os)
        return d, nu, aux
    ds, ns, aux = [], [], []
    for st in range(0, n, chunk):
        sl = slice(st, min(st + chunk, n))
        d, nu, a = m([E[sl] for E in Es], [O[sl] for O in Os])
        ds.append(d)
        ns.append(nu)
        if a is not None:
            aux.append(a)
    return torch.cat(ds), torch.cat(ns), (torch.cat(aux) if aux else None)


def train_one(factory, TR, ES_, y_tr, y_es, lr, wd, epochs, patience, seed, batch=0):
    torch.manual_seed(seed)
    m = factory().to(DEV)
    # 편향·LN·1차원 파라미터 무감쇠 — v1 에서 wd 가 기준선 계수를 눌러 0.9814 로 왜곡됐던 버그
    decay, nodecay = [], []
    for _, p_ in m.named_parameters():
        (nodecay if p_.ndim < 2 else decay).append(p_)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': wd},
                             {'params': nodecay, 'weight_decay': 0.0}], lr=lr)
    lossf = nn.CrossEntropyLoss()
    yt = torch.tensor(y_tr, dtype=torch.long, device=DEV)
    ye = torch.tensor(y_es, dtype=torch.long, device=DEV)
    n = len(y_tr)
    full = batch <= 0 or batch >= n
    best, state, bad, ep = np.inf, None, 0, 0
    for ep in range(epochs):
        m.train()
        if full:
            opt.zero_grad()
            d, nu, _ = m(*TR)
            lossf(logits3(d, nu), yt).backward()
            opt.step()
        else:
            # 미니배치 — cpool/bertffn 처럼 활성이 입력 크기에 비례하는 팔은 전배치가 안 들어간다.
            # 한 epoch = 전 행 1회 통과이므로 patience 의 의미는 그대로다.
            perm = torch.randperm(n, device=DEV)
            for st in range(0, n, batch):
                sel = perm[st:st + batch]
                opt.zero_grad()
                d, nu, _ = m([E[sel] for E in TR[0]], [O[sel] for O in TR[1]])
                lossf(logits3(d, nu), yt[sel]).backward()
                opt.step()
        m.eval()
        with torch.no_grad():
            d, nu, _ = fwd_chunked(m, ES_[0], ES_[1], batch)
            v = float(lossf(logits3(d, nu), ye))
        if v < best - 1e-6:
            best, bad = v, 0
            state = {k: t.detach().clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if state is not None:
        m.load_state_dict(state)
    return m.eval(), ep + 1


def run_arm(content, arch, TRN, S2, foldT, args):
    """한 팔: train fold 앙상블 학습 -> stage2 (d,nu) fold 평균.
    반환 dict: P(stage2 확률), ins(학습행 in-sample ll), alpha, corr, n_par, ep."""
    y_tr, y2 = TRN['y'], S2['y']
    toksT, toks2 = TRN[content], S2[content]
    es_idx = np.where(~S2['is_hold'])[0]                       # valid = ES 전용
    n2 = len(y2)
    d_acc, nu_acc = np.zeros(n2), np.zeros(n2)
    AL, corrs, ins, eps, n_par = None, [], [], [], 0
    rng = np.random.default_rng(SEED)
    folds = sorted(set(foldT)) if args.folds > 1 else [-1]
    for k in folds:
        grad = np.where(foldT != k)[0] if k >= 0 else np.arange(len(y_tr))
        Es_t, Os_t, Es_2, Os_2 = {}, {}, {}, {}
        for nm in EXPERTS:
            Et, Ot = toksT[nm]
            E2, O2 = toks2[nm]
            mu = Et[grad].mean(0)
            sd = np.clip(Et[grad].std(0), 1e-6, None)
            rms = np.clip(np.sqrt((Ot[grad] ** 2).mean(0)), 1e-6, None)
            Es_t[nm], Os_t[nm] = (Et - mu) / sd, Ot / rms
            Es_2[nm], Os_2[nm] = (E2 - mu) / sd, O2 / rms
        dims = [(Es_t[nm].shape[1], Os_t[nm].shape[1]) for nm in EXPERTS]
        T = lambda a, idx=None: torch.tensor(a if idx is None else a[idx],
                                             dtype=torch.float32, device=DEV)
        TRb = ([T(Es_t[nm], grad) for nm in EXPERTS], [T(Os_t[nm], grad) for nm in EXPERTS])
        ESb = ([T(Es_2[nm], es_idx) for nm in EXPERTS], [T(Os_2[nm], es_idx) for nm in EXPERTS])
        factory = lambda: FuseHead(dims, arch, d=args.d, heads=args.heads, layers=args.layers,
                                   hidden=args.hidden, big=args.big, ffn_mult=args.ffn_mult,
                                   drop=args.drop, rank=args.rank)
        m, ne = train_one(factory, TRb, ESb, y_tr[grad], y2[es_idx], args.lr, args.wd,
                          args.epochs, args.patience, SEED + max(k, 0), args.batch)
        n_par = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            S2b = ([T(Es_2[nm]) for nm in EXPERTS], [T(Os_2[nm]) for nm in EXPERTS])
            d, nu, aux = fwd_chunked(m, S2b[0], S2b[1], args.batch)
            d_acc += d.cpu().numpy() / len(folds)
            nu_acc += nu.cpu().numpy() / len(folds)
            # in-sample 진단 — 그래디언트 행 5k 표본의 손실. holdout 과의 격차 = 일반화 갭
            sub = grad[rng.permutation(len(grad))[:5000]]
            di, ni, _ = fwd_chunked(m, [T(Es_t[nm], sub) for nm in EXPERTS],
                                    [T(Os_t[nm], sub) for nm in EXPERTS], args.batch)
            Pi = torch.softmax(logits3(di, ni), 1).cpu().numpy()
            ins.append(float(-np.log(np.clip(Pi[np.arange(len(sub)), y_tr[sub]],
                                             EPS, 1)).mean()))
            if aux is not None:
                a = aux.cpu().numpy()
                if arch in ('attn', 'cpool'):
                    if AL is None:
                        AL = np.zeros((n2, a.shape[1]))
                    AL += a / len(folds)
                elif arch == 'agree':
                    r = args.rank
                    row = []
                    for i, nm in enumerate(EXPERTS):
                        mo, to = S2['own'][nm]
                        za = a[:, i * r:(i + 1) * r]
                        zb = a[:, len(EXPERTS) * r + i * r:len(EXPERTS) * r + (i + 1) * r]
                        cc = lambda M, v: (np.nan if v is None else
                                           max(abs(np.corrcoef(M[:, j], v)[0, 1])
                                               for j in range(M.shape[1])))
                        row += [cc(za, mo), cc(zb, to)]
                    corrs.append(row)
        eps.append(ne)
        del TRb, ESb, S2b, m
        if DEV == 'cuda':
            torch.cuda.empty_cache()
    if corrs:
        C = np.array(corrs)
        # TB 승부 열은 설계상 전량 NaN(2-class 라 승부 로짓이 없다) — nanmean 경고 없이 처리
        corr_out = np.array([np.nan if np.isnan(c).all() else np.nanmean(c) for c in C.T])
    else:
        corr_out = None
    return dict(P=compose(d_acc, nu_acc), ins=float(np.mean(ins)), alpha=AL,
                corr=corr_out, n_par=n_par, ep=int(np.mean(eps)))


# ══════════════════════════════════════════════════════ 평가
def per_row(P, y):
    P = np.clip(P, EPS, 1.0)
    P = P / P.sum(1, keepdims=True)
    return -np.log(P[np.arange(len(y)), y])


def axis_report(P, y):
    P = np.clip(P, EPS, 1.0)
    P = P / P.sum(1, keepdims=True)
    tot = float(per_row(P, y).mean())
    it = y == 2
    t = np.clip(P[:, 2], EPS, 1 - EPS)
    tie = float(-np.where(it, np.log(t), np.log(1 - t)).mean())
    nz = ~it
    rA = np.clip(P[nz, 0] / (P[nz, 0] + P[nz, 1]), EPS, 1 - EPS)
    win = float(-np.where(y[nz] == 0, np.log(rA), np.log(1 - rA)).mean())
    return tot, tie, win


def baseline_stage2(Le, Lo, y, fold2):
    """로짓만의 축분리 결합 — stage2 5-fold, LBFGS. Davidson κ=1 과 동일 모형이라
    0.9705 가 나와야 배선이 맞다. holdout 예측이 out-of-fold 라 팔들과의 비교가 공정하다."""
    N = len(y)
    tie = y == 2
    nz = ~tie
    y_t, y_w = tie.astype(np.float64), (y == 0).astype(np.float64)
    d, nu = np.zeros(N), np.zeros(N)
    for k in range(K2):
        tr, te = fold2 != k, fold2 == k
        vW, ok1 = RH._fit(Lo[tr & nz], y_w[tr & nz], np.zeros(2), False)
        vT, ok2 = RH._fit(Le[tr], y_t[tr], np.zeros(3), True)
        assert ok1 and ok2, '기준선 LBFGS 미수렴'
        d[te] = Lo[te] @ vW[:2]
        nu[te] = Le[te] @ vT[:3] + vT[3]
    return compose(d, nu)


# ══════════════════════════════════════════════════════ 셀프체크 (합성 소형)
def selfcheck(args):
    """전 팔 배선을 합성 데이터로 끝까지 태운다 — 파일 적재 없이 초 단위."""
    rng = np.random.default_rng(0)
    nT, n2 = 300, 120
    mk = lambda n: {'ABE': (rng.normal(0, 1, (n, 10)).astype(np.float32),
                            rng.normal(0, 1, (n, 10)).astype(np.float32)),
                    'TB3': (rng.normal(0, 1, (n, 14)).astype(np.float32),
                            rng.normal(0, 1, (n, 14)).astype(np.float32)),
                    'Arith': (rng.normal(0, 1, (n, 6)).astype(np.float32),
                              rng.normal(0, 1, (n, 5)).astype(np.float32))}
    TRN = {'y': rng.integers(0, 3, nT), 'raw': mk(nT), 'eng': mk(nT)}
    S2 = {'y': rng.integers(0, 3, n2), 'is_hold': np.arange(n2) >= n2 // 2,
          'raw': mk(n2), 'eng': mk(n2),
          'own': {'ABE': (rng.normal(0, 1, n2), rng.normal(0, 1, n2)),
                  'TB3': (None, rng.normal(0, 1, n2)),
                  'Arith': (rng.normal(0, 1, n2), rng.normal(0, 1, n2))}}
    foldT = np.arange(nT) % 2
    args.folds, args.epochs, args.patience, args.batch = 2, 12, 6, 128
    for key, (cn, arch) in ARM_CFG.items():
        r = run_arm(cn, arch, TRN, S2, foldT, args)
        assert np.isfinite(r['P']).all() and abs(r['P'].sum(1) - 1).max() < 1e-5
        print(f'  [{key:>2}] {arch:<8} P OK | in-sample {r["ins"]:.3f} | '
              f'파라미터 {r["n_par"]:,} | α {"—" if r["alpha"] is None else r["alpha"].shape} | '
              f'corr {"—" if r["corr"] is None else np.round(r["corr"], 2)}')
    Le = rng.normal(0, 1, (n2, 3))
    Lo = rng.normal(0, 1, (n2, 2))
    P0 = baseline_stage2(Le, Lo, S2['y'], np.arange(n2) % K2)
    assert np.isfinite(P0).all()
    print(f'  [ 0] baseline OK | 총 {axis_report(P0, S2["y"])[0]:.3f}')
    print('셀프체크 전부 통과')


# ══════════════════════════════════════════════════════ 본체
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--abe-step', type=int, default=3000)
    ap.add_argument('--tb-step', type=int, default=4000)
    ap.add_argument('--arms', default='0,1,2,3,4,5,6,7,8,9,10,11,12')
    ap.add_argument('--folds', type=int, default=5, help='train 앙상블 fold 수 (1 = 단일 적합)')
    ap.add_argument('--d', type=int, default=64, help='어텐션/cpool 병목 차원')
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--hidden', type=int, default=128, help='concat MLP 은닉 폭')
    ap.add_argument('--big', type=int, default=1024, help='bighead/bertffn 은닉 폭 (원 헤드 규모)')
    ap.add_argument('--ffn-mult', type=int, default=4, help='bertffn 확장 배수 (BERT 관례 4)')
    ap.add_argument('--rank', type=int, default=4, help='합의도 팔 전문가별 잠재 차원')
    ap.add_argument('--drop', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-2)
    ap.add_argument('--epochs', type=int, default=800)
    ap.add_argument('--patience', type=int, default=60)
    ap.add_argument('--batch', type=int, default=0,
                    help='미니배치 크기 (0 = 전배치). cpool/bertffn 은 활성이 입력 크기에 '
                         '비례해 eng 전배치(37k행 x 13.5k열)가 16GB 카드에 안 들어간다. '
                         '평가·예측 경로도 같은 크기로 쪼갠다. **전배치와 최적화 궤적이 달라지므로 '
                         '섞어 쓰지 말고 전 팔을 같은 값으로 돌릴 것**')
    ap.add_argument('--tag', default='', help='출력 npz 접미사')
    ap.add_argument('--selfcheck', action='store_true', help='합성 소형 데이터로 배선 검사')
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    worst = equivariance_check()
    print(f'[검산] 등변성 6구조 max 위반 {worst:.2e}  OK — TTA 불필요, 응답 순서 불변\n')
    if args.selfcheck:
        selfcheck(args)
        return

    t0 = time.time()
    print('자산 적재 중 (train + stage2 풀링, Arith 피쳐 x2, 공학피쳐 계산) ...')
    TRN = build_tokens('train', args.abe_step, args.tb_step)
    S2 = build_tokens('stage2', args.abe_step, args.tb_step)
    S2['is_hold'] = np.arange(len(S2['y'])) >= S2['n_first']
    R = RH.load_blocks(args.abe_step, args.tb_step)             # 기준선 로짓 + own 진단용
    assert (R['ids'] == S2['ids']).all(), 'stage2 id 불일치'
    Le, Lo = R['EV']['logit'].astype(np.float64), R['OD']['logit'].astype(np.float64)
    S2['own'] = {'ABE': (Lo[:, 0], Le[:, 1]), 'TB3': (None, Le[:, 0]),
                 'Arith': (Lo[:, 1], Le[:, 2])}
    y_tr, y2, hm = TRN['y'], S2['y'], S2['is_hold']
    print(f'  적재 {time.time() - t0:.0f}s | train {len(y_tr):,} / stage2 {len(y2):,} '
          f'(ES=valid {int((~hm).sum()):,} / 평가=holdout {int(hm.sum()):,}) | device {DEV}')
    for cn in ('raw', 'eng'):
        dd = {nm: (TRN[cn][nm][0].shape[1], TRN[cn][nm][1].shape[1]) for nm in EXPERTS}
        print(f'  {cn:<4} 토큰 (짝, 홀): ' + '  '.join(f'{nm} {v}' for nm, v in dd.items()) +
              f'  합 {sum(a + b for a, b in dd.values()):,}')
    print(f'  하이퍼  d {args.d} hidden {args.hidden} big {args.big} ffn x{args.ffn_mult} '
          f'rank {args.rank} | folds {args.folds} lr {args.lr} wd {args.wd} '
          f'epochs<={args.epochs} patience {args.patience} '
          f'batch {"전배치" if args.batch <= 0 else args.batch}')
    print('  로짓 규율: train 로짓은 in-sample 이라 아예 안 쓴다 — 전 융합 팔이 풀러만 본다.\n')

    from sklearn.model_selection import StratifiedKFold
    foldT = np.zeros(len(y_tr), int)
    if args.folds > 1:
        for k, (_, te) in enumerate(StratifiedKFold(args.folds, shuffle=True,
                                                    random_state=SEED)
                                    .split(np.zeros((len(y_tr), 1)), y_tr)):
            foldT[te] = k
    fold2 = np.zeros(len(y2), int)
    for k, (_, te) in enumerate(StratifiedKFold(K2, shuffle=True, random_state=SEED)
                                .split(np.zeros((len(y2), 1)), y2)):
        fold2[te] = k

    want = [a.strip() for a in args.arms.split(',') if a.strip()]
    store, rows = {}, []
    for key in want:
        t1 = time.time()
        if key == '0':
            P = baseline_stage2(Le, Lo, y2, fold2)
            res = dict(P=P, ins=float('nan'), alpha=None, corr=None, n_par=6, ep=0)
        else:
            cn, arch = ARM_CFG[key]
            print(f'[{key}] {NM[key]}  적합 중 ...', flush=True)
            res = run_arm(cn, arch, TRN, S2, foldT, args)
        P = res['P']
        pr = per_row(P, y2)
        store[key] = (res, pr)
        tot_h, tie_h, win_h = axis_report(P[hm], y2[hm])
        tot_a, _, _ = axis_report(P, y2)
        base = store.get(want[0])
        dci = '기준' if key == want[0] else RH.ci_str(*RH.paired_ci(base[1][hm], pr[hm]))
        gap = '' if np.isnan(res['ins']) else f'{tot_h - res["ins"]:+.3f}'
        print(f'    holdout 총 {tot_h:.4f} (tie {tie_h:.4f} 승부 {win_h:.4f}) | '
              f'전체 {tot_a:.4f} | {dci} | in-sample {res["ins"]:.4f} 갭 {gap} | '
              f'파라미터 {res["n_par"]:,} | epoch {res["ep"]} | {time.time() - t1:.0f}s')
        rows.append([NM[key], f'{res["n_par"]:,}',
                     '—' if np.isnan(res['ins']) else f'{res["ins"]:.4f}',
                     f'{tot_h:.4f}', f'{tie_h:.4f}', f'{win_h:.4f}', gap or '—',
                     dci, f'{res["ep"]}', f'{time.time() - t1:.0f}s'])

    RH.head('1', '팔별 결과 — 판정은 holdout 5,748 (무접촉). in-sample 갭 = 일반화+인코더 암기')
    RH.table(['구성', '파라미터', '학습행 ll', 'holdout 총', 'tie축', '승부축(행당)',
              '갭(hold−in)', 'Δ vs 기준 (holdout, 양수=이득)', 'epoch', '시간'],
             rows, ['l', 'r', 'r', 'r', 'r', 'r', 'r', 'l', 'r', 'r'])
    print('  기준선 참고치: stage2 전체 Davidson κ=1 = 0.9705. 갭이 큰 팔은 train 표현의')
    print('  인코더 암기 방향에 끌려간 것이므로 wd/drop 을 키워 재시도할 것.')

    att = {k: v for k, v in store.items()
           if v[0]['alpha'] is not None}
    if att:
        RH.head('2', 'readout 어텐션 — 결과별 전문가 α (stage2, fold 평균)')
        for k, (res, _) in att.items():
            AL = res['alpha']
            rows = []
            for lab, m_ in (('전체', np.ones(len(y2), bool)), ('A승', y2 == 0),
                            ('B승', y2 == 1), ('tie', y2 == 2)):
                rows.append([lab] + [f'{AL[m_, j].mean():.3f}' for j in range(AL.shape[1])])
            print(f'  ── {NM[k]}')
            RH.table(['결과'] + list(EXPERTS), rows, ['l'] + ['r'] * len(EXPERTS))
            print()

    agr = {k: v for k, v in store.items() if v[0]['corr'] is not None}
    if agr:
        RH.head('3', '합의도 팔 — 잠재가 자기 로짓을 재발견했나 (stage2, fold 평균 max|ρ|)')
        print('  v1(8.6k행)에서 0.44~0.55 였다. 행이 5.4배로 늘었으니 진단 B 상한(0.95~0.998)에')
        print('  가까워져야 정상이고, 안 가까워지면 rank 나 lr 문제다.\n')
        rows = []
        for k, (res, _) in agr.items():
            c = res['corr']
            rows.append([NM[k]] + [('—' if np.isnan(v) else f'{v:.3f}') for v in c])
        RH.table(['팔', 'ABE승부', 'ABE tie', 'TB승부', 'TB tie', 'Ar승부', 'Ar tie'],
                 rows, ['l'] + ['r'] * 6)

    out = BOOST / f'newhead_train{("_" + args.tag) if args.tag else ""}.npz'
    np.savez_compressed(out, ids=S2['ids'], y=y2, is_hold=hm,
                        **{f'P__{NM[k]}': store[k][0]['P'] for k in store})
    print(f'\nsaved: {out}')


if __name__ == '__main__':
    main()
