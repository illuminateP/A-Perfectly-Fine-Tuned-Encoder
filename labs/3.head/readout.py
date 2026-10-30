"""tie축 헤드 구조 격자 — 체인(라운드별 유닛·앵커 유무) × 판독 × 폭 h.

현행 헤드는 `out = f(x⁺) + f(x⁻)`, `f = w(tanh(W1x) ⊙ σ(W2x))` 다. `w` 가 선형이라
합과 자리를 바꿀 수 있고, 따라서 현행 구조는 사실상 이렇다:

    out = w·( g(x⁺) + g(x⁻) )          g = 은닉 h차원
              └─────┬─────┘
          **정확히 짝인 h차원 표현**  (tie축이 짝 축이므로 이게 옳은 표현이다)

이 파일이 여는 축 셋:

[판독]  대칭화 **뒤** 의 판독을 바꾼다.
        LIN     w(u)                     현행과 동치
        LIN_SK  w([u | 로짓])            로짓 직통 선형 경로. 결합기가 이 선형항을
                                         정확히 상쇄할 수 있으므로 **배포 함수족은 안 바뀐다** —
                                         바뀌는 건 "헤드가 무엇을 학습하느냐"(용량 배분)뿐이다.
        FFN     w(u + W2·gelu(W1·u))     BERT식 블록. 잔차라 초기값에서 LIN 을 **포함** 한다.
        BASE    현행 경로(`sym_fn=None`) 그대로 — 판독 재배치가 무해한지 보는 하네스 대조.
        ! FFN 을 대칭화 **앞** 에 놓으면 FFN(g(x⁺))+FFN(g(x⁻)) ≠ FFN(g(x⁺)+g(x⁻)) 라
          짝 표현 위에서 비선형을 못 만든다. 그 배치(깊이 d2)는 이미 졌다.

[체인]  라운드마다 유닛을 따로 고른다. 앵커를 끌 수도 있다.
        `이름=U1/U2/U3`            앵커 있는 부스팅 — 라운드 k 는 1..k−1 의 **TR 잔차** 를 맞춘다
        `이름=U1/U2/U3:noanchor`   앵커 없음 — 각 망이 y 를 직접 학습(시드만 100씩 어긋남).
                                   열 수·파라미터는 같고 앵커만 없는 **교체 대조** 다.
        `이름=U`                   라운드 수만큼 복제

[폭]    `--hs` 로 은닉 폭을 쓸어본다. 기록에 남은 스윕은 {32,64,128,256}·GAT(σ) 유닛
        시절 한 줄뿐이고 개별 숫자가 없다. σ 가 은닉의 절반을 게이트로 쓰므로
        σ 없는 유닛에서 최적 h 가 다를 수 있다.

전부 tie축만 건드린다. 승부축은 전 칸이 같은 1회 적합을 공유하므로 Δ 는 자동으로
tie축 순수분이다. 위생: 헤드는 TR 46,729 에서만 학습하고(ES 도 TR 의 15%),
stage2 는 결합기 5-fold 교차적합과 평가에만 쓴다.
"""
import argparse
import hashlib
import json
import time

import numpy as np
import torch
import torch.nn as nn

import tie_ladder as TL
import tiegrid as TG

FINAL = TL.FINAL
REF_TAG, REF_VAL = TG.REF_TAG, TG.REF_VAL
ROUNDS = 3


# ══════════════════════════════════════════════ 몸통 (은닉 h차원 표현)
class BodyGAT(nn.Module):
    """tanh(W1x) ⊙ σ(W2x) — 현행 게이트."""

    def __init__(self, d, h):
        super().__init__()
        self.W1, self.W2 = nn.Linear(d, h), nn.Linear(d, h)
        self.out_dim = h

    def forward(self, x):
        return torch.tanh(self.W1(x)) * torch.sigmoid(self.W2(x))


class BodyTT(nn.Module):
    """tanh(W1x) ⊙ tanh(W2x) — 사영 두 개를 각각 포화시켜 곱한다(합성 아님)."""

    def __init__(self, d, h):
        super().__init__()
        self.W1, self.W2 = nn.Linear(d, h), nn.Linear(d, h)
        self.out_dim = h

    def forward(self, x):
        return torch.tanh(self.W1(x)) * torch.tanh(self.W2(x))


class BodyBL(nn.Module):
    """W1x ⊙ W2x — 순수 곱."""

    def __init__(self, d, h):
        super().__init__()
        self.W1, self.W2 = nn.Linear(d, h), nn.Linear(d, h)
        self.out_dim = h

    def forward(self, x):
        return self.W1(x) * self.W2(x)


class BodyTANH(nn.Module):
    """tanh(W1x) — 사영 **하나**, 곱 없음. 단조 활성 족의 대표.

    대칭화 뒤 p=0 이면 tanh(q)+tanh(−q) = 0 이라 홀 성분이 통째로 소멸한다.
    파라미터를 곱 유닛(2dh)과 맞추려고 폭을 2배로 잡는다.
    """

    def __init__(self, d, h):
        super().__init__()
        self.W1 = nn.Linear(d, 2 * h)
        self.out_dim = 2 * h

    def forward(self, x):
        return torch.tanh(self.W1(x))


class BodySQ(nn.Module):
    """(W1x)² — 짝 2차. BILIN 을 W1=W2 로 붕괴시킨 판. 폭 2배."""

    def __init__(self, d, h):
        super().__init__()
        self.W1 = nn.Linear(d, 2 * h)
        self.out_dim = 2 * h

    def forward(self, x):
        return self.W1(x) ** 2


class BodyABS(nn.Module):
    """|W1x| — 짝 1차. 폭 2배."""

    def __init__(self, d, h):
        super().__init__()
        self.W1 = nn.Linear(d, 2 * h)
        self.out_dim = 2 * h

    def forward(self, x):
        return torch.abs(self.W1(x))


class BodyMLP(nn.Module):
    """gelu(W1x) — 1은닉층 FFN 의 몸통. 곱 없음. 폭 2배."""

    def __init__(self, d, h):
        super().__init__()
        self.W1 = nn.Linear(d, 2 * h)
        self.out_dim = 2 * h

    def forward(self, x):
        return torch.nn.functional.gelu(self.W1(x))


BODIES = {'GAT': BodyGAT, 'TANHTANH': BodyTT, 'BILIN': BodyBL, 'TANH': BodyTANH,
          'SQ': BodySQ, 'ABS': BodyABS, 'MLP': BodyMLP}


# ══════════════════════════════════════════════ 판독 (짝 표현 -> 스칼라)
class HeadLin(nn.Module):
    """현행 — 선형 사영 하나."""

    def __init__(self, d_in):
        super().__init__()
        self.w = nn.Linear(d_in, 1)

    def forward(self, u):
        return self.w(u)


class HeadFFN(nn.Module):
    """BERT식 FFN 블록 + 잔차, 그 뒤 선형 판독.  w( u + W2·gelu(W1·u) ),  W1: d→4d

    `w` 0 초기화라 출력은 0에서 시작하고(앵커 계약), 잔차 덕에 초기값에서 블록이 항등이라
    **FFN 팔이 LIN 팔을 포함** 한다 — 지면 표현력이 아니라 최적화·과적합이 원인이다.
    LayerNorm 은 뺐다: 행별 정규화는 짝 표현의 **크기** 를 버리는데 그게 이 축이 재는 양이다.
    """

    def __init__(self, d_in, mult=4):
        super().__init__()
        self.f1 = nn.Linear(d_in, mult * d_in)
        self.f2 = nn.Linear(mult * d_in, d_in)
        self.w = nn.Linear(d_in, 1)

    def forward(self, u):
        return self.w(u + self.f2(torch.nn.functional.gelu(self.f1(u))))


HEADS = {'LIN': HeadLin, 'FFN': HeadFFN}


class SplitNet(nn.Module):
    """몸통 + 판독. `fit_gat` 의 0 초기화 계약(`.w`)을 프로퍼티로 노출한다.

    body_idx  몸통이 보는 입력 열. None 이면 전부(풀링 SVD + 로짓).
              풀링 열만 주면 몸통은 **순수 풀링 기계** 가 되고 로짓은 판독 직통만 탄다.
    skip_idx  판독 입력에 이어붙일 열(원 로짓). None 이면 없음.

    ! `self.w = self.head.w` 로 대입하면 nn.Module 이 같은 파라미터를 **두 번 등록** 해
      옵티마이저가 이중 갱신한다. 프로퍼티는 등록하지 않는다.
    """

    def __init__(self, d, h, body_kind, head_kind, skip_idx, body_idx=None):
        super().__init__()
        self.body_idx = body_idx
        self.body = BODIES[body_kind](d if body_idx is None else len(body_idx), h)
        self.skip_idx = skip_idx
        n_sk = 0 if skip_idx is None else len(skip_idx)
        self.head = HEADS[head_kind](self.body.out_dim + n_sk)

    @property
    def w(self):
        return self.head.w

    def forward(self, x):                     # BASE 팔(원본 대칭화)용 경로
        u = self.body(x if self.body_idx is None else x[:, self.body_idx])
        if self.skip_idx is not None:
            u = torch.cat([u, x[:, self.skip_idx]], 1)
        return self.head(u)


def sym_post(f, Ux, Vx, sgn):
    """판독을 대칭화 **뒤** 로 옮긴 out(). `TL.fit_gat(sym_fn=...)` 에 넘긴다.

        u  = g(x⁺) + sgn·g(x⁻)      정확히 짝(sgn=+1) / 홀(sgn=−1)
        sk = L(x⁺) + sgn·L(x⁻)      홀 로짓은 sgn=+1 에서 정확히 0이 된다
        out = head([u | sk])
    """
    xp = torch.cat([Ux, Vx], 1)
    xm = torch.cat([Ux, -Vx], 1)
    bp = xp if f.body_idx is None else xp[:, f.body_idx]
    bm = xm if f.body_idx is None else xm[:, f.body_idx]
    u = f.body(bp) + sgn * f.body(bm)
    if f.skip_idx is not None:
        u = torch.cat([u, xp[:, f.skip_idx] + sgn * xm[:, f.skip_idx]], 1)
    return f.head(u).squeeze(-1)


# ══════════════════════════════════════════════ 판독 팔
# (이름, 판독종류, 로짓스킵, 대칭화뒤판독, 몸통입력, r1로짓앵커)
#   몸통입력 'all'  = 풀링 SVD + 로짓 8   (현행 — 로짓이 비선형 안으로 들어간다)
#            'pool' = 풀링 SVD 만          (로짓은 판독 직통 전용)
#   r1로짓앵커 True = r1 의 s0 를 로짓 기저 점수(TR 적합)로 — 헤드가 처음부터 로짓 잔차만
#            배운다. 판독 직결의 학습-동역학 효과를 앵커로 대체하는 팔 (§3.5 보정 노트의
#            "기둥식 앵커링" 실측판. s0 는 로짓의 선형 함수라 결합기 무벌점 블록이
#            표현 가능 — 배포 함수족 불변)
ARMS = [('BASE',    'LIN', False, False, 'all',  False),  # 현행 경로 그대로 (하네스 대조)
        ('LIN',     'LIN', False, True,  'all',  False),  # 현행과 동치
        ('LIN_SK',  'LIN', True,  True,  'pool', False),  # ★ §3.6 확정형 — 직결
        ('PURE',    'LIN', False, True,  'pool', False),  # ★ 로짓 0 — 결합기만
        ('ANC',     'LIN', False, True,  'pool', True),   # ★ 로짓 0 + r1 잔차 표적
        ('SK_FFN',  'FFN', True,  True,  'pool', False),  # [u|로짓] 뒤 FFN
        ('LIN_SKB', 'LIN', True,  True,  'all',  False),  # 로짓 양쪽 (구 readout 의 LIN_SK)
        ('FFN_B',   'FFN', False, True,  'all',  False),  # 구 readout 의 FFN
        ('FFN_SKB', 'FFN', True,  True,  'all',  False)]  # 구 readout 의 FFN_SK
ARM_MAP = {a[0]: a for a in ARMS}

CHAINS = {}          # 이름 -> (라운드별 유닛 목록, 앵커여부).  --chains 로 채운다


def make_factory(unit, head_kind, skip_idx, body_idx=None):
    def _f(d, h, _u=unit, _k=head_kind, _s=skip_idx, _b=body_idx):
        return SplitNet(d, h, _u, _k, _s, _b)
    return _f


def chain(A, Kd, h, chain_name, arm, seed, tag, skip_idx, pool_idx, rounds=ROUNDS,
          seed_step=100, r1_cache=None, s0=None):
    """체인 하나 — 라운드마다 열 하나. 자기종결은 끈다(접두사 사다리를 위해).

    앵커 있음: 라운드 k 의 s0 = 1..k−1 누적 TR 점수 → **TR 잔차** 를 맞춘다.
    앵커 없음: s0=None 고정, 시드만 seed_step 씩 어긋난 독립 망 R 개.
    s0: 판독 팔의 r1로짓앵커 플래그가 켜져 있으면 r1 의 앵커로 쓴다 (로짓 기저 TR 점수).

    r1_cache: 라운드 1 은 (r1유닛, h, 판독, 시드) 로 완전히 결정된다 (ANC 의 s0 는
        판독 이름에 묶여 있으므로 키가 갈린다). 같은 r1 을 공유하는 체인들이
        **비트 동일한** 열을 재학습하지 않게 캐시한다.
    """
    units, anchored = CHAINS[chain_name]
    _, head_kind, use_skip, post, body_src, anc_r1 = ARM_MAP[arm]
    sk = skip_idx if use_skip else None
    bi = pool_idx if body_src == 'pool' else None
    assert not anc_r1 or s0 is not None, f'{arm}: r1 앵커 팔인데 s0 가 없다'
    cu_t = np.ascontiguousarray(np.hstack([A['CUt'][:, :Kd], A['Lu_tr']]), np.float32)
    cv_t = np.ascontiguousarray(np.hstack([A['CVt'][:, :Kd], A['Lv_tr']]), np.float32)
    cu_s = np.ascontiguousarray(np.hstack([A['CUs'][:, :Kd], A['Lu_s2']]), np.float32)
    cv_s = np.ascontiguousarray(np.hstack([A['CVs'][:, :Kd], A['Lv_s2']]), np.float32)
    z, cols, prev = (s0 if anc_r1 else None), [], None
    for rd in range(1, rounds + 1):
        sd = seed + seed_step * (rd - 1) if not anchored else seed
        key = (units[0], h, arm, seed) if rd == 1 else None
        hit = r1_cache is not None and key is not None and key in r1_cache
        if hit:
            r_s2, aux = r1_cache[key]
        else:
            net = make_factory(units[rd - 1], head_kind, sk, bi)
            r_s2, aux = TL.fit_gat(cu_t, cv_t, A['y01m'], z if anchored else None,
                                   True, sd, cu_s, cv_s, hidden=h, net=net,
                                   sym_fn=(sym_post if post else None))
            if key is not None and r1_cache is not None:
                r1_cache[key] = (r_s2, aux)
        cols.append(r_s2)
        gain = None if prev is None else prev - aux['es_ll']
        print(f'      [{tag}] r{rd}: es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]}'
              f' | {units[rd - 1]}' + (' | r1 캐시' if hit else '')
              + (' | s0앵커' if rd == 1 and anc_r1 else '')
              + ('' if anchored else f' | 독립 seed {sd}')
              + (f' gain {gain:+.5f}' if gain is not None else ''))
        prev, z = aux['es_ll'], aux['z_tr']
    return np.column_stack(cols)


def evaluate(U, win, G, n_gat):
    """결합기 재적합 -> holdout·clean 행손실. tiegrid.eval_cell 의 tie축 경로와 같다."""
    X = np.hstack([U['tie_base'], G[:, :n_gat]])
    n_logit = U['tie_base'].shape[1]
    rows = U['cu']
    y01 = (U['y'] == 2).astype(np.float64)
    z, dep = TL.combiner(X, y01, U['fold'], rows, bias=True, n_logit=n_logit)
    ll = TL.rowloss(z[rows], y01[rows])
    row, lt, _ = TL.compose_ll(U, ll, win['ll'], rows, win['rows'])
    row_all = TG.compose_all(U, ll, win['ll'], rows, win['rows'])
    return row, row_all, dict(tot=float(row.mean()), tot_all=float(row_all.mean()),
                              tie=lt, n_col=X.shape[1], lam=dep['lam'],
                              w_g=[float(dep['v'][n_logit + i]) for i in range(n_gat)])


def parse_chains(spec, rounds):
    """'이름=U1/U2/U3[:noanchor]' 쉼표 목록 -> CHAINS."""
    for s in (x for x in spec.split(',') if x):
        nm, _, body = s.partition('=')
        body, _, flag = body.partition(':')
        units = [u for u in body.split('/') if u]
        assert nm and units, f'--chains 형식은 이름=U1/U2/U3[:noanchor] — {s}'
        assert set(units) <= set(BODIES), f'{s}: 모르는 유닛 {set(units) - set(BODIES)}'
        assert flag in ('', 'noanchor'), f'{s}: 모르는 플래그 {flag!r}'
        if len(units) == 1:
            units = units * rounds
        assert len(units) == rounds, f'{s}: 유닛 {len(units)}개인데 --rounds {rounds}'
        CHAINS[nm] = (units, flag != 'noanchor')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bundle', default='temp/heads_repro')
    ap.add_argument('--ref', default='temp/tie_ladder_es7k.rowloss.npz')
    ap.add_argument('--k', type=int, default=512)
    ap.add_argument('--hs', default='128', help='은닉 폭 목록')
    ap.add_argument('--chains', required=True,
                    help="'이름=U1/U2/U3[:noanchor]' 쉼표 목록. 유닛 1개면 라운드 수만큼 복제")
    ap.add_argument('--readouts', default='LIN')
    ap.add_argument('--base-chain', default='',
                    help="현행 경로(BASE) 대조를 붙일 체인 하나. ''면 생략")
    ap.add_argument('--ref-chain', default='',
                    help='체인 대조표의 기준 체인. 비우면 --chains 의 첫 번째')
    ap.add_argument('--ref-h', type=int, default=0,
                    help='폭 대조표의 기준 h. 0이면 --hs 의 첫 번째')
    ap.add_argument('--no-r1-cache', action='store_true',
                    help='r1 재사용을 끈다 (같은 r1 을 매번 다시 학습 — 검증용)')
    ap.add_argument('--seeds', default='42,1,2,3,4')
    ap.add_argument('--rounds', type=int, default=ROUNDS)
    ap.add_argument('--out', default='temp/readout')
    args = ap.parse_args()

    R = args.rounds
    parse_chains(args.chains, R)
    names = [x.partition('=')[0] for x in args.chains.split(',') if x]
    arms = [a for a in args.readouts.split(',') if a]
    hs = [int(x) for x in args.hs.split(',') if x]
    seeds = [int(s) for s in args.seeds.split(',') if s]
    assert set(arms) <= set(ARM_MAP), f'모르는 판독 {set(arms) - set(ARM_MAP)}'
    assert not args.base_chain or args.base_chain in CHAINS, '--base-chain 이 --chains 에 없다'
    ref_chain = args.ref_chain or names[0]
    assert ref_chain in CHAINS, f'--ref-chain {ref_chain} 이 --chains 에 없다'
    ref_h = args.ref_h or hs[0]
    assert ref_h in hs, f'--ref-h {ref_h} 가 --hs 에 없다'

    print('[적재]')
    U = TL.build(FINAL / args.bundle)
    win = TL.fit_win(U)
    A = TL.gat_assets(U)
    rl = np.load(FINAL / args.ref)
    assert REF_TAG in rl.files, f'{args.ref} 에 {REF_TAG} 없음'
    ref_row = rl[REF_TAG]
    print(f'  기준 {REF_TAG} = {ref_row.mean():.5f} (기록 {REF_VAL:.5f}, '
          f'Δ {ref_row.mean() - REF_VAL:+.5f}) | 기둥 없음 · 승부축 공유')

    # 입력 배치 x = [CUt K | Lu n_lu | CVt K | Lv n_lv]
    K, n_lu, n_lv = args.k, A['Lu_tr'].shape[1], A['Lv_tr'].shape[1]
    idx_lu = list(range(K, K + n_lu))
    idx_lv = list(range(2 * K + n_lu, 2 * K + n_lu + n_lv))
    skip_idx = idx_lu + idx_lv
    pool_idx = list(range(0, K)) + list(range(K + n_lu, 2 * K + n_lu))
    assert len(pool_idx) == 2 * K and not set(pool_idx) & set(skip_idx), '풀링/로짓 열 분리'
    print(f'  로짓 열 짝 {idx_lu} · 홀 {idx_lv} (홀은 대칭화에서 정확히 0) | '
          f'풀링 열 {len(pool_idx)}개')

    plan = [(c, a, h) for h in hs for c in names for a in arms]
    if args.base_chain:
        plan.insert(0, (args.base_chain, 'BASE', hs[0]))
    print(f'\n{"="*78}\n[계획] {len(plan)}구성 x {len(seeds)}시드 = {len(plan)*len(seeds)} 체인'
          f'  (K{K} · {R}라운드 · 자기종결 끔)\n{"="*78}')
    for nm in names:
        u, anc = CHAINS[nm]
        print(f'  {nm:9s} {" / ".join(u):34s} 앵커 {"O (잔차학습)" if anc else "X (독립학습)"}')
    for a in dict.fromkeys([p[1] for p in plan]):
        _, hk, sk, po, bs, anc1 = ARM_MAP[a]
        print(f'  판독 {a:8s} {hk:4s} · 몸통입력 '
              f'{"풀링+로짓 " + str(2 * K + n_lu + n_lv) if bs == "all" else "풀링만 " + str(2 * K):16s}'
              f'· 로짓직통 {"O" if sk else "X"} · 대칭화뒤판독 {"O" if po else "X (현행 경로)"}'
              + (' · r1 로짓앵커' if anc1 else ''))
    print(f'  폭 h {hs}')

    s0 = None
    if any(ARM_MAP[p[1]][5] for p in plan):
        X0 = A['Lu_tr'].astype(np.float64)
        v0, ok0 = TL._fit(X0, A['y01m'], np.zeros(X0.shape[1]), True)
        s0 = X0 @ v0[:X0.shape[1]] + v0[X0.shape[1]]
        print(f'  r1 로짓앵커 s0 — tie로짓 {X0.shape[1]}열 TR 적합(절편 포함) '
              f'w {np.round(v0[:X0.shape[1]], 3)} {"수렴" if ok0 else "** 미수렴"}')

    res, rows, rows_all = {}, {}, {}
    r1_cache = None if args.no_r1_cache else {}
    # 결합기 칸 캐시 — 열 내용 해시로 건다. r1 을 공유하는 체인들의 g1 은 비트 동일하므로
    # (예: GG·GT·GB 의 g1) 아홉 번 재적합할 이유가 없다. 내용 해시라 항상 안전하다.
    cell_cache, n_hit = {}, 0
    for c, a, h in plan:
        for sd in seeds:
            tag = f'{c}-{a}-h{h}-s{sd}'
            print(f'  [{tag}]')
            t0 = time.time()
            G = chain(A, K, h, c, a, sd, tag, skip_idx, pool_idx,
                      rounds=R, r1_cache=r1_cache, s0=s0)
            secs = time.time() - t0
            for k in range(1, R + 1):
                t = f'{tag}-g{k}'
                ck = hashlib.sha1(np.ascontiguousarray(G[:, :k]).tobytes()).hexdigest()
                hit = ck in cell_cache
                if hit:
                    row, row_all, d0 = cell_cache[ck]
                    n_hit += 1
                else:
                    row, row_all, d0 = evaluate(U, win, G, k)
                    cell_cache[ck] = (row, row_all, d0)
                m, lo, hi = TL.paired_ci(row, ref_row)
                d = dict(d0)
                d.update(chain=c, arm=a, h=h, seed=sd, n_gat=k, chain_sec=round(secs, 1),
                         units=CHAINS[c][0], anchored=CHAINS[c][1],
                         d_mean=m, d_lo=lo, d_hi=hi)
                res[t], rows[t], rows_all[t] = d, row, row_all
                print(f'    g{k} -> holdout {d["tot"]:.5f} · clean {d["tot_all"]:.5f}  '
                      f'Δ {TL.ci_str(m, lo, hi)}  (λ {d["lam"]:g})'
                      + ('  [칸 캐시]' if hit else ''))
    print(f'\n  칸 캐시 적중 {n_hit} / {len(plan) * len(seeds) * R} '
          f'(같은 열 집합의 재적합을 건너뛴 횟수)')

    # ── 저장은 표보다 **먼저**. 요약 계산이 죽어도 체인 결과를 잃지 않는다.
    op = FINAL / args.out
    np.savez_compressed(op.with_suffix('.rowloss.npz'), **rows, ref=ref_row,
                        **{f'ALL__{k}': v for k, v in rows_all.items()})
    with open(op.with_suffix('.json'), 'w') as f:
        json.dump({'ref_tag': REF_TAG, 'ref': float(ref_row.mean()), 'cells': res},
                  f, ensure_ascii=False, indent=1)
    print(f'\n저장: {op}.rowloss.npz / {op}.json')

    mean = lambda c, a, h, k: np.mean([res[f'{c}-{a}-h{h}-s{s}-g{k}']['tot'] for s in seeds])
    vec = lambda c, a, h, k: np.array([res[f'{c}-{a}-h{h}-s{s}-g{k}']['tot'] for s in seeds])
    rmat = lambda c, a, h, k: np.mean([rows[f'{c}-{a}-h{h}-s{s}-g{k}'] for s in seeds], axis=0)

    def cmp(c1, a1, h1, c2, a2, h2, k):
        """c1 − c2 를 시드 paired 로. (행부트스트랩 CI, 시드평균 Δ, 시드부호)"""
        m, lo, hi = TL.paired_ci(rmat(c1, a1, h1, k), rmat(c2, a2, h2, k))
        d = vec(c1, a1, h1, k) - vec(c2, a2, h2, k)
        return f'{TL.ci_str(m, lo, hi)}  시드 {d.mean():+.5f} {int((d < 0).sum())}/{len(d)} 유리'

    if R > 1:
        print(f'\n{"="*78}\n[접두사 사다리] 구성별 · 라운드별 {len(seeds)}시드 평균\n{"="*78}')
        for c, a, h in plan:
            print(f'  [{c}·{a}·h{h}]  {"접두사":10s}{"tie축":>10s}{"holdout":>10s}{"sd":>9s}'
                  f'{"clean":>10s}   직전 대비')
            prev = None
            for k in range(1, R + 1):
                tg = [f'{c}-{a}-h{h}-s{s}-g{k}' for s in seeds]
                th = [res[t]['tot'] for t in tg]
                cur = np.mean([rows[t] for t in tg], axis=0)
                line = (f'{"":8s}{"g1..g" + str(k):10s}'
                        f'{np.mean([res[t]["tie"] for t in tg]):10.5f}{np.mean(th):10.5f}'
                        f'{np.std(th, ddof=1):9.5f}'
                        f'{np.mean([res[t]["tot_all"] for t in tg]):10.5f}')
                print(line + ('   —' if prev is None
                              else '   ' + TL.ci_str(*TL.paired_ci(prev, cur))))
                prev = cur

    print(f'\n{"="*78}\n[체인 대조] 기준 {ref_chain} · {len(seeds)}시드 · 같은 시드끼리 짝지음'
          f'\n{"="*78}')
    for a in arms:
        for h in hs:
            print(f'  판독 {a} · h{h}')
            print(f'{"":4s}{"체인":10s}' + ''.join(f'{"g"+str(k):>10s}' for k in range(1, R + 1))
                  + f'   g{R} 기준대비 (음수 = 그 체인이 낫다)')
            for c in names:
                line = f'{"":4s}{c:10s}' + ''.join(f'{mean(c, a, h, k):10.5f}'
                                                   for k in range(1, R + 1))
                print(line + ('   기준' if c == ref_chain
                              else '   ' + cmp(c, a, h, ref_chain, a, h, R)))

    if len(arms) >= 2:
        print(f'\n{"="*78}\n[판독 대조] 같은 체인 안에서 {arms[0]} 대비\n{"="*78}')
        for h in hs:
            for c in names:
                for a in arms[1:]:
                    print(f'  h{h} {c:10s} {a:8s} {mean(c, a, h, R):.5f}   '
                          + cmp(c, a, h, c, arms[0], h, R))

    if len(hs) >= 2:
        print(f'\n{"="*78}\n[폭 대조] 같은 체인·판독 안에서 h{ref_h} 대비\n{"="*78}')
        for c in names:
            for a in arms:
                print(f'  {c:10s} {a:8s}  ' + '  '.join(f'h{h}:{mean(c, a, h, R):.5f}'
                                                        for h in hs))
                for h in hs:
                    if h != ref_h:
                        print(f'{"":22s}h{h:<5d} ' + cmp(c, a, h, c, a, ref_h, R))

    if args.base_chain:
        b = args.base_chain
        print(f'\n  [하네스 대조] {b}  BASE(현행 경로) {mean(b, "BASE", hs[0], R):.5f}'
              f' -> {arms[0]}(재배치) {mean(b, arms[0], hs[0], R):.5f}'
              f'   {cmp(b, arms[0], hs[0], b, "BASE", hs[0], R)}')
        print('    두 팔은 수학적으로 같고 절편만 2b→b 로 재매개화된다. 0 이 아니면')
        print('    아래 판독 Δ 에서 이 값을 먼저 빼고 읽는다.')

    print(f'\n{"="*78}\n[순위] Δ vs {REF_TAG} ({ref_row.mean():.5f}) '
          f'— 음수 = 그 칸이 낫다 / 문턱 0.0005\n{"="*78}')
    print(f'{"칸":30s}{"총 holdout":>11s}{"열":>5s}{"λ":>8s}  Δ')
    for t in sorted(res, key=lambda x: res[x]['tot']):
        d = res[t]
        print(f'{t:30s}{d["tot"]:11.5f}{d["n_col"]:5d}{d["lam"]:8g}  '
              f'{TL.ci_str(d["d_mean"], d["d_lo"], d["d_hi"])}')
    print(f'\n(저장은 표 앞에서 이미 끝났다: {op}.rowloss.npz / {op}.json)')


if __name__ == '__main__':
    main()
