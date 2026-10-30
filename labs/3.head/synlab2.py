"""synlab2.py — u/v 분해의 **제거 실험** (synlab.py 의 설계 교정).

synlab.py 의 결함: NOSYM/RAW 가 유닛(양가지 전입력·절편 → 승부 파라미터 2배)·SVD(joint 1024)·정규화·결합기
절편을 함께 바꿨다 — 제거가 아니라 재설계라 "u/v 가 필요한가"에 답하지 못한다. 여기서는 **같은 망 · 같은
파라미터 · 같은 하이퍼파라미터 · 같은 로짓 기저 · 같은 결합기** 를 두고, 풀링 좌표 (u,v) ↔ (A,B) 와 대칭화
on/off 두 요인만 바꾼다.

2×2 (층 1 학습 · 시드당 1회 · 헤드 파라미터 전 팔 동일: tie 262,537 ×2R / win 262,409 ×4)
                    대칭화 on                              대칭화 off
    (u,v) 좌표      BASE     노트북 현행                    UV-NOSYM  sym_post 만 제거 (반사 없이 한 번 통과)
    (A,B) 좌표      AB-SYM   f(A,B) ∓ f(B,A)                AB        u/v 도 대칭화도 없음 — 순수 제거

(A,B) 좌표의 구성 — (u,v) 자리에 (A,B) 를 **그대로** 넣는다
    tie   블록별 SVD-512 (BASE 의 u-SVD · v-SVD 자리). 기저·통계는 [A;B] 행을 쌓아 **한 벌**로 적합한다 —
          A·B 는 같은 종류의 벡터이고, 그래야 자리 맞바꿈이 좌표에서 정확히 블록 교환이 된다. 몸통 [C_A 512 | C_B 512].
    win   소스별 raw (A,B) 1,024 — z-score 통계 역시 [A;B] 쌓아 한 벌. 유닛은 같은 WinNet: σ(G·A)⊙tanh(V·B).
    유일한 강제 편차 — BASE 의 홀 블록 무중심(RMS) 규칙은 반대칭 보존이 목적이라 홀 블록이 없으면 근거가 없다.
    두 블록 모두 중심화 z-score 다.
    대칭화 off 팔은 반사가 없으니 Lu/Lv 8칸이 전부 판독에 살아 있다. 판독 절편(win 0 동결) · 결합기(tie 절편 O /
    win 절편 X · 무중심 RMS) 는 BASE 그대로. 로짓은 (m,t) 기저·정규화 그대로 — 층 2 로짓 기저의 효과는
    synlab BASE/Lraw (+0.00116 ★ 5/5) 로 이미 따로 분리돼 있다.

층 2 파생 (학습 없음 · 같은 헤드 열에 결합기만 다시)
    /b        대칭화 off 팔만 — W4 가 홀이 아니라 평균이 0 이 아닐 수 있다. "절편 없는 승부 결합기가 불리했다"는
              반론을 막기 위해 승부 결합기 절편 O (전열 중심화) 로 한 번 더 잰다.
    /percol   전문가별 격자 — 로짓 열마다 λ_j (2묶음 최적에서 좌표하강 2회) · 헤드는 묶음 하나   (penfold 규약)
    /ard      1/w² 재가중 5회 (§3.7 의 ard) — 균일 q0 를 격자에서 고른 뒤 전 열 적응                   (penfold 규약)
    축마다 선호 방식이 다를 수 있어 **축별로** 잰다 — tot 는 축 가법이라 (tie 방식, win 방식) 어떤 조합도 표에서
    합성된다. 축별 최적을 holdout 에서 고르는 것은 낙관 선택이므로 그 줄은 참고로만 찍고, 판정은 blk2 대비
    paired Δ (시드 5 · 행 부트스트랩 CI) 로 한다. BASE 에서는 BA +0.00018 · BR +0.00061★ 로 이미 기각된 방식들이다
    (penfold, 2026-08-21) — 대칭화 off 팔은 열 구조가 달라 다시 잰다 (사용자 결정, 2026-08-22).
    세 방식 모두 같은 격자 {0,…,1e5} · 같은 안쪽 3-fold · 배포 = fold 성분별 중앙값 (ard 는 q0 중앙값).

결합기 = 노트북 셀 60 `_blk2_combiner` (synlab.blk2) · 격자 {0,…,1e5}². BLAS 스레드를 1 로 제한한다 —
4k×7 적합에서 16스레드 오버헤드가 37× (41 ms → 1.1 ms/적합) 였다. 수치는 합산 순서 차이 수준(≤1e-12).

읽는 법 — tot = tie ll + 0.691 × win ll (holdout 5,748). 양수 Δ = 그 팔이 나쁨. 문턱 (plan §5): paired bootstrap
95% CI 0 배제 + |Δ| ≥ 0.0005 + 시드 5/5. 요인 분해: 좌표 효과 = AB-SYM − BASE · 대칭화 효과 = UV-NOSYM − BASE ·
상호작용 = AB − AB-SYM − UV-NOSYM + BASE.

검증 — BASE-s* 는 synlab.json / dest_fold.json 의 BASE-s* 와 일치해야 한다 (tie 비트 · win 은 synlab 과).
       AB-SYM 배선 가드: swap_post 가 tie 짝 / win 홀을 정확히 만드는지 난수 입력으로 확인한다.

실행:  cd final/temp && python synlab2.py                         (ML2 · GPU · 4팔 × 5시드 ≈ 40분)
       nohup python -u synlab2.py > log_synlab2.txt 2>&1 &   tail -f log_synlab2.txt
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dest_fold as D                                                             # noqa: E402
from dest_fold import (FINAL, MODELS, SEED, H, TIE_ROUNDS, SRCS, DEVICE, R_NZ,     # noqa: E402
                       REF_TOL, rowloss, assemble, score, build, load_train_logits,
                       load_svd_cache, zstats, raw_uv, TieNet, WinNet, sym_post)
import synlab as S                                                                # noqa: E402
from synlab import (LAMS, SD_TIE, SD_WIN, blk2, enc_logits, bases, arith_mean,     # noqa: E402
                    assets_uv, plain_post, sym_even, sym_odd, fit_head, _split_idx)
import penfold as PF                                                              # noqa: E402

PF.GRID = list(LAMS)                  # 세 벌점 방식이 같은 격자 {0,…,1e5} 를 쓴다 (penfold 기본은 1e4 상단)
KINDS = ('percol', 'ard')             # blk2 는 본 팔 (synlab.blk2 · 노트북 규칙), 나머지 둘은 penfold.combiner
K_AB = 512                            # 블록별 SVD 랭크 — BASE 의 u/v 512 와 같다
SVD_AB = FINAL / 'temp/head_cache/synlab2_svdAB512.npz'
ARMS = {                              # 팔: (좌표, 대칭화)
    'BASE':     ('uv', True),
    'UV-NOSYM': ('uv', False),
    'AB-SYM':   ('ab', True),
    'AB':       ('ab', False),
}
LABEL = {
    'BASE':       'BASE — (u,v) + 대칭화 (노트북 현행)',
    'UV-NOSYM':   'UV-NOSYM — (u,v) 좌표 유지 · sym_post 만 제거 (같은 망, 반사 없이 한 번 통과)',
    'AB-SYM':     'AB-SYM — (A,B) 좌표 · 대칭화 f(A,B) ∓ f(B,A) 유지 (좌표 효과만)',
    'AB':         'AB — (A,B) 좌표 · 대칭화 없음 (u/v 순수 제거)',
    'UV-NOSYM/b': 'UV-NOSYM + 층2 승부 결합기 절편·중심화',
    'AB/b':       'AB + 층2 승부 결합기 절편·중심화',
}
KIND_LABEL = {'blk2': '2묶음 [λ_L 로짓 | λ_g 헤드] (노트북)', 'percol': '전문가별 격자 (열별 λ_j · 좌표하강)',
              'ard': 'ard (1/w² 재가중 5회)'}
for _a in ARMS:
    for _k in KINDS:
        LABEL[f'{_a}/{_k}'] = f'{_a} + 층2 {KIND_LABEL[_k]}'
AB_SPEC = [('abe2__checkpoint-3000', ('ea',), ('eb',)),
           ('tb2__checkpoint-4000', ('a_f', 'b_s'), ('b_f', 'a_s')),
           (f'abe1__checkpoint-{D.ABE1_STEP}', ('ea',), ('eb',)),
           (f'tb1__checkpoint-{D.TB1_STEP}', ('x_f',), ('x_s',))]


# ══════════════════════════════════════════ (A,B) 좌표
def ab_assets():
    """(A,B) 공급 — 소스별 raw (A,B) z-score (승부) + 블록별 SVD-512 화이트닝 좌표 (tie, 캐시).

    A/B 판은 dest_fold.raw_uv 와 같은 뷰평균 접기 (tb2: A=(a_f+b_s)/2 · B=(b_f+a_s)/2).
    z-score 통계 · SVD 기저 · 성분 RMS 는 전부 [A;B] 행을 쌓아 한 벌로 적합한다 (TR 전용) —
    그래야 자리 맞바꿈 (A↔B) 이 정규화·사영 뒤에도 정확히 블록 교환이다 (AB-SYM 의 전제).
    """
    At, Bt, As, Bs = [], [], [], []
    for tag, ka, kb in AB_SPEC:
        zt, zs = np.load(MODELS / f'pooled/{tag}__train.npz'), np.load(MODELS / f'pooled/{tag}.npz')
        g = lambda z, pre, ks: sum(z[f'{pre}_{k}'].astype(np.float32) for k in ks) / len(ks)
        c = lambda z, ks: sum(np.vstack([z[f'valid_{k}'], z[f'hold_{k}']]).astype(np.float32)
                              for k in ks) / len(ks)
        At.append(g(zt, 'train', ka))
        Bt.append(g(zt, 'train', kb))
        As.append(c(zs, ka))
        Bs.append(c(zs, kb))
    At, Bt, As, Bs = (np.hstack(x) for x in (At, Bt, As, Bs))     # (46729, 4096) ×2 · (10748, 4096) ×2
    assert At.shape == Bt.shape == (46729, 4096) and As.shape == Bs.shape == (10748, 4096), '[A|B] 형상'
    n_tr = At.shape[0]
    St = np.vstack([At, Bt])                                        # (93458, 4096) — 통계·기저 적합용
    mu = St.mean(0, dtype=np.float64)
    sc = np.clip(St.std(0, dtype=np.float64), 1e-9, None)
    del St
    for X in (At, Bt, As, Bs):                                      # 제자리 z-score (두 블록 모두 중심화)
        X -= mu.astype(np.float32)
        X /= sc.astype(np.float32)
    SUP = {}
    for i, s in enumerate(SRCS):
        sl = slice(i * 1024, (i + 1) * 1024)
        SUP[s] = dict(a_tr=np.ascontiguousarray(At[:, sl]), b_tr=np.ascontiguousarray(Bt[:, sl]),
                      a_s2=np.ascontiguousarray(As[:, sl]), b_s2=np.ascontiguousarray(Bs[:, sl]))
    print(f'  (A,B) 소스별 공급 — 4소스 × (A,B) 1,024 z-score (통계 [A;B] 쌓음)  [TR {n_tr:,}행]')

    if SVD_AB.exists():
        z = np.load(SVD_AB)
        assert z['CA_tr'].shape == (n_tr, K_AB) and z['CA_s2'].shape == (10748, K_AB), 'SVD 캐시 형상'
        assert float(np.abs(z['mu'] - mu).max()) < 1e-5 and float(np.abs(z['sc'] - sc).max()) < 1e-5, \
            'SVD 캐시의 z-score 통계가 현재 풀링과 다르다 — 캐시를 지워라'
        C = {k: z[k] for k in ('CA_tr', 'CB_tr', 'CA_s2', 'CB_s2')}
        print(f'  블록별 SVD-{K_AB} 캐시 {SVD_AB.name}  보존 분산 {float(z["kept"]):.3f}')
    else:
        t0 = time.time()
        X64 = np.vstack([At, Bt]).astype(np.float64)               # (93458, 4096) · 이미 z-score
        mu2 = X64.mean(0)                                          # ≈ 0 — svd_keep 의 중심화 절차와 동일
        X64 -= mu2
        total = float((X64 ** 2).sum())
        _, Sv, Vt_ = randomized_svd(X64, n_components=K_AB, random_state=SEED)
        W = Vt_.T
        Ct = X64 @ W
        scc = np.clip(np.sqrt((Ct ** 2).mean(0)), 1e-9, None)     # 화이트닝 — 성분별 RMS ([A;B] 행)
        proj = lambda X: (((X.astype(np.float64) - mu2) @ W) / scc).astype(np.float32)
        C = dict(CA_tr=(Ct[:n_tr] / scc).astype(np.float32), CB_tr=(Ct[n_tr:] / scc).astype(np.float32),
                 CA_s2=proj(As), CB_s2=proj(Bs))
        kept = float((Sv ** 2).sum() / total)
        np.savez(SVD_AB, **C, mu=mu, sc=sc, mu2=mu2, W=W, c_sc=scc, K_AB=K_AB, kept=kept)
        del X64, Ct
        print(f'  블록별 SVD-{K_AB} 계산 {(time.time() - t0) / 60:.1f}분 · 보존 분산 {kept:.3f} → 캐시 {SVD_AB.name}')
    del At, Bt, As, Bs
    gc.collect()
    return SUP, C


def assets_ab(C, TR, sd_tr, sd_s2):
    """AB 팔 헤드 재료 — 좌표만 (C_A, C_B) 로 바꾸고 로짓 (m,t)·정규화는 assets_uv 와 같다."""
    A = assets_uv({'CUt': C['CA_tr'], 'CVt': C['CB_tr'], 'CUs': C['CA_s2'], 'CVs': C['CB_s2']}, TR, sd_tr, sd_s2)
    return A


# ══════════════════════════════════════════ (A,B) 좌표의 대칭화
def swap_post(f, Ux, Vx, sgn, Ku, Kv):
    """(A,B) 좌표의 대칭화 — x = [A_pool Ku | Lu | B_pool Kv | Lv] · 자리 맞바꿈 = A_pool↔B_pool · Lu 불변 · Lv 부호 반전.

    u = g(x) + sgn·g(swap x) · sk = L(x) + sgn·L(swap x). sgn=+1(tie) 이면 Lv 칸이, −1(win) 이면 Lu 칸이 정확히 0 —
    sym_post 와 같은 칸이 죽는다. (u,v) 좌표의 반사 (u, −v) 와 같은 연산을 (A,B) 좌표에서 한 것.
    """
    assert Ku == Kv, 'A/B 블록 폭이 같아야 교환이 성립한다'
    x = torch.cat([Ux, Vx], 1)
    xs = torch.cat([Vx[:, :Kv], Ux[:, Ku:], Ux[:, :Ku], -Vx[:, Kv:]], 1)
    u = f.body(x[:, f.body_idx]) + sgn * f.body(xs[:, f.body_idx])
    u = torch.cat([u, x[:, f.skip_idx] + sgn * xs[:, f.skip_idx]], 1)
    return f.head(u).squeeze(-1)


def out_fn_for(coords, sym, even, Ku, Kv):
    if not sym:
        return plain_post
    if coords == 'uv':
        return sym_even if even else sym_odd
    return lambda f, U, V, _s=(+1 if even else -1), _ku=Ku, _kv=Kv: swap_post(f, U, V, _s, _ku, _kv)


# ══════════════════════════════════════════ 헤드 — 같은 망, 판독 경로만 팔별
def tie_head(A, seed, coords, sym):
    Ku, nu, Kv, nv = A['Up_tr'].shape[1], A['Us_tr'].shape[1], A['Vp_tr'].shape[1], A['Vs_tr'].shape[1]
    sk, pool = _split_idx(Ku, nu, Kv, nv)
    ct = np.ascontiguousarray(np.hstack([A['Up_tr'], A['Us_tr']]), np.float32)
    cv = np.ascontiguousarray(np.hstack([A['Vp_tr'], A['Vs_tr']]), np.float32)
    cs = np.ascontiguousarray(np.hstack([A['Up_s2'], A['Us_s2']]), np.float32)
    vs = np.ascontiguousarray(np.hstack([A['Vp_s2'], A['Vs_s2']]), np.float32)
    out_fn = out_fn_for(coords, sym, True, Ku, Kv)
    z, cols, states = None, [], []
    for rd, unit in enumerate(('TANHTANH', 'BILIN')[:TIE_ROUNDS], 1):
        net = lambda d, h, _u=unit: TieNet(d, h, _u, sk, pool)
        r, aux = fit_head(ct, cv, A['y01m'], z, seed, cs, vs, net, out_fn)
        cols.append(r)
        states.append(aux['state'])
        z = aux['z_tr']
        print(f'      tie r{rd} {unit:9s} es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]}')
    return np.column_stack(cols), states


def win_head(A, SUP, seed, coords, sym):
    """소스별 4체인 · 같은 WinNet (게이트 = 블록 1 · 값 = 블록 2 무절편 · 판독 절편 0 동결). 블록 1/2 = (u,v) 또는 (A,B)."""
    nu, nv = A['Us_tr'].shape[1], A['Vs_tr'].shape[1]
    m = A['y_tr'] != 2
    y01w = (A['y_tr'] == 0).astype(np.float64)[m]
    k1, k2, k3, k4 = ('u_tr', 'v_tr', 'u_s2', 'v_s2') if coords == 'uv' else ('a_tr', 'b_tr', 'a_s2', 'b_s2')
    cols, states = [], []
    for si, s in enumerate(SRCS):
        R = SUP[s]
        du, dv = R[k1].shape[1], R[k2].shape[1]
        sk, pool = _split_idx(du, nu, dv, nv)
        cu_t = np.ascontiguousarray(np.hstack([R[k1][m], A['Us_tr'][m]]), np.float32)
        cv_t = np.ascontiguousarray(np.hstack([R[k2][m], A['Vs_tr'][m]]), np.float32)
        cu_s = np.ascontiguousarray(np.hstack([R[k3], A['Us_s2']]), np.float32)
        cv_s = np.ascontiguousarray(np.hstack([R[k4], A['Vs_s2']]), np.float32)
        net = lambda d, h, _s=sk, _p=pool, _du=du: WinNet(d, h, _s, _p, _du)
        out_fn = out_fn_for(coords, sym, False, du, dv)
        r_s2, aux = fit_head(cu_t, cv_t, y01w, None, seed + si, cu_s, cv_s, net, out_fn)
        cols.append(r_s2)
        states.append(aux['state'])
        print(f'      win r1 {s:5s} es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]}')
        del cu_t, cv_t, cu_s, cv_s
    gc.collect()
    return np.column_stack(cols), states


# ══════════════════════════════════════════ 배선 가드
@torch.no_grad()
def wiring_ab():
    """AB-SYM 의 swap_post 가 tie 짝 / win 홀을 정확히 만드는지, 대칭화 off 는 위반이 남는지 — 난수 입력."""
    g = torch.Generator().manual_seed(0)
    sk, pool = _split_idx(512, 5, 512, 3)
    tie = TieNet(1032, H, 'TANHTANH', sk, pool)
    nn.init.normal_(tie.w.weight)
    Ux, Vx = torch.randn(64, 517, generator=g), torch.randn(64, 515, generator=g)
    # 자리 맞바꿈: A_pool↔B_pool · Lu 불변 · Lv 반전 → (Ux', Vx')
    Ux2 = torch.cat([Vx[:, :512], Ux[:, 512:]], 1)
    Vx2 = torch.cat([Ux[:, :512], -Vx[:, 512:]], 1)
    ev = float((swap_post(tie, Ux, Vx, +1, 512, 512) - swap_post(tie, Ux2, Vx2, +1, 512, 512)).abs().max())
    ev0 = float((plain_post(tie, Ux, Vx) - plain_post(tie, Ux2, Vx2)).abs().max())
    skw, poolw = _split_idx(1024, 5, 1024, 3)
    win = WinNet(2056, H, skw, poolw, 1024)
    nn.init.normal_(win.w.weight)
    Uw, Vw = torch.randn(64, 1029, generator=g), torch.randn(64, 1027, generator=g)
    Uw2 = torch.cat([Vw[:, :1024], Uw[:, 1024:]], 1)
    Vw2 = torch.cat([Uw[:, :1024], -Vw[:, 1024:]], 1)
    od = float((swap_post(win, Uw, Vw, -1, 1024, 1024) + swap_post(win, Uw2, Vw2, -1, 1024, 1024)).abs().max())
    od0 = float((plain_post(win, Uw, Vw) + plain_post(win, Uw2, Vw2)).abs().max())
    n_tie, n_win = sum(p.numel() for p in tie.parameters()), sum(p.numel() for p in win.parameters())
    print(f'[배선 AB] 파라미터 tie {n_tie:,} · win {n_win:,} (BASE 와 동일) | AB-SYM tie 짝 위반 {ev:.1e} · '
          f'win 홀 위반 {od:.1e} | 대칭화 off 위반 tie {ev0:.2e} · win {od0:.2e} (0 이 아닌 것이 정상)')
    assert n_tie == 262537 and n_win == 262409, '파라미터 수 계약 (BASE 와 달라졌다)'
    assert ev < 1e-4 and od < 1e-4, 'AB-SYM 패리티 계약 위반'
    assert ev0 > 1e-3 and od0 > 1e-3, '대칭화 off 인데 패리티가 성립한다 — 배선 오류'


# ══════════════════════════════════════════ 층 2
def layer2(U, Lt, Lw, Gt, W, bias_win, kind='blk2'):
    """층 2 — kind 'blk2' 는 synlab.blk2 (노트북 셀 60), 'percol'/'ard' 는 penfold.combiner (같은 격자)."""
    y, fold, cu, nz, hold = U['y'], U['fold'], U['cu'], U['nz'], U['hold']
    yT, yA = (y == 2).astype(np.float64), (y == 0).astype(np.float64)
    Xt, Xw = np.hstack([Lt, Gt]), np.hstack([Lw, W])
    with threadpool_limits(limits=1):                 # 4k×7 적합에 16 BLAS 스레드는 37× 오버헤드
        if kind == 'blk2':
            z_tie, dt = blk2(Xt, yT, fold, cu, Lt.shape[1], True)
            z_win, dw = blk2(Xw, yA, fold, cu & nz, Lw.shape[1], bias_win)
        else:
            z_tie, dt = PF.combiner(Xt, yT, fold, cu, True, Lt.shape[1], kind)
            z_win, dw = PF.combiner(Xw, yA, fold, cu & nz, bias_win, Lw.shape[1], kind)
    row, tie_ll, win_ll = score(U, z_tie, z_win, hold)
    lp = assemble(z_tie[hold & nz], z_win[hold & nz])
    assert abs(float(np.abs(np.exp(lp).sum(1) - 1).max())) < 1e-9, '조립 확률합 위반'
    assert abs(tie_ll + R_NZ * win_ll - float(row.mean())) < 1e-10, '축 분해 항등 위반'
    cell = dict(tot=float(row.mean()), tie=tie_ll, win=win_ll, kind=kind,
                tie_lam=dt['lam'], tie_picked=dt['picked'], win_lam=dw['lam'], win_picked=dw['picked'],
                tie_w=dt['v'], win_w=dw['v'], win_bias=bool(bias_win))
    if kind == 'ard':
        cell['tie_q0'], cell['win_q0'] = dt['q0'], dw['q0']
    return cell, row, rowloss(z_tie[hold], yT[hold]), rowloss(z_win[hold & nz], yA[hold & nz])


def _lam_str(c):
    """시드 표의 λ 표기 — blk2 는 (λ_L, λ_g) · percol 은 로짓 열별 벡터 + 헤드 · ard 는 q0 중앙값."""
    k = c.get('kind', 'blk2')
    if k == 'blk2':
        return f'{tuple(c["tie_lam"])} / {tuple(c["win_lam"])}'
    if k == 'ard':
        return f'q0 {float(np.median(c["tie_q0"])):g} / {float(np.median(c["win_q0"])):g}'
    f = lambda v, nl: '[' + ' '.join(f'{x:g}' for x in v[:nl]) + f' | {v[nl]:g}]'
    return f'{f(c["tie_lam"], len(SD_TIE))} / {f(c["win_lam"], len(SD_WIN))}'


def paired(res, rows_, seeds, key, base='BASE'):
    print(f'\n  [{key} − {base}]  {LABEL.get(key, key)}')
    print(f'  {"시드":6s}{"Δtie":>10s}{"Δwin":>10s}{"Δtot":>10s}')
    for sd in seeds:
        b, a = res[f'{base}-s{sd}'], res[f'{key}-s{sd}']
        print(f'  {sd:<6d}{a["tie"] - b["tie"]:+10.5f}{a["win"] - b["win"]:+10.5f}{a["tot"] - b["tot"]:+10.5f}')
    rng = np.random.default_rng(SEED)
    out = {}
    for ax, k in (('tie', 'tie'), ('win', 'win'), ('tot', 'row')):
        a = np.mean([rows_[f'{k}_{key}-s{s}'] for s in seeds], 0)
        b = np.mean([rows_[f'{k}_{base}-s{s}'] for s in seeds], 0)
        d = a - b
        bs = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(1)
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        per = np.array([res[f'{key}-s{s}'][ax] - res[f'{base}-s{s}'][ax] for s in seeds])
        verdict = ('유의 ★' if (lo > 0 or hi < 0) else '노이즈') + (' · 문턱 ≥0.0005' if abs(d.mean()) >= 0.0005 else '')
        print(f'  {ax:4s} Δ {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}] {verdict:22s} 시드 악화 {int((per > 0).sum())}/{len(seeds)}')
        out[ax] = dict(mean=float(d.mean()), lo=lo, hi=hi, worse=int((per > 0).sum()))
    return out


# ══════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seeds', default='42,1,2,3,4')
    ap.add_argument('--arms', default=','.join(ARMS), help='BASE,UV-NOSYM,AB-SYM,AB 중 부분집합')
    ap.add_argument('--out', default='temp/synlab2')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s]
    arms = [a for a in args.arms.split(',') if a]
    assert set(arms) <= set(ARMS), f'팔은 {list(ARMS)} 중에서'
    t_all = time.time()
    print(f'[격자] 양축 λ ∈ {[f"{g:g}" for g in LAMS]}  (blk2 · 축별 ×10 확장)')
    print('[팔]   ' + '\n       '.join(f'{a:9s} {LABEL[a]}' for a in arms))

    print('\n[적재]')
    U = build()
    TR = load_train_logits()
    tr_ids, y_tr = TR[0], TR[1]
    E_s2, E_tr = enc_logits(U['ids'], 's2'), enc_logits(tr_ids, 'tr', y_tr)
    Mar_s2, Mar_tr = arith_mean(U, tr_ids)
    sd_s2, _ = bases(E_s2, Mar_s2)
    sd_tr, _ = bases(E_tr, Mar_tr)
    assert np.array_equal(sd_s2['tie'][:, :4], U['enc_tie']) and np.array_equal(sd_tr['tie'][:, :4], TR[2])
    print(f'  로짓 (m,t) 기저 — tie {sd_s2["tie"].shape[1]} / win {sd_s2["win"].shape[1]} (전 팔 공통) | 장치 {DEVICE}')

    need_uv = any(ARMS[a][0] == 'uv' for a in arms)
    need_ab = any(ARMS[a][0] == 'ab' for a in arms)
    Z = load_svd_cache() if need_uv else None
    RAWUV = raw_uv(Z) if need_uv else None
    SUP_AB, C_AB = ab_assets() if need_ab else (None, None)
    D.wiring_checks(5, 3)
    wiring_ab()

    REF = {}
    for nm in ('synlab', 'dest_fold'):
        p = FINAL / f'temp/{nm}.json'
        if p.exists():
            REF[nm] = json.load(open(p))['cells']

    res, rows_, store, keys = {}, {}, {}, []
    for arm in arms:
        coords, sym = ARMS[arm]
        A = assets_uv(Z, TR, sd_tr, sd_s2) if coords == 'uv' else assets_ab(C_AB, TR, sd_tr, sd_s2)
        SUPa = RAWUV if coords == 'uv' else SUP_AB
        # (접미사, 승부 절편, 벌점 방식) — 본 팔 blk2 · /b · /percol · /ard
        variants = [('', False, 'blk2')] + ([('/b', True, 'blk2')] if not sym else []) \
            + [(f'/{k}', False, k) for k in KINDS]
        for sd in seeds:
            tag = f'{arm}-s{sd}'
            print(f'\n{"=" * 78}\n[{tag}]  {LABEL[arm]}\n{"=" * 78}')
            t0 = time.time()
            Gt, st_t = tie_head(A, sd, coords, sym)
            W, st_w = win_head(A, SUPa, sd, coords, sym)
            sec_h = time.time() - t0
            for suf, bw, kind in variants:
                key = f'{arm}{suf}'
                if key not in keys:
                    keys.append(key)
                t1 = time.time()
                cell, row, lt, lw = layer2(U, sd_s2['tie'], sd_s2['win'], Gt, W, bw, kind)
                res[f'{key}-s{sd}'] = dict(arm=key, seed=sd, **cell,
                                           sec_heads=round(sec_h, 1), sec_l2=round(time.time() - t1, 1))
                rows_[f'row_{key}-s{sd}'], rows_[f'tie_{key}-s{sd}'], rows_[f'win_{key}-s{sd}'] = row, lt, lw
                line = (f'  {key:15s} tie {cell["tie"]:.5f} · win {cell["win"]:.5f}{" (절편)" if bw else ""} '
                        f'→ **{cell["tot"]:.5f}**   λ {_lam_str(cell)}  [{res[f"{key}-s{sd}"]["sec_l2"]:.0f}s]')
                if key == 'BASE':
                    for nm, cells in REF.items():
                        if tag in cells:
                            d_t, d_w = cell['tie'] - cells[tag]['tie'], cell['win'] - cells[tag]['win']
                            ok_t = abs(d_t) < REF_TOL
                            ok_w = abs(d_w) < REF_TOL if nm == 'synlab' else True
                            line += (f'\n              {nm} 대조  tie Δ {d_t:+.1e} {"일치" if ok_t else "*** 불일치"}'
                                     + (f' · win Δ {d_w:+.1e} {"일치" if ok_w else "*** 불일치"}' if nm == 'synlab'
                                        else f' · win Δ {d_w:+.5f} (λ_L 고정 0 대비 — 다른 것이 정상)'))
                print(line)
            print(f'  [{sec_h:.0f}s 헤드]')
            store[f'Gt_{tag}'], store[f'W_{tag}'] = Gt.astype(np.float32), W.astype(np.float32)
            torch.save({'tie': st_t, 'win': st_w, 'arm': arm, 'seed': sd, 'coords': coords, 'sym': sym,
                        'tie_names': SD_TIE, 'win_names': SD_WIN}, FINAL / f'{args.out}_heads_{tag}.pt')

    op = FINAL / args.out
    np.savez_compressed(op.with_suffix('.rowloss.npz'), **rows_, **store)
    summary = {}
    for key in keys:
        g = lambda k: np.array([res[f'{key}-s{sd}'][k] for sd in seeds])
        print(f'\n{"=" * 78}\n[{key}] {LABEL[key]}\n{len(seeds)}시드 · holdout 5,748\n{"=" * 78}')
        print(f'  {"시드":6s}{"tie ll":>10s}{"win ll":>10s}{"총 3-class":>12s}   λ(tie) / λ(win)')
        for sd in seeds:
            c = res[f'{key}-s{sd}']
            print(f'  {sd:<6d}{c["tie"]:10.5f}{c["win"]:10.5f}{c["tot"]:12.5f}   {_lam_str(c)}')
        sdv = g('tot').std(ddof=1) if len(seeds) > 1 else 0.0
        print(f'  {"평균":6s}{g("tie").mean():10.5f}{g("win").mean():10.5f}{g("tot").mean():12.5f}   (시드 sd {sdv:.5f})')
        summary[key] = dict(tie=float(g('tie').mean()), win=float(g('win').mean()), tot=float(g('tot').mean()), sd=float(sdv))

    # ── 벌점 방식 × 축 — 팔마다, blk2 대비 paired (시드 5 · 행 부트스트랩)
    print(f'\n{"=" * 78}\n[벌점 방식 × 축] 팔별 · blk2(본 팔) 대비 paired Δ · 양수 = 그 방식이 나쁨\n{"=" * 78}')
    rng = np.random.default_rng(SEED)
    for arm in arms:
        print(f'\n  [{arm}] {LABEL[arm]}')
        print(f'  {"방식":8s}{"tie ll":>9s}{"Δtie":>9s}{"95% CI":>20s} 악화 {"win ll":>9s}{"Δwin":>9s}{"95% CI":>20s} 악화 {"tot(동일방식)":>14s}')
        best = {}
        for kind in ('blk2',) + KINDS:
            key = arm if kind == 'blk2' else f'{arm}/{kind}'
            s = summary[key]
            line = f'  {kind:8s}{s["tie"]:9.5f}'
            pk = {}
            for ax in ('tie', 'win'):
                if kind == 'blk2':
                    line += f'{"—":>9s}{"":>20s}     ' if ax == 'tie' else f'{s["win"]:9.5f}{"—":>9s}{"":>20s}     '
                    continue
                a = np.mean([rows_[f'{ax}_{key}-s{sd}'] for sd in seeds], 0)
                b = np.mean([rows_[f'{ax}_{arm}-s{sd}'] for sd in seeds], 0)
                d = a - b
                bs = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(1)
                lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
                per = np.array([res[f'{key}-s{sd}'][ax] - res[f'{arm}-s{sd}'][ax] for sd in seeds])
                mark = '★' if (lo > 0 or hi < 0) else ' '
                pk[ax] = dict(mean=float(d.mean()), lo=lo, hi=hi, worse=int((per > 0).sum()))
                line += (f'{s["win"]:9.5f}' if ax == 'win' else '') + \
                    f'{d.mean():+9.5f} [{lo:+.5f},{hi:+.5f}]{mark} {int((per > 0).sum())}/{len(seeds)} '
            line += f'{s["tot"]:14.5f}'
            print(line)
            if kind != 'blk2':
                summary[key]['vs_blk2'] = pk
            best[kind] = (s['tie'], s['win'])
        bt = min(best, key=lambda k: best[k][0])
        bw_ = min(best, key=lambda k: best[k][1])
        print(f'  축별 최적 (holdout 선택 — 낙관, 참고만): tie={bt} · win={bw_} → tot {best[bt][0] + R_NZ * best[bw_][1]:.5f}  '
              f'(blk2 양축 {best["blk2"][0] + R_NZ * best["blk2"][1]:.5f})')
    print('  ! BASE 에서는 percol +0.00018 · ard +0.00061★ (penfold 2026-08-21) 가 재현돼야 한다 — 아니면 격자/이식 차이를 의심')

    HEAD_KEYS = [k for k in keys if '/' not in k or k.endswith('/b')]      # 헤드 팔 + /b 만 BASE 대비
    if 'BASE' in keys:
        print(f'\n{"=" * 78}\n[paired vs BASE] {len(seeds)}시드 · 행별 부트스트랩 95% CI · 양수 = 그 팔이 나쁨\n{"=" * 78}')
        for key in HEAD_KEYS:
            if key != 'BASE':
                summary[key]['vs_BASE'] = paired(res, rows_, seeds, key)
        print(f'\n{"=" * 78}\n[요약] tot 시드 평균 · Δ vs BASE  (헤드 파라미터 전 팔 동일 · 층2 blk2)\n{"=" * 78}')
        print(f'  {"팔":12s}{"tie":>9s}{"win":>9s}{"tot":>9s}{"Δtot":>10s}  {"95% CI":>20s}  악화')
        for key in HEAD_KEYS:
            s = summary[key]
            if key == 'BASE':
                print(f'  {key:12s}{s["tie"]:9.5f}{s["win"]:9.5f}{s["tot"]:9.5f}{"—":>10s}')
            else:
                v = s['vs_BASE']['tot']
                print(f'  {key:12s}{s["tie"]:9.5f}{s["win"]:9.5f}{s["tot"]:9.5f}{v["mean"]:+10.5f}  '
                      f'[{v["lo"]:+.5f},{v["hi"]:+.5f}]  {v["worse"]}/{len(seeds)}')
        if {'BASE', 'UV-NOSYM', 'AB-SYM', 'AB'} <= set(keys):
            S_ = summary
            print('\n  2×2 (tot 시드 평균)             대칭화 on       대칭화 off')
            print(f'  {"(u,v) 좌표":26s}{S_["BASE"]["tot"]:13.5f}{S_["UV-NOSYM"]["tot"]:16.5f}')
            print(f'  {"(A,B) 좌표":26s}{S_["AB-SYM"]["tot"]:13.5f}{S_["AB"]["tot"]:16.5f}')
            for ax in ('tie', 'win', 'tot'):
                b, n, c, a = (S_[k][ax] for k in ('BASE', 'UV-NOSYM', 'AB-SYM', 'AB'))
                print(f'  {ax:4s} 좌표 효과 (AB-SYM−BASE) {c - b:+.5f} · 대칭화 효과 (UV-NOSYM−BASE) {n - b:+.5f} · '
                      f'순수 제거 (AB−BASE) {a - b:+.5f} · 상호작용 {a - c - n + b:+.5f}')
        print('  ! tot = tie + 0.691 × win. 문턱: CI 0 배제 + |Δ| ≥ 0.0005 + 시드 5/5')

    with open(op.with_suffix('.json'), 'w') as f:
        json.dump({'cells': res, 'summary': summary, 'seeds': seeds, 'arms': arms, 'keys': keys, 'lams': LAMS,
                   'labels': {k: LABEL[k] for k in keys}, 'design': {a: dict(coords=c, sym=s) for a, (c, s) in ARMS.items()},
                   'names': dict(tie=SD_TIE + [f'g{i}' for i in range(1, TIE_ROUNDS + 1)], win=SD_WIN + list(SRCS)),
                   'params': dict(tie=262537, win=262409), 'K_AB': K_AB, 'kinds': ['blk2', *KINDS]},
                  f, ensure_ascii=False, indent=1)
    print(f'\n저장: {op}.rowloss.npz / {op}.json / {args.out}_heads_*.pt')
    print(f'총 {((time.time() - t_all) / 60):.1f}분')


if __name__ == '__main__':
    main()
