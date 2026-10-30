"""synlab.py — 통합 헤드의 패리티 구조(u/v 분해 · sym_post · 홀 결합기)를 양축에서 떼어 본다.

!! 설계 결함 (2026-08-22 실행 후 판정) — NOSYM/RAW 는 좌표만 뗀 것이 아니라 유닛(양가지 전입력·절편 → 승부
   파라미터 2배)·SVD(joint 1024)·정규화·결합기 절편을 함께 바꿨다. 제거가 아니라 재설계라 "u/v 가 필요한가"에
   답하지 못한다. 순수 제거 실험은 temp/synlab2.py (같은 망·같은 파라미터, 좌표 × 대칭화 2×2).
   이 파일의 결과 중 유효한 것: BASE (= 노트북 현행, dest_fold 와 비트 일치) · BASE/Lraw (층 2 로짓 기저만 교체,
   +0.00116 ★ 5/5 — 이것은 순수 제거다).

질문: 스왑 TTA 로 인코더 단에서 자리 효과가 이미 상쇄된 입력이 오는데, 헤드에서 또 u/v 로 갈라
대칭화할 필요가 있는가? — 구조를 단계적으로 떼어 같은 시드 · 같은 결합기 규칙으로 잰다.
기준은 노트북 현행(APFTE §7: 양축 blk2 결합기 · 격자 0~1e5). winlab4 의 AUG 팔은 게이트=u · 값=v
타입 유닛을 그대로 둔 채 skip/절편만 풀었으므로 몸통은 여전히 구조적 홀이었다 — 분해 자체를 뗀
실험은 이것이 처음이다.

팔 (층 1 학습 · 시드당 1회)
    BASE   노트북 현행 — u/v SVD-512+512 · TieNet 2R / WinNet GATE σ(G·u)⊙tanh(V·v) · sym_post
           · 로짓 패리티 기저 (m 홀 / t 짝) · 결합기 tie 절편 O / win 절편 X (홀)
    NOSYM  u/v 좌표 · 로짓 기저는 BASE 그대로 두고 **대칭화만 뗀다** — 몸통을 x 한 번만 통과(반사 없음),
           판독 절편 자유, 로짓 8칸 전부 생존. 승부 유닛은 σ(G·x+b)⊙tanh(V·x+b) 로 두 가지가 전입력
           [u|v] 를 읽는다 (게이트=u · 값=v 타입 유닛은 그 자체가 홀이라 sym_post 를 떼어도 홀이다).
           결합기 win 절편 O (헤드 열이 더는 홀이 아니다).
    RAW    분해 없음 — 풀링 [A|B] 8,192 → z-score → joint SVD-1024 화이트닝 (tie 몸통 폭 1,024 동일)
           · 승부 공급 소스별 [A|B] 2,048 z-score · 로짓은 원시 (a,b) = (z_A−z_tie, z_B−z_tie)
           (softmax 불변 오프셋만 뺀다 — 패리티 연산이 아니다; TB 는 s 한 열) · 무제약 헤드 · 결합기 양축 절편 O.
           tie 로짓 8 [s_tb2 s_tb1 a_abe2 b_abe2 a_abe1 b_abe1 a_ar b_ar] · win 로짓 6 (ABE1·ABE2·Arith 의 a,b).
           (m,t) ↔ (a,b) 는 전단사 (m = a−b, t = −logaddexp(a,b)) 라 정보량은 같고 **정렬만 다르다**.

층 2 파생 (학습 없음 · 같은 헤드 열에 결합기만 다시)
    BASE/Lraw   BASE 헤드 + 원시 로짓 (a,b) · 절편 결합기      → 층 2 로짓 기저만의 효과
    RAW/Lsd3    RAW 헤드 + 패리티 로짓 (m,t) · 절편 결합기      → RAW 손실 중 층 2 기저 몫을 분리

결합기 = 노트북 셀 60 `_blk2_combiner` 이식 — 양축 2묶음 [λ_L×로짓 | λ_g×헤드] · 격자 {0,1,…,1e5}² ·
축별 ×10 자동확장 · 배포 λ = fold 선택의 성분별 중앙값. (dest_fold.win_combiner 는 λ_L 고정 0 이라 안 쓴다.)

파라미터 — tie 는 세 팔 모두 262,537 × 2R 로 같다. 승부는 BASE 262,409 / NOSYM·RAW 524,681 (두 가지가
전입력 2,048 을 읽어 2배). 용량 증설은 이 프로젝트에서 8연패라 교란으로는 약하지만 해석 시 감안한다.

읽는 법 — tot = tie ll + 0.691 × win ll (holdout 5,748 · 리더보드 단위). 양수 Δ = 그 팔이 나쁨.
          판정 문턱 (plan §5): paired bootstrap 95% CI 가 0 배제 + |Δ| ≥ 0.0005 + 시드 부호 5/5.

검증 — BASE-s* 의 tie ll 은 temp/dest_fold.json 의 BASE-s* 와 비트 일치해야 한다 (같은 헤드 · 같은 tie
       결합기). win 은 blk2 로 바뀌어 다르다 (Δ 를 찍는다). 어긋나면 이식 오류다.

실행:  cd final/temp && python synlab.py                          (ML2 · GPU · 3팔 × 5시드 ≈ 70분)
       python synlab.py --arms BASE,RAW                            (판정만 · ≈ 45분)
       nohup python -u synlab.py > log_synlab.txt 2>&1 &   tail -f log_synlab.txt
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
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dest_fold as D                                                             # noqa: E402
from dest_fold import (FINAL, MODELS, SEED, K, INNER, LAM_CEIL, H, TIE_ROUNDS,     # noqa: E402
                       SRCS, DEVICE, R_NZ, REF_TOL, LR, EPOCHS, PATIENCE, WD,
                       rowloss, _fit, _z, _scores, assemble, score, sd3, _tta,
                       build, load_train_logits, arith_fold_logits, load_svd_cache,
                       zstats, raw_uv, _es_split, HeadLin, TieNet, WinNet, sym_post)

LAMS = list(D.set_grid(1e5))          # 양축 공통 {0, 1, 10, …, 1e5} — 노트북 셀 58 과 같다
K_AB = 1024                           # RAW tie joint SVD 랭크 = BASE 의 512 + 512
SVD_AB = FINAL / 'temp/head_cache/synlab_svdAB1024.npz'
SD_TIE = ['s_tb2', 's_tb1', 't_abe2', 't_abe1', 't_ar']
SD_WIN = ['m_abe2', 'm_abe1', 'm_ar']
RW_ALL = ['s_tb2', 's_tb1', 'a_abe2', 'b_abe2', 'a_abe1', 'b_abe1', 'a_ar', 'b_ar']
RW_WIN = RW_ALL[2:]
ARMS_ALL = ['BASE', 'NOSYM', 'RAW']
LABEL = {
    'BASE':      'BASE — u/v 분해 + sym_post + 패리티 결합기 (노트북 현행)',
    'NOSYM':     'NOSYM — u/v 좌표·(m,t) 로짓 유지, sym_post 제거 · 절편 자유 · 양가지 전입력 (대칭을 학습에 맡김)',
    'RAW':       'RAW — [A|B] 좌표 · 원시 로짓 (a,b) · 무제약 헤드 · 절편 결합기 (분해 없음)',
    'BASE/Lraw': 'BASE 헤드 + 층2 원시 로짓 (a,b)·절편   (층2 기저만 교체)',
    'RAW/Lsd3':  'RAW 헤드 + 층2 패리티 로짓 (m,t)·절편  (층2 기저만 복원)',
}
ENC = [('abe2', 'abe2__checkpoint-3000', 3), ('tb2', 'tb2__checkpoint-4000', 2),
       ('abe1', f'abe1__checkpoint-{D.ABE1_STEP}', 3), ('tb1', f'tb1__checkpoint-{D.TB1_STEP}', 2)]
AB_SPEC = [('abe2__checkpoint-3000', ('ea',), ('eb',)),
           ('tb2__checkpoint-4000', ('a_f', 'b_s'), ('b_f', 'a_s')),
           (f'abe1__checkpoint-{D.ABE1_STEP}', ('ea',), ('eb',)),
           (f'tb1__checkpoint-{D.TB1_STEP}', ('x_f',), ('x_s',))]


# ══════════════════════════════════════════ 결합기 — 노트북 셀 60 이식
def blk2(X, y01, fold, rows, nl, bias):
    """두 묶음 릿지 결합기 (APFTE 셀 60 `_blk2_combiner` 글자 단위 이식).

    pen = [λ_L × 로짓 nl | λ_g × 헤드] · (λ_L, λ_g) ∈ LAMS² · 큰 λ 부터 웜스타트 · 축별 자동확장.
    bias=True → 전 열 중심화+표준화 (짝 축 / 무제약) · bias=False → 무중심 RMS (홀 축, 반대칭 보존).
    배포 (λ_L, λ_g) = 바깥 fold 선택의 성분별 중앙값.
    """
    n_even = X.shape[1] if bias else 0
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
        sc, warm = {}, [None] * INNER

        def ev(q):
            if q in sc:
                return
            tot = 0.0
            for j in range(INNER):
                A, B = Zi[j]
                pen = np.concatenate([np.full(nl, q[0]), np.full(A.shape[1] - nl, q[1])])
                v, _ok = _fit(A, ytr[inner != j], pen, bias, warm[j])
                warm[j] = v
                tot += float(rowloss(_scores(v, B, n_even, bias), ytr[inner == j]).sum())
            sc[q] = tot

        gl, gg = list(LAMS), list(LAMS)
        for a in sorted(gl, reverse=True):
            for b in sorted(gg, reverse=True):
                ev((a, b))
        while True:                                   # 격자 상단이 뽑히면 그 축만 ×10
            q = min(sc, key=sc.get)
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
        v, _ok = _fit(A, ytr, pen, bias)
        z[te] = _scores(v, B, n_even, bias)
    lam = (float(np.median([q[0] for q in picked])), float(np.median([q[1] for q in picked])))
    A, _ = _z(X[rows], X[rows], n_even)
    pen = np.concatenate([np.full(nl, lam[0]), np.full(X.shape[1] - nl, lam[1])])
    v, ok = _fit(A, y01[rows], pen, bias)
    assert ok, f'{"tie" if bias else "win"} 결합기 배포 적합 미수렴'
    return z, dict(lam=lam, picked=[tuple(map(float, q)) for q in picked],
                   v=[float(x) for x in v])


# ══════════════════════════════════════════ 로짓 — 두 기저
def enc_logits(ids, split, y=None):
    """인코더 4종의 뷰평균 로짓 (fwd+swp)/2 — dest_fold 와 같은 접기. split ∈ {'s2','tr'}."""
    out = {}
    for tag, f, nc in ENC:
        if split == 's2':
            L, z = _tta(f'logits/{f}.npz', nc)
            assert (np.concatenate([z['valid_ids'], z['hold_ids']]) == ids).all(), f'{tag} id 정렬'
        else:
            z = np.load(MODELS / f'logits/{f}__train.npz')
            assert (z['train_ids'] == ids).all() and (z['train_labels'] == y).all(), f'{tag} train 정렬'
            L = ((z['train_fwd'] + z['train_swp']) / 2).astype(np.float64)
            assert L.shape == (len(ids), nc) and np.isfinite(L).all(), f'{tag} 형상'
        out[tag] = L
    return out


def raw3(L):
    """원시 기저 — (a, b) = (z_A − z_tie, z_B − z_tie). softmax 불변 오프셋만 제거, 자리 맞바꾸면 a ↔ b."""
    return L[:, 0] - L[:, 2], L[:, 1] - L[:, 2]


def bases(E, Mar):
    """sd = 패리티 기저 (m 홀 · t 짝, 노트북 현행) · rw = 원시 기저 (a, b). 같은 13 원시 로짓에서 softmax
    불변 5방향을 뺀 8열씩 — 전단사이므로 정보량은 같고 정렬만 다르다 (아래 assert 가 그 항등을 찍는다)."""
    s2 = E['tb2'][:, 1] - E['tb2'][:, 0]
    s1 = E['tb1'][:, 1] - E['tb1'][:, 0]
    m2, t2 = sd3(E['abe2'])
    m1, t1 = sd3(E['abe1'])
    ma, ta = sd3(Mar)
    a2, b2 = raw3(E['abe2'])
    a1, b1 = raw3(E['abe1'])
    aa, ba = raw3(Mar)
    sd = dict(tie=np.column_stack([s2, s1, t2, t1, ta]), win=np.column_stack([m2, m1, ma]))
    rw_all = np.column_stack([s2, s1, a2, b2, a1, b1, aa, ba])
    rw = dict(tie=rw_all, win=np.ascontiguousarray(rw_all[:, 2:]))
    for d in (sd, rw):
        for k, v in d.items():
            assert np.isfinite(v).all(), f'{k} 비유한 로짓'
    assert np.allclose(sd['win'][:, 0], a2 - b2) and np.allclose(sd['tie'][:, 2], -np.logaddexp(a2, b2)), \
        '(m,t) ↔ (a,b) 전단사 항등 위반'
    return sd, rw


def arith_mean(U, tr_ids):
    """Arith 마진 평균 (n,3) — stage2 · TR. 출처는 dest_fold.arith_fold_logits 한 곳 (BASE 팔과 동일)."""
    fid, M = arith_fold_logits()
    i_s2, i_tr = fid.get_indexer(U['ids']), fid.get_indexer(tr_ids)
    assert (i_s2 >= 0).all() and (i_tr >= 0).all(), 'arith id 커버리지'
    Mm = M.mean(0)
    return Mm[i_s2], Mm[i_tr]


# ══════════════════════════════════════════ [A|B] 좌표 (RAW 팔)
def ab_assets():
    """RAW 공급 — 소스별 [A|B] 2,048 z-score (승부) + joint SVD-1024 화이트닝 좌표 (tie, 캐시).

    A/B 판은 dest_fold.raw_uv 와 같은 뷰평균 접기 (tb2: A=(a_f+b_s)/2 · B=(b_f+a_s)/2). 통계는 TR 전용.
    열 순서 [A: abe2 tb2 abe1 tb1 | B: abe2 tb2 abe1 tb1] = 8,192.
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
    Xt, Xs = np.hstack(At + Bt), np.hstack(As + Bs)              # (46729, 8192) · (10748, 8192) f32
    del At, Bt, As, Bs
    gc.collect()
    assert Xt.shape == (46729, 8192) and Xs.shape == (10748, 8192), f'[A|B] 형상 {Xt.shape} {Xs.shape}'
    mu = Xt.mean(0, dtype=np.float64)
    sc = np.clip(Xt.std(0, dtype=np.float64), 1e-9, None)
    for X in (Xt, Xs):                                           # 제자리 z-score (중심화 — 패리티 없음)
        X -= mu.astype(np.float32)
        X /= sc.astype(np.float32)
    SUP = {}
    for i, s in enumerate(SRCS):
        sa, sb = slice(i * 1024, (i + 1) * 1024), slice(4096 + i * 1024, 4096 + (i + 1) * 1024)
        SUP[s] = dict(ab_tr=np.ascontiguousarray(np.hstack([Xt[:, sa], Xt[:, sb]])),
                      ab_s2=np.ascontiguousarray(np.hstack([Xs[:, sa], Xs[:, sb]])))
    print(f'  [A|B] 소스별 공급 — 4소스 × 2,048 z-score  [TR {Xt.shape[0]:,}행]')

    if SVD_AB.exists():
        z = np.load(SVD_AB)
        assert z['C_tr'].shape == (46729, K_AB) and z['C_s2'].shape == (10748, K_AB), 'SVD 캐시 형상'
        assert float(np.abs(z['mu'] - mu).max()) < 1e-5 and float(np.abs(z['sc'] - sc).max()) < 1e-5, \
            'SVD 캐시의 z-score 통계가 현재 풀링과 다르다 — 캐시를 지워라'
        C_tr, C_s2 = z['C_tr'], z['C_s2']
        print(f'  joint SVD-{K_AB} 캐시 {SVD_AB.name}  보존 분산 {float(z["kept"]):.3f}')
    else:
        t0 = time.time()
        X64 = Xt.astype(np.float64)
        mu2 = X64.mean(0)                                        # ≈ 0 (이미 중심화) — svd_keep 의 center 와 동일 절차
        X64 -= mu2
        total = float((X64 ** 2).sum())
        _, S, Vt_ = randomized_svd(X64, n_components=K_AB, random_state=SEED)
        W = Vt_.T
        Ct = X64 @ W
        Cs = (Xs.astype(np.float64) - mu2) @ W
        scc = np.clip(np.sqrt((Ct ** 2).mean(0)), 1e-9, None)   # 화이트닝 — 성분별 TR-RMS (BASE 와 같은 α=1)
        C_tr, C_s2 = (Ct / scc).astype(np.float32), (Cs / scc).astype(np.float32)
        kept = float((S ** 2).sum() / total)
        np.savez(SVD_AB, C_tr=C_tr, C_s2=C_s2, mu=mu, sc=sc, mu2=mu2, W=W, c_sc=scc, K_AB=K_AB, kept=kept)
        del X64, Ct, Cs
        print(f'  joint SVD-{K_AB} 계산 {(time.time() - t0) / 60:.1f}분 · 보존 분산 {kept:.3f} → 캐시 {SVD_AB.name}')
    del Xt, Xs
    gc.collect()
    return SUP, C_tr, C_s2


# ══════════════════════════════════════════ 헤드 재료
def assets_uv(Z, TR, sd_tr, sd_s2):
    """BASE · NOSYM — dest_fold.head_assets 와 같은 재료 (u/v SVD 캐시 · Lu z-score · Lv 무중심 RMS)."""
    mu_u, sc_u = zstats(sd_tr['tie'], True)
    _, sc_v = zstats(sd_tr['win'], False)
    return dict(y_tr=TR[1], y01m=(TR[1] == 2).astype(np.float64),
                Up_tr=Z['CUt'].astype(np.float32), Us_tr=(sd_tr['tie'] - mu_u) / sc_u,
                Vp_tr=Z['CVt'].astype(np.float32), Vs_tr=sd_tr['win'] / sc_v,
                Up_s2=Z['CUs'].astype(np.float32), Us_s2=(sd_s2['tie'] - mu_u) / sc_u,
                Vp_s2=Z['CVs'].astype(np.float32), Vs_s2=sd_s2['win'] / sc_v)


def assets_ab(C_tr, C_s2, TR, rw_tr, rw_s2):
    """RAW — tie 몸통 joint SVD-1024 · 로짓 원시 8열 전부 z-score (패리티 없음 → 전부 중심화). V 쪽은 폭 0."""
    mu, sc = zstats(rw_tr['tie'], True)
    e_tr, e_s2 = np.zeros((len(TR[1]), 0), np.float32), np.zeros((len(C_s2), 0), np.float32)
    return dict(y_tr=TR[1], y01m=(TR[1] == 2).astype(np.float64),
                Up_tr=C_tr, Us_tr=(rw_tr['tie'] - mu) / sc, Vp_tr=e_tr, Vs_tr=e_tr,
                Up_s2=C_s2, Us_s2=(rw_s2['tie'] - mu) / sc, Vp_s2=e_s2, Vs_s2=e_s2)


# ══════════════════════════════════════════ 헤드 (층 1)
class FreeWinBody(nn.Module):
    """무제약 게이트 — σ(G·x+b) ⊙ tanh(V·x+b). 두 가지 모두 전입력을 읽고 절편이 있다 (패리티 타입 없음).
    BASE 의 σ(G·u)⊙tanh(V·v) 는 G 가 u 방향만, V 가 v 방향만 보고 절편 0 인 특수해로 이 족에 포함된다."""

    def __init__(self, d, h):
        super().__init__()
        self.G, self.V = nn.Linear(d, h), nn.Linear(d, h)
        self.out_dim = h

    def forward(self, x):
        return torch.sigmoid(self.G(x)) * torch.tanh(self.V(x))


class FreeWinNet(nn.Module):
    """무제약 승부 헤드 — 판독 절편 학습 가능 (WinNet 은 0 동결)."""

    def __init__(self, d, h, skip_idx, body_idx):
        super().__init__()
        self.body_idx, self.skip_idx = body_idx, skip_idx
        self.body = FreeWinBody(len(body_idx), h)
        self.head = HeadLin(self.body.out_dim + len(skip_idx))

    @property
    def w(self):
        return self.head.w


def plain_post(f, Ux, Vx):
    """무제약 판독 — 몸통을 x 한 번만 통과 (반사 없음). skip 칸 전부 · 절편이 살아 있다."""
    x = torch.cat([Ux, Vx], 1)
    u = torch.cat([f.body(x[:, f.body_idx]), x[:, f.skip_idx]], 1)
    return f.head(u).squeeze(-1)


def sym_even(f, Ux, Vx):
    return sym_post(f, Ux, Vx, +1)


def sym_odd(f, Ux, Vx):
    return sym_post(f, Ux, Vx, -1)


def fit_head(CUt_, CVt_, y01, s0, seed, CUs_, CVs_, net, out_fn, hidden=H):
    """dest_fold.fit_head 그대로 — 판독 경로만 out_fn (sym_even/sym_odd = BASE · plain_post = 무제약).
    BASE 경로는 연산 순서가 같아 dest_fold 와 비트 일치해야 한다."""
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
    out = lambda Ux, Vx: out_fn(f, Ux, Vx)

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


def _split_idx(Ku, nu, Kv, nv):
    """x = [U_pool Ku | U_skip nu | V_pool Kv | V_skip nv] 의 몸통/판독 열. (Ku=Kv=512, 5, 3) 이면 dest_fold 와 같다."""
    sk = list(range(Ku, Ku + nu)) + list(range(Ku + nu + Kv, Ku + nu + Kv + nv))
    pool = list(range(0, Ku)) + list(range(Ku + nu, Ku + nu + Kv))
    assert not set(pool) & set(sk) and len(pool) + len(sk) == Ku + nu + Kv + nv, '열 분리'
    return sk, pool


def tie_head(A, seed, free):
    """TANHTANH → BILIN(TR 잔차앵커) · h128 · 2R — dest_fold.tie_head 의 일반화. free 면 plain_post."""
    sk, pool = _split_idx(A['Up_tr'].shape[1], A['Us_tr'].shape[1], A['Vp_tr'].shape[1], A['Vs_tr'].shape[1])
    ct = np.ascontiguousarray(np.hstack([A['Up_tr'], A['Us_tr']]), np.float32)
    cv = np.ascontiguousarray(np.hstack([A['Vp_tr'], A['Vs_tr']]), np.float32)
    cs = np.ascontiguousarray(np.hstack([A['Up_s2'], A['Us_s2']]), np.float32)
    vs = np.ascontiguousarray(np.hstack([A['Vp_s2'], A['Vs_s2']]), np.float32)
    out_fn = plain_post if free else sym_even
    z, cols, states = None, [], []
    for rd, unit in enumerate(('TANHTANH', 'BILIN')[:TIE_ROUNDS], 1):
        net = lambda d, h, _u=unit: TieNet(d, h, _u, sk, pool)
        r, aux = fit_head(ct, cv, A['y01m'], z, seed, cs, vs, net, out_fn)
        cols.append(r)
        states.append(aux['state'])
        z = aux['z_tr']
        print(f'      tie r{rd} {unit:9s} es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]}')
    return np.column_stack(cols), states


def win_head(A, SUP, seed, arm):
    """소스별 4체인 · h128 · 1R — dest_fold.win_head 의 일반화.
    BASE: WinNet + sym_odd (u,v 공급) · NOSYM: FreeWinNet + plain_post (u,v 공급) · RAW: FreeWinNet + plain_post ([A|B] 공급)."""
    nu, nv = A['Us_tr'].shape[1], A['Vs_tr'].shape[1]
    m = A['y_tr'] != 2                       # 승부 헤드는 비-tie 행에서만 학습한다
    y01w = (A['y_tr'] == 0).astype(np.float64)[m]
    cols, states = [], []
    for si, s in enumerate(SRCS):
        R = SUP[s]
        if arm == 'RAW':
            e_tr, e_s2 = np.zeros((len(m), 0), np.float32), np.zeros((len(R['ab_s2']), 0), np.float32)
            Pu_tr, Pv_tr, Pu_s2, Pv_s2 = R['ab_tr'], e_tr, R['ab_s2'], e_s2
        else:
            Pu_tr, Pv_tr, Pu_s2, Pv_s2 = R['u_tr'], R['v_tr'], R['u_s2'], R['v_s2']
        du, dv = Pu_tr.shape[1], Pv_tr.shape[1]
        sk, pool = _split_idx(du, nu, dv, nv)
        cu_t = np.ascontiguousarray(np.hstack([Pu_tr[m], A['Us_tr'][m]]), np.float32)
        cv_t = np.ascontiguousarray(np.hstack([Pv_tr[m], A['Vs_tr'][m]]), np.float32)
        cu_s = np.ascontiguousarray(np.hstack([Pu_s2, A['Us_s2']]), np.float32)
        cv_s = np.ascontiguousarray(np.hstack([Pv_s2, A['Vs_s2']]), np.float32)
        if arm == 'BASE':
            net = lambda d, h, _s=sk, _p=pool, _du=du: WinNet(d, h, _s, _p, _du)
            out_fn = sym_odd
        else:
            net = lambda d, h, _s=sk, _p=pool: FreeWinNet(d, h, _s, _p)
            out_fn = plain_post
        r_s2, aux = fit_head(cu_t, cv_t, y01w, None, seed + si, cu_s, cv_s, net, out_fn)
        cols.append(r_s2)
        states.append(aux['state'])
        print(f'      win r1 {s:5s} es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]}')
        del cu_t, cv_t, cu_s, cv_s
    gc.collect()
    return np.column_stack(cols), states


# ══════════════════════════════════════════ 배선 가드
@torch.no_grad()
def wiring_free():
    """무제약 망의 계약 — 파라미터 수 · 0초기 출력 0. 패리티 위반은 **있어야 정상**이라 값만 찍는다."""
    g = torch.Generator().manual_seed(0)
    sk, pool = _split_idx(512, 5, 512, 3)
    tie = TieNet(1032, H, 'TANHTANH', sk, pool)                    # NOSYM tie — BASE 와 같은 망, 판독만 다름
    n_tie = sum(p.numel() for p in tie.parameters())
    sk8, pool8 = _split_idx(K_AB, 8, 0, 0)
    tie8 = TieNet(K_AB + 8, H, 'TANHTANH', sk8, pool8)               # RAW tie — 몸통 1,024 + 로짓 8
    n_tie8 = sum(p.numel() for p in tie8.parameters())
    skw, poolw = _split_idx(2048, 8, 0, 0)
    win = FreeWinNet(2056, H, skw, poolw)                           # NOSYM/RAW win — 전입력 2,048 + 로짓 8
    n_win = sum(p.numel() for p in win.parameters())
    Ux, Vx = torch.randn(64, 517, generator=g), torch.randn(64, 515, generator=g)
    nn.init.zeros_(tie.w.weight)
    nn.init.zeros_(tie.w.bias)
    z0 = float(plain_post(tie, Ux, Vx).abs().max())
    nn.init.normal_(tie.w.weight)
    ev = float((plain_post(tie, Ux, Vx) - plain_post(tie, Ux, -Vx)).abs().max())
    print(f'[배선 free] 파라미터 tie {n_tie:,} (NOSYM) / {n_tie8:,} (RAW) · win {n_win:,} | '
          f'0초기 출력 {z0:.1e} · NOSYM tie 짝 위반 {ev:.2e} (무제약이라 0 이 아닌 것이 정상)')
    assert n_tie == 262537 and n_tie8 == 262537 and n_win == 524681, '무제약 망 파라미터 수 계약'
    assert z0 == 0.0, '0초기 출력 계약'


# ══════════════════════════════════════════ 층 2 + 채점
def layer2(U, Lt, Lw, Gt, W, bias_win):
    y, fold, cu, nz, hold = U['y'], U['fold'], U['cu'], U['nz'], U['hold']
    yT, yA = (y == 2).astype(np.float64), (y == 0).astype(np.float64)
    z_tie, dt = blk2(np.hstack([Lt, Gt]), yT, fold, cu, Lt.shape[1], True)
    z_win, dw = blk2(np.hstack([Lw, W]), yA, fold, cu & nz, Lw.shape[1], bias_win)
    row, tie_ll, win_ll = score(U, z_tie, z_win, hold)
    lp = assemble(z_tie[hold & nz], z_win[hold & nz])
    assert abs(float(np.abs(np.exp(lp).sum(1) - 1).max())) < 1e-9, '조립 확률합 위반'
    assert abs(tie_ll + R_NZ * win_ll - float(row.mean())) < 1e-10, '축 분해 항등 위반'
    cell = dict(tot=float(row.mean()), tie=tie_ll, win=win_ll,
                tie_lam=dt['lam'], tie_picked=dt['picked'], win_lam=dw['lam'], win_picked=dw['picked'],
                tie_w=dt['v'], win_w=dw['v'], n_tie_logit=int(Lt.shape[1]), n_win_logit=int(Lw.shape[1]),
                win_bias=bool(bias_win))
    return cell, row, rowloss(z_tie[hold], yT[hold]), rowloss(z_win[hold & nz], yA[hold & nz])


def paired(res, rows_, seeds, key, base='BASE'):
    """key − base · 시드 paired · 행별 부트스트랩 95% CI (시드 평균 행손실)."""
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
        sig = lo > 0 or hi < 0
        verdict = ('유의 ★' if sig else '노이즈') + (' · 문턱 ≥0.0005' if abs(d.mean()) >= 0.0005 else '')
        print(f'  {ax:4s} Δ {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}] {verdict:22s} 시드 악화 {int((per > 0).sum())}/{len(seeds)}')
        out[ax] = dict(mean=float(d.mean()), lo=lo, hi=hi, worse=int((per > 0).sum()))
    return out


# ══════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seeds', default='42,1,2,3,4')
    ap.add_argument('--arms', default=','.join(ARMS_ALL), help='BASE,NOSYM,RAW 중 부분집합 (paired 는 BASE 필요)')
    ap.add_argument('--out', default='temp/synlab')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s]
    arms = [a for a in args.arms.split(',') if a]
    assert set(arms) <= set(ARMS_ALL), f'팔은 {ARMS_ALL} 중에서'
    t_all = time.time()
    print(f'[격자] 양축 λ ∈ {[f"{g:g}" for g in LAMS]}  (blk2 · 축별 ×10 확장 · 상한 {LAM_CEIL:g})')
    print('[팔]   ' + '\n       '.join(f'{a:6s} {LABEL[a]}' for a in arms))

    print('\n[적재]')
    U = build()
    TR = load_train_logits()
    tr_ids, y_tr = TR[0], TR[1]
    E_s2, E_tr = enc_logits(U['ids'], 's2'), enc_logits(tr_ids, 'tr', y_tr)
    Mar_s2, Mar_tr = arith_mean(U, tr_ids)
    sd_s2, rw_s2 = bases(E_s2, Mar_s2)
    sd_tr, rw_tr = bases(E_tr, Mar_tr)
    # 패리티 기저는 dest_fold 의 것과 비트 일치해야 한다 (같은 접기)
    assert np.array_equal(sd_s2['tie'][:, :4], U['enc_tie']) and np.array_equal(sd_s2['win'][:, :2], U['enc_win'])
    assert np.array_equal(sd_tr['tie'][:, :4], TR[2]) and np.array_equal(sd_tr['win'][:, :2], TR[3])
    print(f'  로짓 기저 — 패리티 (m,t) tie {sd_s2["tie"].shape[1]} / win {sd_s2["win"].shape[1]} · '
          f'원시 (a,b) tie {rw_s2["tie"].shape[1]} / win {rw_s2["win"].shape[1]}  | 장치 {DEVICE}')

    Z = load_svd_cache()
    RAWUV = raw_uv(Z) if {'BASE', 'NOSYM'} & set(arms) else None
    SUP_AB, C_tr, C_s2 = ab_assets() if 'RAW' in arms else (None, None, None)
    D.wiring_checks(5, 3)
    wiring_free()

    dfj = FINAL / 'temp/dest_fold.json'
    DF = json.load(open(dfj))['cells'] if dfj.exists() else {}

    ASSET = {'BASE': lambda: assets_uv(Z, TR, sd_tr, sd_s2),
             'NOSYM': lambda: assets_uv(Z, TR, sd_tr, sd_s2),
             'RAW': lambda: assets_ab(C_tr, C_s2, TR, rw_tr, rw_s2)}
    # 팔별 (접미사, 층2 tie 로짓, 층2 win 로짓, win 절편) — 첫 항이 본 팔, 나머지는 층2 파생
    VARIANTS = {'BASE': [('', sd_s2['tie'], sd_s2['win'], False), ('/Lraw', rw_s2['tie'], rw_s2['win'], True)],
                'NOSYM': [('', sd_s2['tie'], sd_s2['win'], True)],
                'RAW': [('', rw_s2['tie'], rw_s2['win'], True), ('/Lsd3', sd_s2['tie'], sd_s2['win'], True)]}
    NAMES = {'BASE': (SD_TIE, SD_WIN), 'BASE/Lraw': (RW_ALL, RW_WIN), 'NOSYM': (SD_TIE, SD_WIN),
             'RAW': (RW_ALL, RW_WIN), 'RAW/Lsd3': (SD_TIE, SD_WIN)}

    res, rows_, store, keys = {}, {}, {}, []
    for arm in arms:
        A = ASSET[arm]()
        SUPa = SUP_AB if arm == 'RAW' else RAWUV
        free = arm != 'BASE'
        for sd in seeds:
            tag = f'{arm}-s{sd}'
            print(f'\n{"=" * 78}\n[{tag}]  {LABEL[arm]}\n{"=" * 78}')
            t0 = time.time()
            Gt, st_t = tie_head(A, sd, free)
            W, st_w = win_head(A, SUPa, sd, arm)
            sec_h = time.time() - t0
            for suf, Lt, Lw, bw in VARIANTS[arm]:
                key = f'{arm}{suf}'
                if key not in keys:
                    keys.append(key)
                t1 = time.time()
                cell, row, lt, lw = layer2(U, Lt, Lw, Gt, W, bw)
                res[f'{key}-s{sd}'] = dict(arm=key, seed=sd, **cell,
                                           sec_heads=round(sec_h, 1), sec_l2=round(time.time() - t1, 1))
                rows_[f'row_{key}-s{sd}'], rows_[f'tie_{key}-s{sd}'], rows_[f'win_{key}-s{sd}'] = row, lt, lw
                line = (f'  {key:10s} tie {cell["tie"]:.5f} (λ {tuple(cell["tie_lam"])}) · '
                        f'win {cell["win"]:.5f} (λ {tuple(cell["win_lam"])}{"" if bw else " · 무절편"}) '
                        f'→ **{cell["tot"]:.5f}**')
                if key == 'BASE' and tag in DF:
                    d_t, d_w = cell['tie'] - DF[tag]['tie'], cell['win'] - DF[tag]['win']
                    line += (f'\n             dest_fold 대조  tie Δ {d_t:+.1e} ' + ('비트 일치' if abs(d_t) < REF_TOL else '*** 불일치 — 이식 오류')
                             + f' · win Δ {d_w:+.5f} (blk2 vs λ_L 고정 0 — 다른 것이 정상)')
                print(line)
            print(f'  [{sec_h:.0f}s 헤드]')
            store[f'Gt_{tag}'], store[f'W_{tag}'] = Gt.astype(np.float32), W.astype(np.float32)
            torch.save({'tie': st_t, 'win': st_w, 'arm': arm, 'seed': sd,
                        'tie_names': NAMES[arm][0], 'win_names': NAMES[arm][1]},
                       FINAL / f'{args.out}_heads_{tag}.pt')

    op = FINAL / args.out
    np.savez_compressed(op.with_suffix('.rowloss.npz'), **rows_, **store)
    summary = {}

    # ── 팔별 시드 표
    for key in keys:
        g = lambda k: np.array([res[f'{key}-s{sd}'][k] for sd in seeds])
        print(f'\n{"=" * 78}\n[{key}] {LABEL[key]}\n{len(seeds)}시드 · holdout 5,748\n{"=" * 78}')
        print(f'  {"시드":6s}{"tie ll":>10s}{"win ll":>10s}{"총 3-class":>12s}   λ(tie) / λ(win)')
        for sd in seeds:
            c = res[f'{key}-s{sd}']
            print(f'  {sd:<6d}{c["tie"]:10.5f}{c["win"]:10.5f}{c["tot"]:12.5f}   '
                  f'{tuple(c["tie_lam"])} / {tuple(c["win_lam"])}')
        sdv = g('tot').std(ddof=1) if len(seeds) > 1 else 0.0
        print(f'  {"평균":6s}{g("tie").mean():10.5f}{g("win").mean():10.5f}{g("tot").mean():12.5f}   (시드 sd {sdv:.5f})')
        summary[key] = dict(tie=float(g('tie').mean()), win=float(g('win').mean()), tot=float(g('tot').mean()), sd=float(sdv))

    # ── paired — BASE 가 있을 때만
    if 'BASE' in keys:
        print(f'\n{"=" * 78}\n[paired vs BASE] {len(seeds)}시드 · 행별 부트스트랩 95% CI · 양수 = 그 팔이 나쁨\n{"=" * 78}')
        for key in keys:
            if key != 'BASE':
                summary[key]['vs_BASE'] = paired(res, rows_, seeds, key)
        if 'NOSYM' in keys and 'RAW' in keys:
            summary['RAW']['vs_NOSYM'] = paired(res, rows_, seeds, 'RAW', base='NOSYM')
        print(f'\n{"=" * 78}\n[요약] tot 시드 평균 · Δ vs BASE\n{"=" * 78}')
        print(f'  {"팔":10s}{"tie":>9s}{"win":>9s}{"tot":>9s}{"Δtot":>10s}  {"95% CI":>20s}  악화')
        for key in keys:
            s = summary[key]
            if key == 'BASE':
                print(f'  {key:10s}{s["tie"]:9.5f}{s["win"]:9.5f}{s["tot"]:9.5f}{"—":>10s}')
            else:
                v = s['vs_BASE']['tot']
                print(f'  {key:10s}{s["tie"]:9.5f}{s["win"]:9.5f}{s["tot"]:9.5f}{v["mean"]:+10.5f}  '
                      f'[{v["lo"]:+.5f},{v["hi"]:+.5f}]  {v["worse"]}/{len(seeds)}')
        if {'BASE', 'BASE/Lraw', 'RAW', 'RAW/Lsd3'} <= set(keys):
            print('\n  층 2 로짓 기저 2×2 (tot 시드 평균):')
            print(f'  {"":12s}{"층2 (m,t) 패리티":>18s}{"층2 (a,b) 원시":>18s}')
            print(f'  {"BASE 헤드":12s}{summary["BASE"]["tot"]:18.5f}{summary["BASE/Lraw"]["tot"]:18.5f}')
            print(f'  {"RAW 헤드":12s}{summary["RAW/Lsd3"]["tot"]:18.5f}{summary["RAW"]["tot"]:18.5f}')
        print('  ! tot = tie + 0.691 × win — 리더보드 단위는 tot 행. 문턱: CI 0 배제 + |Δ| ≥ 0.0005 + 시드 5/5')
        print('  ! 승부 파라미터 BASE 262,409 / NOSYM·RAW 524,681 — 무제약 팔이 이기면 용량 교란을 먼저 의심한다')

    with open(op.with_suffix('.json'), 'w') as f:
        json.dump({'cells': res, 'summary': summary, 'seeds': seeds, 'arms': arms, 'keys': keys, 'lams': LAMS,
                   'labels': {k: LABEL[k] for k in keys},
                   'names': {k: dict(tie=list(NAMES[k][0]) + [f'g{i}' for i in range(1, TIE_ROUNDS + 1)],
                                     win=list(NAMES[k][1]) + list(SRCS)) for k in keys},
                   'params': dict(tie=262537, win_base=262409, win_free=524681), 'K_AB': K_AB},
                  f, ensure_ascii=False, indent=1)
    print(f'\n저장: {op}.rowloss.npz / {op}.json / {args.out}_heads_*.pt')
    print(f'총 {((time.time() - t_all) / 60):.1f}분')


if __name__ == '__main__':
    main()
