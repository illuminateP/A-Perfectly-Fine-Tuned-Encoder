"""synlab3.py — 층 0 까지 내려간 제거: 인코더 스왑 뷰(TTA) 없이, 정방향 한 번의 출력만으로 통합 헤드를 학습한다.

synlab2 의 AB 팔(헤드 단 u/v·대칭화 제거)에서 한 단계 더 — **모델 전체에서 대칭을 전부 뗀다.** 남는 대칭은
각 인코더 학습 시의 swap 증강뿐이다. 동일 조건: 같은 망·같은 파라미터(tie 262,537×2 / win 262,409×4)·
같은 하이퍼파라미터·같은 (m,t) 로짓 기저·같은 결합기(blk2 · /b · /percol · /ard).

FWD 팔의 입력 (정방향 뷰만)
    로짓    abe2·abe1 fwd 3 · tb2·tb1 fwd 2 · arith fwd 마진 (swap 판 없이 5폴드 평균)  → sd3 로 (m,t)
            (기저는 AB 와 같게 두어 층 2 기저 효과(synlab BASE/Lraw +0.00116★)와 섞이지 않게 한다)
    풀링    abe2 (ea, eb) · tb2 (a_f, b_f) · abe1 (ea, eb) · tb1 (x_f, —)
            ! TB1 은 전체 시퀀스 풀링이라 정방향 뷰에 x_f 하나뿐이다 — BASE/AB 의 (A,B) 쌍은 곧 (x_f, x_s) 두 뷰,
              즉 스왑 뷰 = 두 번째 인코더 패스 = 두 번째 벡터다. 뷰평균을 떼면 TB1 에서는 입력 블록 하나가 빠진다.
              FWD 에서: 승부 체인은 같은 WinNet 에 x_f 를 두 블록에 넣고(σ(G·x_f)⊙tanh(V·x_f) — 유닛 형태를 안 바꾸는
              유일한 방법), tie 는 A-스택 4,096 = [abe2 ea | tb2 a_f | abe1 ea | tb1 x_f] · B-스택 3,072 (tb1 없음).
    좌표    스택별 z-score + 중심화 SVD-512 (각자 TR 통계 — 대칭이 없으니 [A;B] 쌓기 근거도 없다) → 몸통 [C_A 512 | C_B 512].
            승부 공급 소스별 (P1, P2) 1,024 z-score.
    헤드    TieNet · WinNet 그대로, 반사 없음 (plain_post) — synlab2 의 AB 와 같은 경로.

두 팔 — TB1 의 정보 손실과 뷰평균의 가치를 가른다
    FWD       전부 정방향 (위 그대로). FWD − AB 에는 "뷰평균의 가치" 와 "TB1 두 번째 뷰의 정보" 가 섞여 있다.
    FWD+TB1   나머지는 전부 정방향, TB1 만 (x_f, x_s) 두 뷰를 AB 와 같은 (A,B) 블록으로 유지 (B-스택 4,096).
              TB1 로짓은 정방향 그대로. → FWD+TB1 − AB = 입력 블록 수를 같게 둔 뷰평균(로짓·TB2 풀링·Arith)의 가치,
              FWD − FWD+TB1 = TB1 두 번째 뷰의 정보 가치.

기준 — synlab2 산출물(json · rowloss)에서 BASE · AB (+ /b, /percol, /ard) 를 읽는다 (같은 시드·같은 결합기).
    사다리   BASE (전부 on) → AB (층 0 TTA 만 on) → FWD+TB1 (TB1 두 뷰만) → FWD (전부 off)
    ! FWD 의 m 열은 정방향 자리 편향을 품는다 — 무중심 RMS 정규화·절편 없는 승부 결합기는 그 편향을 못 뺀다.
      /b (승부 결합기 절편·중심화) 가 그 반론을 막는다. 비교는 /b 끼리도 본다.

실행:  cd final/temp && python synlab3.py                          (ML2 · GPU · 2팔 × 5시드 ≈ 22분 · synlab2.json 필요)
       python synlab3.py --arms FWD                                 (한 팔만)
       nohup python -u synlab3.py > log_synlab3.txt 2>&1 &   tail -f log_synlab3.txt
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dest_fold as D                                                             # noqa: E402
from dest_fold import (FINAL, MODELS, SEED, TIE_ROUNDS, SRCS, DEVICE, R_NZ,        # noqa: E402
                       NFOLD, CK, FEAT_CACHE, DATA_DIR, build, load_train_logits, zstats)
from arith._base import build_swap, catalog_cols                                  # noqa: E402
from synlab import LAMS, SD_TIE, SD_WIN, ENC, bases, assets_uv                     # noqa: E402
import synlab2 as S2                                                              # noqa: E402
from synlab2 import KINDS, layer2, _lam_str, tie_head, win_head, paired           # noqa: E402

K_FWD = 512
SVD_FWD = {'FWD': FINAL / 'temp/head_cache/synlab3_svdFWD512.npz',
           'FWD+TB1': FINAL / 'temp/head_cache/synlab3_svdFWDtb1_512.npz'}
REF_JSON, REF_ROWS = FINAL / 'temp/synlab2.json', FINAL / 'temp/synlab2.rowloss.npz'
REF_KEYS = ['BASE', 'AB', 'AB/b', 'AB/percol', 'AB/ard']
ARMS = {'FWD': False, 'FWD+TB1': True}          # 팔 → TB1 두 뷰 유지 여부
LABEL = {
    'FWD':        'FWD — 정방향 뷰만 (층 0 TTA 없음) · (A,B) 좌표 · 대칭화 없음 — 모델 전체 대칭 제거',
    'FWD+TB1':    'FWD+TB1 — 전부 정방향이되 TB1 풀링만 (x_f, x_s) 두 뷰 유지 (입력 블록 수 = AB)',
    'BASE':       'BASE — (u,v) + 대칭화 + 층 0 TTA (노트북 현행) [synlab2]',
    'AB':         'AB — 층 0 TTA 만 유지 · 헤드 대칭 없음 [synlab2]',
    'AB/b':       'AB + 층2 승부 결합기 절편 [synlab2]',
    'AB/percol':  'AB + 층2 전문가별 격자 [synlab2]',
    'AB/ard':     'AB + 층2 ard [synlab2]',
}
for _a in ARMS:
    LABEL[f'{_a}/b'] = f'{_a} + 층2 승부 결합기 절편·중심화'
    LABEL[f'{_a}/percol'] = f'{_a} + 층2 전문가별 격자'
    LABEL[f'{_a}/ard'] = f'{_a} + 층2 ard'
S2.LABEL.update(LABEL)                       # paired() 가 라벨을 찾는다
# 정방향 풀링 키 — (P1, P2). tb1 은 x_f 하나뿐 (None → FWD: P1 복제 / FWD+TB1: x_s)
FWD_POOL = [('abe2__checkpoint-3000', 'ea', 'eb'), ('tb2__checkpoint-4000', 'a_f', 'b_f'),
            (f'abe1__checkpoint-{D.ABE1_STEP}', 'ea', 'eb'), (f'tb1__checkpoint-{D.TB1_STEP}', 'x_f', None)]


# ══════════════════════════════════════════ 정방향 로짓
def enc_logits_fwd(ids, split, y=None):
    """인코더 4종의 정방향 로짓 — fwd 만 (뷰평균 없음). 형상·id 는 enc_logits 와 같은 대조."""
    out = {}
    for tag, f, nc in ENC:
        if split == 's2':
            z = np.load(MODELS / f'logits/{f}.npz')
            assert (np.concatenate([z['valid_ids'], z['hold_ids']]) == ids).all(), f'{tag} id 정렬'
            L = np.vstack([z['valid_fwd'], z['hold_fwd']]).astype(np.float64)
        else:
            z = np.load(MODELS / f'logits/{f}__train.npz')
            assert (z['train_ids'] == ids).all() and (z['train_labels'] == y).all(), f'{tag} train 정렬'
            L = z['train_fwd'].astype(np.float64)
        assert L.shape == (len(ids), nc) and np.isfinite(L).all(), f'{tag} 형상'
        out[tag] = L
    return out


def arith_fwd_mean(U, tr_ids):
    """Arith 정방향 마진 5폴드 평균 (n,3) — dest_fold.arith_fold_logits 와 같은 피쳐·부스터, swap 판만 없다."""
    import xgboost as xgb
    assert FEAT_CACHE.exists(), f'{FEAT_CACHE} 가 없다'
    z = np.load(FEAT_CACHE, allow_pickle=False)
    ids, have, Xall = z['ids'], [str(c) for c in z['cols']], z['X']
    meta = json.load(open(CK / 'meta.json'))
    cols = list(meta['cols'])
    X = np.ascontiguousarray(Xall[:, [have.index(c) for c in cols]].astype(np.float32))
    assert np.isfinite(X).all(), '비유한 피쳐값'
    _, swap_series = catalog_cols(DATA_DIR / 'feature_catalog.csv')
    build_swap(cols, swap_series)                                    # 계약 검증만 (swap 판은 쓰지 않는다)
    dX = xgb.DMatrix(X)
    M = np.zeros((NFOLD, len(X), 3), np.float64)
    for k in range(1, NFOLD + 1):
        m = xgb.XGBClassifier()
        m.load_model(str(CK / f'xgb_fold{k}.json'))
        bst, bi = m.get_booster(), getattr(m, 'best_iteration', None)
        kw = dict(output_margin=True)
        if bi is not None:
            kw['iteration_range'] = (0, bi + 1)
        M[k - 1] = bst.predict(dX, **kw)
    fid = pd.Index(ids)
    i_s2, i_tr = fid.get_indexer(U['ids']), fid.get_indexer(tr_ids)
    assert (i_s2 >= 0).all() and (i_tr >= 0).all(), 'arith id 커버리지'
    Mm = M.mean(0)
    print(f'  arith 정방향 마진 {M.shape} (swap 판 없음) · checkpoint-ab224')
    return Mm[i_s2], Mm[i_tr]


# ══════════════════════════════════════════ 정방향 풀링 좌표
def fwd_assets(arm):
    """정방향 공급 — 소스별 (P1,P2) z-score (승부) + 스택별 SVD-512 (tie, 캐시). 통계는 전부 TR · 스택별 각자.
    arm='FWD': tb1 은 x_f 하나 (승부 복제 · tie B-스택 3,072). arm='FWD+TB1': tb1 (x_f, x_s) 두 뷰 (B-스택 4,096)."""
    tb1_both = ARMS[arm]
    P = {}
    for (tag, k1, k2), s in zip(FWD_POOL, SRCS):
        if k2 is None and tb1_both:
            k2 = 'x_s'                                               # FWD+TB1 — 스왑 뷰 풀링을 B 블록으로
        zt, zs = np.load(MODELS / f'pooled/{tag}__train.npz'), np.load(MODELS / f'pooled/{tag}.npz')
        g = lambda z, k: z[f'train_{k}'].astype(np.float32)
        c = lambda z, k: np.vstack([z[f'valid_{k}'], z[f'hold_{k}']]).astype(np.float32)
        P[s] = dict(p1_tr=g(zt, k1), p1_s2=c(zs, k1),
                    p2_tr=g(zt, k2) if k2 else None, p2_s2=c(zs, k2) if k2 else None)
    n_tr, n_s2 = P['abe2']['p1_tr'].shape[0], P['abe2']['p1_s2'].shape[0]
    assert (n_tr, n_s2) == (46729, 10748), f'풀링 행 {n_tr} {n_s2}'

    SUP = {}
    for s in SRCS:
        R = P[s]
        mu1, sc1 = zstats(R['p1_tr'].astype(np.float64), True)
        a_tr, a_s2 = ((R['p1_tr'] - mu1) / sc1).astype(np.float32), ((R['p1_s2'] - mu1) / sc1).astype(np.float32)
        if R['p2_tr'] is None:                                       # tb1 (FWD) — x_f 를 두 블록에 (유닛 형태 유지)
            b_tr, b_s2 = a_tr, a_s2
        else:
            mu2, sc2 = zstats(R['p2_tr'].astype(np.float64), True)
            b_tr, b_s2 = ((R['p2_tr'] - mu2) / sc2).astype(np.float32), ((R['p2_s2'] - mu2) / sc2).astype(np.float32)
        SUP[s] = dict(a_tr=a_tr, b_tr=b_tr, a_s2=a_s2, b_s2=b_s2)
    print(f'  [{arm}] 소스별 공급 — abe2 (ea,eb) · tb2 (a_f,b_f) · abe1 (ea,eb) · '
          f'tb1 {"(x_f,x_s 두 뷰)" if tb1_both else "(x_f, x_f 복제)"}  z-score 1,024')

    path, wb = SVD_FWD[arm], 4096 if tb1_both else 3072
    if path.exists():
        z = np.load(path)
        assert z['CA_tr'].shape == (n_tr, K_FWD) and z['CB_tr'].shape == (n_tr, K_FWD), 'SVD 캐시 형상'
        assert int(z['width_b']) == wb, f'SVD 캐시 B-스택 폭 {int(z["width_b"])} != {wb}'
        C = {k: z[k] for k in ('CA_tr', 'CB_tr', 'CA_s2', 'CB_s2')}
        print(f'  스택별 SVD-{K_FWD} 캐시 {path.name}  보존 분산 A {float(z["kept_a"]):.3f} · B {float(z["kept_b"]):.3f}')
    else:
        t0 = time.time()
        A_tr = np.hstack([P[s]['p1_tr'] for s in SRCS])                       # 4,096 (tb1 x_f 포함)
        A_s2 = np.hstack([P[s]['p1_s2'] for s in SRCS])
        B_tr = np.hstack([P[s]['p2_tr'] for s in SRCS if P[s]['p2_tr'] is not None])   # 3,072 / 4,096
        B_s2 = np.hstack([P[s]['p2_s2'] for s in SRCS if P[s]['p2_s2'] is not None])
        assert A_tr.shape[1] == 4096 and B_tr.shape[1] == wb, f'스택 폭 {A_tr.shape[1]} {B_tr.shape[1]} (기대 B {wb})'
        C, kept = {}, {}
        for nm, Xt, Xs in (('A', A_tr, A_s2), ('B', B_tr, B_s2)):
            mu, sc = zstats(Xt.astype(np.float64), True)
            X64 = (Xt.astype(np.float64) - mu) / sc
            mu2 = X64.mean(0)
            X64 -= mu2
            total = float((X64 ** 2).sum())
            _, Sv, Vt_ = randomized_svd(X64, n_components=K_FWD, random_state=SEED)
            W = Vt_.T
            Ct = X64 @ W
            scc = np.clip(np.sqrt((Ct ** 2).mean(0)), 1e-9, None)
            C[f'C{nm}_tr'] = (Ct / scc).astype(np.float32)
            C[f'C{nm}_s2'] = ((((Xs.astype(np.float64) - mu) / sc - mu2) @ W) / scc).astype(np.float32)
            kept[nm] = float((Sv ** 2).sum() / total)
            del X64, Ct
        np.savez(path, **C, K_FWD=K_FWD, kept_a=kept['A'], kept_b=kept['B'], width_b=wb)
        del A_tr, A_s2, B_tr, B_s2
        print(f'  스택별 SVD-{K_FWD} 계산 {(time.time() - t0) / 60:.1f}분 · 보존 분산 A {kept["A"]:.3f} · B {kept["B"]:.3f} → 캐시 {path.name}')
    del P
    gc.collect()
    return SUP, C


# ══════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seeds', default='42,1,2,3,4')
    ap.add_argument('--arms', default=','.join(ARMS), help='FWD,FWD+TB1 중 부분집합')
    ap.add_argument('--out', default='temp/synlab3')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s]
    arms = [a for a in args.arms.split(',') if a]
    assert set(arms) <= set(ARMS), f'팔은 {list(ARMS)} 중에서'
    t_all = time.time()
    assert REF_JSON.exists() and REF_ROWS.exists(), 'synlab2.json / synlab2.rowloss.npz 가 필요하다 — synlab2.py 를 먼저 돌려라'
    REF = json.load(open(REF_JSON))
    assert all(s in REF['seeds'] for s in seeds), f'synlab2 시드 {REF["seeds"]} 가 {seeds} 를 덮지 않는다'
    zr = np.load(REF_ROWS)
    print(f'[격자] 양축 λ ∈ {[f"{g:g}" for g in LAMS]}  · 기준 팔 {REF_KEYS} ← synlab2')
    print('[팔]   ' + '\n       '.join(f'{a:8s} {LABEL[a]}' for a in arms))

    print('\n[적재]')
    U = build()
    TR = load_train_logits()
    tr_ids, y_tr = TR[0], TR[1]
    E_s2, E_tr = enc_logits_fwd(U['ids'], 's2'), enc_logits_fwd(tr_ids, 'tr', y_tr)
    Mar_s2, Mar_tr = arith_fwd_mean(U, tr_ids)
    sd_s2, _ = bases(E_s2, Mar_s2)
    sd_tr, _ = bases(E_tr, Mar_tr)
    # 정방향 vs 뷰평균 로짓의 거리 — 자리 편향의 크기 (참고)
    d_m = float(np.abs(sd_s2['win'][:, :2] - U['enc_win']).mean())
    d_t = float(np.abs(sd_s2['tie'][:, :4] - U['enc_tie']).mean())
    print(f'  정방향 로짓 (m,t) — 뷰평균 대비 평균 |Δ| m {d_m:.4f} · t {d_t:.4f} · m 평균 (자리 편향) '
          f'{sd_s2["win"][:, :2].mean(0).round(4).tolist()} vs TTA {U["enc_win"].mean(0).round(4).tolist()} | 장치 {DEVICE}')
    D.wiring_checks(5, 3)

    # 기준 팔 — synlab2 에서
    res, rows_, store = {}, {}, {}
    for key in REF_KEYS:
        for sd in seeds:
            res[f'{key}-s{sd}'] = REF['cells'][f'{key}-s{sd}']
            for k in ('row', 'tie', 'win'):
                rows_[f'{k}_{key}-s{sd}'] = zr[f'{k}_{key}-s{sd}']

    variants = [('', False, 'blk2'), ('/b', True, 'blk2')] + [(f'/{k}', False, k) for k in KINDS]
    keys = []
    for arm in arms:
        SUP, C = fwd_assets(arm)
        A = assets_uv({'CUt': C['CA_tr'], 'CVt': C['CB_tr'], 'CUs': C['CA_s2'], 'CVs': C['CB_s2']}, TR, sd_tr, sd_s2)
        keys += [arm + v[0] for v in variants]
        for sd in seeds:
            tag = f'{arm}-s{sd}'
            print(f'\n{"=" * 78}\n[{tag}]  {LABEL[arm]}\n{"=" * 78}')
            t0 = time.time()
            Gt, st_t = tie_head(A, sd, 'ab', False)
            W, st_w = win_head(A, SUP, sd, 'ab', False)
            sec_h = time.time() - t0
            for suf, bw, kind in variants:
                key = f'{arm}{suf}'
                t1 = time.time()
                cell, row, lt, lw = layer2(U, sd_s2['tie'], sd_s2['win'], Gt, W, bw, kind)
                res[f'{key}-s{sd}'] = dict(arm=key, seed=sd, **cell, sec_heads=round(sec_h, 1), sec_l2=round(time.time() - t1, 1))
                rows_[f'row_{key}-s{sd}'], rows_[f'tie_{key}-s{sd}'], rows_[f'win_{key}-s{sd}'] = row, lt, lw
                print(f'  {key:15s} tie {cell["tie"]:.5f} · win {cell["win"]:.5f}{" (절편)" if bw else ""} '
                      f'→ **{cell["tot"]:.5f}**   λ {_lam_str(cell)}  [{res[f"{key}-s{sd}"]["sec_l2"]:.0f}s]')
            print(f'  [{sec_h:.0f}s 헤드]   (참고 같은 시드: BASE {res[f"BASE-s{sd}"]["tot"]:.5f} · AB {res[f"AB-s{sd}"]["tot"]:.5f})')
            store[f'Gt_{tag}'], store[f'W_{tag}'] = Gt.astype(np.float32), W.astype(np.float32)
            torch.save({'tie': st_t, 'win': st_w, 'arm': arm, 'seed': sd, 'coords': 'fwd', 'sym': False,
                        'tb1_both': ARMS[arm], 'tie_names': SD_TIE, 'win_names': SD_WIN},
                       FINAL / f'{args.out}_heads_{tag}.pt')
        del SUP, C, A
        gc.collect()

    op = FINAL / args.out
    np.savez_compressed(op.with_suffix('.rowloss.npz'), **{k: v for k, v in rows_.items() if 'FWD' in k}, **store)
    summary = {}
    for key in keys + REF_KEYS:
        g = lambda k: np.array([res[f'{key}-s{sd}'][k] for sd in seeds])
        summary[key] = dict(tie=float(g('tie').mean()), win=float(g('win').mean()), tot=float(g('tot').mean()),
                            sd=float(g('tot').std(ddof=1)) if len(seeds) > 1 else 0.0)
    rng = np.random.default_rng(SEED)
    for arm in arms:
        print(f'\n{"=" * 78}\n[{arm}] {LABEL[arm]}\n{len(seeds)}시드 · holdout 5,748\n{"=" * 78}')
        print(f'  {"시드":6s}{"tie ll":>10s}{"win ll":>10s}{"총 3-class":>12s}   λ(tie) / λ(win)')
        for sd in seeds:
            c = res[f'{arm}-s{sd}']
            print(f'  {sd:<6d}{c["tie"]:10.5f}{c["win"]:10.5f}{c["tot"]:12.5f}   {_lam_str(c)}')
        s = summary[arm]
        print(f'  {"평균":6s}{s["tie"]:10.5f}{s["win"]:10.5f}{s["tot"]:12.5f}   (시드 sd {s["sd"]:.5f})')

        print(f'\n  [벌점 방식 × 축] {arm} · blk2 대비 paired Δ')
        print(f'  {"방식":8s}{"tie ll":>9s}{"Δtie":>9s}{"95% CI":>20s} 악화 {"win ll":>9s}{"Δwin":>9s}{"95% CI":>20s} 악화 {"tot":>9s}')
        for kind in ('blk2',) + KINDS:
            key = arm if kind == 'blk2' else f'{arm}/{kind}'
            s = summary[key]
            line = f'  {kind:8s}{s["tie"]:9.5f}'
            if kind == 'blk2':
                line += f'{"—":>9s}{"":>20s}     {s["win"]:9.5f}{"—":>9s}{"":>20s}     '
            else:
                for ax in ('tie', 'win'):
                    a = np.mean([rows_[f'{ax}_{key}-s{sd}'] for sd in seeds], 0)
                    b = np.mean([rows_[f'{ax}_{arm}-s{sd}'] for sd in seeds], 0)
                    d = a - b
                    bs = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(1)
                    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
                    per = np.array([res[f'{key}-s{sd}'][ax] - res[f'{arm}-s{sd}'][ax] for sd in seeds])
                    line += (f'{s["win"]:9.5f}' if ax == 'win' else '') + \
                        f'{d.mean():+9.5f} [{lo:+.5f},{hi:+.5f}]{"★" if (lo > 0 or hi < 0) else " "} {int((per > 0).sum())}/{len(seeds)} '
                    summary[key].setdefault('vs_blk2', {})[ax] = dict(mean=float(d.mean()), lo=lo, hi=hi, worse=int((per > 0).sum()))
            print(line + f'{s["tot"]:9.5f}')

    print(f'\n{"=" * 78}\n[paired] {len(seeds)}시드 · 행별 부트스트랩 95% CI · 양수 = 앞 팔이 나쁨\n{"=" * 78}')
    PAIRS = [('AB', 'BASE')]
    if 'FWD+TB1' in arms:
        PAIRS += [('FWD+TB1', 'AB'), ('FWD+TB1/b', 'AB/b'), ('FWD+TB1', 'BASE')]
    if 'FWD' in arms:
        PAIRS += [('FWD', 'AB'), ('FWD/b', 'AB/b'), ('FWD', 'BASE')]
    if set(arms) == set(ARMS):
        PAIRS += [('FWD', 'FWD+TB1'), ('FWD/b', 'FWD+TB1/b')]
    for a, b in PAIRS:
        summary[a][f'vs_{b}'] = paired(res, rows_, seeds, a, base=b)

    print(f'\n{"=" * 78}\n[사다리] tot 시드 평균 — 대칭을 한 층씩 뗀다\n{"=" * 78}')
    print(f'  {"팔":12s}{"tie":>9s}{"win":>9s}{"tot":>9s}   무엇이 남아 있나')
    LADDER = [('BASE', '층0 TTA + 층1 (u,v)·대칭화 + 층2 홀 결합기'), ('AB', '층0 TTA 만 (헤드 대칭 없음)'),
              ('FWD+TB1', 'TB1 두 뷰 풀링만 (뷰평균 없음)'), ('FWD', '없음 (인코더 학습 시 swap 증강만)'),
              ('AB/b', '층0 TTA · 층2 승부 절편'), ('FWD+TB1/b', 'TB1 두 뷰 · 층2 승부 절편'), ('FWD/b', '없음 · 층2 승부 절편')]
    for key, what in LADDER:
        if key in summary:
            s = summary[key]
            print(f'  {key:12s}{s["tie"]:9.5f}{s["win"]:9.5f}{s["tot"]:9.5f}   {what}')

    def _d(a, b):
        v = summary[a][f'vs_{b}']
        return (f'{v["tot"]["mean"]:+.5f} [{v["tot"]["lo"]:+.5f},{v["tot"]["hi"]:+.5f}] 악화 {v["tot"]["worse"]}/{len(seeds)}'
                f'   (tie {v["tie"]["mean"]:+.5f} · win {v["win"]["mean"]:+.5f})')
    print(f'\n  헤드 대칭 (AB − BASE)            {_d("AB", "BASE")}')
    if 'FWD+TB1' in arms:
        print(f'  뷰평균·블록 동수 (FWD+TB1 − AB)  {_d("FWD+TB1", "AB")}')
    if 'FWD' in arms:
        print(f'  전부 제거 (FWD − AB)             {_d("FWD", "AB")}')
        print(f'  전체 (FWD − BASE)                {_d("FWD", "BASE")}')
    if set(arms) == set(ARMS):
        print(f'  TB1 두 번째 뷰 (FWD − FWD+TB1)   {_d("FWD", "FWD+TB1")}')
    print('  ! 층0 효과는 "헤드 대칭이 없을 때"의 값이다 — BASE 에서 층0 TTA 만 빼는 팔은 u/v 가 근사가 되어 정의가 다르다.')
    print('  ! tot = tie + 0.691 × win. 문턱: CI 0 배제 + |Δ| ≥ 0.0005 + 시드 5/5')

    with open(op.with_suffix('.json'), 'w') as f:
        json.dump({'cells': {k: v for k, v in res.items() if 'FWD' in k}, 'summary': summary, 'seeds': seeds,
                   'arms': arms, 'keys': keys, 'ref_keys': REF_KEYS, 'ref': str(REF_JSON.name), 'lams': LAMS,
                   'labels': LABEL, 'params': dict(tie=262537, win=262409), 'K_FWD': K_FWD,
                   'kinds': ['blk2', *KINDS]}, f, ensure_ascii=False, indent=1)
    print(f'\n저장: {op}.rowloss.npz / {op}.json / {args.out}_heads_*.pt')
    print(f'총 {((time.time() - t_all) / 60):.1f}분')


if __name__ == '__main__':
    main()
