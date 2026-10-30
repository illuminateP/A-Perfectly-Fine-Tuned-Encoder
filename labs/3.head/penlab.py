"""penlab.py — 벌점을 **fold 마다 고르는 것**과 **고정하는 것**의 직접 비교 + 전이 검정.

(구 tie_ladder 판은 penlab_v1_tieladder.py 로 보존. 결과 penlab*.json/npz 도 그대로 둔다.)

무엇이 문제인가
    현행 계약은 fold 마다 안쪽 3-fold CV 가 λ 를 다시 고른다. 우리는 그걸 고정값으로 바꾸면
    홀드아웃 점수가 좋아진다는 것을 봤지만, **그 고정값 자체가 홀드아웃을 본 절차에서 나왔다.**
    그래서 "고정이 더 좋다"는 홀드아웃 점수로 인증할 수 없다 (순환).

    이 파일은 두 체제를 같은 조건에 놓고, **선택에 쓰이지 않은 행에서** 비교한다.

두 체제 (같은 열 · 같은 fold 추첨 · 같은 헤드)
    adaptive     fold 마다 안쪽 3-fold CV 가 λ 재선택 — destnation 계약 원본. **기준선**
                 tie = LAMS0 × LAMS 30점 2-D 격자 (축별 자동확장) · win = LAMS 1-D · m3 무벌점
    fix-group    전 fold 고정 (100,10)/10 — lam_ceiling 오라클이 지목한 그룹 λ (자유도 2)
    fix-B        fix-group + {t_abe1, m_abe2} → 1e5   (무라벨 R² 로 고른 열)
    fix-H        fix-group + {s_tb1, t_abe1, m_abe2, W_tb1} → 1e5   (홀드아웃 유래)

행 세그먼트 — 벌점 선택은 전부 hold 5,748 에서만 했다
    hold      5,748  선택셋. 여기 Δ 는 낙관을 포함한다 (참조용)
    calib2k   2,000  cu 안 · 비-holdout. **다섯 전문가 전부 청정** · OOF z
    dirty3k   3,000  cu 밖. V3 49,729 에 3,000/3,000 포함 → **ABE1 만 in-sample**
                     (현행 train.csv 46,729 에는 0개 — ABE2·TB1·TB2·Arith 청정) · fold 평균 z
    out5k     5,000  calib2k + dirty3k — 최대 검정력 (SE 비 1.07)
    train46k 46,729 `--train` 일 때만. 검정력은 최고(SE 비 0.35)지만 **12/14 열이 in-sample**
                     (인코더 4태그·헤드가 train 에서 학습) → 최적 벌점 자체가 「낮은 λ·제외
                     없음」으로 **이동**한다. 선택은 여기서 하지 않고 **평가만** 한다.
                     사전 등록 예측: adaptive > fix-group > fix-B > fix-H (제외 열 수에 비례).
                     그 순서가 나오면 편향 예측대로라 정보 없음 · 어긋나면 그게 정보다.

    ! dirty3k 는 ABE1 열(t_abe1·m_abe1·W_abe1·g1·g2)이 과대평가된다 → ABE1 열을 버리는
      fix-B/fix-H 에 **불리한** 보수적 검정. fix-group 은 아무 열도 안 버리므로 영향이 작다.

무엇을 저장하나 — **벌점 열 자체**
    penlab.json   fix-* 의 벌점 벡터(각 7열) + adaptive 가 (헤드시드, 추첨, fold) 마다
                  실제로 고른 벌점 벡터 전부. 열 이름과 함께 저장해 직접 대조할 수 있다.
    penlab.npz    세그먼트별 행손실 (체제 × 조합) — 사후 paired 분석용

실행:  conda activate ML2 && cd final && python temp/penlab.py
       python temp/penlab.py --hs 5 --reps 5        # 헤드시드까지 평균 (~45분)
예상:  기본 **헤드시드 42 × 추첨 4개(42~45) = 4조합** · adaptive 조합당 ~85초 → **~8분**
       헤드 s42 는 dest_universe 산출(ab224 계보)을 캐시에서 그대로 쓴다 — 학습 없음.
       ! 헤드시드 1개이므로 헤드 초기화 잡음이 평균되지 않는다. 기준·셀이 같은 헤드를
         공유해 paired 에서 대부분 소거되지만, 부호 검사는 추첨 축(4개)에서 한다.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

FINAL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FINAL / 'temp'))
import finlab as FL                                  # noqa: E402  검증된 현행 계약 기계

TIE_NAMES, WIN_NAMES = FL.TIE_NAMES, FL.WIN_NAMES
EXCL = FL.EXCL


def _fwd(net_fn, state, cu_, cv_):
    """저장된 state 로 헤드를 재구성해 임의 입력에 순전파. destnation 은 건드리지 않는다.
    (fit_head 가 state 를 반환하므로 train 측 열을 이렇게 되살릴 수 있다)"""
    d = cu_.shape[1] + cv_.shape[1]
    net = net_fn(d, FL.D.H).to(FL.D.DEVICE)
    net.load_state_dict(state)
    net.eval()
    T = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=torch.float32,
                               device=FL.D.DEVICE)
    with torch.no_grad():
        return FL.D.sym_post(net, T(cu_), T(cv_), 1 if isinstance(net, FL.D.TieNet)
                             else -1).cpu().numpy().astype(np.float64)


def heads_both(seed=42):
    """헤드를 학습해 **stage2 열과 train 열을 동시에** 돌려준다 (캐시).

    train 측 열은 fit_head 의 state 로 재순전파해 얻는다 — destnation 수정 없음.
    ! train 46,729 는 헤드의 학습셋이므로 그 열은 in-sample 이다 (이 파일의 전제).
    """
    cache = FINAL / f'temp/penlab_trainheads_s{seed}.npz'
    if cache.exists():
        z = np.load(cache)
        print(f'  train 헤드 열 캐시 재사용 — {cache.name}')
        return {k2: z[k2].astype(np.float64) for k2 in z.files}
    print('  train 측 헤드 열 생성 — SVD·풀링 적재 + 헤드 학습 (~6분)')
    U0 = FL.build()
    Z = FL.D.load_svd_cache()
    FL.D.load_train_logits = FL._train_logits_ab224          # ab224 계보
    A = FL.D.head_assets(U0, Z)
    RAW = FL.D.raw_uv(Z)

    ct = np.ascontiguousarray(np.hstack([A['CUt'], A['Lu_tr']]), np.float32)
    cv = np.ascontiguousarray(np.hstack([A['CVt'], A['Lv_tr']]), np.float32)
    cs = np.ascontiguousarray(np.hstack([A['CUs'], A['Lu_s2']]), np.float32)
    vs = np.ascontiguousarray(np.hstack([A['CVs'], A['Lv_s2']]), np.float32)
    Kd, nu, nv = A['CUt'].shape[1], A['Lu_tr'].shape[1], A['Lv_tr'].shape[1]
    sk = list(range(Kd, Kd + nu)) + list(range(2 * Kd + nu, 2 * Kd + nu + nv))
    pool = list(range(0, Kd)) + list(range(Kd + nu, 2 * Kd + nu))
    zt_anchor, g_s2, g_tr = None, [], []
    for unit in ('TANHTANH', 'BILIN')[:FL.D.TIE_ROUNDS]:
        nf = lambda d, h, _u=unit: FL.D.TieNet(d, h, _u, sk, pool)
        r, aux = FL.D.fit_head(ct, cv, A['y01m'], zt_anchor, True, seed, cs, vs, nf)
        g_s2.append(r)
        g_tr.append(_fwd(nf, aux['state'], ct, cv))
        zt_anchor = aux['z_tr']
        print(f'    tie {unit:9s} es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]}')

    m = A['y_tr'] != 2
    y01w = (A['y_tr'] == 0).astype(np.float64)[m]
    w_s2, w_tr = [], []
    for si, sname in enumerate(FL.D.SRCS):
        R = RAW[sname]
        du, dv = R['u_tr'].shape[1], R['v_tr'].shape[1]
        skw = list(range(du, du + nu)) + list(range(du + nu + dv, du + nu + dv + nv))
        pw_ = list(range(0, du)) + list(range(du + nu, du + nu + dv))
        cu_t = np.ascontiguousarray(np.hstack([R['u_tr'][m], A['Lu_tr'][m]]), np.float32)
        cv_t = np.ascontiguousarray(np.hstack([R['v_tr'][m], A['Lv_tr'][m]]), np.float32)
        cu_a = np.ascontiguousarray(np.hstack([R['u_tr'], A['Lu_tr']]), np.float32)
        cv_a = np.ascontiguousarray(np.hstack([R['v_tr'], A['Lv_tr']]), np.float32)
        cu_s = np.ascontiguousarray(np.hstack([R['u_s2'], A['Lu_s2']]), np.float32)
        cv_s = np.ascontiguousarray(np.hstack([R['v_s2'], A['Lv_s2']]), np.float32)
        nf = lambda d, h, _s=skw, _p=pw_, _du=du: FL.D.WinNet(d, h, _s, _p, _du)
        r, aux = FL.D.fit_head(cu_t, cv_t, y01w, None, False, seed + si, cu_s, cv_s, nf)
        w_s2.append(r)
        w_tr.append(_fwd(nf, aux['state'], cu_a, cv_a))     # 전 train 행 (비-tie 마스크 없이)
        print(f'    win {sname:5s} es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]}')
        del cu_t, cv_t, cu_a, cv_a, cu_s, cv_s
        gc.collect()

    ids_tr, y_tr, tie_tr, win_tr = FL._train_logits_ab224()
    out = dict(Gt_s2=np.column_stack(g_s2), W_s2=np.column_stack(w_s2),
               Gt_tr=np.column_stack(g_tr), W_tr=np.column_stack(w_tr),
               XT_tr=np.hstack([tie_tr, np.column_stack(g_tr)]),
               XW_tr=np.hstack([win_tr, np.column_stack(w_tr)]),
               y_tr=y_tr, ids_tr=ids_tr)
    np.savez_compressed(cache, **{k2: v.astype(np.float32) if v.dtype == np.float64 else v
                                  for k2, v in out.items()})
    del A, RAW, Z
    gc.collect()
    print(f'  저장: {cache.name}  XT_tr {out["XT_tr"].shape} · XW_tr {out["XW_tr"].shape}')
    return {k2: v.astype(np.float64) if k2 != 'ids_tr' else v for k2, v in out.items()}


def train_pred(XT_s2, XW_s2, y, nz, cu, fold7, pen_t, pen_w, XT_tr, XW_tr, adaptive=False):
    """stage2 cu 에 적합한 결합기로 **train 46,729 행을 예측**한다 (fold 평균).

    `_z` 정규화 통계는 적합 행(cu 의 4/5)에서 나오고 그대로 train 열에 적용된다 —
    dirty3k 를 다루는 방식과 같은 규약이다. 선택은 여기서 하지 않는다: 벌점은 인자로 받는다.
    """
    yT, yA = (y == 2).astype(np.float64), (y == 0).astype(np.float64)
    idx7 = np.where(cu)[0]
    out = {}
    for tag, X, y01, pen, even, rows7, Xtr_ in (
            ('t', XT_s2, yT, pen_t, True, np.ones(len(idx7), bool), XT_tr),
            ('w', XW_s2, yA, pen_w, False, nz[cu], XW_tr)):
        n_even = X.shape[1] if even else 0
        acc = np.zeros(len(Xtr_))
        for k in range(FL.K):
            tr7 = (fold7 != k) & rows7
            tri = idx7[tr7]
            A, B = FL.D._z(X[tri], Xtr_, n_even)
            p = pen if not adaptive else pen[(tag, k)]
            v, ok = FL.D._fit(A, y01[tri], p, even)
            assert ok, f'{tag} fold{k} 미수렴'
            acc += FL.D._scores(v, B, n_even, even)
        out[tag] = acc / FL.K
    return out['t'], out['w']


# ── 1a·1b 오라클 벡터 (finlab exp1 런 출력 · 2026-08-17) ──────────────
#    finlab.json 은 그 뒤 부분 실행(--exps f/g 등)에 덮여 exp1 항목이 없다. 런 출력의 값을
#    출처와 함께 박아 둔다. '제외급' = λ≥1e5 · '무벌점' = 0 (그때 출력의 태그).
#      1a  tie λ=(220, 30) · win λ=15                      미세 격자 오라클
#      1b  tie s_tb2=291 s_tb1=제외급 t_abe2=220 t_abe1=제외급 t_ar=157 g1=30 g2=227
#          win m_abe2=제외급 m_abe1=59 m_ar=무벌점 W_abe2=7.18 W_tb2=25 W_abe1=59.3 W_tb1=160
LAM_1A = ((220.0, 30.0), 15.0)
PEN_1B_TIE = [291.0, EXCL, 220.0, EXCL, 157.0, 30.0, 227.0]
PEN_1B_WIN = [EXCL, 59.0, 0.0, 7.18, 25.0, 59.3, 160.0]


def fixed_vecs():
    """고정 체제의 벌점 벡터. 계약 λ 기준 + 제외 열은 1e5.

    fix-group  (100,10)/10  — 계약 그룹 λ · adaptive 선택의 **전체 최빈값**이자 lam_ceiling 오라클
    fix-T      + s_tb1·t_abe1·m_abe2 제외      (열정리만)
    fix-B      + t_abe1·m_abe2 제외            (무라벨 R² 로 고른 2열)
    fix-H      + s_tb1·t_abe1·m_abe2·W_tb1 제외 (홀드아웃 유래 4열)
    fix-fine   (220,30)/15  — 1a 미세 격자 오라클 (그룹 λ, 자유도 2)
    fix-1b     열별 연속 오라클 (자유도 14) — 캠페인 최대 천장 −0.00073 을 낸 벡터
    ! fix-median 은 추첨마다 달라지므로 여기서 만들지 않고 루프 안에서 adaptive 선택의
      성분 중앙값으로 만든다 (destnation 이 실제로 배포하는 λ).
    """
    out = {}
    for nm, key in (('fix-group', None), ('fix-T', 'T'), ('fix-B', 'B'), ('fix-H', 'H')):
        et, ew = FL.VEC[key] if key else (frozenset(), frozenset())
        out[nm] = FL.pen_vec(et, ew, False)
    (a, b), w = LAM_1A
    out['fix-fine'] = (np.array([a] * 5 + [b] * 2), np.array([0.0] * 3 + [w] * 4))
    out['fix-1b'] = (np.array(PEN_1B_TIE), np.array(PEN_1B_WIN))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--hs', type=int, default=1)      # 헤드 시드 개수 (42,1,2,5,6 순)
    ap.add_argument('--reps', type=int, default=4)    # fold 추첨 개수 (42..)
    ap.add_argument('--train', action='store_true',
                    help='train 46,729 세그먼트 추가 (12/14 열 in-sample — 편향 주의)')
    ap.add_argument('--out', default='temp/penlab')
    args = ap.parse_args()
    head_seeds = list(FL.HS_DEFAULT)[:args.hs]
    fold_seeds = list(range(42, 42 + args.reps))
    t_all = time.time()

    print('[적재]')
    U = FL.build()
    cu, y, nz, hold = U['cu'], U['y'], U['nz'], U['hold']
    yT, yA = (y == 2).astype(np.float64), (y == 0).astype(np.float64)
    SEG = {'hold': hold, 'calib2k': cu & ~hold, 'dirty3k': ~cu, 'out5k': ~hold}
    for k2, v2 in SEG.items():
        print(f'  {k2:9s} {int(v2.sum()):>6,}행 · 비-tie {int((v2 & nz).sum()):>5,} · '
              f'SE 비 {(int(hold.sum()) / int(v2.sum())) ** 0.5:.2f}')
    HD = FL._heads_ab224(U, head_seeds)
    TR = None
    if args.train:
        assert head_seeds == [42], '--train 은 헤드시드 42 전용 (train 열이 그 헤드로 만들어진다)'
        TR = heads_both(42)
        y_trn = TR['y_tr'].astype(np.int64)
        nz_trn = y_trn != 2
        yT_trn = (y_trn == 2).astype(np.float64)
        yA_trn = (y_trn == 0).astype(np.float64)
        print(f'  train46k {len(y_trn):,}행 · 비-tie {int(nz_trn.sum()):,} · '
              f'SE 비 {(int(hold.sum()) / len(y_trn)) ** 0.5:.2f}')
        print('  ! train 은 12/14 열이 in-sample — **선택이 아니라 평가만** 한다.')
        print('    사전 등록 예측: 편향이 「낮은 λ · 제외 없음」을 선호하므로')
        print('    adaptive > fix-group > fix-B > fix-H (fix-* 전부 양수, 제외 열 수에 비례).')
        print('    이 순서가 나오면 편향 예측대로라 정보가 없고, 어긋나면 그게 정보다.')
    FIX = fixed_vecs()
    folds = {fs: FL.fold_of(y[cu], fs) for fs in fold_seeds}
    print(f'\n[체제] 기준 adaptive(계약 원본) vs {list(FIX)}')
    print(f'  헤드시드 {head_seeds} × 추첨 {fold_seeds} = {len(head_seeds) * len(folds)}조합')
    tag = lambda p: '0' if p == 0 else ('제외' if p >= EXCL else f'{p:g}')
    for nm, (pt, pw) in FIX.items():
        print(f'  {nm:10s} tie ' + ' '.join(f'{n}={tag(p)}' for n, p in zip(TIE_NAMES, pt)))
        print(f'  {"":10s} win ' + ' '.join(f'{n}={tag(p)}' for n, p in zip(WIN_NAMES, pw)))

    SEG_ALL = list(SEG) + (['train46k'] if args.train else [])
    ALL_NM = list(FIX) + ['fix-median', 'adaptive']
    rows = {nm: {sg: {} for sg in SEG_ALL} for nm in ALL_NM}
    tots = {nm: {} for nm in ALL_NM}
    med_seen = {}
    ad_picks = {}
    for hs in head_seeds:
        Gt, Wc = HD[hs]
        XT = np.hstack([U['tie_base'], Gt])
        XW = np.hstack([U['win_base'], Wc])
        t0 = time.time()
        for fs, fd7 in folds.items():
            zt, zw, pk = FL._run_adaptive(U, fd7, XT, XW, y, nz, cu)
            ad_picks[f's{hs}_f{fs}'] = pk
            for sg, m in SEG.items():
                rows['adaptive'][sg][(hs, fs)] = FL._seg_rows(zt, zw, yT, yA, m, nz)
            tots['adaptive'][(hs, fs)] = float(rows['adaptive']['hold'][(hs, fs)].mean())
            if TR is not None:                        # adaptive 는 fold 별 선택 벌점을 그대로
                pmap = {('t', k): np.array(pk['t'][k]) for k in range(FL.K)}
                pmap.update({('w', k): np.array(pk['w'][k]) for k in range(FL.K)})
                zt_t, zw_t = train_pred(XT, XW, y, nz, cu, fd7, pmap, pmap,
                                        TR['XT_tr'], TR['XW_tr'], adaptive=True)
                rows['adaptive']['train46k'][(hs, fs)] = FL._seg_rows(
                    zt_t, zw_t, yT_trn, yA_trn, np.ones(len(y_trn), bool), nz_trn)
            # fix-median — 이 추첨의 adaptive 선택 성분 중앙값 (destnation 배포 λ)
            FIX_D = dict(FIX)
            FIX_D['fix-median'] = (np.median(np.array(pk['t']), axis=0),
                                   np.median(np.array(pk['w']), axis=0))
            med_seen[f's{hs}_f{fs}'] = [FIX_D['fix-median'][0].tolist(),
                                        FIX_D['fix-median'][1].tolist()]
            for nm, (pt, pw) in FIX_D.items():
                zt, zw = FL._run_full(U, fd7, XT, XW, pt, pw, y, nz, cu)
                for sg, m in SEG.items():
                    rows[nm][sg][(hs, fs)] = FL._seg_rows(zt, zw, yT, yA, m, nz)
                tots[nm][(hs, fs)] = float(rows[nm]['hold'][(hs, fs)].mean())
                if TR is not None:
                    zt_t, zw_t = train_pred(XT, XW, y, nz, cu, fd7, pt, pw,
                                            TR['XT_tr'], TR['XW_tr'])
                    rows[nm]['train46k'][(hs, fs)] = FL._seg_rows(
                        zt_t, zw_t, yT_trn, yA_trn, np.ones(len(y_trn), bool), nz_trn)
        print(f'  헤드 시드 {hs} 완료 [{(time.time() - t0) / 60:.1f}분]')

    keys = sorted(rows['adaptive']['hold'])
    # ── adaptive 가 실제로 고른 벌점 열 ──
    print(f'\n{"=" * 78}\n[adaptive 가 고른 벌점] {len(keys)}조합 × 5 fold = '
          f'{len(keys) * 5}개 벡터\n{"=" * 78}')
    for ax, nms in (('t', TIE_NAMES), ('w', WIN_NAMES)):
        allp = np.array([v for pk in ad_picks.values() for v in pk[ax]])
        print(f'  {"tie" if ax == "t" else "win"} 축 — 열별 선택 분포 (고유값: 횟수)')
        for j, nm in enumerate(nms):
            u, c = np.unique(allp[:, j], return_counts=True)
            print(f'    {nm:8s} ' + ' · '.join(f'{tag(x)}:{n}' for x, n in zip(u, c)))
        print(f'    → 서로 다른 벡터 {len(np.unique(allp, axis=0))}종 / {len(allp)}개')

    # ── 체제 비교 (adaptive 기준 paired) ──
    print(f'\n{"=" * 78}\n[체제 비교] adaptive 기준 Δ · 음수 = 고정이 더 좋다\n{"=" * 78}')
    print(f'  {"체제":10s}' + ''.join(f'{sg:>22s}' for sg in SEG_ALL))
    print(f'  {"":10s}' + ''.join(f'{"Δ":>11s}{"t":>11s}' for _ in SEG_ALL))
    out = {}
    for nm in [x for x in ALL_NM if x != 'adaptive']:
        line, o = f'  {nm:10s}', {}
        for sg in SEG_ALL:
            dr = np.mean([rows[nm][sg][k2] - rows['adaptive'][sg][k2] for k2 in keys], axis=0)
            se = float(dr.std(ddof=1) / np.sqrt(len(dr)))
            t = float(dr.mean() / se) if se > 0 else 0.0
            line += f'{dr.mean():+11.6f}{t:>11.2f}'
            o[sg] = dict(d=float(dr.mean()), t=t, se=se, n=int(len(dr)))
        print(line)
        out[nm] = o
    # 부호 검사 축 — 헤드시드가 2개 이상이면 그 축, 1개면 추첨 축
    by_hs = len(head_seeds) > 1
    axis_nm = '헤드시드별 부호' if by_hs else '추첨별 부호'
    print(f'\n  {"체제":10s}{"hold 절대 ll":>16s}{axis_nm:>18s}')
    for nm in ['adaptive'] + [x for x in ALL_NM if x != 'adaptive']:
        v = np.array([tots[nm][k2] for k2 in keys])
        if nm == 'adaptive':
            print(f'  {nm:10s}{v.mean():16.5f}{"(기준)":>18s}')
            continue
        if by_hs:
            per = [float(np.mean([tots[nm][(h, f_)] - tots['adaptive'][(h, f_)]
                                 for f_ in folds])) for h in head_seeds]
        else:
            per = [float(np.mean([tots[nm][(h, f_)] - tots['adaptive'][(h, f_)]
                                 for h in head_seeds])) for f_ in folds]
        sg_ = ('전부−' if all(d < 0 for d in per)
               else '전부+' if all(d > 0 for d in per) else '혼합')
        print(f'  {nm:10s}{v.mean():16.5f}{sg_:>18s}   ' + ' '.join(f'{d:+.5f}' for d in per))

    op = FINAL / args.out
    json.dump({'fixed': {nm: {'tie': [float(x) for x in pt], 'win': [float(x) for x in pw]}
                         for nm, (pt, pw) in FIX.items()},
               'tie_names': TIE_NAMES, 'win_names': WIN_NAMES,
               'adaptive_picks': {k2: {a: [[float(x) for x in v] for v in vv]
                                       for a, vv in pk.items()}
                                  for k2, pk in ad_picks.items()},
               'head_seeds': head_seeds, 'fold_seeds': fold_seeds,
               'segs': {k2: int(v2.sum()) for k2, v2 in SEG.items()},
               'median_by_draw': med_seen,
               'cmp': out, 'hold_tot': {nm: {f'{h}_{f_}': tots[nm][(h, f_)]
                                             for h, f_ in keys} for nm in tots}},
              open(op.with_suffix('.json'), 'w'), ensure_ascii=False, indent=1)
    np.savez_compressed(op.with_suffix('.npz'),
                        **{f'{nm}__{sg}__s{h}_f{f_}': rows[nm][sg][(h, f_)]
                           for nm in rows for sg in SEG_ALL for h, f_ in keys})
    print(f'\n저장: {op}.json (벌점 열 전부) / {op}.npz (세그먼트 행손실)')
    print('읽는 법')
    print('  · hold 는 선택셋 — 고정 체제에 유리하게 편향돼 있다 (고정값이 여기서 나왔다)')
    print('  · calib2k 가 유일하게 완전 청정 — 여기 부호가 진짜 방향이다')
    print('  · out5k 는 검정력 최대(SE 비 1.07) 다만 dirty3k 의 ABE1 편향이 섞인다')
    print('  · fix-group 은 열을 버리지 않으므로 ABE1 편향에 거의 영향받지 않는다 →')
    print('    **fix-group 의 calib2k·out5k 부호가 「λ 동결」 질문의 답이다**')
    print(f'\n총 {(time.time() - t_all) / 60:.1f}분')


if __name__ == '__main__':
    main()
