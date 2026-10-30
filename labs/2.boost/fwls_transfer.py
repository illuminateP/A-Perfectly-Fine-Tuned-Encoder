"""fwls_transfer.py — train(46,729)으로 FWLS 를 적합해 stage2 로 이식하면 성립하는가.

가설 (사용자 제안)
  ABE·TB·Arith 는 훈련 중 서로를 본 적이 없으므로 train 로짓을 블렌딩 근거로 재활용해도
  된다 — FWLS 를 train 으로 적합하고 stage2 로 채점하자.

반론 (실측 근거)
  FWLS 의 지도 신호는 행별 전문가 정답 여부다. train 에서 인코더는 in-sample:
      ABE-3000 승부 AUC 0.8509(in) vs 0.7503(out)  격차 +0.1005
      TB3-4000 tie  AUC 0.7245(in) vs 0.6563(out)  격차 +0.0683
      Arith 는 train OOF 저장(raw_oof_train__*)이라 격차 0 — **비대칭 오염**
  train 세계에서는 ABE·TB 가 실제보다 강해 보이므로 기저 가중치부터 과신 쪽으로 왜곡되고,
  게이트는 실력 국면과 암기 국면을 구분하지 못한다. …는 예측이고, 여기서 직접 잰다.

사다리 (전부 같은 stage2 행에서 paired 비교)
  C0  stage2 5-fold 교차적합 (기존 프로토콜)             — 앵커
  C1  F0(기저 로짓 가중만)을 train 적합 -> stage2 채점    — 5 파라미터 수준의 이식 왜곡
  C2  게이트 팔을 train 적합 -> stage2 채점               — 본런 후보 2팔 + kitchen-sink
  판정: C1 < C0 유의면 암기 왜곡 실증(방향 폐쇄). C1 ≈ C0 이고 C2 가 C0 의 F0 을
  이기면 제안 채택 — 겸사겸사 본런 후보(태도·기타/반복)의 독립 방향 확인도 된다.

계약 주의
  train 에 존재하는 로짓은 단일 ckpt(abe-3000, tb3-4000)뿐이라 **양쪽 모두 단일 ckpt** 로
  맞춘다. 본선(ABE 7-ckpt 평균)의 0.9705/0.5739/0.5740 과 직접 비교하지 말 것 —
  이 파일 안의 C0 이 유일한 비교 기준이다. rankgauss 는 57k 공통 적합(라벨 무관).

사용 (전체 약 15~25분. 빠른 검산: --only F0)
  python Boost/fwls_transfer.py 2>&1 | tee Boost/log_transfer.txt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import reprhead as RH
import fwls as FW

sys.stdout.reconfigure(line_buffering=True)

LLM = Path('.')
BOOST, ARITH = LLM / 'Boost', LLM / 'Arith'
SEED, K, INNER = RH.SEED, RH.K, RH.INNER
HALF = 5000
ABE_STEP, TB_STEP = 3000, 4000
LAMS = FW.LAMS
CAND = {'win': '태도·기타', 'tie': '반복'}      # 본런에서 문턱 언저리였던 가족 (사후 선택 후보)


def tta(z, pre):
    return ((z[f'{pre}_fwd'] + z[f'{pre}_swp']) / 2).astype(np.float64)


def load_side(side):
    """side='train'|'stage2'. 반환: ids, y, M[m3,m_ar], T[s_tie,t3,t_ar]"""
    zr = np.load(ARITH / 'outputs/arith_experts_raw.npz')
    if side == 'train':
        za = np.load(BOOST / f'logits/abe__checkpoint-{ABE_STEP}__train.npz')
        zt = np.load(BOOST / f'logits/tb3__checkpoint-{TB_STEP}__train.npz')
        L_ABE, L_TB = tta(za, 'train'), tta(zt, 'train')
        ids, y = za['train_ids'], za['train_labels'].astype(np.int64)
        assert (zt['train_ids'] == ids).all() and (zt['train_labels'] == y).all(), 'TB train 정렬'
        assert (zr['ids_train'] == ids).all(), 'Arith train 정렬'
        L_ARI = np.mean([zr[f'raw_oof_train__{m}'].astype(np.float64)
                         for m in FW.ARITH_MODELS], 0)          # OOF — train 에서도 깨끗한 유일 전문가
    else:
        za = np.load(BOOST / f'logits/abe__checkpoint-{ABE_STEP}.npz')
        zt = np.load(BOOST / f'logits/tb3__checkpoint-{TB_STEP}.npz')
        L_ABE = np.vstack([tta(za, 'valid'), tta(za, 'hold')])
        L_TB = np.vstack([tta(zt, 'valid'), tta(zt, 'hold')])
        ids = np.concatenate([za['valid_ids'], za['hold_ids']])
        y = np.concatenate([za['valid_labels'], za['hold_labels']]).astype(np.int64)
        assert (np.concatenate([zt['valid_ids'], zt['hold_ids']]) == ids).all(), 'TB stage2 정렬'
        assert (np.concatenate([zr['ids_valid'], zr['ids_hold']]) == ids).all(), 'Arith stage2 정렬'
        L_ARI = np.mean([np.vstack([zr[f'raw_valid__{m}'], zr[f'raw_hold__{m}']]).astype(np.float64)
                         for m in FW.ARITH_MODELS], 0)
    sd = lambda B: (B[:, 0] - B[:, 1], B[:, 2] - np.logaddexp(B[:, 0], B[:, 1]))
    m3, t3 = sd(L_ABE)
    m_ar, t_ar = sd(L_ARI)
    s_tie = L_TB[:, 1] - L_TB[:, 0]
    return ids, y, np.column_stack([m3, m_ar]), np.column_stack([s_tie, t3, t_ar])


def fit_transfer(Xtr, ytr, Xte, bias, n_logit, lams):
    """train 안쪽 3-fold 로 λ 선택(상한 없는 자동확장) -> train 전체 재적합 -> 외부 채점.
    RH.crossfit 의 안쪽 루프와 같은 규율, 바깥 fold 만 없다 (평가는 외부 셋이므로)."""
    n_even = Xtr.shape[1] if bias else 0
    lam, n_bad = lams[0], 0
    if len(lams) > 1:
        inner = np.zeros(len(ytr), int)
        for j, (_, iv) in enumerate(StratifiedKFold(INNER, shuffle=True, random_state=SEED)
                                    .split(np.zeros((len(ytr), 1)), ytr)):
            inner[iv] = j
        Zi = [RH._z(Xtr[inner != j], Xtr[inner == j], n_even) for j in range(INNER)]
        score = {}

        def eval_lam(lam_, warm):
            nonlocal n_bad
            tot = 0.0
            for j in range(INNER):
                A, B = Zi[j]
                v, ok = RH._fit(A, ytr[inner != j], RH._penvec(A.shape[1], n_logit, lam_), bias,
                                warm[j])
                warm[j] = v
                n_bad += (not ok)
                tot += RH.rowloss(RH._scores(v, B, n_even, bias, False)[0], ytr[inner == j]).sum()
            score[lam_] = tot

        warm = [None] * INNER
        for lam_ in sorted(set(lams), reverse=True):
            eval_lam(lam_, warm)
        while min(score, key=score.get) == max(score) and max(score) < RH.LAM_CEIL:
            warm = [None] * INNER
            eval_lam(max(score) * 10, warm)
        lam = min(score, key=score.get)
    A, B = RH._z(Xtr, Xte, n_even)
    v, ok = RH._fit(A, ytr, RH._penvec(A.shape[1], n_logit, lam), bias)
    n_bad += (not ok)
    z, _ = RH._scores(v, B, n_even, bias, False)
    return z, lam, v[:n_logit], n_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='', help='팔 코드 부분일치 필터 (예: F0)')
    args = ap.parse_args()
    only = [s.strip() for s in args.only.split(',') if s.strip()]
    t_start = time.time()

    print('자산 적재 중 (train/stage2 단일 ckpt 로짓 + Arith OOF + 피쳐) ...')
    id_tr, y_tr, M_tr, T_tr = load_side('train')
    id_s2, y_s2, M_s2, T_s2 = load_side('stage2')
    assert len(y_tr) == 46729 and len(y_s2) == 10748

    cat = pd.read_csv(ARITH / 'feature_catalog.csv')
    feat = pd.read_parquet(ARITH / 'features_57k.parquet')
    fidx = pd.Index(feat['id'])
    pos_tr, pos_s2 = fidx.get_indexer(id_tr), fidx.get_indexer(id_s2)
    assert (pos_tr >= 0).all() and (pos_s2 >= 0).all()
    E_tr, evn, _, _ = RH.parity_split_arith(feat, pos_tr, cat)
    E_s2, evn2, _, _ = RH.parity_split_arith(feat, pos_s2, cat)
    assert evn == evn2, '짝피쳐 열 정의 불일치'
    EA = np.vstack([E_tr, E_s2])                        # 공통 rankgauss — 라벨 무관, 이식 일관성
    UA = np.column_stack([FW.rankgauss(EA[:, j]) for j in range(EA.shape[1])])
    U_tr, U_s2 = UA[:len(E_tr)], UA[len(E_tr):]

    fam_of = cat.set_index('feature')['유형']

    def col_family(name):
        if name.startswith('log '):
            name = name[4:]
        if name.startswith('('):
            name = name[1:].replace('-', '+').split('+')[0]
        return str(fam_of.get(name, '기타'))

    fam_e = [col_family(c) for c in evn]

    fold = np.zeros(len(y_s2), int)
    for k, (_, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=SEED)
                                .split(np.zeros((len(y_s2), 1)), y_s2)):
        fold[te] = k
    hold = np.arange(len(y_s2)) >= HALF
    nz_tr, nz_s2 = y_tr != 2, y_s2 != 2

    print(f'train {len(y_tr):,} (비-tie {int(nz_tr.sum()):,}) | stage2 {len(y_s2):,} | '
          f'짝피쳐 {U_tr.shape[1]} (57k 공통 rankgauss)')
    print('계약: 단일 ckpt(abe-3000/tb3-4000) — 본선 7-ckpt 수치와 직접 비교 금지, C0 이 기준\n')

    axis_res = {}
    for axis in ('win', 'tie'):
        bias = axis == 'tie'
        lnames = ['m3', 'm_ar'] if axis == 'win' else ['s_tie', 't3', 't_ar']
        L_tr, L_s2 = (M_tr, M_s2) if axis == 'win' else (T_tr, T_s2)
        y01_tr = (y_tr == 0).astype(np.float64) if axis == 'win' else (y_tr == 2).astype(np.float64)
        y01_s2 = (y_s2 == 0).astype(np.float64) if axis == 'win' else (y_s2 == 2).astype(np.float64)
        rows_s2 = nz_s2 if axis == 'win' else None
        fit_tr = nz_tr if axis == 'win' else np.ones(len(y_tr), bool)
        mask_s2 = nz_s2 if axis == 'win' else np.ones(len(y_s2), bool)
        nl = L_tr.shape[1]

        fam = CAND[axis]
        idx = [j for j, g in enumerate(fam_e) if g == fam]
        arms = [('F0', '로짓만', np.zeros((0,)), None)]
        Xf_tr, bf = FW.build_inter(U_tr[:, idx], [evn[j] for j in idx], [fam] * len(idx), L_tr, lnames)
        Xf_s2, _ = FW.build_inter(U_s2[:, idx], [evn[j] for j in idx], [fam] * len(idx), L_s2, lnames)
        arms.append((f'후보({fam})', f'본런 문턱 언저리 가족 {len(idx)}피쳐', Xf_tr, Xf_s2))
        Xa_tr, ba = FW.build_inter(U_tr, evn, fam_e, L_tr, lnames)
        Xa_s2, _ = FW.build_inter(U_s2, evn, fam_e, L_s2, lnames)
        arms.append(('F3', f'kitchen-sink {Xa_tr.shape[1]}열 — 4.3배 표본의 수혜 예상 지점',
                     Xa_tr, Xa_s2))
        if only:
            arms = [a for a in arms if any(o in a[0] for o in only)] or arms[:1]
            if arms[0][0] != 'F0':
                arms = [('F0', '로짓만', np.zeros((0,)), None)] + arms

        ax_nm = '승부축' if axis == 'win' else 'tie축'
        RH.head('1' if axis == 'win' else '2',
                f'{ax_nm} — stage2 교차적합(C0) vs train 적합->stage2 이식(C1/C2), 같은 행 paired')
        res = []
        for code, label, Xi_tr, Xi_s2 in arms:
            X_tr = L_tr if code == 'F0' else np.hstack([L_tr, Xi_tr])
            X_s2 = L_s2 if code == 'F0' else np.hstack([L_s2, Xi_s2])
            lams = [0.0] if X_tr.shape[1] == nl else LAMS
            print(f'  [{code}] {label}  ({X_tr.shape[1]}열)')

            zc, _, lamc, nbad_c, _, _, wlog_c = RH.crossfit(
                X_s2, y01_s2, fold, bias, lams, rows=rows_s2,
                n_even=X_s2.shape[1] if bias else 0, n_logit=nl)
            llc = RH.rowloss(zc[mask_s2], y01_s2[mask_s2])

            zt_, lam_t, w_tr, nbad_t = fit_transfer(
                X_tr[fit_tr], y01_tr[fit_tr], X_s2, bias, nl, lams)
            llt = RH.rowloss(zt_[mask_s2], y01_s2[mask_s2])

            hm = hold[mask_s2]
            d = RH.paired_ci(llc, llt)                  # 양수 = train-fit 이 낫다
            warn = f'  ** 미수렴 {nbad_c + nbad_t}' if (nbad_c + nbad_t) else ''
            print(f'        C0 교차적합   {llc.mean():.4f} (hold {llc[hm].mean():.4f}) λ중앙 {np.median(lamc):g}')
            print(f'        train 이식    {llt.mean():.4f} (hold {llt[hm].mean():.4f}) λ {lam_t:g}')
            print(f'        Δ(양수=이식 이득) {RH.ci_str(*d)}{warn}   [누적 {(time.time() - t_start) / 60:.0f}분]')
            if code == 'F0':
                wc = np.array(wlog_c).mean(0)
                print('        기저 가중치(정규화 공간)  stage2: '
                      + '  '.join(f'{n_} {v:+.3f}' for n_, v in zip(lnames, wc))
                      + '  |  train: '
                      + '  '.join(f'{n_} {v:+.3f}' for n_, v in zip(lnames, w_tr))
                      + '   <- 인코더 열 과신 여부를 여기서 본다')
            res.append(dict(code=code, label=label, llc=llc, llt=llt, d=d))
        axis_res[axis] = res
        print()

    RH.head('3', '종합 — 총 ll = tie축 + (1−tie율)·승부축.  같은 팔끼리 조립')
    nzf = float(nz_s2.mean())
    nzf_h = float(nz_s2[hold].mean())
    rows_tbl = []
    for i in range(min(len(axis_res['win']), len(axis_res['tie']))):
        rw, rt = axis_res['win'][i], axis_res['tie'][i]
        for tag, kw in (('C0 교차적합', 'llc'), ('train 이식', 'llt')):
            tot = float(rt[kw].mean() + nzf * rw[kw].mean())
            tot_h = float(rt[kw][hold].mean() + nzf_h * rw[kw][hold[nz_s2]].mean())
            rows_tbl.append([f"{rw['code']}·{tag}", f'{tot:.4f}', f'{tot_h:.4f}'])
    RH.table(['구성 (단일 ckpt 계약)', '총(전체)', '총(holdout)'], rows_tbl, align=['l', 'r', 'r'])
    print('\n  판정: F0 의 Δ 가 유의 음수면 5 파라미터에서 이미 암기 왜곡 실증 — train-fit 방향 폐쇄.')
    print('  F0 이 버티고 게이트 팔 Δ 가 양수면 제안 성립 + 본런 후보의 방향 확인을 겸한다.')
    print('  (후보 팔은 본런 33팔에서 사후 선택된 것이라 여기서 이겨도 확인이지 발견이 아니다)')


if __name__ == '__main__':
    main()
