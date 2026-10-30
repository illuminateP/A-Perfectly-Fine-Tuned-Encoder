"""arith_insample.py — 단계 1: arith XGB 를 **동일 조건(in-sample)** 으로 재학습한다.

왜: 층2 벌점 모양을 train 에서 학습(ardlab.py)하려면 열별 오염이 균등해야 한다.
deberta 4열은 train 46,729 를 학습한 모델의 in-sample 로짓인데 arith 열만 5-fold
OOF 라서 train 우주에서 관련도 순위가 뒤집힌다(실측: 라벨상관 배율 deberta
×1.14~1.31, t_ar ×0.93 — t_ar 가 clean 2위에서 train 4위로 밀린다). 여기서
deberta 와 같은 조건 — **train 전 행을 학습하고 train 전 행을 예측** — 의 XGB 열을
만들어 그 비대칭을 없앤다.

동일 조건의 정의 (Arith/Train.ipynb 프로토콜에서 fold 만 제거):
    파라미터  lr 0.03 · n 4000 · depth 8 · min_child_weight 8 · subsample 0.8 ·
             colsample 0.8 · reg_lambda 1.0 · hist · seed 42     (make_xgb 그대로)
    증강     원본+swap 쌍둥이 2배 (라벨 a↔b 교환)                 (그대로)
    ES       origin/valid.csv 5,000 · rounds 200                 (그대로 — deberta
             의 체크포인트 선택도 valid 기준이라 조건이 일치한다)
    학습 행   46,729 전 행 (5-fold OOF 대신)                      ← 유일한 변경
    예측     train 전 행 swap-TTA 평균                            (그대로, 단 in-sample)

출력: final/temp/arith_xgb_insample_train.npz — logp 로 저장한다. sd3 는 행별 상수
     이동에 불변이므로(t(B+c)=t(B)) 저장 npz 의 raw 계열과 정확히 같은 t 열이 된다.
예상: XGB 1회 적합 — 원 5-fold 가 총 ~8분이었으니 1적합 ≈ 3~6분 (CPU).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ARITH_PRJ = Path('./Arith')
FINAL = Path(__file__).resolve().parent.parent
NPZ_REF = FINAL / 'models/ARITH/outputs/arith_experts_raw.npz'   # 하네스가 읽는 npz
OUT = FINAL / 'temp/arith_xgb_insample_train.npz'
SEED, N_XGB = 42, 4000
META_COLS = ['id', 'role', 'label', 'is_tie']
PARAMS = dict(objective='multi:softprob', num_class=3, learning_rate=0.03,
              n_estimators=N_XGB, max_depth=8, min_child_weight=8,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
              tree_method='hist', eval_metric='mlogloss',
              early_stopping_rounds=200, random_state=SEED, n_jobs=-1, verbosity=0)


def labels_of(d):
    yy = np.select([d['winner_model_a'] == 1, d['winner_model_b'] == 1,
                    d['winner_tie'] == 1], [0, 1, 2], default=-1).astype(np.int64)
    assert (yy != -1).all(), 'invalid labels!'
    return yy


def build_swap(cols, swap_series):
    """Train.ipynb cell 2 그대로 — 카탈로그 swap 컬럼이 진실의 원천."""
    neg = [c for c in cols if swap_series[c] == '부호반전']
    recip = [c for c in cols if swap_series[c] == '역수']
    pairs = []
    for c in cols:
        v = swap_series[c]
        if isinstance(v, str) and v.startswith('교환:') and c < v.split(':', 1)[1]:
            partner = v.split(':', 1)[1]
            assert partner in set(cols), f'교환 파트너 {partner} 미포함'
            pairs.append((c, partner))
    inv = [c for c in cols if swap_series[c] == '불변']
    assert len(neg) + len(recip) + len(inv) + 2 * len(pairs) == len(cols), 'swap 분류 누락!'
    idx = {c: i for i, c in enumerate(cols)}
    neg_i = np.array([idx[c] for c in neg], int)
    rec_i = np.array([idx[c] for c in recip], int)
    a_i = np.array([idx[a] for a, _ in pairs], int)
    b_i = np.array([idx[b] for _, b in pairs], int)

    def swap_X(X):
        Xs = X.copy()
        if len(neg_i):
            Xs[:, neg_i] *= -1.0
        if len(rec_i):
            Xs[:, rec_i] = 1.0 / np.clip(Xs[:, rec_i], 1e-12, None)
        if len(a_i):
            Xs[:, a_i], Xs[:, b_i] = Xs[:, b_i].copy(), Xs[:, a_i].copy()
        return Xs
    return swap_X


def swap_p3(p):
    return p[:, [1, 0, 2]]


def swap_label(yy):
    return np.where(yy == 0, 1, np.where(yy == 1, 0, 2)).astype(yy.dtype)


def sd3(B):
    return B[:, 0] - B[:, 1], B[:, 2] - np.logaddexp(B[:, 0], B[:, 1])


def row_ll(p, yy):
    p = np.clip(np.asarray(p, np.float64), 1e-15, 1.0)
    p = p / p.sum(1, keepdims=True)
    return float(-np.log(p[np.arange(len(yy)), yy]).mean())


def main():
    print('[적재]')
    parts = {}
    for name, n_expect in [('train', 46729), ('valid', 5000), ('holdout', 5748)]:
        d = pd.read_csv(ARITH_PRJ / 'origin' / f'{name}.csv')
        assert len(d) == n_expect, f'{name}.csv {len(d)}행 != {n_expect}'
        d['role'] = name
        parts[name] = d
    df = pd.concat([parts['train'], parts['valid'], parts['holdout']], ignore_index=True)
    y = labels_of(df)
    role = df['role'].to_numpy()
    m_tr, m_va = role == 'train', role == 'valid'
    y_tr, y_va = y[m_tr], y[m_va]

    feat = pd.read_parquet(ARITH_PRJ / 'features_57k.parquet')
    assert (feat['id'].to_numpy() == df['id'].to_numpy()).all(), 'features_57k 정렬 불일치!'
    assert (feat['label'].to_numpy() == y).all(), '라벨 불일치!'
    F = feat.drop(columns=META_COLS)

    catalog = pd.read_csv(ARITH_PRJ / 'feature_catalog.csv').fillna({'제외사유': ''})
    UCOLS = catalog[catalog['사용'] == 1]['feature'].tolist()

    zar = np.load(NPZ_REF)
    assert list(zar['cols']) == UCOLS, '피쳐 집합이 하네스 npz 와 불일치 — 카탈로그가 바뀌었다'
    ids_tr = df.loc[m_tr, 'id'].to_numpy()
    assert (zar['ids_train'] == ids_tr).all() and (zar['y_train'] == y_tr).all(), 'train 정렬'
    assert (zar['ids_valid'] == df.loc[m_va, 'id'].to_numpy()).all(), 'valid 정렬'

    swap_X = build_swap(UCOLS, catalog.set_index('feature')['swap'])
    X_tr = F.loc[m_tr, UCOLS].to_numpy(np.float32)
    X_va = F.loc[m_va, UCOLS].to_numpy(np.float32)
    chk = X_tr[:200].copy()
    assert np.allclose(swap_X(swap_X(chk)), chk, atol=1e-5), 'involution 실패!'
    print(f'  피쳐 {len(UCOLS)}종 · train {len(y_tr):,} · ES valid {len(y_va):,}')

    print('[학습] 전 행 ×2(swap) · ES valid — 원 fold 모델 best_iter 262~399 참고')
    t0 = time.time()
    m = xgb.XGBClassifier(**PARAMS)
    Xf = np.vstack([X_tr, swap_X(X_tr)])
    yf = np.hstack([y_tr, swap_label(y_tr)])
    m.fit(Xf, yf, eval_set=[(X_va, y_va)], verbose=False)
    best = int(getattr(m, 'best_iteration', 0) or 0)
    print(f'  best_iteration {best} [{(time.time() - t0) / 60:.1f}분]')

    tta = lambda Xp: (m.predict_proba(Xp) + swap_p3(m.predict_proba(swap_X(Xp)))) / 2
    p_ins = tta(X_tr)
    p_va = tta(X_va)
    logp = np.log(np.clip(p_ins, 1e-15, 1.0)).astype(np.float32)

    # ── 진단 — 달성된 오염 수준을 바로 본다 ──
    p_oof = np.exp(zar['logp_oof_train__xgb'].astype(np.float64))
    yt = (y_tr == 2).astype(np.float64)
    _, t_ins = sd3(logp.astype(np.float64))
    _, t_oof = sd3(zar['raw_oof_train__xgb'].astype(np.float64))
    c_ins = np.corrcoef(t_ins, yt)[0, 1]
    c_oof = np.corrcoef(t_oof, yt)[0, 1]
    print('\n[진단]')
    print(f'  train ll   in-sample {row_ll(p_ins, y_tr):.4f}  vs  OOF {row_ll(p_oof, y_tr):.4f}'
          f'   ← 차이가 암기량')
    print(f'  valid ll   단일모델 {row_ll(p_va, y_va):.4f}  vs  5-fold 앙상블 '
          f'{row_ll(np.exp(zar["logp_valid__xgb"].astype(np.float64)), y_va):.4f}'
          f'   ← 프로토콜 온전성 (근소 열세가 정상)')
    print(f'  t_ar 라벨상관(train)   in-sample {c_ins:.4f}  vs  OOF {c_oof:.4f}'
          f'   (deberta 열은 0.29~0.38 — 이 사이로 오면 균등화 성공)')

    np.savez_compressed(OUT, ids_train=ids_tr, y_train=y_tr,
                        logp_ins_train__xgb=logp, best_iteration=best,
                        protocol=json.dumps({k: str(v) for k, v in PARAMS.items()}))
    print(f'\n저장: {OUT}')


if __name__ == '__main__':
    main()
