"""arith.train — 학습 모듈. stage1 3분할에서 xgb 5-fold 를 학습해 체크포인트를 낸다.
Training module: fit the xgb 5-fold booster on the stage1 split and write a checkpoint.

계약 / contract
    분할    train 46,729 (학습·OOF·in-sample 마진) · valid 5,000 (ES; stage1 캠페인 세트)
            · holdout 5,748 (평가). stage1 valid 는 배포 번들 밖에 있어 재학습은 로컬 전용이다.
            The 5,000-row valid ships outside the deploy bundle, so retraining is local-only.
    fold    arith_folds.csv (train id 순서와 정렬돼야 한다)
    증강    fold 학습 행마다 swap 판을 덧붙인다 (라벨도 A<->B 뒤집어서)
    OOF     지표용으로만 낸다 (저장 안 함) — 저장된 체크포인트로 다시 계산해 배포 파일을 검증한다
    산출    xgb_fold1..5.json · artifacts/{tfidf,svd}.joblib · meta.json

경로는 전부 호출자가 준다 — 이 모듈도 나머지 패키지와 같이 저장소 배치를 모른다.
Every path comes from the caller; like the rest of the package this module knows no layout.

    import arith
    ck = arith.train(ckpt=MODEL_DIR/'ARITH'/'checkpoint-ab224',
                     catalog=MODEL_DIR/'ARITH'/'feature_catalog_ab224.csv',
                     emb=MODEL_DIR/'ARITH'/'emb_model',
                     data_dir=STAGE1_DIR,            # train·valid(5,000)·holdout.csv 가 있는 곳
                     folds=DATASET_DIR/'arith_folds.csv')
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ._base import (CONTRACT_FAMILY, NFOLD, SEED, build_swap, catalog_cols, labels_of, raw_ll,
                    swap_label, _tq)
from .predict import _load_fold, raw_margins
from .preprocess import extract

# 분할 크기 — valid 가 진짜 가드다. ES 는 stage1 캠페인의 valid 5,000 이다; 2,000행짜리
# valid 가 걸리면 그건 번들 origin 의 stage2 clean 부분집합이라 여기서 멈춘다.
# Split sizes. The valid count is the guard that matters: ES uses the stage1 campaign's
# 5,000-row valid; a 2,000-row valid is the bundle origin's stage2 clean subset and stops here.
N_TRAIN, N_VALID, N_HOLDOUT = 46729, 5000, 5748

N_XGB = 4000            # boosting cap (the step recorded in the checkpoint tag)
ES_ROUNDS = 200         # xgb early stopping rounds

# 계약 부스터의 하이퍼파라미터 — checkpoint-4000 계보 그대로. meta.json 에 그대로 기록한다:
# 부스터 JSON 은 학습 설정을 저장하지 않아(feature_names 조차 비어 있다) 여기 적힌 값이
# 사라지면 같은 모델을 다시 만들 방법이 없다.
# The contract booster's hyperparameters, recorded verbatim into meta.json — the booster JSON
# does not persist training config (even feature_names comes back empty), so if these are lost
# there is no way to rebuild the same model.
XGB_PARAMS = dict(objective='multi:softprob', num_class=3, learning_rate=0.03,
                  n_estimators=N_XGB, max_depth=8, min_child_weight=8, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=1.0, tree_method='hist',
                  eval_metric='mlogloss')


def _xgb_model(seed):
    import xgboost as xgb
    return xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=ES_ROUNDS,
                             random_state=seed, n_jobs=-1, verbosity=0)


def load_splits(data_dir, folds):
    """번들 origin 3분할 -> (df, y, m_tr, m_va, m_ho, fold 배열). 유일한 학습 데이터 원천.
    The single training data source; returns the concatenated frame and its role masks."""
    src = Path(data_dir)
    parts = {}
    for name, n_exp in (('train', N_TRAIN), ('valid', N_VALID), ('holdout', N_HOLDOUT)):
        p = src / f'{name}.csv'
        assert p.exists(), f'{p} not found — data_dir must hold train/valid/holdout.csv'
        d = pd.read_csv(p)
        if name == 'valid' and len(d) != n_exp:
            raise AssertionError(
                f'{p} has {len(d):,} rows, but the contract ES set has {n_exp:,}.\n'
                '  A 2,000-row valid.csv is the bundle origin\'s stage2 clean subset for the\n'
                '  combiner, not the stage1 ES set. Early-stopping on it puts Arith on\n'
                '  different terms from the encoder ckpts — point data_dir at the stage1 split.')
        assert len(d) == n_exp, f'{name}.csv has {len(d):,} rows != {n_exp:,}'
        d['role'] = name
        parts[name] = d
    df = pd.concat([parts[n] for n in ('train', 'valid', 'holdout')], ignore_index=True)
    assert df['id'].is_unique, 'duplicated id across the three splits'
    y = labels_of(df)
    role = df['role'].to_numpy()
    m_tr, m_va, m_ho = role == 'train', role == 'valid', role == 'holdout'

    fd = pd.read_csv(folds)
    assert (fd['id'].to_numpy() == df.loc[m_tr, 'id'].to_numpy()).all(), \
        f'{folds} is not aligned with {src}/train.csv — same ids in the same order are required'
    return df, y, m_tr, m_va, m_ho, fd['fold'].to_numpy()


def train(ckpt, catalog, emb, data_dir, folds, cols=None, seed=SEED, overwrite=False,
          verbose=True):
    """**학습 진입점** — stage1 에서 xgb 5-fold 를 학습해 배포 가능한 체크포인트를 만든다.
    Training entry point: fit the 5-fold booster and write the deployable checkpoint.

    cols 기본값은 카탈로그 전량(ab224 224종)이고, 순서도 카탈로그 순서다 — 체크포인트가
    사라져도 계약이 복원되는 유일한 경로라 meta 가 아니라 카탈로그를 기준으로 삼는다.
    cols defaults to the whole catalog in catalog order. The catalog, not a checkpoint's meta,
    is the reference: it is the only file from which the contract survives a deleted checkpoint.
    """
    ck, cat = Path(ckpt), Path(catalog)
    assert overwrite or not ck.exists(), f'{ck} already exists — pass overwrite=True to replace it'

    df, y, m_tr, m_va, m_ho, fold = load_splits(data_dir, folds)
    if verbose:
        print(f'splits <- {data_dir}')
        print(f'  train {int(m_tr.sum()):,} / valid(ES) {int(m_va.sum()):,} '
              f'/ holdout {int(m_ho.sum()):,}   — the same valid the encoder ckpts were chosen on')

    ck.mkdir(parents=True, exist_ok=True)
    (ck / 'artifacts').mkdir(exist_ok=True)

    # ── [1/3] 피쳐 — 항상 지금 정의로 새로 뽑는다 / always extracted fresh ──────────────
    if verbose:
        print('\n[1/3] features (fresh extraction, no cache)')
    F, tfidf, svd = extract(df, ckpt=ck, emb=emb, fit_idx=np.where(m_tr)[0], verbose=verbose)
    import joblib
    joblib.dump(tfidf, ck / 'artifacts/tfidf.joblib')
    joblib.dump(svd, ck / 'artifacts/svd.joblib')
    if verbose:
        print(f'  artifacts -> {ck.name}/artifacts (fit on the train rows only)')

    all_cols, swap_rules = catalog_cols(cat)
    cols = list(all_cols) if cols is None else list(cols)
    unknown = [c for c in cols if c not in set(all_cols)]
    assert not unknown, f'{len(unknown)} feature(s) outside the catalog: {unknown[:6]}'
    miss = [c for c in cols if c not in F.columns]
    assert not miss, f'{len(miss)} feature(s) missing from the extraction: {miss[:6]}'
    X = F[cols].to_numpy(np.float32)
    X_tr, X_va, X_ho = X[m_tr], X[m_va], X[m_ho]
    y_tr, y_va, y_ho = y[m_tr], y[m_va], y[m_ho]
    swap_X, sw = build_swap(cols, swap_rules)
    assert np.allclose(swap_X(swap_X(X_ho[:200])), X_ho[:200], atol=1e-5), 'swap involution failed'
    if verbose:
        print(f'  {len(cols)}/{len(all_cols)} features | swap: {len(sw["pairs"])} exchange pairs / '
              f'{len(sw["neg"])} sign-flip / {len(sw["recip"])} reciprocal / {len(sw["inv"])} invariant')

    # ── [2/3] 5-fold — 폴드마다 swap 판을 덧붙여 학습 / swap-augmented folds ──────────
    if verbose:
        print(f'\n[2/3] 5-fold {CONTRACT_FAMILY} (seed {seed})')
    t0 = time.time()
    iters = []
    for k in _tq(np.sort(np.unique(fold)), CONTRACT_FAMILY):
        tr = fold != k
        Xf = np.vstack([X_tr[tr], swap_X(X_tr[tr])])
        yf = np.hstack([y_tr[tr], swap_label(y_tr[tr])])
        m = _xgb_model(seed)
        m.fit(Xf, yf, eval_set=[(X_va, y_va)], verbose=False)
        m.save_model(str(ck / f'xgb_fold{int(k) + 1}.json'))
        iters.append(int(getattr(m, 'best_iteration', 0) or 0))
        del m

    # OOF 는 저장된 체크포인트로 다시 낸다 — 학습 객체가 아니라 **배포될 파일**을 검증한다.
    # OOF is recomputed from the saved folds, so the metrics describe the files that ship.
    oof = np.zeros((len(X_tr), 3), np.float64)
    for k in np.sort(np.unique(fold)):
        va = fold == k
        pred, _ = _load_fold(ck, int(k) + 1)
        oof[va] = (pred(X_tr[va]) + pred(swap_X(X_tr[va]))[:, [1, 0, 2]]) / 2
    L_va = raw_margins(X_va, ckpt=ck, swap_X=swap_X)
    L_ho = raw_margins(X_ho, ckpt=ck, swap_X=swap_X)
    metrics = {CONTRACT_FAMILY: {'oof_train_raw_ll': raw_ll(oof, y_tr),
                                 'valid_raw_ll': raw_ll(L_va, y_va),
                                 'holdout_raw_ll': raw_ll(L_ho, y_ho)}}
    mm = metrics[CONTRACT_FAMILY]
    if verbose:
        print(f'  best_iters {iters} | OOF {mm["oof_train_raw_ll"]:.4f} '
              f'| valid {mm["valid_raw_ll"]:.4f} | holdout {mm["holdout_raw_ll"]:.4f} '
              f'| {(time.time() - t0) / 60:.1f} min')

    # ── [3/3] meta.json — 체크포인트의 config / the checkpoint's config ────────────────
    import sklearn
    import xgboost
    meta = {
        'version': 3, 'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'step': N_XGB, 'tag': ck.name, 'n_folds': NFOLD, 'use_swap': True,
        'families': [CONTRACT_FAMILY], 'contract_family': CONTRACT_FAMILY,
        'n_cols': len(cols), 'n_cols_full': len(all_cols), 'cols': cols, 'seed': seed,
        'es': {'set': str(Path(data_dir) / 'valid.csv'), 'rows': int(m_va.sum()),
               'rounds': ES_ROUNDS,
               'note': 'stage1 ES — the same 5,000-row valid used to pick the '
                       'abe2/tb2/abe1/tb1 ckpts; ships outside the bundle (local retraining only)'},
        'caps': {CONTRACT_FAMILY: N_XGB},
        'params': {CONTRACT_FAMILY: XGB_PARAMS},
        'best_iters': {CONTRACT_FAMILY: iters},
        'metrics_raw': metrics,
        'source': {'features': 'arith.preprocess (definitions inlined)', 'parquet': False,
                   'data_dir': str(data_dir), 'folds': str(folds), 'catalog': str(cat)},
        'libs': {'xgboost': xgboost.__version__, 'sklearn': sklearn.__version__},
    }
    (ck / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    if verbose:
        print(f'\n[3/3] -> {ck}\n     xgb_fold1..{NFOLD} + meta.json + artifacts/')
        print('stage2/test margins are not saved — predict re-extracts them every time.')
    return ck
