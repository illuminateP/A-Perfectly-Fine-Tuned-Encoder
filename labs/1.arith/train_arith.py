"""train_arith.py — Arith(산술 전문가) 단독 학습·추론 모듈. **parquet 를 쓰지 않는다.**

무엇을 하나
    stage1 자산(`origin/{train,valid,holdout}.csv`)에서 255 피쳐를 **직접** 뽑아
    5-fold 부스터를 학습하고 `models/ARITH/checkpoint-{step}/` 에 저장한다.
    stage2·Kaggle test 로짓은 저장하지 않는다 — 예측할 때마다 이 모듈이 다시 뽑는다.

왜 parquet 를 버리나
    `features_57k.parquet` 는 이미 아는 57,477 행의 캐시일 뿐이다. Kaggle 비공개 test 의
    id 는 거기 하나도 없으므로 제출 경로에서는 쓸 수가 없다. 캐시를 남겨 두면 학습은
    캐시로, 추론은 실시간 추출로 도는 **두 경로**가 생기고 그 둘이 갈라져도 아무도 못 잡는다.
    (실측: 실시간 추출 0.02582 s/행 — 50,000 행 21.5 분. 캐시 대비 정합 max|Δ| 4.17e-07)

피쳐 정의의 원천
    `models/ARITH/Features.ipynb` 의 [정의] 셀 1~4 를 exec 해서 그대로 쓴다. 복사하면
    노트북과 조용히 갈라지므로 복사하지 않는다.

상태형 아티팩트 (TF-IDF / SVD)
    train 행으로만 fit 한다. 학습 시 새로 fit 해 **체크포인트 안에** 저장하고, 추론 시엔
    그 파일로 transform 만 한다. 체크포인트가 자기완결이라 예전 `ARITH/artifacts/` 에
    의존하지 않는다.

checkpoint-{step} 의 step
    GBDT 에는 global step 이 없다. 인코더 규약(`checkpoint-3000`)과 모양을 맞추기 위한
    라벨이고, 기본값은 계약 부스터 xgb 의 부스팅 상한 `n_estimators=4000` 이다.
    `--step` 으로 바꿀 수 있다.

학습 데이터는 stage1 분할 (valid 5,000행)
    `LLM_CLS/origin/` (또는 `Arith/origin/`) 를 읽는다 — 후보 중 **valid 가 5,000행**인 곳을
    자동으로 고른다. abe2/tb2/abe1/tb1 이 ES·체크포인트 선택에 쓴 것과
    **같은 세트**라야 전문가끼리 조건이 같다.
    `final/origin/valid.csv` 2,000행은 stage2 결합기용 clean 부분집합(ABE1 오염 3,000행
    제외)이지 stage1 ES 가 아니다. 그걸로 학습하면 Arith 만 다른 조건이 되므로
    행 수가 5,000 이 아니면 즉시 멈춘다. `--data-dir` 로 바꿀 수 있다.
    학습은 제출 경로가 아니라 컨테이너 밖을 읽어도 된다 — 실리는 것은 checkpoint 뿐이다.

사용
    # 학습 (GPU 권장 — 피쳐 추출 임베딩 단계 때문)
    python models/train_arith.py --train
    python models/train_arith.py --train --models xgb --step 4000

    # 저장 자산과 대조 (구 npz 가 아직 있을 때만)
    python models/train_arith.py --verify --step 4000

    # 노트북에서
    import sys; sys.path.append(str(MODELS))
    import train_arith as TA
    L_ar = TA.predict_raw(df_inf, TA.ckpt_dir(4000), model='xgb')   # raw 마진 (n,3)
    oof  = TA.load_oof(TA.ckpt_dir(4000), 'xgb')                    # train OOF raw 마진
"""
import argparse
import contextlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = Path(__file__).resolve().parent
ARITH = MODELS / 'ARITH'
FINAL = MODELS.parent
ORIGIN = FINAL / 'origin'                       # 카탈로그·folds 의 원천 (컨테이너 안)

# 학습 데이터는 **stage1 분할**을 쓴다 — valid 5,000행. abe2/tb2/abe1/tb1 이 ES·ckpt 선택에
# 쓴 것과 같은 세트라야 전문가끼리 같은 조건이 된다.
#   final/origin/valid.csv 는 2,000행이다. 그건 **stage2 결합기 적합용** clean 부분집합
#   (ABE1 오염 3,000행 제외)이지 stage1 ES 세트가 아니다. 여기에 쓰면 Arith 만 다른
#   조건으로 학습된다.
# 학습은 제출 경로가 아니므로 컨테이너 밖을 읽어도 된다 — 컨테이너에 실리는 것은
# checkpoint(가중치·artifacts·meta·oof)뿐이다.
#
# 위치를 하드코딩하지 않는다 — 후보를 훑어 **valid 가 5,000행인 곳**을 고른다.
# 디렉터리를 옮겨도 안 깨지고, 잘못된 분할을 조용히 집는 일도 없다.
STAGE1_CANDIDATES = (FINAL.parent / 'origin', FINAL.parent / 'Arith/origin', ORIGIN)
ES_ROWS = 5000

SEED = 42
NFOLD = 5
N_FEAT = 259                    # 정의 셀이 내는 전량. 이 중 카탈로그 '사용'==1 인 255 를 쓴다
DEF_STEP = 4000                 # = 계약 부스터 xgb 의 n_estimators (인코더 규약과 모양 통일)
FAMILIES = ('lgbm', 'xgb', 'cat')

N_LGB, N_XGB, N_CAT = 3000, 4000, 4000
ES_ROUNDS = {'lgbm': 100, 'xgb': 200, 'cat': 200}


# 학습 전용 스크래치. **제출 경로는 절대 읽지 않는다** — temp/ 는 컨테이너에 안 올라간다.
# 피쳐 추출이 57,477행에 ~20분이라 피쳐 사다리 실험을 매번 다시 뽑을 수는 없어서 둔다.
# Features.ipynb 정의 셀의 sha 를 같이 저장해, 정의가 바뀌면 캐시를 자동 무효화한다.
FEATS_CACHE = FINAL / 'temp/arith_feats_stage1'


def ckpt_dir(step=DEF_STEP, tag=None):
    return ARITH / (f'checkpoint-{step}' if not tag else f'checkpoint-{step}-{tag}')


def defs_sha():
    """정의 셀 1~4 의 내용 해시 — 캐시 무효화 키."""
    import hashlib
    nb = json.loads((ARITH / 'Features.ipynb').read_text(encoding='utf-8'))
    src = ''.join(''.join(nb['cells'][i]['source']) for i in (1, 2, 3, 4))
    return hashlib.sha256(src.encode('utf-8')).hexdigest()[:16]


# ══════════════════════════════════════════════ 피쳐 정의 (Features.ipynb)
@contextlib.contextmanager
def _chdir(p):
    """Features.ipynb 는 상대경로로 쓰여 있다 — 그 블록만 CWD 를 옮긴다."""
    old = os.getcwd()
    os.chdir(p)
    try:
        yield
    finally:
        os.chdir(old)


def load_defs():
    """Features.ipynb 의 [정의] 셀 1~4 만 실행해 추출 함수를 그대로 얻는다."""
    nb = json.loads((ARITH / 'Features.ipynb').read_text(encoding='utf-8'))
    ns = {'__name__': 'arith_feature_defs'}
    with _chdir(ARITH):
        for i in (1, 2, 3, 4):
            c = nb['cells'][i]
            src = ''.join(c['source'])
            assert c['cell_type'] == 'code', f'Features.ipynb 셀 {i} 가 코드가 아니다'
            assert '[정의]' in src or '── 설정 ──' in src, f'셀 {i} 가 정의 셀이 아니다'
            exec(compile(src, f'Features.ipynb#cell{i}', 'exec'), ns)
    need = ('parse_columns', 'load_embedder', 'encode_texts', 'emb_frames',
            'bertscore_features', 'build_stat_features', 'build_unit_pair_frame')
    miss = [f for f in need if f not in ns]
    assert not miss, f'정의 셀에서 함수 누락: {miss}'
    return ns


def catalog_cols():
    """계약 피쳐 255종과 swap 규약. 원천은 origin/feature_catalog.csv."""
    cat = pd.read_csv(ORIGIN / 'feature_catalog.csv')
    use = cat[cat['사용'] == 1]
    return use['feature'].tolist(), cat.set_index('feature')['swap']


def extract(df, ckpt=None, fit_idx=None, ns=None, verbose=True):
    """텍스트 프레임 -> 259 피쳐 DataFrame. parquet 를 읽지 않는다.

    ckpt 를 주면 그 안의 tfidf/svd 로 **transform 만** 한다 (추론).
    fit_idx 를 주면 그 행으로 TF-IDF/SVD 를 **fit** 하고 (tfidf, svd) 를 함께 돌려준다 (학습).
    """
    import joblib
    assert (ckpt is None) != (fit_idx is None), 'ckpt(추론) 와 fit_idx(학습) 중 하나만'
    ns = ns or load_defs()
    tfidf = svd = None
    if ckpt is not None:
        tfidf = joblib.load(Path(ckpt) / 'artifacts/tfidf.joblib')
        svd = joblib.load(Path(ckpt) / 'artifacts/svd.joblib')

    t0 = time.time()
    with _chdir(ARITH):
        P, A, B = ns['parse_columns'](df)
        tok, mod, dev = ns['load_embedder']('emb_model')
        if verbose:
            print(f'  임베더 device={dev} | {len(df):,}행')
        eq = ns['encode_texts'](tok, mod, P, dev, desc='embed P')
        ea = ns['encode_texts'](tok, mod, A, dev, desc='embed A')
        eb = ns['encode_texts'](tok, mod, B, dev, desc='embed B')
        emb5, emb3 = ns['emb_frames'](eq, ea, eb)
        bs12 = ns['bertscore_features'](tok, mod, P, A, B, dev)
        del mod
        try:
            import torch
            if dev != 'cpu':
                torch.cuda.empty_cache()
        except ImportError:
            pass
        stat44, tfidf, svd = ns['build_stat_features'](P, A, B, emb3, tok, tfidf=tfidf,
                                                       svd=svd, fit_idx=fit_idx)
        up = ns['build_unit_pair_frame'](P, A, B)

    F = pd.concat([f.reset_index(drop=True) for f in (emb5, bs12, stat44, up)], axis=1)
    assert F.shape[1] == N_FEAT, f'{F.shape[1]}종 != {N_FEAT} — 정의 셀과 계약이 어긋났다'
    assert not F.columns.duplicated().any(), '피쳐 컬럼 중복'
    assert np.isfinite(F.to_numpy(np.float32)).all(), '피쳐에 NaN/inf'
    if verbose:
        dt = time.time() - t0
        print(f'  추출 완료 {F.shape} {dt / 60:.1f}분 ({dt / len(df):.5f} s/행)')
    return F, tfidf, svd


def build_swap(cols, swap_series):
    """A<->B 교환 시 피쳐 거동. 카탈로그의 swap 열이 진실의 원천."""
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
    assert len(neg) + len(recip) + len(inv) + 2 * len(pairs) == len(cols), 'swap 분류 누락'
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
    return swap_X, {'neg': neg, 'recip': recip, 'inv': inv, 'pairs': pairs}


def swap_label(y):
    return np.where(y == 0, 1, np.where(y == 1, 0, 2)).astype(y.dtype)


def labels_of(d):
    y = np.select([d['winner_model_a'] == 1, d['winner_model_b'] == 1, d['winner_tie'] == 1],
                  [0, 1, 2], default=-1).astype(np.int64)
    assert (y != -1).all(), '라벨이 3열 중 하나로 떨어지지 않는다'
    return y


# ══════════════════════════════════════════════ 부스터
def _factories():
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb
    return {
        'lgbm': lambda: lgb.LGBMClassifier(
            objective='multiclass', num_class=3, learning_rate=0.03, n_estimators=N_LGB,
            num_leaves=63, min_child_samples=100, colsample_bytree=0.8, subsample=0.8,
            subsample_freq=1, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1),
        'xgb': lambda: xgb.XGBClassifier(
            objective='multi:softprob', num_class=3, learning_rate=0.03, n_estimators=N_XGB,
            max_depth=8, min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, tree_method='hist', eval_metric='mlogloss',
            early_stopping_rounds=ES_ROUNDS['xgb'], random_state=SEED, n_jobs=-1, verbosity=0),
        'cat': lambda: cb.CatBoostClassifier(
            loss_function='MultiClass', eval_metric='MultiClass', iterations=N_CAT,
            learning_rate=0.03, depth=8, l2_leaf_reg=4.0,
            early_stopping_rounds=ES_ROUNDS['cat'], bootstrap_type='Bernoulli', subsample=0.8,
            border_count=254, random_seed=SEED, verbose=0, allow_writing_files=False),
    }


def _fit_one(name, m, Xf, yf, X_es, y_es):
    import inspect
    import lightgbm as lgb
    if name == 'lgbm':
        new = 'eval_X' in inspect.signature(lgb.LGBMClassifier.fit).parameters
        ekw = {'eval_X': X_es, 'eval_y': y_es} if new else {'eval_set': [(X_es, y_es)]}
        m.fit(Xf, yf, **ekw, callbacks=[lgb.early_stopping(ES_ROUNDS['lgbm'], verbose=False)])
    elif name == 'xgb':
        m.fit(Xf, yf, eval_set=[(X_es, y_es)], verbose=False)
    else:
        m.fit(Xf, yf, eval_set=(X_es, y_es), use_best_model=True)


def _best_iter(name, m):
    if name == 'lgbm':
        return int(m.booster_.best_iteration or 0)
    if name == 'xgb':
        return int(getattr(m, 'best_iteration', 0) or 0)
    return int(m.get_best_iteration() or 0)


def _save_fold(name, m, ck, k):
    if name == 'lgbm':
        m.booster_.save_model(str(ck / f'lgbm_fold{k}.txt'))
    elif name == 'xgb':
        m.save_model(str(ck / f'xgb_fold{k}.json'))
    else:
        m.save_model(str(ck / f'cat_fold{k}.cbm'))


def _load_fold(name, ck, k):
    """(예측함수, best_iter) — raw 마진을 내는 클로저를 돌려준다."""
    if name == 'xgb':
        import xgboost as xgb
        m = xgb.XGBClassifier()
        m.load_model(str(ck / f'xgb_fold{k}.json'))
        bst, bi = m.get_booster(), getattr(m, 'best_iteration', None)
        kw = dict(output_margin=True)
        if bi is not None:
            kw['iteration_range'] = (0, bi + 1)
        return (lambda X: bst.predict(xgb.DMatrix(X), **kw)), bi
    if name == 'lgbm':
        import lightgbm as lgb
        bst = lgb.Booster(model_file=str(ck / f'lgbm_fold{k}.txt'))
        bi = bst.best_iteration or None
        return (lambda X: bst.predict(X, raw_score=True, num_iteration=bi)), bi
    import catboost as cb
    m = cb.CatBoostClassifier()
    m.load_model(str(ck / f'cat_fold{k}.cbm'))
    return (lambda X: np.asarray(m.predict(X, prediction_type='RawFormulaVal'), np.float64)), None


def raw_margins(X, ck, model='xgb', nfold=NFOLD, swap_X=None):
    """fold 평균 + swap-TTA **마진 공간** 평균. 노트북 test 경로와 같은 계산."""
    assert swap_X is not None, 'swap_X 가 필요하다'
    Xs = swap_X(X)
    out = np.zeros((len(X), 3), np.float64)
    for k in range(1, nfold + 1):
        pred, _ = _load_fold(model, Path(ck), k)
        out += (pred(X) + pred(Xs)[:, [1, 0, 2]]) / 2 / nfold
    return out


def predict_raw(df, ck, model='xgb', ns=None, verbose=True):
    """텍스트 프레임 -> raw 마진 (n,3). 피쳐 추출부터 전부 여기서 한다."""
    ck = Path(ck)
    cols = json.loads((ck / 'meta.json').read_text(encoding='utf-8'))['cols']
    _, swap_series = catalog_cols()
    F, _, _ = extract(df, ckpt=ck, ns=ns, verbose=verbose)
    miss = [c for c in cols if c not in F.columns]
    assert not miss, f'추출 결과에 계약 피쳐 누락 {len(miss)}종: {miss[:6]}'
    X = F[cols].to_numpy(np.float32)
    swap_X, _ = build_swap(cols, swap_series)
    assert np.allclose(swap_X(swap_X(X[:200])), X[:200], atol=1e-5), 'swap involution 실패'
    return raw_margins(X, ck, model=model, swap_X=swap_X)


def load_oof(ck, model='xgb'):
    """train OOF raw 마진 — 학습 때만 만들 수 있어 체크포인트에 동봉한다."""
    z = np.load(Path(ck) / 'oof_train.npz')
    return z[f'raw_oof_train__{model}'].astype(np.float64), z['ids_train'], z['y_train']


def take_from_cache(ids, cols):
    """스크래치 캐시에서 지정 id·열을 그 순서대로 잘라 낸다 (학습·실험 전용).

    제출 경로는 이걸 쓰지 않는다 — `predict_raw()` 가 매번 실시간 추출한다.
    캐시는 그 추출의 저장본이고 `defs_sha` 로 정의 셀 변경을 감지하므로 값은 같다.
    """
    z = np.load(FEATS_CACHE / 'F.npz')
    have = [str(c) for c in z['cols']]
    pos = pd.Index(z['ids']).get_indexer(np.asarray(ids))
    assert (pos >= 0).all(), f'캐시에 없는 id {int((pos < 0).sum())}개 — 캐시를 다시 만들어라'
    ci = [have.index(c) for c in cols]
    return np.ascontiguousarray(z['X'][np.ix_(pos, ci)])


def margins_for(ids, ck, model='xgb'):
    """캐시된 피쳐로 지정 id 의 raw 마진을 낸다 (fold평균 + swap-TTA)."""
    ck = Path(ck)
    cols = json.loads((ck / 'meta.json').read_text(encoding='utf-8'))['cols']
    _, swap_series = catalog_cols()
    swap_X, _ = build_swap(cols, swap_series)
    X = take_from_cache(ids, cols)
    return raw_margins(X, ck, model=model, swap_X=swap_X)


def raw_ll(L, y):
    zs = L - L.max(1, keepdims=True)
    p = np.exp(zs)
    p /= p.sum(1, keepdims=True)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean())


# ══════════════════════════════════════════════ stage1 자산 + 피쳐 캐시
def stage1_dir(data_dir=None):
    """학습용 3분할 디렉터리를 고른다 — valid 가 정확히 5,000행인 후보.

    헤더만 세므로 176MB train.csv 를 읽지 않는다.
    """
    if data_dir is not None:
        return Path(data_dir)
    seen = []
    for d in STAGE1_CANDIDATES:
        v = d / 'valid.csv'
        if not (v.exists() and (d / 'train.csv').exists() and (d / 'holdout.csv').exists()):
            continue
        n = sum(1 for _ in v.open(encoding='utf-8', newline='')) - 1   # 헤더 제외
        seen.append((d, n))
        if n == ES_ROWS:
            return d
    raise AssertionError(
        f'stage1 분할(valid {ES_ROWS:,}행)을 못 찾았다. 후보와 실제 행 수:\n  '
        + '\n  '.join(f'{d} -> valid {n:,}행' for d, n in seen or [(d, -1) for d in
                                                                   STAGE1_CANDIDATES])
        + '\n  --data-dir 로 직접 지정할 수 있다.')


def stage1_frames(data_dir=None):
    """stage1 3분할 -> (df, y, m_tr, m_va, m_ho, folds, src). 유일한 데이터 원천."""
    src = stage1_dir(data_dir)
    parts = {}
    for name, n_exp in (('train', 46729), ('valid', 5000), ('holdout', 5748)):
        d = pd.read_csv(src / f'{name}.csv')
        if name == 'valid' and len(d) != n_exp:
            raise AssertionError(
                f'{src / "valid.csv"} 가 {len(d):,}행이다 (stage1 ES 는 {n_exp:,}행).\n'
                '  final/origin/valid.csv 2,000행은 **stage2 결합기용 clean 부분집합**이지\n'
                '  stage1 ES 가 아니다. 그걸로 학습하면 Arith 만 abe2/tb2 와 다른 조건이 된다.\n'
                f'  후보: {[str(c) for c in STAGE1_CANDIDATES]}')
        assert len(d) == n_exp, f'{name}.csv {len(d)}행 != {n_exp}'
        d['role'] = name
        parts[name] = d
    df = pd.concat([parts[n] for n in ('train', 'valid', 'holdout')], ignore_index=True)
    assert df['id'].is_unique, 'id 중복'
    y = labels_of(df)
    role = df['role'].to_numpy()
    m_tr, m_va, m_ho = role == 'train', role == 'valid', role == 'holdout'

    # folds 는 컨테이너 안 origin 이 원천 — train id 순서가 양쪽에서 같아야 한다
    folds_df = pd.read_csv(ORIGIN / 'arith_folds.csv')
    assert (folds_df['id'].to_numpy() == df.loc[m_tr, 'id'].to_numpy()).all(), \
        f'origin/arith_folds.csv 가 {src}/train.csv 와 정렬되지 않는다'
    return df, y, m_tr, m_va, m_ho, folds_df['fold'].to_numpy(), src


def feature_matrix(df, m_tr, use_cache=True, ns=None):
    """stage1 전 행의 259 피쳐 + (tfidf, svd). 캐시가 유효하면 재사용한다.

    캐시는 **학습 전용 스크래치**다. `predict_raw()` 는 절대 읽지 않는다 — 제출 경로가
    캐시를 타기 시작하면 학습과 추론이 다른 계산이 되고, test 3행엔 라벨이 없어 못 잡는다.
    """
    import joblib
    sha = defs_sha()
    fz, ft, fs = (FEATS_CACHE / 'F.npz', FEATS_CACHE / 'tfidf.joblib',
                  FEATS_CACHE / 'svd.joblib')
    if use_cache and fz.exists() and ft.exists() and fs.exists():
        z = np.load(fz, allow_pickle=False)
        meta = json.loads((FEATS_CACHE / 'meta.json').read_text(encoding='utf-8'))
        ok = (meta.get('defs_sha') == sha and len(z['ids']) == len(df)
              and (z['ids'] == df['id'].to_numpy()).all())
        if ok:
            F = pd.DataFrame(z['X'], columns=[str(c) for c in z['cols']])
            print(f'  피쳐 캐시 사용 ({FEATS_CACHE.name}, defs_sha {sha}) {F.shape}')
            return F, joblib.load(ft), joblib.load(fs)
        print('  피쳐 캐시 무효 (정의 sha 또는 id 불일치) -> 재추출')

    ns = ns or load_defs()
    F, tfidf, svd = extract(df, fit_idx=np.where(m_tr)[0], ns=ns)
    if use_cache:
        FEATS_CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fz, ids=df['id'].to_numpy(),
                            cols=np.array(list(F.columns)), X=F.to_numpy(np.float32))
        joblib.dump(tfidf, ft)
        joblib.dump(svd, fs)
        (FEATS_CACHE / 'meta.json').write_text(json.dumps(
            {'defs_sha': sha, 'rows': len(df), 'cols': int(F.shape[1]),
             'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
             'note': '학습 전용 스크래치 — 제출 경로는 읽지 않는다. 지워도 재생성된다.'},
            ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'  피쳐 캐시 저장 -> {FEATS_CACHE}')
    return F, tfidf, svd


# ══════════════════════════════════════════════ 학습
def train(step=DEF_STEP, families=('xgb',), overwrite=False, cols=None, tag=None,
          use_cache=True, data_dir=None):
    from tqdm import tqdm
    ck = ckpt_dir(step, tag)
    assert overwrite or not ck.exists(), f'{ck} 가 이미 있다 — --overwrite 로 덮어써라'

    df, y, m_tr, m_va, m_ho, folds, src = stage1_frames(data_dir)
    print(f'stage1 <- {src}')
    print(f'  train {int(m_tr.sum()):,} / valid(ES) {int(m_va.sum()):,} '
          f'/ holdout {int(m_ho.sum()):,}   — abe2/tb2/abe1/tb1 과 같은 세트')

    ck.mkdir(parents=True, exist_ok=True)
    (ck / 'artifacts').mkdir(exist_ok=True)

    # ── 피쳐 ──
    print('\n[1/3] 피쳐 (parquet 미사용)')
    F, tfidf, svd = feature_matrix(df, m_tr, use_cache=use_cache)
    import joblib
    joblib.dump(tfidf, ck / 'artifacts/tfidf.joblib')
    joblib.dump(svd, ck / 'artifacts/svd.joblib')
    print(f'  artifacts 저장 -> {ck.name}/artifacts (train 행으로만 fit)')

    all_cols, swap_series = catalog_cols()
    cols = list(all_cols) if cols is None else list(cols)
    unknown = [c for c in cols if c not in set(all_cols)]
    assert not unknown, f'카탈로그 사용 목록 밖의 피쳐: {unknown[:6]}'
    miss = [c for c in cols if c not in F.columns]
    assert not miss, f'추출 결과에 피쳐 누락 {len(miss)}종: {miss[:6]}'
    X = F[cols].to_numpy(np.float32)
    X_tr, X_va, X_ho = X[m_tr], X[m_va], X[m_ho]
    y_tr, y_va, y_ho = y[m_tr], y[m_va], y[m_ho]
    swap_X, sw = build_swap(cols, swap_series)
    assert np.allclose(swap_X(swap_X(X_ho[:200])), X_ho[:200], atol=1e-5), 'involution 실패'
    print(f'  피쳐 {len(cols)}/{len(all_cols)}종 | swap 교환 {len(sw["pairs"])}쌍 / '
          f'부호반전 {len(sw["neg"])} / 역수 {len(sw["recip"])} / 불변 {len(sw["inv"])}')

    # ── 학습 ──
    print(f'\n[2/3] 5-fold 학습 — {", ".join(families)}')
    fac = _factories()
    best_iters, oof_raw, metrics = {}, {}, {}
    for name in families:
        t0 = time.time()
        oof = np.zeros((len(X_tr), 3), np.float64)
        iters = []
        for k in tqdm(np.sort(np.unique(folds)), desc=name, file=sys.stdout, smoothing=0):
            tr, va = folds != k, folds == k
            Xf = np.vstack([X_tr[tr], swap_X(X_tr[tr])])        # swap 증강
            yf = np.hstack([y_tr[tr], swap_label(y_tr[tr])])
            m = fac[name]()
            _fit_one(name, m, Xf, yf, X_va, y_va)
            _save_fold(name, m, ck, int(k) + 1)
            iters.append(_best_iter(name, m))
            del m
        best_iters[name] = iters
        # OOF 는 저장된 체크포인트로 다시 낸다 — 학습 객체가 아니라 **배포될 파일**을 검증한다
        for k in np.sort(np.unique(folds)):
            va = folds == k
            pred, _ = _load_fold(name, ck, int(k) + 1)
            Xv, Xvs = X_tr[va], swap_X(X_tr[va])
            oof[va] = (pred(Xv) + pred(Xvs)[:, [1, 0, 2]]) / 2
        oof_raw[name] = oof.astype(np.float32)
        L_ho = raw_margins(X_ho, ck, model=name, swap_X=swap_X)
        L_va = raw_margins(X_va, ck, model=name, swap_X=swap_X)
        metrics[name] = {'oof_train_raw_ll': raw_ll(oof, y_tr),
                         'valid_raw_ll': raw_ll(L_va, y_va),
                         'holdout_raw_ll': raw_ll(L_ho, y_ho)}
        print(f'  {name}: best_iters {iters} | OOF {metrics[name]["oof_train_raw_ll"]:.4f} '
              f'| valid {metrics[name]["valid_raw_ll"]:.4f} '
              f'| holdout {metrics[name]["holdout_raw_ll"]:.4f} | {(time.time() - t0) / 60:.1f}분')

    # ── 저장 ──
    print('\n[3/3] 저장')
    np.savez_compressed(ck / 'oof_train.npz',
                        ids_train=df.loc[m_tr, 'id'].to_numpy(), y_train=y_tr,
                        cols=np.array(cols),
                        **{f'raw_oof_train__{n}': oof_raw[n] for n in families})
    import catboost
    import lightgbm
    import sklearn
    import xgboost
    meta = {
        'version': 3, 'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'step': step, 'tag': tag, 'n_folds': NFOLD, 'use_swap': True,
        'families': list(families), 'contract_family': 'xgb',
        'n_cols': len(cols), 'n_cols_full': len(all_cols), 'cols': cols,
        'es': {'set': str(src / 'valid.csv'), 'rows': int(m_va.sum()), 'rounds': ES_ROUNDS,
               'note': 'stage1 ES — abe2/tb2/abe1/tb1 의 ckpt 선택과 같은 valid 5,000행'},
        'caps': {'lgbm': N_LGB, 'xgb': N_XGB, 'cat': N_CAT},
        'best_iters': best_iters,
        'metrics_raw': metrics,
        'source': {'features': 'Features.ipynb 정의 셀 1~4 (exec)', 'parquet': False,
                   'data_dir': str(src), 'folds': 'origin/arith_folds.csv',
                   'catalog': 'origin/feature_catalog.csv'},
        'libs': {'lightgbm': lightgbm.__version__, 'xgboost': xgboost.__version__,
                 'catboost': catboost.__version__, 'sklearn': sklearn.__version__},
    }
    (ck / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                  encoding='utf-8')
    print(f'  -> {ck}')
    print(f'     {", ".join(f"{n}_fold1..{NFOLD}" for n in families)} + meta.json '
          f'+ oof_train.npz + artifacts/')
    print('\nstage2·test 로짓은 저장하지 않는다 — predict_raw() 가 매번 다시 뽑는다.')
    return ck


def verify(step=DEF_STEP, model='xgb', tag=None):
    """구 자산(outputs/arith_experts_raw.npz)이 아직 있으면 holdout raw ll 을 대조한다."""
    ck = ckpt_dir(step, tag)
    old = ARITH / 'outputs/arith_experts_raw.npz'
    meta = json.loads((ck / 'meta.json').read_text(encoding='utf-8'))
    print(f'신규 {ck.name} | 계약 {model} | 피쳐 {meta.get("n_cols")}종')
    for k, v in meta['metrics_raw'].get(model, {}).items():
        print(f'    {k:20s} {v:.5f}')
    if not old.exists():
        print('  구 npz 없음 — 대조 생략')
        return
    z = np.load(old)
    y_h, y_v = z['y_hold'], z['y_valid']
    print('구 자산 (ES valid 5,000행)')
    print(f'    holdout_raw_ll       {raw_ll(z[f"raw_hold__{model}"].astype(np.float64), y_h):.5f}')
    print(f'    valid_raw_ll         {raw_ll(z[f"raw_valid__{model}"].astype(np.float64), y_v):.5f}')
    print('  * ES 세트가 다르므로 일치가 아니라 **근접**을 본다. 판정은 사용자.')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--step', type=int, default=DEF_STEP)
    ap.add_argument('--models', default='xgb', help=f'쉼표 구분 — {"|".join(FAMILIES)} 또는 all')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--tag', default=None, help='checkpoint-{step}-{tag} 로 저장 (사다리 실험용)')
    ap.add_argument('--cols-file', default=None, help='피쳐 부분집합 JSON (리스트 또는 {"cols":[...]})')
    ap.add_argument('--no-cache', action='store_true', help='피쳐 캐시를 쓰지도 만들지도 않는다')
    ap.add_argument('--extract-only', action='store_true', help='피쳐만 뽑아 캐시에 채운다')
    ap.add_argument('--data-dir', default=None,
                    help='stage1 3분할 위치 (기본: 후보 중 valid 5,000행인 곳 자동 선택 — '
                         + ' / '.join(str(c) for c in STAGE1_CANDIDATES) + ')')
    a = ap.parse_args()

    fams = tuple(FAMILIES) if a.models == 'all' else tuple(
        s.strip() for s in a.models.split(',') if s.strip())
    bad = [f for f in fams if f not in FAMILIES]
    assert not bad, f'모르는 부스터: {bad}'

    cols = None
    if a.cols_file:
        obj = json.loads(Path(a.cols_file).read_text(encoding='utf-8'))
        cols = obj['cols'] if isinstance(obj, dict) else obj
        print(f'피쳐 부분집합 {len(cols)}종 <- {a.cols_file}')

    if a.extract_only:
        df, _, m_tr, _, _, _, src = stage1_frames(a.data_dir)
        print(f'stage1 <- {src}  ({len(df):,}행)')
        feature_matrix(df, m_tr, use_cache=not a.no_cache)
        return
    if a.train:
        train(step=a.step, families=fams, overwrite=a.overwrite, cols=cols, tag=a.tag,
              use_cache=not a.no_cache, data_dir=a.data_dir)
    if a.verify:
        verify(step=a.step, model=fams[0], tag=a.tag)
    if not (a.train or a.verify):
        ap.print_help()


if __name__ == '__main__':
    main()
