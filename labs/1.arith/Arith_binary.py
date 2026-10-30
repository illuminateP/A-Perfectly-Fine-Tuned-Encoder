"""Arith_binary.py — tie 전용 산술 전문가. 255피쳐로 tie/비-tie 이진 모델을 새로 학습한다.

왜 만드나
  지금 쓰는 tie 신호 `t_ar` 은 **3-class 목적함수의 부산물**이다. 트리 용량을 승부축과 나눠 쓰므로
  tie 판정에 온전히 쓰이지 않는다. 이진 목적으로 다시 학습하면 46,729행 전부를 tie 에만 쓴다.
  근거: Arith 의 tie 신호는 단독 ll 0.5841 로 tb3(0.5915)보다 이미 낫다 — 전용 학습이면 더 갈 여지.

이진 문제의 구조적 이점
  tie 라벨은 **swap 불변**이다(A↔B 를 바꿔도 tie 는 tie). 그래서 3-class 때와 달리
  증강이 `(swap_X(X), 같은 라벨)` 이고, 예측 TTA 도 단순 평균이다 — 라벨 되돌림이 없다.
  이건 대칭을 근사가 아니라 **구조로** 넣는 것이라 3-class 판보다 규율이 깨끗하다.

전수조사에서 나온 결함 2건을 여기서 표적 수정한다 (전체 재추출 없이)
  1. `_tok_lens` 가 truncation=True, max_length=512 라 토큰 길이가 검열돼 있다
     (실측: a 14.86% / b 14.88% / prompt 3.02% 가 512 도달). 파생 10열이 함께 오염.
     -> `--fix-tok` (기본 on) 이 그 10열만 **무절단으로 재계산**한다. 토크나이저 1패스뿐이라
        features_57k.parquet 도, 3-class Arith 도, 하류 산출물도 건드리지 않는다.
  2. 완전중복 2쌍이 둘 다 사용=1 이다 (ρ = 1.000000):
        diff_keyword_coverage == d_kw_coverage
        abs_diff_keyword_coverage == ad_kw_coverage
     -> `--drop-dup` (기본 on) 이 뒤쪽 하나씩 뺀다. 트리에는 무해하지만 남길 이유도 없다.

  꼬리(max|z| 최대 207, 93열이 20 초과)는 **결함이 아니다** — 트리 분할은 순위 기반이라 무해하다.
  선형 소비자(Davidson)가 랭크-가우스로 받으면 되는 문제라 여기서는 손대지 않는다.

규율 (Train.ipynb 와 동일)
  훈련 = train.csv 5-fold (arith_folds.csv 그대로), ES = valid.csv 고정, 평가 = holdout.csv 전용.
  stage2(valid+hold) 는 그래디언트에 안 들어간다.

내보내기  Arith/outputs/arith_tie.npz
  tie_raw_{oof_train,valid,hold}__{model}  (n,)  softmax 이전 raw margin (= logit p(tie))
  ids_*/y_* /cols/meta.  Davidson 의 S 채널에 그대로 꽂을 수 있는 형식이다.

사용 (수십 분 — 직접 실행할 것)
  python Arith/Arith_binary.py                    # lgbm+xgb+cat
  python Arith/Arith_binary.py --models lgbm      # 빠른 확인
  python Arith/Arith_binary.py --no-fix-tok       # 토큰길이 수정 없이 (현행 피쳐 그대로)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

ARITH = Path('./Arith')
SEED = 42
EPS = 1e-15
META_COLS = ['id', 'role', 'label', 'is_tie']
DUP_DROP = ['d_kw_coverage', 'ad_kw_coverage']      # diff_/abs_diff_keyword_coverage 와 ρ=1.000000
EXT = {'lgbm': 'txt', 'xgb': 'json', 'cat': 'cbm'}


# ══════════════════════════════════════════════════════ 결함 1 — 토큰 길이 검열 해제
def refit_token_len(F, ids):
    """truncation 없이 토큰 길이를 다시 재고, 그 10열을 제자리에 갈아끼운다.

    build_stat_features 의 정의를 그대로 따른다:
        {prompt,a,b}_token_len, ratio_token_len, {a,b}_len_vs_prompt,
        {diff,abs_diff}_token_len, {diff,abs_diff}_len_vs_prompt
    """
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ARITH))
    from audit_features import load_defs                                   # 정의 셀 재사용
    ns = load_defs()

    parts = []
    for name in ('train', 'valid', 'holdout'):
        d = pd.read_csv(ARITH / f'origin/{name}.csv')
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    assert (d['id'].to_numpy() == ids).all(), 'origin 3분할 순서가 parquet 과 다르다'
    P, A, B = ns['parse_columns'](d)

    tok = AutoTokenizer.from_pretrained(str(ARITH / 'emb_model'))
    def lens(texts, tag):
        out = np.empty(len(texts), np.float32)
        for i in range(0, len(texts), 1024):
            enc = tok([t if t else ' ' for t in texts[i:i + 1024]], truncation=False,
                      verbose=False)
            out[i:i + 1024] = [len(x) for x in enc['input_ids']]
            if i % 20480 == 0:
                print(f'    {tag} {i:,}/{len(texts):,}', end='\r')
        return out

    t0 = time.time()
    lp, la, lb = lens(P, 'P'), lens(A, 'A'), lens(B, 'B')
    print(f'    무절단 토큰화 완료 ({time.time() - t0:.0f}s)                    ')

    new = {'prompt_token_len': lp, 'a_token_len': la, 'b_token_len': lb,
           'ratio_token_len': (la + 1.0) / (lb + 1.0),
           'a_len_vs_prompt': la / (lp + 1.0), 'b_len_vs_prompt': lb / (lp + 1.0)}
    for base in ('token_len', 'len_vs_prompt'):
        dd = (new[f'a_{base}'] - new[f'b_{base}']).astype(np.float32)
        new[f'diff_{base}'], new[f'abs_diff_{base}'] = dd, np.abs(dd)
    old_sat = float((F['a_token_len'].to_numpy() >= 512).mean())
    for k, v in new.items():
        assert k in F.columns, f'{k} 가 피쳐에 없다 — 정의가 바뀌었다'
        F[k] = np.nan_to_num(v.astype(np.float32), posinf=0, neginf=0)
    print(f'    갈아끼운 열 {len(new)}개 | a_token_len 512 도달 {old_sat:.2%} -> 0% '
          f'(최댓값 {la.max():.0f})')
    return F, list(new)


# ══════════════════════════════════════════════════════ swap 매핑 (Train.ipynb 셀 2 와 동일)
def build_swap(cols, swap_series):
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
    ni = np.array([idx[c] for c in neg], int)
    ri = np.array([idx[c] for c in recip], int)
    ai = np.array([idx[a] for a, _ in pairs], int)
    bi = np.array([idx[b] for _, b in pairs], int)

    def swap_X(X):
        Xs = X.copy()
        if len(ni):
            Xs[:, ni] *= -1.0
        if len(ri):
            Xs[:, ri] = 1.0 / np.clip(Xs[:, ri], 1e-12, None)
        if len(ai):
            Xs[:, ai], Xs[:, bi] = Xs[:, bi].copy(), Xs[:, ai].copy()
        return Xs
    return swap_X, {'pairs': len(pairs), 'neg': len(neg), 'recip': len(recip), 'inv': len(inv)}


# ══════════════════════════════════════════════════════ 부스터 (이진)
def make(name, n_est):
    if name == 'lgbm':
        import lightgbm as lgb
        return lgb.LGBMClassifier(objective='binary', learning_rate=0.03, n_estimators=n_est,
                                  num_leaves=63, min_child_samples=100, colsample_bytree=0.8,
                                  subsample=0.8, subsample_freq=1, reg_lambda=1.0,
                                  random_state=SEED, n_jobs=-1, verbosity=-1)
    if name == 'xgb':
        import xgboost as xgb
        return xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.03,
                                 n_estimators=n_est, max_depth=8, min_child_weight=8,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                 tree_method='hist', eval_metric='logloss',
                                 early_stopping_rounds=200, random_state=SEED,
                                 n_jobs=-1, verbosity=0)
    import catboost as cb
    return cb.CatBoostClassifier(loss_function='Logloss', eval_metric='Logloss',
                                 iterations=n_est, learning_rate=0.03, depth=8, l2_leaf_reg=4.0,
                                 early_stopping_rounds=200, bootstrap_type='Bernoulli',
                                 subsample=0.8, border_count=254, random_seed=SEED,
                                 verbose=0, allow_writing_files=False)


def fit_one(name, m, X, y, Xe, ye):
    import inspect
    import lightgbm as lgb
    if name == 'lgbm':
        new = 'eval_X' in inspect.signature(lgb.LGBMClassifier.fit).parameters
        ekw = {'eval_X': Xe, 'eval_y': ye} if new else {'eval_set': [(Xe, ye)]}
        m.fit(X, y, **ekw, callbacks=[lgb.early_stopping(200, verbose=False)])
    elif name == 'xgb':
        m.fit(X, y, eval_set=[(Xe, ye)], verbose=False)
    else:
        m.fit(X, y, eval_set=(Xe, ye), use_best_model=True)
    return m


def best_iter(name, m):
    if name == 'lgbm':
        return int(m.booster_.best_iteration or 0)
    if name == 'xgb':
        return int(getattr(m, 'best_iteration', 0) or 0)
    return int(m.get_best_iteration() or 0)


def raw_margin(name, m, X):
    """softmax 이전 raw margin = logit p(tie). 확률로 평균하면 되돌릴 수 없으므로 여기서 낸다."""
    if name == 'lgbm':
        z = m.predict_proba(X, raw_score=True)
    elif name == 'xgb':
        z = m.predict(X, output_margin=True)
    else:
        z = m.predict(X, prediction_type='RawFormulaVal')
    z = np.asarray(z, np.float64).ravel()
    assert z.shape == (len(X),) and np.isfinite(z).all(), f'{name} raw 형상/유한성'
    return z


def tta(name, m, X, swap_X):
    """tie 는 swap 불변이라 되돌림 없이 그냥 평균한다 — 대칭이 구조로 보장된다."""
    return (raw_margin(name, m, X) + raw_margin(name, m, swap_X(X))) / 2


# ══════════════════════════════════════════════════════ 지표
def tie_report(z, t):
    p = 1.0 / (1.0 + np.exp(-z))
    return {'tie ll': float(log_loss(t, np.clip(p, EPS, 1 - EPS), labels=[0, 1])),
            'tie AUC': float(roc_auc_score(t, z)),
            'acc%': float(((p > 0.5).astype(int) == t).mean() * 100),
            'recall%': float((p[t == 1] > 0.5).mean() * 100)}


def paired_ci(a, b, n_boot=2000, seed=SEED):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def ci_str(m, lo, hi, thresh=0.0005):
    sig = '유의' if (lo > 0 or hi < 0) else '노이즈'
    return f'{m:+.4f} [{lo:+.4f},{hi:+.4f}] {sig}' + (' ★' if sig == '유의' and abs(m) >= thresh else '')


def bce_rows(z, t):
    p = np.clip(1.0 / (1.0 + np.exp(-z)), EPS, 1 - EPS)
    return -np.where(t == 1, np.log(p), np.log(1 - p))


# ══════════════════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='lgbm,xgb,cat')
    ap.add_argument('--n-est', type=int, default=4000)
    ap.add_argument('--no-fix-tok', dest='fix_tok', action='store_false',
                    help='토큰 길이 512 검열을 그대로 둔다 (현행 피쳐 재현용)')
    ap.add_argument('--no-drop-dup', dest='drop_dup', action='store_false')
    ap.add_argument('--out', default='outputs/arith_tie.npz')
    args = ap.parse_args()
    MODELS = [m.strip() for m in args.models.split(',') if m.strip()]
    assert all(m in EXT for m in MODELS), f'알 수 없는 모델: {MODELS}'

    feat = pd.read_parquet(ARITH / 'features_57k.parquet')
    cat = pd.read_csv(ARITH / 'feature_catalog.csv')
    role, y = feat['role'].to_numpy(), feat['label'].to_numpy()
    ids = feat['id'].to_numpy()
    t = (y == 2).astype(np.int64)                       # 이진 타깃 — swap 불변이다
    m_tr, m_va, m_ho = role == 'train', role == 'valid', role == 'holdout'
    assert [m_tr.sum(), m_va.sum(), m_ho.sum()] == [46729, 5000, 5748], '분할 크기 불일치'
    print(f'train {m_tr.sum():,} / valid {m_va.sum():,} / hold {m_ho.sum():,} | '
          f'tie 비율 {t.mean():.4f} (train {t[m_tr].mean():.4f})')

    F = feat.drop(columns=META_COLS)
    fixed = []
    if args.fix_tok:
        print('\n[결함 수정 1] 토큰 길이 무절단 재계산')
        F, fixed = refit_token_len(F, ids)

    cols = cat[cat['사용'] == 1]['feature'].tolist()
    if args.drop_dup:
        drop = [c for c in DUP_DROP if c in cols]
        cols = [c for c in cols if c not in drop]
        print(f'[결함 수정 2] 완전중복 제거: {drop}  -> {len(cols)}피쳐')
    else:
        print(f'피쳐 {len(cols)}종')

    sw_series = cat.set_index('feature')['swap']
    swap_X, info = build_swap(cols, sw_series)
    X = F[cols].to_numpy(np.float32)
    assert np.isfinite(X).all(), 'NaN/inf'
    assert np.allclose(swap_X(swap_X(X[:200])), X[:200], atol=1e-5), 'swap involution 실패'
    print(f'swap: 교환 {info["pairs"]}쌍 / 부호반전 {info["neg"]} / 역수 {info["recip"]} / '
          f'불변 {info["inv"]}  (involution 통과)')

    folds_df = pd.read_csv(ARITH / 'arith_folds.csv')
    assert (folds_df['id'].to_numpy() == ids[m_tr]).all(), 'folds 정렬 불일치'
    folds = folds_df['fold'].to_numpy()
    X_tr, X_va, X_ho = X[m_tr], X[m_va], X[m_ho]
    t_tr, t_va, t_ho = t[m_tr], t[m_va], t[m_ho]

    R = {}
    for name in MODELS:
        t0 = time.time()
        oof = np.zeros(len(X_tr))
        z_va, z_ho = np.zeros(len(X_va)), np.zeros(len(X_ho))
        iters = []
        for k in np.sort(np.unique(folds)):
            tr, va = folds != k, folds == k
            # tie 라벨은 swap 불변 -> 증강은 (교환뷰, 같은 라벨) 이다. 되돌림이 없다.
            Xf = np.vstack([X_tr[tr], swap_X(X_tr[tr])])
            tf = np.hstack([t_tr[tr], t_tr[tr]])
            m = fit_one(name, make(name, args.n_est), Xf, tf, X_va, t_va)
            oof[va] = tta(name, m, X_tr[va], swap_X)
            z_va += tta(name, m, X_va, swap_X) / 5
            z_ho += tta(name, m, X_ho, swap_X) / 5
            iters.append(best_iter(name, m))
            d = ARITH / 'model'
            (m.booster_.save_model(str(d / f'tie_{name}_fold{k + 1}.txt')) if name == 'lgbm'
             else m.save_model(str(d / f'tie_{name}_fold{k + 1}.{EXT[name]}')))
        R[name] = {'oof': oof, 'valid': z_va, 'hold': z_ho, 'iters': iters}
        r = tie_report(z_ho, t_ho)
        print(f'  {name:<5} holdout tie ll {r["tie ll"]:.4f}  AUC {r["tie AUC"]:.4f}  '
              f'acc {r["acc%"]:.2f}%  recall {r["recall%"]:.2f}%  iters {iters} '
              f'({time.time() - t0:.0f}s)')

    z_mean = np.mean([R[n]['hold'] for n in MODELS], axis=0)      # raw 공간 평균 = 기하평균
    R['평균'] = {'hold': z_mean,
                 'valid': np.mean([R[n]['valid'] for n in MODELS], axis=0),
                 'oof': np.mean([R[n]['oof'] for n in MODELS], axis=0), 'iters': []}

    # ── 기존 3-class Arith 의 tie 신호와 대조 (holdout 전용) ──
    print(f'\n{"=" * 96}\n[판정] holdout 5,748 — 기존 3-class 부산물 t_ar 대비')
    base_p = ARITH / 'outputs/arith_experts_raw.npz'
    rows = [[n, f'{tie_report(R[n]["hold"], t_ho)["tie ll"]:.4f}',
             f'{tie_report(R[n]["hold"], t_ho)["tie AUC"]:.4f}', '']
            for n in list(MODELS) + ['평균']]
    if base_p.exists():
        za = np.load(base_p)
        blk = np.mean([za[f'raw_hold__{m}'] for m in ('lgbm', 'xgb', 'cat')], axis=0)
        t_ar = blk[:, 2] - np.logaddexp(blk[:, 0], blk[:, 1])
        rb = tie_report(t_ar, t_ho)
        base_rows = bce_rows(t_ar, t_ho)
        for r in rows:
            r[3] = ci_str(*paired_ci(base_rows, bce_rows(R[r[0]]['hold'], t_ho)))
        rows.append(['t_ar (3-class 부산물)', f'{rb["tie ll"]:.4f}', f'{rb["tie AUC"]:.4f}', '기준'])
    print(f'{"모델":<22}{"tie ll":>9}{"tie AUC":>10}  Δ vs t_ar (양수 = 이진 전용이 이득)')
    print('─' * 78)
    for r in rows:
        print(f'{r[0]:<22}{r[1]:>9}{r[2]:>10}  {r[3]}')
    print('\n  holdout 5,748 은 SE ≈ 0.0058 이라 CI 폭이 stage2 의 1.37배다 — 문턱 판정에 감안할 것.')

    out = ARITH / args.out
    np.savez_compressed(
        out, cols=np.array(cols), fixed_cols=np.array(fixed),
        ids_train=ids[m_tr], y_train=y[m_tr], ids_valid=ids[m_va], y_valid=y[m_va],
        ids_hold=ids[m_ho], y_hold=y[m_ho],
        **{f'tie_raw_oof_train__{n}': R[n]['oof'].astype(np.float32) for n in R},
        **{f'tie_raw_valid__{n}': R[n]['valid'].astype(np.float32) for n in R},
        **{f'tie_raw_hold__{n}': R[n]['hold'].astype(np.float32) for n in R})
    meta = {'created': time.strftime('%Y-%m-%d %H:%M'), 'task': 'binary tie',
            'n_features': len(cols), 'models': MODELS, 'n_est': args.n_est,
            'fix_tok': args.fix_tok, 'fixed_cols': fixed,
            'drop_dup': args.drop_dup, 'dropped': DUP_DROP if args.drop_dup else [],
            'best_iters': {n: R[n]['iters'] for n in MODELS},
            'holdout': {n: tie_report(R[n]['hold'], t_ho) for n in R},
            'note': 'tie 라벨은 swap 불변 -> 증강·TTA 에 라벨 되돌림이 없다'}
    (ARITH / 'model/tie_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                               encoding='utf-8')
    print(f'\nsaved: {out}  |  model/tie_{{lgbm,xgb,cat}}_fold1..5  |  model/tie_meta.json')


if __name__ == '__main__':
    main()
