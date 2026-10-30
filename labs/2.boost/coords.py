"""coords.py — "스케일이 다른 로짓을 그냥 결합한 게 문제냐"를 두 갈래로 분리해 잰다.

논점 분리
  (A) 스케일    트리는 열별 단조변환에 불변이다(분할은 문턱 하나). 열마다 임의의 양수배·평행이동을
                걸고 재학습해 결과가 같은지 본다. 같으면 **스케일은 논점이 아니다**가 실증된다.
                (LGBM/XGB hist·CatBoost border 는 분위수 기반이라 이론상 완전 동일해야 한다.)

  (B) 좌표계    이쪽이 진짜 논점이다. softmax 로짓은 행별 가산상수가 자유라 z_a 하나는 의미가 없고
                의미 있는 건 차이뿐이다:
                    s = z_tie − logaddexp(z_a, z_b)   = logit p(tie)      tie축 단독 결정
                    d = z_a − z_b                     = logit p(A|비-tie) 승부축 단독 결정
                그런데 지금은 z_a, z_b 를 **따로** 준다. 축정렬 분할은 z_a − z_b 라는 대각선을
                못 긋고 계단으로 근사한다. TB 2열도 실제 자유도는 1(s_tb)뿐이다.
                (s, d) 는 정보상 무손실이지만 **트리 등가가 아니다** — 그래서 실측이 필요하다.

arm
  raw       현행 14열 (ABE 3 + TB 2 + Arith 9)
  rep       재매개화 9열 (s_abe, d_abe, s_tb, Arith 3모델 x (s, d))
  rep+|d|   rep + 마진 |d| 4열 = 13열. |d| 는 swap 불변량이라 '확신 세기'를 대칭적으로 준다
  raw+rep   23열. 트리가 좌표계를 직접 고르게 둔다 (중복은 트리가 감당한다)

swap 규율
  raw 는 A<->B 가 열 치환이지만 rep 좌표에서는 **s 불변 / d 부호반전**이다. 두 경우를 한 코드로
  다루려고 swap 을 (순열, 부호) 쌍으로 표현하고 involution 을 매번 검사한다. 틀리면 증강·TTA 가
  조용히 비대칭을 만든다.

판정
  Boost.ipynb 와 같은 하네스 — stage2 10,748 5-fold, fold 마다 평가 20% / ES 10%, swap 증강 + TTA,
  fold 평균 보고. raw 대비 paired bootstrap 95% CI, 문턱 0.0005.

사용 (2분 넘음 — 직접 실행할 것)
  python Boost/coords.py                      # cat, 4 arm + 스케일 실증
  python Boost/coords.py --boosters lgbm      # 빠른 확인
  python Boost/coords.py --boosters lgbm,cat --skip-scale
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

LLM = Path('.')
BOOST = LLM / 'Boost'
ARITH = LLM / 'Arith'
SEED = 42
EPS = 1e-15
N_SPLITS = 5
INNER_FRAC = 0.10
ARITH_MODELS = ('lgbm', 'xgb', 'cat')


# ══════════════════════════════════════════════════════ 입력
def load():
    z = np.load(BOOST / 'boost_oof.npz')
    ids, y = z['ids'], z['y']
    meta = json.loads((BOOST / 'model/boost_meta.json').read_text(encoding='utf-8'))
    ex = meta['experts']

    def stack(tag, steps):
        acc = []
        for s in steps:
            w = np.load(BOOST / f'logits/{tag}__checkpoint-{s}.npz')
            assert (np.concatenate([w['valid_ids'], w['hold_ids']]) == ids).all(), f'{tag} id 불일치'
            assert (np.concatenate([w['valid_labels'], w['hold_labels']]) == y).all(), f'{tag} 라벨'
            acc.append(np.vstack([(w['valid_fwd'] + w['valid_swp']) / 2,
                                  (w['hold_fwd'] + w['hold_swp']) / 2]).astype(np.float64))
        return np.mean(acc, axis=0)

    L_abe = stack('abe', ex['abe'].get('steps') or [ex['abe']['step']])
    L_tb = stack('tb3', ex['tb'].get('steps') or [ex['tb']['step']])

    def arith(path, pre):
        za = np.load(path)
        assert (np.concatenate([za['ids_valid'], za['ids_hold']]) == ids).all(), f'{path} id 불일치'
        return [np.vstack([za[f'{pre}_valid__{m}'], za[f'{pre}_hold__{m}']]).astype(np.float64)
                for m in ARITH_MODELS]

    L_ari = arith(ARITH / 'outputs/arith_experts.npz', 'logp')       # 기존 규약(확률평균 후 log)
    raw_p = ARITH / 'outputs/arith_experts_raw.npz'                  # 새 규약(raw margin 평균)
    L_raw = arith(raw_p, 'raw') if raw_p.exists() else None
    return ids, y, L_abe, L_tb, L_ari, L_raw


def sd_of(blk):
    """3-class 블록 -> (s, d). 자유도 2 라 무손실이고 축을 하나씩 단독 결정한다."""
    return blk[:, 2] - np.logaddexp(blk[:, 0], blk[:, 1]), blk[:, 0] - blk[:, 1]


def build_arms(L_abe, L_tb, L_ari, L_raw=None):
    """arm 마다 (X, 열이름, swap 순열, swap 부호) 를 만든다.

    L_raw 를 주면 Arith 블록을 **raw margin 규약**으로 바꾼 arm(`ari-raw`)을 함께 만든다.
    단일 모델이면 raw 와 logp 는 행별 평행이동 차이뿐이지만, **평균을 낸 뒤에는 다르다** —
    raw 평균 = 확률의 기하평균, logp 평균 = 산술평균의 로그. 그 차이를 재는 arm 이다.
    """
    C = {}                                        # 이름 -> (값, swap 부호)
    C['abe_z_a'], C['abe_z_b'], C['abe_z_tie'] = (L_abe[:, 0], 0), (L_abe[:, 1], 0), (L_abe[:, 2], +1)
    C['tb_z_notie'], C['tb_z_tie'] = (L_tb[:, 0], +1), (L_tb[:, 1], +1)
    PAIR = {'abe_z_a': 'abe_z_b', 'abe_z_b': 'abe_z_a'}
    for pre, blocks in (('ari', L_ari), ('arw', L_raw)):
        if blocks is None:
            continue
        for m, B in zip(ARITH_MODELS, blocks):
            C[f'{pre}_{m}_a'], C[f'{pre}_{m}_b'] = (B[:, 0], 0), (B[:, 1], 0)
            C[f'{pre}_{m}_tie'] = (B[:, 2], +1)
            PAIR[f'{pre}_{m}_a'] = f'{pre}_{m}_b'
            PAIR[f'{pre}_{m}_b'] = f'{pre}_{m}_a'
    s, d = sd_of(L_abe)
    C['s_abe'], C['d_abe'], C['|d_abe|'] = (s, +1), (d, -1), (np.abs(d), +1)
    C['s_tb'] = (L_tb[:, 1] - L_tb[:, 0], +1)
    for m, B in zip(ARITH_MODELS, L_ari):
        s, d = sd_of(B)
        C[f's_{m}'], C[f'd_{m}'], C[f'|d_{m}|'] = (s, +1), (d, -1), (np.abs(d), +1)

    ENC = ['abe_z_a', 'abe_z_b', 'abe_z_tie', 'tb_z_notie', 'tb_z_tie']
    RAW = ENC + [f'ari_{m}_{c}' for m in ARITH_MODELS for c in ('a', 'b', 'tie')]
    ARW = ENC + [f'arw_{m}_{c}' for m in ARITH_MODELS for c in ('a', 'b', 'tie')]
    REP = ['s_abe', 'd_abe', 's_tb'] + [f'{k}_{m}' for m in ARITH_MODELS for k in ('s', 'd')]
    ABS = ['|d_abe|'] + [f'|d_{m}|' for m in ARITH_MODELS]

    def make(names):
        X = np.column_stack([C[n][0] for n in names])
        idx = {n: i for i, n in enumerate(names)}
        perm = np.arange(len(names))
        sign = np.ones(len(names))
        for i, n in enumerate(names):
            sg = C[n][1]
            if sg == 0:                            # 교환쌍 — 짝이 같은 arm 안에 있어야 한다
                p = PAIR[n]
                assert p in idx, f'{n} 의 교환 파트너 {p} 가 arm 에 없다'
                perm[i] = idx[p]
            else:
                sign[i] = sg
        assert (perm[perm] == np.arange(len(names))).all(), 'swap 순열이 involution 이 아니다'
        assert np.allclose(sign[perm] * sign, 1.0), 'swap 부호가 involution 이 아니다'
        chk = X[:50]
        assert np.allclose((chk[:, perm] * sign)[:, perm] * sign, chk), 'swap∘swap != 항등'
        return X, names, perm, sign

    arms = {'raw': make(RAW), 'rep': make(REP), 'rep+|d|': make(REP + ABS),
            'raw+rep': make(RAW + REP)}
    if L_raw is not None:
        arms['ari-raw'] = make(ARW)               # 새 설계: Arith 도 raw 로짓
        arms['ari-both'] = make(RAW + ARW[5:])    # 두 규약 동시 — 트리가 직접 고르게
    return arms


# ══════════════════════════════════════════════════════ 지표
def per_row(p, y):
    p = np.clip(np.asarray(p, np.float64), EPS, 1.0)
    p = p / p.sum(1, keepdims=True)
    return -np.log(p[np.arange(len(y)), y])


def axis_report(p, y):
    p = np.clip(np.asarray(p, np.float64), EPS, 1.0)
    p = p / p.sum(1, keepdims=True)
    total = float(per_row(p, y).mean())
    it = y == 2
    t = np.clip(p[:, 2], EPS, 1 - EPS)
    tie = float(-np.where(it, np.log(t), np.log(1 - t)).mean())
    nz = ~it
    rA = np.clip(p[nz, 0] / (p[nz, 0] + p[nz, 1]), EPS, 1 - EPS)
    win = float(-np.where(y[nz] == 0, np.log(rA), np.log(1 - rA)).sum() / len(y))
    assert abs(total - tie - win) < 1e-6, '축 분해 검산 실패'
    return {'총 ll': total, 'tie축': tie, '승부축': win,
            'tie AUC': float(roc_auc_score(it.astype(int), t)),
            '승부 AUC': float(roc_auc_score((y[nz] == 0).astype(int), rA)),
            'acc%': float((p.argmax(1) == y).mean() * 100)}


def paired_ci(a, b, n_boot=2000, seed=SEED):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def fmt_ci(m, lo, hi, thresh=0.0005):
    sig = '유의' if (lo > 0 or hi < 0) else '노이즈'
    return f'{m:+.4f} [{lo:+.4f},{hi:+.4f}] {sig}' + (' ★' if sig == '유의' and abs(m) >= thresh else '')


# ══════════════════════════════════════════════════════ 부스터 (Boost.ipynb 와 동일 파라미터)
def make_lgbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective='multiclass', num_class=3, learning_rate=0.03,
                              n_estimators=3000, num_leaves=15, min_child_samples=60,
                              colsample_bytree=0.9, subsample=0.9, subsample_freq=1,
                              reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1)


def make_xgb():
    import xgboost as xgb
    return xgb.XGBClassifier(objective='multi:softprob', num_class=3, learning_rate=0.03,
                             n_estimators=3000, max_depth=4, min_child_weight=20,
                             subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                             tree_method='hist', eval_metric='mlogloss',
                             early_stopping_rounds=100, random_state=SEED, n_jobs=-1, verbosity=0)


def make_cat():
    import catboost as cb
    return cb.CatBoostClassifier(loss_function='MultiClass', eval_metric='MultiClass',
                                 iterations=3000, learning_rate=0.03, depth=4,
                                 l2_leaf_reg=6.0, early_stopping_rounds=100,
                                 bootstrap_type='Bernoulli', subsample=0.9, border_count=254,
                                 random_seed=SEED, verbose=0, allow_writing_files=False)


FACTORY = {'lgbm': make_lgbm, 'xgb': make_xgb, 'cat': make_cat}


def fit_one(name, m, Xf, yf, X_es, y_es):
    import inspect
    import lightgbm as lgb
    if name == 'lgbm':
        new = 'eval_X' in inspect.signature(lgb.LGBMClassifier.fit).parameters
        ekw = {'eval_X': X_es, 'eval_y': y_es} if new else {'eval_set': [(X_es, y_es)]}
        m.fit(Xf, yf, **ekw, callbacks=[lgb.early_stopping(100, verbose=False)])
    elif name == 'xgb':
        m.fit(Xf, yf, eval_set=[(X_es, y_es)], verbose=False)
    else:
        m.fit(Xf, yf, eval_set=(X_es, y_es), use_best_model=True)
    return m


def swap_y(y):
    return np.where(y == 0, 1, np.where(y == 1, 0, 2)).astype(y.dtype)


def cv_boost(name, X, y, perm, sign):
    """Boost.ipynb 의 cv_boost 와 같은 분할·증강·TTA. swap 만 (순열, 부호) 로 일반화했다."""
    n_inner = int(round(INNER_FRAC * len(y)))
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), 3))
    rows, iters = [], []
    sw = lambda Z: Z[:, perm] * sign
    for k, (rest, te) in enumerate(skf.split(X, y)):
        tr, es = train_test_split(rest, test_size=n_inner, stratify=y[rest],
                                  random_state=SEED + k, shuffle=True)
        Xf = np.vstack([X[tr], sw(X[tr])])
        yf = np.hstack([y[tr], swap_y(y[tr])])
        m = fit_one(name, FACTORY[name](), Xf, yf, X[es], y[es])
        p = (m.predict_proba(X[te]) + m.predict_proba(sw(X[te]))[:, [1, 0, 2]]) / 2
        oof[te] = p
        rows.append(axis_report(p, y[te]))
        if name == 'lgbm':
            iters.append(int(m.booster_.best_iteration or 0))
        elif name == 'xgb':
            iters.append(int(getattr(m, 'best_iteration', 0) or 0))
        else:
            iters.append(int(m.get_best_iteration() or 0))
    fd = pd.DataFrame(rows)
    return {'mean': fd.mean(), 'std': fd.std(ddof=1), 'oof': oof, 'iters': iters}


# ══════════════════════════════════════════════════════ (A) 스케일 실증
def scale_test(name, X, y, perm, sign, cols):
    """열마다 임의의 양수배 + 평행이동. 트리가 불변이면 총 ll 이 그대로여야 한다.

    주의: swap 이 교환쌍을 섞으므로 **쌍에는 같은 변환**을 걸어야 증강이 깨지지 않는다.
    (a 열만 10배 하면 swap 뷰의 a 열은 b 열에서 오므로 분포가 어긋난다.)
    """
    rng = np.random.default_rng(SEED)
    a = np.exp(rng.normal(0, 1.0, X.shape[1]))          # 0.1~10 배
    b = rng.normal(0, 2.0, X.shape[1])
    # 변환이 swap 과 교환법칙을 지켜야 한다: (Xa+b)[perm]*sign == (X[perm]*sign)a + b
    #   -> a[perm] == a (교환쌍은 같은 배율)  and  b[perm]*sign == b (부호반전 열은 b=0)
    for i, j in enumerate(perm):
        a[i] = a[min(i, j)]
        b[i] = b[min(i, j)]
    b = np.where(sign < 0, 0.0, b)
    assert np.allclose(a[perm], a) and np.allclose(b[perm] * sign, b), '변환이 swap 과 안 맞는다'
    Xs = X * a + b
    print(f'\n  [스케일 실증] 열마다 x{a.min():.2f}~x{a.max():.2f} 배 + 평행이동 후 재학습')
    r0 = cv_boost(name, X, y, perm, sign)
    r1 = cv_boost(name, Xs, y, perm, sign)
    d = float(abs(r0['mean']['총 ll'] - r1['mean']['총 ll']))
    # 완전 비트일치는 기대하지 않는다 — 히스토그램 경계를 float32 로 다시 잡으므로 1e-6 급
    # 수치오차가 남는다. 판정 문턱은 채택 문턱(0.0005)의 1/5 인 1e-4 로 둔다.
    print(f'    원본 {r0["mean"]["총 ll"]:.6f} | 변환 {r1["mean"]["총 ll"]:.6f} | |Δ| {d:.2e}  '
          + ('-> 스케일은 논점이 아니다 (수치오차 수준)' if d < 1e-4
             else '-> 실제 차이 발생 — 트리 불변성 가정 재검토'))
    print(f'    (참고 열 표준편차: 원본 {np.std(X, 0).min():.3f}~{np.std(X, 0).max():.3f} '
          f'-> 변환 {np.std(Xs, 0).min():.3f}~{np.std(Xs, 0).max():.3f}, 열 {len(cols)}개)')
    return r0


# ══════════════════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boosters', default='cat', help='쉼표 구분 (lgbm,xgb,cat)')
    ap.add_argument('--skip-scale', action='store_true', help='(A) 스케일 실증 생략')
    ap.add_argument('--out', default='coords_oof.npz')
    args = ap.parse_args()
    names = [s.strip() for s in args.boosters.split(',') if s.strip()]
    assert all(n in FACTORY for n in names), f'알 수 없는 부스터: {names}'

    ids, y, L_abe, L_tb, L_ari, L_raw = load()
    ARMS = build_arms(L_abe, L_tb, L_ari, L_raw)
    print(f'stage2 {len(ids):,}행 | 부스터 {names} | swap involution 전 arm 통과'
          + ('' if L_raw is not None else '\n  ! arith_experts_raw.npz 없음 — '
                                          'Boost/arith_raw.py 를 먼저 실행하면 ari-raw arm 이 생긴다'))
    for k, (X, cols, _, sg) in ARMS.items():
        print(f'  {k:<9} {X.shape[1]:>2}열  부호반전 {int((sg < 0).sum())}개  {cols}')

    R = {}
    for name in names:
        print(f'\n{"=" * 96}\n[{name}]')
        base = None
        if not args.skip_scale:
            X, cols, perm, sign = ARMS['raw']
            base = scale_test(name, X, y, perm, sign, cols)
        for arm, (X, cols, perm, sign) in ARMS.items():
            if arm == 'raw' and base is not None:
                R[(name, arm)] = base
            else:
                t0 = time.time()
                R[(name, arm)] = cv_boost(name, X, y, perm, sign)
                print(f'  {arm:<9} 학습 {time.time() - t0:.0f}s  iters {R[(name, arm)]["iters"]}')

    print(f'\n{"=" * 96}\n[(B) 좌표계 비교] fold 평균 ± std, Δ 는 raw 대비 (양수 = 재매개화가 이득)')
    print(f'  {"부스터":<6} {"arm":<9} {"열":>3} {"총 ll":>8} {"±std":>7} {"tie축":>8} {"승부축":>8} '
          f'{"승부AUC":>8}  Δ 와 95% CI')
    for name in names:
        base_ll = per_row(R[(name, 'raw')]['oof'], y)
        for arm in ARMS:
            r = R[(name, arm)]
            ci = '' if arm == 'raw' else fmt_ci(*paired_ci(base_ll, per_row(r['oof'], y)))
            print(f'  {name:<6} {arm:<9} {ARMS[arm][0].shape[1]:>3} {r["mean"]["총 ll"]:8.4f} '
                  f'{r["std"]["총 ll"]:7.4f} {r["mean"]["tie축"]:8.4f} {r["mean"]["승부축"]:8.4f} '
                  f'{r["mean"]["승부 AUC"]:8.4f}  {ci}')

    np.savez_compressed(BOOST / args.out, ids=ids, y=y,
                        **{f'oof__{n}__{a}': R[(n, a)]['oof'].astype(np.float32) for n, a in R})
    print(f'\nsaved: {BOOST / args.out}')
    print('\n[읽는 법] 스케일 실증에서 |Δ| ≈ 0 이면 "스케일이 문제"는 기각이다.')
    print('  rep 이 raw 를 문턱 0.0005 넘겨 이기면 축정렬 기하가 실제로 손해였다는 뜻 —')
    print('  그 경우 승부축이 먼저 내려가야 한다(d 를 직접 준 효과). tie축만 움직이면 우연이다.')
    print('  raw+rep 만 이기고 rep 이 지면 좌표계가 아니라 열 수가 이득이었다는 뜻이다.')


if __name__ == '__main__':
    main()
