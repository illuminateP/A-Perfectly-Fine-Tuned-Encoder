"""fit_arith.py — Arith(xgb) 가 **실제로 쓰는 피쳐**를 확정하고, 안 쓰는 피쳐를 닫았을 때의
영향을 검산한다. GPU 불필요.

배경
  Arith 는 255 피쳐 × 5-fold 부스터다. 계약상 선택되는 부스터는 **xgb** (valid raw ll
  xgb 1.01373 < cat 1.01537 < lgbm 1.01597, 온도 피팅 없음) — 노트북 test 경로가 xgb 를
  하드코딩한 것과 일치한다.

무엇을 재나
  1) **추론 구간 기준** 사용 피쳐. get_score 는 모델의 전 트리를 세지만, 추론은
     iteration_range=(0, best_iter+1) 로만 돈다. 그 뒤 트리에서만 등장하는 피쳐는
     사실상 미사용이므로 부분 모델(bst[0:bi+1])에서 센다.
  2) **영향은 측정이 아니라 항등이다.** 추론 구간에서 한 번도 분기하지 않는 열은 값이
     무엇이든 라우팅에 관여하지 않는다 -> 그 열을 0 이나 NaN 으로 채워도 마진이 **비트
     단위로 같다**. 따라서 GAT/LIN 을 태운 holdout ll 도 정확히 같다. 이 스크립트는
     그 항등을 assert 로 강제한다 (0 채움 / NaN 채움 두 가지, swap-TTA 포함).
  3) 저장 자산 재현 — 로컬 예측이 arith_experts_raw.npz 의 raw_hold__xgb 와 맞는지.
     여기가 어긋나면 위 항등 검산도 신뢰할 수 없다.
  4) 비용 계층 — 미사용 피쳐가 어느 의존성(무상태 / MiniLM / TF-IDF / 토크나이저)에
     속하는지. 무상태뿐이면 **닫아도 추출 시간은 1초도 안 준다.**

주의 — 열을 진짜로 빼려면 재학습이 필요하다
  DMatrix 는 meta['cols'] 255열 순서를 그대로 요구한다. 재학습 없이 얻을 수 있는 것은
  "추출을 건너뛰고 자리는 상수로 채운다" 뿐이고, 그건 위 항등 덕에 무손실이다.

사용
  python final/fit_arith.py                 # 전체 진단 + 항등 검산
  python final/fit_arith.py --dump closed.json   # 닫아도 되는 열 목록을 파일로
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

FINAL = Path(__file__).resolve().parent.parent      # 이 파일은 final/temp/ 에 있다
ARITH = FINAL / 'models/ARITH'
ORIGIN = FINAL / 'origin'
NFOLD = 5


def load_folds(cols):
    """5-fold xgb 를 (booster, best_iteration) 으로. feature 이름 환원까지."""
    out = []
    for k in range(1, NFOLD + 1):
        m = xgb.XGBClassifier()
        m.load_model(str(ARITH / f'model/xgb_fold{k}.json'))
        bi = getattr(m, 'best_iteration', None)
        bst = m.get_booster()
        names = bst.feature_names or cols
        out.append((bst, bi, names))
    return out


def used_features(folds, cols):
    """추론 구간(0..best_iter)에서 최소 한 번 분기한 피쳐의 fold 별 집합과 총 gain."""
    per, gain = [], {}
    for bst, bi, names in folds:
        sub = bst[0:(bi + 1)] if bi is not None else bst
        g = sub.get_score(importance_type='total_gain')
        if g and list(g)[0][0] == 'f' and list(g)[0][1:].isdigit():
            g = {names[int(x[1:])]: v for x, v in g.items()}
        per.append(set(g))
        for x, v in g.items():
            gain[x] = gain.get(x, 0.0) + v
    return per, gain


def build_swap(cols, swap_series):
    """A<->B 교환 시 피쳐 거동 — 노트북 셀과 같은 구현."""
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


def margins(folds, X, swap_X):
    """노트북 셀 71 과 같은 경로 — raw 마진 fold 평균 + swap-TTA."""
    out = np.zeros((len(X), 3))
    Xs = swap_X(X)
    for bst, bi, _ in folds:
        kw = dict(output_margin=True)
        if bi is not None:
            kw['iteration_range'] = (0, bi + 1)
        out += (bst.predict(xgb.DMatrix(X), **kw)
                + bst.predict(xgb.DMatrix(Xs), **kw)[:, [1, 0, 2]]) / 2 / NFOLD
    return out


def raw_ll(L, y):
    zs = L - L.max(1, keepdims=True)
    p = np.exp(zs)
    p /= p.sum(1, keepdims=True)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', help='닫아도 되는 열 목록을 JSON 으로 저장')
    args = ap.parse_args()

    cols = json.loads((ARITH / 'model/meta.json').read_text(encoding='utf-8'))['cols']
    folds = load_folds(cols)
    print(f'xgb {NFOLD}-fold 적재 | 피쳐 {len(cols)}')
    print('  best_iteration: ' + ', '.join(str(bi) for _, bi, _ in folds))

    # ── 1) 사용 피쳐 ──────────────────────────────────────────
    per, gain = used_features(folds, cols)
    for k, s in enumerate(per, 1):
        print(f'  fold{k}: 추론 구간 사용 {len(s)}')
    U = set().union(*per)
    I = set.intersection(*per)
    closed = [c for c in cols if c not in U]
    print(f'\n합집합 {len(U)} | 전 fold 공통 {len(I)} | **미사용 {len(closed)}**')
    for c in closed:
        print(f'    {c}')

    # ── 2) 비용 계층 ──────────────────────────────────────────
    docs = pd.read_csv(ORIGIN / 'feature_docs.csv').set_index('name')
    cat = pd.read_csv(ORIGIN / 'feature_catalog.csv').set_index('feature')
    dep = cat['의존성'] if '의존성' in cat.columns else docs['의존성']
    print('\n의존성별 (전체 / 미사용):')
    tot = dep.reindex(cols).value_counts(dropna=False)
    sub = dep.reindex(closed).value_counts(dropna=False)
    for kdep in tot.index:
        print(f'    {str(kdep):10s} {tot[kdep]:4d} / {int(sub.get(kdep, 0)):3d}')
    exp_dep = {d for d in tot.index if str(d) != '무상태'}
    costly_closed = [c for c in closed if dep.get(c) in exp_dep]
    print(f'  -> 닫아서 추출을 건너뛸 수 있는 **비용 피쳐** {len(costly_closed)}개'
          + (f': {costly_closed}' if costly_closed else ' — 없다. 추출 시간 절감 0.'))

    # ── 3) 저장 자산 재현 + 4) 항등 검산 ──────────────────────
    z = np.load(ARITH / 'outputs/arith_experts_raw.npz')
    ids_h, y_h = z['ids_hold'], z['y_hold']
    F = pd.read_parquet(ARITH / 'features_57k.parquet').set_index('id')
    miss = [i for i in ids_h if i not in F.index]
    assert not miss, f'features_57k 에 holdout id {len(miss)}개 없음'
    X = F.loc[ids_h, cols].to_numpy(np.float32)
    assert np.isfinite(X).all(), '피쳐에 NaN/inf'

    swap_fn, info = build_swap(cols, cat['swap'])
    assert np.allclose(swap_fn(swap_fn(X)), X, atol=1e-5), 'swap involution 실패'
    bad_recip = [c for c in closed if c in info['recip']]
    assert not bad_recip, f'미사용인데 역수 규약: {bad_recip} — 상수 채움이 안전하지 않다'

    L0 = margins(folds, X, swap_fn)
    ref = z['raw_hold__xgb'].astype(np.float64)
    d = float(np.abs(L0 - ref).max())
    print(f'\n저장 자산 재현: max|Δ| {d:.3e}  (raw ll 로컬 {raw_ll(L0, y_h):.5f} / '
          f'저장 {raw_ll(ref, y_h):.5f})')
    assert d < 1e-4, '*** 저장 raw_hold__xgb 와 불일치 — 아래 검산을 신뢰하지 말 것'

    idx_closed = [cols.index(c) for c in closed]
    print('\n미사용 열을 닫았을 때 (마진 비트 동일이어야 한다):')
    for tag, fill in (('0 채움', 0.0), ('NaN 채움', np.nan)):
        Xc = X.copy()
        if idx_closed:
            Xc[:, idx_closed] = fill
        Lc = margins(folds, Xc, swap_fn)
        same = np.array_equal(Lc, L0) if fill == fill else np.array_equal(Lc, L0)
        dd = float(np.nanmax(np.abs(Lc - L0)))
        print(f'    {tag:9s} max|Δ| {dd:.3e}   {"비트 동일 OK" if same else "*** 차이 발생"}')
        assert same, f'{tag}: 미사용 열이 실제로는 예측에 관여한다 — 목록이 틀렸다'

    print('\n=> 미사용 열은 예측에 전혀 관여하지 않는다. GAT/LIN 을 태운 holdout ll 도 '
          '정확히 동일하다 (측정이 아니라 항등).')

    # ── 저기여 피쳐 참고 ──────────────────────────────────────
    tg = sum(gain.values())
    low = sorted(gain.items(), key=lambda kv: kv[1])[:25]
    print(f'\n[참고] 쓰이긴 하나 기여가 미미한 하위 25개 — 누적 total_gain '
          f'{sum(v for _, v in low) / tg * 100:.4f}%')
    for c, v in low:
        print(f'    {c:26s} {v / tg * 100:8.5f}%   {str(dep.get(c))}')
    print('  (이쪽을 닫으려면 **재학습**이 필요하고, 그때는 holdout 판정 런이 의미가 있다.)')

    if args.dump:
        p = Path(args.dump)
        p.write_text(json.dumps({'closed': closed, 'kept': [c for c in cols if c in U],
                                 'n_cols': len(cols)}, ensure_ascii=False, indent=1),
                     encoding='utf-8')
        print(f'\n저장: {p}')


if __name__ == '__main__':
    main()
