"""arith.predict — feature matrix (or text frame) -> raw margins and metrics.
예측 모듈. 피쳐 행렬(또는 텍스트 프레임) -> raw 마진 / 지표.

Margins are fold-averaged with swap-TTA **in margin space**. Averaging in probability
space stays correlated (sd3 corr 0.99999) but drifts by max|Δ| 0.07, so only margin space
reproduces the contract columns bit for bit.
마진 계약 — fold 평균 + swap-TTA 를 **마진 공간에서** 평균한다. 확률 공간 평균은 어긋난다.

Arith is a GBDT and has no pooled vector — it reaches the unified head as logits only.
Arith 는 GBDT 라 풀링벡터가 없다 — 통합 헤드에는 로짓(t_ar·m_ar)으로만 들어간다.

These functions take every path explicitly; the notebook normally goes through
`arith.load(...)` and calls the methods on the returned object.
경로는 전부 인자로 받는다. 노트북은 보통 `arith.load(...)` 를 거쳐 메서드로 부른다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ._base import NFOLD, labels_of, raw_ll
from .preprocess import preprocess


def _load_fold(ck, k):
    """Closure yielding raw margins for one xgb fold (the contract family is xgb only).
    xgb fold 하나의 raw 마진 클로저 — 계약 부스터는 xgb 뿐."""
    import xgboost as xgb
    m = xgb.XGBClassifier()
    m.load_model(str(Path(ck) / f'xgb_fold{k}.json'))
    bst, bi = m.get_booster(), getattr(m, 'best_iteration', None)
    kw = dict(output_margin=True)
    if bi is not None:
        kw['iteration_range'] = (0, bi + 1)
    return (lambda X: bst.predict(xgb.DMatrix(X), **kw)), bi


def raw_margins(X, ckpt, swap_X, nfold=NFOLD):
    """Fold-mean + swap-TTA averaged in margin space — identical to the contract columns.
    fold 평균 + swap-TTA 마진 공간 평균 — 계약 npz 의 stage2 열과 같은 계산."""
    Xs = swap_X(X)
    out = np.zeros((len(X), 3), np.float64)
    for k in range(1, nfold + 1):
        pred, _ = _load_fold(ckpt, k)
        out += (pred(X) + pred(Xs)[:, [1, 0, 2]]) / 2 / nfold
    return out


def predict(data, ckpt, cols, swap_X, emb, verbose=True):
    """Prediction entry point: a text frame is preprocessed first, a matrix is used as is.
    예측 진입점 — 텍스트 프레임이면 전처리부터, 행렬이면 그대로 마진을 낸다.

    Returns raw margins (n, 3) in column order [A, B, tie]; the head takes t_ar / m_ar
    from these with sd3. 반환 raw 마진 (n, 3), 열 순서 [A, B, tie].
    """
    if isinstance(data, pd.DataFrame) and 'prompt' in data.columns:
        _, X = preprocess(data, cols, ckpt=ckpt, emb=emb, verbose=verbose)
    else:
        X = np.ascontiguousarray(np.asarray(data, np.float32))
        assert X.ndim == 2 and X.shape[1] == len(cols), \
            f'matrix width {X.shape[-1]} != {len(cols)} contract columns'
    n = min(len(X), 200)
    assert np.allclose(swap_X(swap_X(X[:n])), X[:n], atol=1e-5), 'swap involution failed'
    return raw_margins(X, ckpt=ckpt, swap_X=swap_X)


def eval(data, ckpt, cols, swap_X, emb, verbose=True):
    """Evaluation entry point: predict, and when labels are present report metrics too.
    평가 진입점 — 예측 + (라벨이 있으면) 무보정 지표까지 한 번에.

    Returns margins (n,3) · proba (n,3) · n, plus raw_ll · acc · tie_ll · win_ll · nz
    when the frame carries winner columns. The tie/win split uses the same definition as
    the unified head's two axes, so this shows which axis Arith is weak on by itself.
    tie/win 분해는 통합 헤드의 두 축과 같은 정의다.
    """
    L = predict(data, ckpt=ckpt, cols=cols, swap_X=swap_X, emb=emb, verbose=verbose)
    z = L - L.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    out = {'margins': L, 'proba': p, 'n': len(L)}
    if isinstance(data, pd.DataFrame) and 'winner_model_a' in data.columns:
        y = labels_of(data)
        m = L[:, 0] - L[:, 1]                                  # odd logit — win axis
        t = L[:, 2] - np.logaddexp(L[:, 0], L[:, 1])           # even logit — tie axis
        nz = y != 2
        s_t = np.where(y == 2, 1.0, -1.0)
        s_m = np.where(y[nz] == 0, 1.0, -1.0)
        out.update(y=y, raw_ll=raw_ll(L, y), acc=float((p.argmax(1) == y).mean()),
                   tie_ll=float(np.log1p(np.exp(-s_t * t)).mean()),
                   win_ll=float(np.log1p(np.exp(-s_m * m[nz])).mean()), nz=int(nz.sum()))
        if verbose:
            print(f'  arith alone — raw ll {out["raw_ll"]:.5f} · acc {out["acc"]:.4f} | '
                  f'tie {out["tie_ll"]:.5f} · win {out["win_ll"]:.5f} '
                  f'({out["nz"]:,} non-tie rows)')
    return out
