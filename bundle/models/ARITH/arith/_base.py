"""arith._base — contract constants + catalog/swap conventions shared by the modules.
계약 상수와 카탈로그·swap 규약. 세 모듈이 공유하는 기반.

No path is derived here — the caller supplies ckpt / catalog / emb, exactly the way
build_model(tag) is handed spec['cfg'] and spec['ckpt']. The package therefore works
wherever the bundle is mounted (local tree, Kaggle input) with no __file__ guessing.
경로를 여기서 만들지 않는다 — 호출자가 ckpt·catalog·emb 를 준다. 인코더의 build_model 이
spec['cfg']·spec['ckpt'] 를 받는 것과 같은 모양이라, 번들 위치가 바뀌어도 코드가 그대로다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Contract constants / 계약 상수 ──
# 열 수는 여기 없다 — preprocess.N_CONTRACT(224) 가 빌더 네 갈래의 합으로 강제한다.
# No column count here: preprocess.N_CONTRACT (224) enforces it as the sum of the builders.
SEED = 42
NFOLD = 5
CONTRACT_FAMILY = 'xgb'
EMB_HF_ID = 'sentence-transformers/all-MiniLM-L6-v2'
EMB_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'   # pinned commit of the official repository / 공식 저장소의 고정 커밋


def _tq(seq, desc):
    return tqdm(seq, desc=desc, file=sys.stdout, smoothing=0, leave=False)


# ══════════════════════════════════════════════ catalog & swap / 카탈로그·swap
def catalog_cols(catalog):
    """Contract features and their swap conventions, read from the given catalog csv.
    계약 피쳐와 swap 규약 — 인자로 받은 카탈로그 csv 에서 읽는다.

    The swap table is required at inference: swap-TTA applies negate / reciprocal /
    exchange / invariant exactly as this table says. The 의미·계산·의존성 columns are
    documentation and never enter the computation.
    swap 규약은 추론 필수다 — swap-TTA 가 이 표대로 적용한다. 설명 열은 계산에 안 쓰인다.
    """
    p = Path(catalog)
    assert p.exists(), f'catalog not found: {p}'
    cat = pd.read_csv(p)
    # ab224 카탈로그는 전 행이 계약 열이다 — 전체 259 카탈로그는 상류(피쳐 선택) 전용.
    # Every row of the ab224 catalog is a contract column; the full catalog lives upstream.
    return cat['feature'].tolist(), cat.set_index('feature')['swap']


def build_swap(cols, swap_series):
    """How each feature behaves when A and B are exchanged — negate / reciprocal / exchange /
    invariant. The caller verifies swap(swap(X)) == X.
    A<->B 교환 시 피쳐 거동. 호출부가 involution 을 검증한다."""
    uncovered = [c for c in cols if c not in swap_series.index]
    assert not uncovered, (f'{len(uncovered)} column(s) have no swap rule: {uncovered[:5]} — '
                           'the catalog does not match the checkpoint contract')
    neg = [c for c in cols if swap_series[c] == '부호반전']
    recip = [c for c in cols if swap_series[c] == '역수']
    pairs = []
    for c in cols:
        v = swap_series[c]
        if isinstance(v, str) and v.startswith('교환:') and c < v.split(':', 1)[1]:
            partner = v.split(':', 1)[1]
            assert partner in set(cols), f'exchange partner {partner} is outside the contract'
            pairs.append((c, partner))
    inv = [c for c in cols if swap_series[c] == '불변']
    assert len(neg) + len(recip) + len(inv) + 2 * len(pairs) == len(cols), \
        'swap classification does not cover every column'
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
    """Labels under A<->B exchange: flip A/B, leave tie alone.
    A<->B 교환 시 라벨 — 0<->1 뒤집고 tie(2) 는 불변."""
    return np.where(y == 0, 1, np.where(y == 1, 0, 2)).astype(y.dtype)


def labels_of(d):
    """Three winner columns -> integer label (0=A, 1=B, 2=tie).
    3열 승자 표기 -> 정수 라벨."""
    y = np.select([d['winner_model_a'] == 1, d['winner_model_b'] == 1, d['winner_tie'] == 1],
                  [0, 1, 2], default=-1).astype(np.int64)
    assert (y != -1).all(), 'some rows map to none of the three winner columns'
    return y


def raw_ll(L, y):
    """Uncalibrated 3-class logloss of the raw margins (no temperature fitting).
    raw 마진의 무보정 3-class 로그손실."""
    zs = L - L.max(1, keepdims=True)
    p = np.exp(zs)
    p /= p.sum(1, keepdims=True)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean())


# ══════════════════════════════════════════════ checkpoint / 체크포인트
def read_meta(ckpt):
    """The checkpoint's meta.json — this is Arith's config, the counterpart of an
    encoder's config.json. 체크포인트 meta.json — Arith 의 config 에 해당한다."""
    return json.loads((Path(ckpt) / 'meta.json').read_text(encoding='utf-8'))


def verify_assets(ckpt, catalog, emb):
    """Check every deploy asset and return the meta dict.
    배포 자산을 확인하고 meta 를 돌려준다 — 폴드·meta·artifacts·카탈로그·임베더.

    Stops here when the catalog and the checkpoint disagree: inferring with a mismatched
    catalog leaves columns without a swap rule and silently breaks the involution.
    카탈로그와 meta 의 열 집합이 어긋나면 여기서 멈춘다 — 그대로 추론하면 involution 이 깨진다.
    """
    ck, cat, em = Path(ckpt), Path(catalog), Path(emb)
    bad = [p for p in ['meta.json', 'artifacts/tfidf.joblib', 'artifacts/svd.joblib']
           + [f'xgb_fold{k}.json' for k in range(1, NFOLD + 1)] if not (ck / p).exists()]
    assert not bad, f'{ck.name}: missing {bad}'
    assert cat.exists(), f'catalog not found: {cat}'
    assert em.exists(), f'embedder directory not found: {em}'
    meta = read_meta(ck)
    assert meta['contract_family'] == CONTRACT_FAMILY, \
        f'contract family {meta["contract_family"]!r} != {CONTRACT_FAMILY!r}'
    assert meta['n_folds'] == NFOLD, f'n_folds {meta["n_folds"]} != {NFOLD}'
    assert len(meta['cols']) == meta['n_cols'], 'meta cols count does not match n_cols'
    build_swap(list(meta['cols']), catalog_cols(cat)[1])   # swap rules present, pairs closed
    return meta
