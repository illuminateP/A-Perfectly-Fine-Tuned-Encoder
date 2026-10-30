"""arith — the Arith expert (ab224: 224 arithmetic features, xgb 5-fold, logits only).
Arith 전문가 패키지 — ab224, 224피쳐 × XGB 5-fold, 로짓으로만 기여.

Loaded the same way an encoder is: hand it a config (meta.json + feature catalog) and
weights (the xgb folds), get back something ready to predict.
인코더와 같은 방식으로 적재한다 — 설정(meta.json + 카탈로그)과 가중치(xgb 폴드)를 주면
바로 예측 가능한 객체가 나온다.

    import arith
    ari = arith.load(ckpt=ARITH_SPEC['ckpt'],
                     catalog=ARITH_SPEC['catalog'],
                     emb=ARITH_SPEC['emb'])
    L   = ari.predict(df)              # raw margins (n, 3)  [A, B, tie]
    res = ari.eval(holdout_df)         # margins + metrics when labels are present
    F, X = ari.preprocess(df)          # text -> the 224 contract columns (frame, matrix)

Three modules / 세 모듈
    arith.preprocess   text -> features; TF-IDF/SVD are transform-only at inference
    arith.predict      fold mean + swap-TTA margins, and the per-axis metrics
    arith.train        stage1 3-split -> a deployable checkpoint (xgb folds + artifacts + meta)

학습도 패키지 안에 있다 — 경로는 여전히 전부 호출자가 주므로 저장소 배치는 모른다.
Training now lives here too; it still takes every path from the caller, so the package knows
no repository layout.

    ck = arith.train(ckpt=..., catalog=..., emb=..., data_dir=..., folds=...)

Contract
    features  ab224 = top-224 by shap contribution, closed under swap-exchange pairs.
              preprocess.py defines these 224 and nothing else — there is no wider candidate
              pool here, so re-selecting features means re-drawing the contract upstream.
              preprocess.py 는 이 224종만 정의한다 — 후보 공간이 없으므로 피쳐 재선택은
              상류의 피쳐 선택 단계에서 계약을 다시 뽑는 일이다.
    booster   xgb 5-fold (seed 42 draw), swap augmentation, early stopping on the stage1
              valid 5,000 — the set the encoder ckpts were selected on. That valid ships
              outside the deploy bundle, so retraining is local-only.
              ES 는 stage1 valid 5,000 (인코더 ckpt 선택과 같은 세트) — 번들 밖에 있어
              재학습은 로컬 전용이다.
    role      logits only (t_ar / m_ar) — a GBDT has no pooled vector
    embedder  every BERT-derived feature shares one budget: 512 tokens, fp32, no char guard,
              and one truncation rule — head 60% + tail 40% for both emb_* and BERTScore
              (they share pack_head_tail). Only the pooling differs. See MAX_SEQ_LEN.
              BERT 가 닿는 피쳐는 전부 같은 예산·같은 절단이다 — 512토큰·fp32·문자 상한
              없음·앞60%+뒤40%. emb_* 와 BERTScore 의 차이는 풀링뿐이다.
    train     the stage1 column is the same fold-mean predict as everywhere else (in-sample).
              The OOF column was retired: experiments showed no measurable difference, and
              in-sample matches the encoders, which also re-run on their own training data.
              stage1 열도 동일한 fold 평균 predict 다 (in-sample). OOF 열은 퇴역 —
              실험상 차이가 없었고, 자기 학습 데이터를 다시 순전파하는 인코더와 조건이 같아진다.

Self-sufficiency
    The code is self-contained. The assets are not: the embedder weights, the checkpoint's
    tfidf/svd artifacts, the feature catalog and the xgb folds all ship in the bundle, and
    every one of them is passed in by the caller. No network, no hub lookup.
    코드는 자급하고 자산은 번들에 의존한다 — 경로는 전부 호출자가 준다. 네트워크 접근 없음.
"""
from pathlib import Path

from . import predict as _predict
from . import preprocess as _preprocess
from ._base import (CONTRACT_FAMILY, EMB_HF_ID, NFOLD, SEED, build_swap, catalog_cols,
                    labels_of, raw_ll, read_meta, swap_label, verify_assets)
from .predict import raw_margins
from .preprocess import extract
from .train import train

__version__ = 'ab224'


class Arith:
    """One loaded Arith expert — resolved paths, verified contract, ready to predict.
    적재된 Arith 전문가 하나 — 경로 해석·계약 검증이 끝나 바로 예측할 수 있다."""

    def __init__(self, ckpt, catalog, emb, verbose=True):
        self.ckpt, self.catalog, self.emb = Path(ckpt), Path(catalog), Path(emb)
        self.meta = verify_assets(self.ckpt, self.catalog, self.emb)
        self.cols = list(self.meta['cols'])
        self.swap_X, self.swap_info = build_swap(self.cols, catalog_cols(self.catalog)[1])
        if verbose:
            print(f'*** arith {self.meta["tag"]} loaded {len(self.cols)} features, '
                  f'{self.meta["n_folds"]}-fold {self.meta["contract_family"]} ***')

    def preprocess(self, df, verbose=True):
        """Text frame -> (contract-column frame, contract-column matrix) — same 224, same order.
        텍스트 프레임 -> (F 계약열 프레임, X 계약열 행렬) — 같은 224열, 같은 순서."""
        return _preprocess.preprocess(df, self.cols, ckpt=self.ckpt, emb=self.emb,
                                      verbose=verbose)

    def predict(self, data, verbose=True):
        """Raw margins (n, 3) in [A, B, tie] order. 원시 마진."""
        return _predict.predict(data, ckpt=self.ckpt, cols=self.cols, swap_X=self.swap_X,
                                emb=self.emb, verbose=verbose)

    def eval(self, data, verbose=True):
        """Margins plus uncalibrated metrics when the frame carries winner columns.
        마진 + 라벨이 있으면 무보정 지표."""
        return _predict.eval(data, ckpt=self.ckpt, cols=self.cols, swap_X=self.swap_X,
                             emb=self.emb, verbose=verbose)

    def __repr__(self):
        return (f'Arith(tag={self.meta["tag"]!r}, n_cols={len(self.cols)}, '
                f'ckpt={self.ckpt})')


def load(ckpt, catalog, emb, verbose=True):
    """Build one Arith expert from its config (meta.json + catalog) and weights (xgb folds).
    설정과 가중치를 받아 Arith 전문가 하나를 만든다 — 인코더의 build_model(tag) 와 같은 자리."""
    return Arith(ckpt, catalog, emb, verbose=verbose)


__all__ = ['Arith', 'load', 'train', 'extract', 'raw_margins', 'read_meta', 'verify_assets',
           'catalog_cols', 'build_swap', 'swap_label', 'labels_of', 'raw_ll',
           'SEED', 'NFOLD', 'CONTRACT_FAMILY', 'EMB_HF_ID', '__version__']
