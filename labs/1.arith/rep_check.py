"""rep_check.py — 반복 피처 후보를 카탈로그에 넣기 전에 검증한다.

기존 카탈로그에 이미 있는 것: mn/mx/ad/d_unique_3gram_ratio (비율), type_token_ratio.
후보는 '비율이 아닌 절대 카운트·구조'다 — 긴 답변에서 비율은 희석되지만 카운트는 안 그런다.

  rep_max_ngram   최빈 단어 3-gram 의 등장 횟수 (절대)
  rep_top_share   최빈 3-gram 이 전체 3-gram 에서 차지하는 비율
  rep_dup_line    중복 줄 비율 (목록·코드 퇴화)
  rep_run         같은 줄이 연속으로 반복된 최대 길이 (생성 루프)
  rep_compress    zlib 압축비 len(z)/len(raw) — 모든 스케일의 반복을 한 숫자로

검증 3단 (언어 불일치를 기각시킨 것과 같은 절차)
  [1] 중복성 — 기존 unique_3gram_ratio·길이와의 상관. |r| 이 높으면 새 정보가 아니다
  [2] 라벨   — 십분위별 tie율 / 승률, 무신호면 기각
  [3] 사각지대 — ABE 마진을 5분위로 통제한 뒤의 lift (Decision_TB §3 형식).
      인코더가 이미 아는 신호면 통제 후 lift 가 사라진다

사용: python Arith/rep_check.py
"""
import json
import re
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ARITH = Path('./Arith')
WORD = re.compile(r'\S+')
MAX_WORDS = 2000                       # 초장문 비용 상한 (분포 p99 이상만 잘린다)


def parse_text(x):
    if pd.isna(x):
        return ''
    try:
        t = json.loads(x)
    except (json.JSONDecodeError, TypeError):
        return str(x)
    return ' '.join(s for s in t if isinstance(s, str)) if isinstance(t, list) else str(t)


def rep_feats(text):
    """반복 5종. 빈 문자열도 안전한 기본값을 낸다."""
    if not text:
        return dict(rep_max_ngram=0.0, rep_top_share=0.0, rep_dup_line=0.0,
                    rep_run=0.0, rep_compress=1.0)
    w = WORD.findall(text)[:MAX_WORDS]
    if len(w) >= 3:
        grams = Counter(zip(w, w[1:], w[2:]))
        top = grams.most_common(1)[0][1]
        n_g = sum(grams.values())
    else:
        top, n_g = 0, 1
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if lines:
        c = Counter(lines)
        dup = 1.0 - len(c) / len(lines)                 # 중복 줄 비율
        run, cur = 1, 1
        for i in range(1, len(lines)):
            cur = cur + 1 if lines[i] == lines[i - 1] else 1
            run = max(run, cur)
    else:
        dup, run = 0.0, 0.0
    raw = text.encode('utf-8', 'ignore')
    comp = len(zlib.compress(raw, 6)) / max(len(raw), 1)
    return dict(rep_max_ngram=float(top), rep_top_share=top / max(n_g, 1),
                rep_dup_line=float(dup), rep_run=float(run), rep_compress=float(comp))


def main():
    df = pd.read_csv(ARITH / 'origin' / 'train.csv')
    y = np.select([df['winner_model_a'] == 1, df['winner_model_b'] == 1, df['winner_tie'] == 1],
                  [0, 1, 2], -1).astype(np.int64)
    tie = y == 2
    A = df['response_a'].map(parse_text)
    B = df['response_b'].map(parse_text)
    fa = pd.DataFrame([rep_feats(t) for t in A])
    fb = pd.DataFrame([rep_feats(t) for t in B])
    bases = list(fa.columns)

    # {mn, mx, ad, d} 4종 전개 — 카탈로그 관례와 동일
    F = {}
    for b in bases:
        a_, b_ = fa[b].to_numpy(), fb[b].to_numpy()
        F[f'mn_{b}'] = np.minimum(a_, b_)
        F[f'mx_{b}'] = np.maximum(a_, b_)
        F[f'ad_{b}'] = np.abs(a_ - b_)
        F[f'd_{b}'] = a_ - b_
    F = pd.DataFrame(F)

    feat = pd.read_parquet(ARITH / 'features_57k.parquet')
    m_tr = feat['role'].to_numpy() == 'train'
    old = feat.loc[m_tr, ['mn_unique_3gram_ratio', 'mx_unique_3gram_ratio',
                          'mn_type_token_ratio', 'mx_char_len', 'mn_char_len']].reset_index(drop=True)

    print('=== [1] 기존 피처와의 상관 (|r| 큰 것 = 새 정보 아님) ===')
    print(f'  {"후보":<20} ' + ' '.join(f'{c[:16]:>17}' for c in old.columns))
    for c in [f'{p}_{b}' for p in ('mn', 'mx') for b in bases]:
        rs = [np.corrcoef(F[c], old[o])[0, 1] for o in old.columns]
        print(f'  {c:<20} ' + ' '.join(f'{r:>+17.3f}' for r in rs))

    print(f'\n=== [2] 라벨 연관 (전체 tie율 {tie.mean():.3f}) ===')
    for c in F.columns:
        v = F[c].to_numpy()
        if np.std(v) == 0:
            print(f'  {c:<20} 분산 0 — 제외')
            continue
        try:
            dec = pd.qcut(v, 10, labels=False, duplicates='drop')
        except ValueError:
            dec = (v > np.median(v)).astype(int)
        tr = pd.Series(tie).groupby(dec).mean()
        rng = tr.max() - tr.min()
        star = ' ★' if rng > 0.05 else ''
        print(f'  {c:<20} tie율 범위 {tr.min():.3f}~{tr.max():.3f} (폭 {rng:.3f}){star}'
              f'   {"["+" ".join(f"{x:.2f}" for x in tr.values)+"]" if len(tr) <= 10 else ""}')

    # [3] ABE 마진 통제 후 lift — 인코더가 이미 아는 신호인지
    za = ARITH / 'logits' / 'abe_train_logits.npz'
    if not za.exists():
        print('\n[3] 생략 — ABE 로짓 캐시 없음')
        return
    z = np.load(za)
    assert (z['ids'] == df['id'].to_numpy()).all()
    zz = (z['fwd'] + z['swp']) / 2
    margin = np.abs(zz[:, 0] - zz[:, 1])
    mq = pd.qcut(margin, 5, labels=False, duplicates='drop')
    print('\n=== [3] ABE 마진 5분위 통제 후 tie율 lift (상위 절반 − 하위 절반) ===')
    print('    Decision_TB §3 기준: pr_jac_word 가 +0.072 로 최강이었다')
    for c in [f'{p}_{b}' for p in ('mn', 'mx', 'ad') for b in bases]:
        v = F[c].to_numpy()
        lifts = []
        for q in range(5):
            sel = mq == q
            med = np.median(v[sel])
            hi, lo = tie[sel & (v > med)], tie[sel & (v <= med)]
            lifts.append(hi.mean() - lo.mean() if len(hi) > 20 and len(lo) > 20 else np.nan)
        avg = np.nanmean(lifts)
        star = ' ★' if abs(avg) > 0.03 else ''
        print(f'  {c:<20} ' + ' '.join(f'{x:+.3f}' if np.isfinite(x) else '  nan' for x in lifts)
              + f'   평균 {avg:+.3f}{star}')


if __name__ == '__main__':
    main()
