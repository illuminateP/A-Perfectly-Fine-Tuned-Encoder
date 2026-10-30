# -*- coding: utf-8 -*-
"""feature.py — 버킷 토큰 후보 산술 피처의 상관 분석.

응답 텍스트에서 ~30개 산술 피처를 뽑아, 어떤 피처가 라벨(0=a승, 1=b승, 2=tie)에
신호를 갖는지 3개 지표로 측정한다:

  1. 승부 축   : Spearman ρ( fa-fb , y_signed )        y_signed = +1(a승)/-1(b승)/0(tie)
  2. 무승부 축 : point-biserial r( -|fa-fb| , is_tie )  "둘이 비슷하면 tie"
  3. 버킷 delta: fa-fb 를 5분위 버킷 → 버킷별 경험 라벨분포로 예측했을 때
                 baseline(사전분포) 대비 logloss 감소량.
                 = "버킷 토큰 1개가 인코더에 전달할 수 있는 정보량"의 직접 근사 (주 결정 지표)

사용법:
  python feature.py                    # origin/train.csv 로 전체 분석
  from feature import extract_features, compute_bucket_edges   # 파이프라인에서 재사용

주의: 분석은 파이프라인과 동일한 SEED=42 10% 분리 후 train 쪽만 사용 (hold-out 오염 방지).
"""
import os
import re
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd

SEED = 42
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(_HERE, '..', 'origin', 'train.csv')
REPORT_MD = os.path.join(_HERE, 'feature_report.md')   # 실행마다 자동 append

# ======================================================================
# 텍스트 파싱 (파이프라인과 동일)
# ======================================================================
def parse_text(x):
    if pd.isna(x):
        return ''
    try:
        turns = json.loads(x)
    except (json.JSONDecodeError, TypeError):
        return str(x)
    if isinstance(turns, list):
        return ' '.join(t for t in turns if isinstance(t, str))
    return str(turns)


# ======================================================================
# 피처 추출 (응답 1개당) — 토크나이저 불필요, 문자 기반
# ======================================================================
_RE_SENT       = re.compile(r'[.!?]+')
_RE_PARA       = re.compile(r'\n\s*\n')
_RE_NUMBERING  = re.compile(r'^\s*\d+[.)]\s')
_RE_NUMBER     = re.compile(r'\d+(?:\.\d+)?')
_RE_EXAMPLE    = re.compile(r'for example|for instance|e\.g\.|example:', re.I)
_RE_FOOTNOTE   = re.compile(r'\[\d+\]')
_RE_MDLINK     = re.compile(r'\[[^\]]+\]\([^)]+\)')
_RE_FORMAL     = re.compile(r'\b(?:furthermore|moreover|consequently|therefore|thus|'
                            r'additionally|nevertheless|hence)\b', re.I)
_RE_UNCERTAIN  = re.compile(r'\b(?:might|may|perhaps|possibly|likely|probably|seems|appears)\b'
                            r'|i think|not sure', re.I)
_RE_POLITE     = re.compile(r'please|thank|apolog|appreciate|glad to|happy to', re.I)
_RE_PASSIVE    = re.compile(r'\b(?:is|are|was|were|be|been|being)\s+'
                            r'(?:\w+ed|made|done|given|taken|seen|known|found|used|shown|'
                            r'written|built|held|kept|sent|told|thought|understood)\b', re.I)
_RE_REFUSAL    = re.compile(r"as an ai|i can'?t|i cannot|i'?m sorry|i apologize|"
                            r"unable to|sorry,? but", re.I)
_RE_YOU        = re.compile(r'\byou\b', re.I)
_RE_WORD       = re.compile(r'\w+')
_RE_POSITIVE   = re.compile(
    r'\b(?:good|great|excellent|best|better|effective|efficient|helpful|useful|valuable|'
    r'important|easy|simple|powerful|amazing|awesome|wonderful|fantastic|perfect|ideal|'
    r'beneficial|brilliant|impressive|outstanding|remarkable|superior|optimal|convenient|'
    r'flexible|reliable|robust|accurate|correct|clear|safe|secure|fast|quick|smooth|'
    r'love|enjoy|recommended?|advantages?|benefits?|successful?|'
    r'improve(?:d|s|ment)?|enhance[ds]?)\b', re.I)
_RE_NEGATIVE   = re.compile(
    r'\b(?:bad|worse|worst|wrong|errors?|mistakes?|fail(?:s|ed|ure|ures)?|problems?|'
    r'problematic|difficult(?:y|ies)?|hard|harder|impossible|unfortunately|issues?|'
    r'risk(?:s|y)?|danger(?:ous)?|harm(?:ful)?|poor(?:ly)?|lack(?:s|ing)?|incorrect|'
    r'inaccurate|negative|weak|slow|broken|bugs?|confusing|confused|unclear|complicated|'
    r'misleading|flaw(?:s|ed)?|terrible|horrible|awful|useless|invalid|unsafe|insecure|'
    r'unstable|inefficient|disadvantages?|drawbacks?|concern(?:s|ing)?|severe|fatal|'
    r'damage[ds]?|threats?|vulnerable|vulnerabilit(?:y|ies))\b', re.I)
_RE_NEGATION   = re.compile(
    r"\b(?:not|no|never|none|nothing|nobody|nowhere|neither|nor|cannot|without|"
    r"hardly|barely|scarcely|unable)\b|n't", re.I)

# 피처 → 카테고리 (보고용)
CATEGORIES = {
    'char_len': 'Length', 'word_count': 'Length',
    'sentence_count': 'Length', 'paragraph_count': 'Length',
    'bullet_count': 'Structure', 'numbering_count': 'Structure',
    'heading_count': 'Structure', 'has_table': 'Structure',
    'code_block_count': 'Structure', 'quote_count': 'Structure',
    'avg_sentence_len': 'Readability', 'avg_word_len': 'Readability',
    'type_token_ratio': 'Readability', 'unique_3gram_ratio': 'Readability',
    'number_count': 'Information', 'url_count': 'Information',
    'example_count': 'Information', 'citation_count': 'Information',
    'formal_words': 'Style', 'uncertainty_words': 'Style',
    'politeness': 'Style', 'passive_ratio': 'Style',
    'positive_words': 'Sentiment', 'negative_words': 'Sentiment',
    'negation_rate': 'Sentiment',
    'refusal': 'Etc', 'you_rate': 'Etc', 'user_mention': 'Etc',
    'newline_count': 'Etc', 'question_marks': 'Etc',
    'non_ascii_ratio': 'Etc', 'is_empty': 'Etc', 'prompt_echo': 'Etc',
}
BINARY_FEATS = {'has_table', 'refusal', 'is_empty'}

# 피처 → 산출 방법 (리포트 표에 사용. 마크다운 표 안이므로 '|' 금지)
DESCRIPTIONS = {
    'char_len':          'log1p(문자 수)',
    'word_count':        'log1p(공백 분리 단어 수)',
    'sentence_count':    'log1p([.!?]+ 매치 수)',
    'paragraph_count':   'log1p(빈 줄로 구분된 블록 수)',
    'bullet_count':      'log1p(행 시작 -, *, • 개수)',
    'numbering_count':   'log1p(행 시작 "숫자." / "숫자)" 개수)',
    'heading_count':     'log1p(행 시작 # 개수)',
    'has_table':         '이진: "---"와 파이프를 함께 가진 행 존재',
    'code_block_count':  'log1p(백틱 3개 코드펜스 쌍 개수)',   # 표 셀 안에 백틱 금지 (렌더러 호환)
    'quote_count':       'log1p(행 시작 > 개수)',
    'avg_sentence_len':  '단어 수 / max(문장 수, 1)',
    'avg_word_len':      '총 문자(공백 제외) / 단어 수',
    'type_token_ratio':  '고유 단어(소문자) / 전체 단어',
    'unique_3gram_ratio': '고유 단어 3-gram / 전체 3-gram (반복 루프 탐지)',
    'number_count':      'log1p(숫자 패턴 \\d+(.\\d+)? 매치 수)',
    'url_count':         'log1p("http" 등장 수)',
    'example_count':     'log1p("for example", "e.g." 등 매치 수)',
    'citation_count':    'log1p([n] 각주 + md링크 + "according to"/"source:")',
    'formal_words':      'furthermore/moreover/therefore 등 격식어 빈도 /100단어',
    'uncertainty_words': 'might/may/perhaps/i think 등 헤징어 /100단어',
    'politeness':        'please/thank/apolog/appreciate 등 /100단어',
    'passive_ratio':     'be동사+과거분사 휴리스틱 매치 /100단어 (POS 태거 없는 근사)',
    'positive_words':    '긍정 어휘사전 ~45어 (good/excellent/reliable/improve 등) /100단어',
    'negative_words':    '부정 어휘사전 ~50어 (bad/error/flaw/vulnerable 등) /100단어',
    'negation_rate':     '부정문 표지 15종 (not/no/never/nothing/without/n\'t 등) /100단어',
    'refusal':           '이진: as an ai, i can\'t, i\'m sorry 등 거부 정규식 존재',
    'you_rate':          '"you" 단어 빈도 /100단어 (사용자 직접 호칭)',
    'user_mention':      'log1p("the user" 등장 수) (3인칭 호칭)',
    'newline_count':     'log1p(\\n 개수) — 토크나이저가 소실시키는 신호',
    'question_marks':    'log1p(? 개수) — 답 대신 되묻기',
    'non_ascii_ratio':   '비ASCII 문자 수 / 전체 문자 수',
    'is_empty':          '이진: 공백 제거 후 빈 문자열',
    'prompt_echo':       '질문 단어집합과 답변 단어집합의 Jaccard (동문서답 탐지)',
}


def extract_features(t):
    """응답 텍스트 1개 → 피처 dict. (prompt_echo 는 쌍 수준에서 별도 계산)"""
    n_char = len(t)
    words  = t.split()
    n_w    = max(len(words), 1)
    lines  = t.split('\n')
    n_sent = len(_RE_SENT.findall(t))
    lower_words = [w.lower() for w in words]

    if len(lower_words) >= 3:
        tri = list(zip(lower_words, lower_words[1:], lower_words[2:]))
        uniq3 = len(set(tri)) / len(tri)
    else:
        uniq3 = 1.0

    per100 = 100.0 / n_w

    return {
        # ---- Length ----
        'char_len':        np.log1p(n_char),
        'word_count':      np.log1p(len(words)),
        'sentence_count':  np.log1p(n_sent),
        'paragraph_count': np.log1p(sum(1 for b in _RE_PARA.split(t) if b.strip())),
        # ---- Structure ----
        'bullet_count':    np.log1p(sum(1 for l in lines if l.lstrip()[:2] in ('- ', '* ', '• '))),
        'numbering_count': np.log1p(sum(1 for l in lines if _RE_NUMBERING.match(l))),
        'heading_count':   np.log1p(sum(1 for l in lines if l.lstrip().startswith('#'))),
        'has_table':       1.0 if any('|' in l and '---' in l for l in lines) else 0.0,
        'code_block_count': np.log1p(t.count('```') // 2),
        'quote_count':     np.log1p(sum(1 for l in lines if l.lstrip().startswith('>'))),
        # ---- Readability ----
        'avg_sentence_len': len(words) / max(n_sent, 1),
        'avg_word_len':     sum(len(w) for w in words) / n_w,
        'type_token_ratio': len(set(lower_words)) / n_w,
        'unique_3gram_ratio': uniq3,
        # ---- Information ----
        'number_count':    np.log1p(len(_RE_NUMBER.findall(t))),
        'url_count':       np.log1p(t.count('http')),
        'example_count':   np.log1p(len(_RE_EXAMPLE.findall(t))),
        'citation_count':  np.log1p(len(_RE_FOOTNOTE.findall(t))
                                    + len(_RE_MDLINK.findall(t))
                                    + t.lower().count('according to')
                                    + t.lower().count('source:')),
        # ---- Style (빈도/100단어) ----
        'formal_words':      len(_RE_FORMAL.findall(t)) * per100,
        'uncertainty_words': len(_RE_UNCERTAIN.findall(t)) * per100,
        'politeness':        len(_RE_POLITE.findall(t)) * per100,
        'passive_ratio':     len(_RE_PASSIVE.findall(t)) * per100,   # POS 태거 없는 휴리스틱 근사
        # ---- Sentiment (빈도/100단어) ----
        'positive_words':    len(_RE_POSITIVE.findall(t)) * per100,
        'negative_words':    len(_RE_NEGATIVE.findall(t)) * per100,
        'negation_rate':     len(_RE_NEGATION.findall(t)) * per100,
        # ---- Etc ----
        'refusal':         1.0 if _RE_REFUSAL.search(t) else 0.0,
        'you_rate':        len(_RE_YOU.findall(t)) * per100,
        'user_mention':    np.log1p(t.lower().count('the user')),
        'newline_count':   np.log1p(t.count('\n')),
        'question_marks':  np.log1p(t.count('?')),
        'non_ascii_ratio': sum(1 for c in t if ord(c) > 127) / max(n_char, 1),
        'is_empty':        1.0 if not t.strip() else 0.0,
    }


def prompt_echo(prompt_text, resp_text):
    """질문 단어 집합과 답변 단어 집합의 Jaccard — 동문서답 탐지."""
    pw = set(w.lower() for w in _RE_WORD.findall(prompt_text))
    rw = set(w.lower() for w in _RE_WORD.findall(resp_text))
    if not pw or not rw:
        return 0.0
    return len(pw & rw) / len(pw | rw)


def extract_frame(texts, prompts=None):
    """텍스트 리스트 → 피처 DataFrame."""
    rows = [extract_features(t) for t in texts]
    df = pd.DataFrame(rows)
    if prompts is not None:
        df['prompt_echo'] = [prompt_echo(p, t) for p, t in zip(prompts, texts)]
    return df


# ======================================================================
# 버킷 유틸 (토큰화/추론 단계에서 재사용)
# ======================================================================
def compute_bucket_edges(vals, k=5):
    """train 분위수 기반 버킷 경계. 반환된 edges 는 np.searchsorted 로 사용."""
    edges = np.unique(np.quantile(vals, np.linspace(0, 1, k + 1)[1:-1]))
    return edges


def bucketize(vals, edges):
    return np.searchsorted(edges, vals, side='right')


# ======================================================================
# 분석
# ======================================================================
def _bucket_logloss_delta(d, y, seed=SEED, k=5, alpha=5.0):
    """fa-fb 를 버킷화해 경험적 라벨분포로 예측했을 때 baseline 대비 logloss 감소량."""
    from sklearn.metrics import log_loss
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(d))
    cut = int(len(d) * 0.8)
    tr_i, te_i = idx[:cut], idx[cut:]
    d_tr, d_te, y_tr, y_te = d[tr_i], d[te_i], y[tr_i], y[te_i]

    # baseline: train 쪽 사전분포
    prior = np.bincount(y_tr, minlength=3) / len(y_tr)
    base = log_loss(y_te, np.tile(prior, (len(y_te), 1)), labels=[0, 1, 2])

    # 이산 피처(고유값 적음)는 분위수 대신 값 자체로 버킷
    if len(np.unique(d_tr)) <= k + 1:
        edges = np.unique(d_tr)[:-1]
    else:
        edges = compute_bucket_edges(d_tr, k)
    b_tr, b_te = bucketize(d_tr, edges), bucketize(d_te, edges)

    n_b = len(edges) + 1
    probs = np.zeros((n_b, 3))
    for b in range(n_b):
        m = b_tr == b
        cnt = np.bincount(y_tr[m], minlength=3).astype(float) if m.any() else np.zeros(3)
        probs[b] = (cnt + alpha) / (cnt.sum() + 3 * alpha)     # 라플라스 평활
    feat_ll = log_loss(y_te, probs[b_te], labels=[0, 1, 2])
    return base - feat_ll


def _write_report(res, kept, dropped, sanity_ll, n_rows, note, path=REPORT_MD):
    """실행 결과로 feature_report.md 전체를 새로 그림 (매 실행 덮어쓰기)."""
    L = []
    L.append('# 산술 피처 상관 분석 리포트')
    L.append('')
    L.append(f'> `python feature.py` 자동 생성 — {datetime.now():%Y-%m-%d %H:%M}'
             + (f'  ({note})' if note else ''))
    L.append(f'> 데이터: train.csv {n_rows:,}행 (SEED={SEED}, 10% hold-out 제외 — 파이프라인과 동일 분리)')
    L.append('')
    L.append('## 지표 정의')
    L.append('')
    L.append('| 지표 | 정의 | 읽는 법 |')
    L.append('|---|---|---|')
    L.append('| 승부 ρ | Spearman( fa−fb, y_signed ), y = +1(a승)/−1(b승)/0(tie) | +면 "이 피처가 큰 쪽이 이김" |')
    L.append('| tie r | point-biserial( −\\|fa−fb\\|, is_tie ) | +면 "둘이 비슷하면 tie" |')
    L.append('| 버킷 Δ | fa−fb 5분위 버킷별 3-클래스(a승/b승/tie) 경험 라벨분포로 예측 시 '
             'baseline(사전분포) 대비 logloss 감소 | 버킷 토큰 1개가 전달 가능한 정보량의 근사 (주 결정 지표) |')
    L.append('')
    L.append('## 전체 순위 (버킷 Δ 내림차순)')
    L.append('')
    L.append('| # | 피처 | 카테고리 | 승부 ρ | tie r | 버킷 Δ | 판정 |')
    L.append('|---|---|---|---|---|---|---|')
    for i, r in res.iterrows():
        mark = '★' if r.bucket_delta >= 0.003 else ''
        L.append(f'| {i+1} | {r.feature} | {r.category} '
                 f'| {r.spearman_win:+.3f} | {r.pb_tie:+.3f} | {r.bucket_delta:+.4f} | {mark} |')

    L.append('')
    L.append('## 산출 방법')
    L.append('')
    L.append('| 피처 | 산출 방법 |')
    L.append('|---|---|')
    for _, r in res.iterrows():                    # 순위 순서 유지
        L.append(f'| {r.feature} | {DESCRIPTIONS.get(r.feature, "")} |')

    L.append('')
    L.append('## 카테고리별 최고 버킷 Δ')
    L.append('')
    cat = res.groupby('category')['bucket_delta'].max().sort_values(ascending=False)
    L.append('| 카테고리 | 최고 Δ |')
    L.append('|---|---|')
    for c, v in cat.items():
        L.append(f'| {c} | {v:+.4f} |')

    n_tok = sum(1 if f in BINARY_FEATS else 5 for f in kept)
    L.append('')
    L.append('## 토큰화 추천 (Δ≥0.003, |ρ|>0.8 중복 제거)')
    L.append('')
    for f in kept:
        k = '이진 1토큰' if f in BINARY_FEATS else '5버킷 5토큰'
        L.append(f'- `{f}` — {k}  (Δ {res.set_index("feature").loc[f, "bucket_delta"]:+.4f})')
    if dropped:
        L.append(f'- 중복 탈락: {", ".join(dropped)}')
    L.append(f'- 총 추가 토큰(브랜치당): ~{n_tok}개')
    L.append('')
    L.append('## sanity check')
    L.append('')
    L.append(f'- 전체 피처 로지스틱 logloss = **{sanity_ll:.4f}**  '
             f'(7피처 앵커 1.0560 / baseline 1.0986)')
    L.append('')
    L.append('## 단서 (항상 유효)')
    L.append('')
    L.append('- 버킷 Δ는 피처 **단독** 정보량이지 인코더 위에 얹었을 때의 **증분**이 아니다.')
    L.append('  char_len 등은 인코더가 부분 관측 가능 → 실제 증분은 표기값보다 작음.')
    L.append('  newline_count 와 (잘린 행의) 원본 길이는 토크나이저/절단 때문에 인코더 불가시 → 증분 온전.')
    L.append('- 진짜 답은 학습 런의 hold-out 비교가 준다.')
    L.append('')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f'\n리포트 갱신 완료 (덮어쓰기): {path}')


def analyze(train_csv=DATA_CSV, verbose=True, report=True, note=''):
    from scipy.stats import spearmanr, pointbiserialr
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import log_loss

    df = pd.read_csv(train_csv)
    df['label'] = np.select(
        [df['winner_model_a'] == 1, df['winner_model_b'] == 1, df['winner_tie'] == 1],
        [0, 1, 2], default=-1)
    assert (df['label'] != -1).all()

    # ★ 파이프라인과 동일한 분리 — hold-out(10%) 쪽은 피처 선정에 사용하지 않음
    df, _holdout = train_test_split(df, test_size=0.10, stratify=df['label'],
                                    random_state=SEED, shuffle=True)
    df = df.reset_index(drop=True)
    print(f'분석 데이터: {len(df):,}행 (SEED={SEED} 10% hold-out 제외)')

    prompts = df['prompt'].map(parse_text).tolist()
    print('피처 추출 중 (response_a)...')
    fa = extract_frame(df['response_a'].map(parse_text).tolist(), prompts)
    print('피처 추출 중 (response_b)...')
    fb = extract_frame(df['response_b'].map(parse_text).tolist(), prompts)

    y = df['label'].values
    y_signed = np.select([y == 0, y == 1], [1.0, -1.0], 0.0)
    is_tie = (y == 2).astype(float)

    names = list(fa.columns)
    D = fa.values - fb.values                       # diff 형태 (분석의 기본 단위)

    # ---------- 피처별 3개 지표 ----------
    rows = []
    for j, name in enumerate(names):
        d = D[:, j]
        if np.std(d) < 1e-12:
            rows.append((name, CATEGORIES.get(name, '?'), 0.0, 0.0, 0.0)); continue
        rho, _ = spearmanr(d, y_signed)
        tie_r, _ = pointbiserialr(is_tie, -np.abs(d)) if np.std(np.abs(d)) > 1e-12 else (0.0, 1.0)
        delta = _bucket_logloss_delta(d, y)
        rows.append((name, CATEGORIES.get(name, '?'), rho, tie_r, delta))

    res = pd.DataFrame(rows, columns=['feature', 'category', 'spearman_win',
                                      'pb_tie', 'bucket_delta'])
    res = res.sort_values('bucket_delta', ascending=False).reset_index(drop=True)

    if verbose:
        print('\n' + '=' * 78)
        print(f'{"피처":<20} {"카테고리":<12} {"승부ρ":>8} {"tie r":>8} {"버킷Δ":>9}')
        print('-' * 78)
        for _, r in res.iterrows():
            mark = ' ★' if r.bucket_delta >= 0.003 else ''
            print(f'{r.feature:<20} {r.category:<12} {r.spearman_win:>+8.3f} '
                  f'{r.pb_tie:>+8.3f} {r.bucket_delta:>+9.4f}{mark}')

        print('\n----- 카테고리별 최고 버킷Δ -----')
        print(res.groupby('category')['bucket_delta'].max().sort_values(ascending=False)
              .round(4).to_string())

    # ---------- 중복 제거: |ρ|>0.8 쌍은 delta 높은 쪽만 ----------
    cand = res[res.bucket_delta >= 0.003].feature.tolist()
    if cand:
        sub = pd.DataFrame(D, columns=names)[cand]
        corr = sub.corr(method='spearman').abs()
        kept = []
        for f in cand:                              # delta 내림차순으로 순회
            if all(corr.loc[f, k] <= 0.8 for k in kept):
                kept.append(f)
    else:
        kept = []
    dropped = [f for f in cand if f not in kept]

    if verbose:
        print(f'\n----- 토큰화 추천 (버킷Δ≥0.003, |ρ|>0.8 중복 제거) -----')
        print(f'  후보 {len(cand)}개 -> 유지 {len(kept)}개  (중복 탈락: {dropped})')
        n_tok = sum(1 if f in BINARY_FEATS else 5 for f in kept)
        for f in kept:
            k = 1 if f in BINARY_FEATS else 5
            print(f'  {f:<20} {"이진 1토큰" if k == 1 else "5버킷 5토큰"}')
        print(f'  총 추가 토큰(브랜치당): ~{n_tok}개')

    # ---------- sanity: 전체 피처 다항 로지스틱 (기존 앵커 ~1.056) ----------
    X = np.concatenate([fa.values, fb.values, D], axis=1)
    X = np.nan_to_num(X, posinf=0, neginf=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=SEED)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    pipe.fit(Xtr, ytr)
    ll = log_loss(yte, pipe.predict_proba(Xte), labels=[0, 1, 2])
    if verbose:
        print(f'\n----- sanity check -----')
        print(f'  전체 피처 로지스틱 logloss: {ll:.4f}   '
              f'(기존 7피처 앵커 1.0560 / baseline 1.0986)')

    if report:
        _write_report(res, kept, dropped, ll, len(df), note)

    return res, kept


if __name__ == '__main__':
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    # 사용법: python feature.py ["실행 메모"]
    analyze(note=sys.argv[1] if len(sys.argv) > 1 else '')
