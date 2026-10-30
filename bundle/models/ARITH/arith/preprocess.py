"""arith.preprocess — 전처리 모듈. 텍스트 프레임 -> 피쳐 행렬.
Preprocessing module: text frame -> feature matrix.

피쳐 정의를 이 모듈에 **인라인**했다 — 노트북 exec 의존이 없다.
이 모듈은 ab224 계약 224종**만** 정의한다. 예전에는 259종을 뽑아 계약 열로 잘랐지만,
쓰이지 않던 35종은 정의째 삭제했다 — 다른 피쳐 집합을 뽑는 경로도, 스위치도 없다.
Only the 224 ab224 contract features are defined here. The module used to extract 259 and
slice; the 35 that no column used are gone from the source. There is no other feature set
and no switch to ask for one.

지운 것 / removed
    quote_count · is_empty          4형 8열 + q_ 2열이 전부 미사용 -> _v3_features 에서 삭제
    ab_emb_sim                      emb_cos 와 같은 배열 (카탈로그 사용=0)
    diff_emb_sim · abs_diff_emb_sim _STAT_DIFF_BASES 에서 emb_sim 제거
    4형 15열 · q_ 7열               SYM_FORMS / Q_FEATURES 표에 남은 조합만 만든다
후보로 되살리려면 피쳐 선택 단계에서 계약을 새로 뽑아야 한다 (여기서 못 되돌린다).
Putting any of them back means re-drawing the contract upstream, not editing this file.

상태를 가진 것은 TF-IDF/SVD 둘뿐이다 — 배포는 체크포인트 아티팩트로 **transform 만**,
학습(train 모듈)은 train 행으로 **fit** 한다. 같은 코드 경로에서 fit_idx 유무로 갈린다.
이 SVD 는 TF-IDF 차분 행렬(30,000차원 희소)의 8성분으로 tfidf_svd_0..7 을 만드는 **피쳐
추출기**다 — 통합 헤드 Layer1 의 풀링 SVD 와는 이름만 같고 아무 관계가 없다.
Only TF-IDF/SVD carry state: transform-only at deploy time, fit on train rows during training.
That SVD is a feature extractor — 8 components of the TF-IDF difference matrix, producing
tfidf_svd_0..7. It shares nothing but its name with the pooled-vector SVD in the head's Layer1.

공개 API / public API
    preprocess(df, cols)  -> (F 계약열 DataFrame, X 계약열 행렬 float32)
    extract(df, ckpt=..., emb=...) -> 224 계약열 DataFrame
        fit_idx 를 주면 (F, tfidf, svd) — TF-IDF/SVD 를 그 행으로 fit 한다 (재학습)
"""
import ast
import json
import math
import os
import re
import time
import weakref
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ._base import EMB_HF_ID, EMB_REVISION, SEED, _tq

# 이 모듈이 만드는 열 수 — 아래 네 갈래의 합이고 ab224 계약과 같아야 한다.
#   emb 5 + BERTScore 12 + stat 41 + unit/pair 166 (4형 133 + 쌍 8 + q_ 25)
# What this module builds, and it must equal the ab224 contract.
N_CONTRACT = 224

# ══════════════════════════════════════════════ [정의 1] 파싱 + 응답 단위 기저 37종 + 쌍 8종 + 4형 전개
# (계약 밖 기저는 삭제) / (bases outside the contract removed)
def parse_text(x):
    # 원본 셀(문자열화된 리스트 또는 생 문자열) → 평문. json → ast 폴백, None 항 스킵.
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ''
    t = str(x).strip()
    if t.startswith('[') and t.endswith(']'):
        for loader in (json.loads, ast.literal_eval):
            try:
                v = loader(t)
            except Exception:
                continue
            if isinstance(v, list):
                return ' '.join(str(u) for u in v if u is not None)
    return t


def parse_columns(d):
    return ([parse_text(x) for x in d['prompt']],
            [parse_text(x) for x in d['response_a']],
            [parse_text(x) for x in d['response_b']])


_RE_SENT      = re.compile(r'[.!?]+')
_RE_PARA      = re.compile(r'\n\s*\n')
_RE_NUMBERING = re.compile(r'^\s*\d+[.)]\s')
_RE_NUMBER    = re.compile(r'\d+(?:\.\d+)?')
_RE_EXAMPLE   = re.compile(r'for example|for instance|e\.g\.|example:', re.I)
_RE_FOOTNOTE  = re.compile(r'\[\d+\]')
_RE_MDLINK    = re.compile(r'\[[^\]]+\]\([^)]+\)')
_RE_FORMAL    = re.compile(r'\b(?:furthermore|moreover|consequently|therefore|thus|'
                           r'additionally|nevertheless|hence)\b', re.I)
_RE_UNCERTAIN = re.compile(r'\b(?:might|may|perhaps|possibly|likely|probably|seems|appears)\b'
                           r'|i think|not sure', re.I)
_RE_POLITE    = re.compile(r'please|thank|apolog|appreciate|glad to|happy to', re.I)
_RE_PASSIVE   = re.compile(r'\b(?:is|are|was|were|be|been|being)\s+'
                           r'(?:\w+ed|made|done|given|taken|seen|known|found|used|shown|'
                           r'written|built|held|kept|sent|told|thought|understood)\b', re.I)
_RE_REFUSAL   = re.compile(r"as an ai|i can'?t|i cannot|i'?m sorry|i apologize|"
                           r"unable to|sorry,? but", re.I)
_RE_YOU       = re.compile(r'\byou\b', re.I)
_RE_WORD      = re.compile(r'\w+')
_RE_POSITIVE  = re.compile(
    r'\b(?:good|great|excellent|best|better|effective|efficient|helpful|useful|valuable|'
    r'important|easy|simple|powerful|amazing|awesome|wonderful|fantastic|perfect|ideal|'
    r'beneficial|brilliant|impressive|outstanding|remarkable|superior|optimal|convenient|'
    r'flexible|reliable|robust|accurate|correct|clear|safe|secure|fast|quick|smooth|'
    r'love|enjoy|recommended?|advantages?|benefits?|successful?|'
    r'improve(?:d|s|ment)?|enhance[ds]?)\b', re.I)
_RE_NEGATIVE  = re.compile(
    r'\b(?:bad|worse|worst|wrong|errors?|mistakes?|fail(?:s|ed|ure|ures)?|problems?|'
    r'problematic|difficult(?:y|ies)?|hard|harder|impossible|unfortunately|issues?|'
    r'risk(?:s|y)?|danger(?:ous)?|harm(?:ful)?|poor(?:ly)?|lack(?:s|ing)?|incorrect|'
    r'inaccurate|negative|weak|slow|broken|bugs?|confusing|confused|unclear|complicated|'
    r'misleading|flaw(?:s|ed)?|terrible|horrible|awful|useless|invalid|unsafe|insecure|'
    r'unstable|inefficient|disadvantages?|drawbacks?|concern(?:s|ing)?|severe|fatal|'
    r'damage[ds]?|threats?|vulnerable|vulnerabilit(?:y|ies))\b', re.I)
_RE_NEGATION  = re.compile(
    r"\b(?:not|no|never|none|nothing|nobody|nowhere|neither|nor|cannot|without|"
    r"hardly|barely|scarcely|unable)\b|n't", re.I)
_RE_EMOJI     = re.compile('[\U0001F000-\U0001FAFF☀-➿️]')


def _v3_features(t):
    n_char = len(t)
    words = t.split()
    n_w = max(len(words), 1)
    lines = t.split('\n')
    n_sent = len(_RE_SENT.findall(t))
    lower_words = [w.lower() for w in words]
    if len(lower_words) >= 3:
        tri = list(zip(lower_words, lower_words[1:], lower_words[2:]))
        uniq3 = len(set(tri)) / len(tri)
        top3 = Counter(tri).most_common(1)[0][1]      # 최빈 3단어 연쇄의 등장 수
        top_share = top3 / len(tri)
    else:
        uniq3, top3, top_share = 1.0, 0, 0.0
    per100 = 100.0 / n_w
    return {
        'char_len': np.log1p(n_char), 'word_count': np.log1p(len(words)),
        'sentence_count': np.log1p(n_sent),
        'paragraph_count': np.log1p(sum(1 for b in _RE_PARA.split(t) if b.strip())),
        'bullet_count': np.log1p(sum(1 for l in lines if l.lstrip()[:2] in ('- ', '* ', '• '))),
        'numbering_count': np.log1p(sum(1 for l in lines if _RE_NUMBERING.match(l))),
        'heading_count': np.log1p(sum(1 for l in lines if l.lstrip().startswith('#'))),
        'has_table': 1.0 if any('|' in l and '---' in l for l in lines) else 0.0,
        'code_block_count': np.log1p(t.count('```') // 2),
        'avg_sentence_len': len(words) / max(n_sent, 1),
        'avg_word_len': sum(len(w) for w in words) / n_w,
        'type_token_ratio': len(set(lower_words)) / n_w,
        'unique_3gram_ratio': uniq3,
        'rep_top_share': top_share, 'rep_max_ngram': np.log1p(top3),
        'number_count': np.log1p(len(_RE_NUMBER.findall(t))),
        'url_count': np.log1p(t.count('http')),
        'example_count': np.log1p(len(_RE_EXAMPLE.findall(t))),
        'citation_count': np.log1p(len(_RE_FOOTNOTE.findall(t)) + len(_RE_MDLINK.findall(t))
                                   + t.lower().count('according to') + t.lower().count('source:')),
        'formal_words': len(_RE_FORMAL.findall(t)) * per100,
        'uncertainty_words': len(_RE_UNCERTAIN.findall(t)) * per100,
        'politeness': len(_RE_POLITE.findall(t)) * per100,
        'passive_ratio': len(_RE_PASSIVE.findall(t)) * per100,
        'positive_words': len(_RE_POSITIVE.findall(t)) * per100,
        'negative_words': len(_RE_NEGATIVE.findall(t)) * per100,
        'negation_rate': len(_RE_NEGATION.findall(t)) * per100,
        'refusal': 1.0 if _RE_REFUSAL.search(t) else 0.0,
        'you_rate': len(_RE_YOU.findall(t)) * per100,
        'user_mention': np.log1p(t.lower().count('the user')),
        'newline_count': np.log1p(t.count('\n')),
        'question_marks': np.log1p(t.count('?')),
        'non_ascii_ratio': sum(1 for c in t if ord(c) > 127) / max(n_char, 1),
    }


def prompt_echo(p, t):
    pw = set(w.lower() for w in _RE_WORD.findall(p))
    rw = set(w.lower() for w in _RE_WORD.findall(t))
    return len(pw & rw) / len(pw | rw) if pw and rw else 0.0


def _extra_features(t, p):
    pw = set(w.lower() for w in _RE_WORD.findall(p) if len(w) > 2)
    rw = set(w.lower() for w in _RE_WORD.findall(t))
    return {'kw_coverage': (len(pw & rw) / len(pw)) if pw else 0.0,
            'len_vs_prompt': np.log1p(len(t) / (len(p) + 1.0)),
            'bold_count': np.log1p(t.count('**') // 2),
            'emoji_count': np.log1p(len(_RE_EMOJI.findall(t)))}


def extract_unit_frame(texts, prompts=None, desc='unit'):
    # prompts 있으면 37종(V3 32 + prompt_echo + 추가 4), 없으면 32종 (프롬프트 자체용)
    if prompts is None:
        return pd.DataFrame([_v3_features(t) for t in _tq(texts, desc)])
    rows = []
    for t, p in _tq(list(zip(texts, prompts)), desc):
        r = _v3_features(t)
        r['prompt_echo'] = prompt_echo(p, t)
        r.update(_extra_features(t, p))
        rows.append(r)
    return pd.DataFrame(rows)


def _word_set(t):
    return set(w.lower() for w in _RE_WORD.findall(t))


def _word_3grams(t):
    ws = [w.lower() for w in _RE_WORD.findall(t)]
    return set(zip(ws, ws[1:], ws[2:])) if len(ws) >= 3 else set()


def _jaccard(sa, sb):
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def _tf_cos(ca, cb):
    if not ca or not cb:
        return 0.0
    num = sum(ca[k] * cb[k] for k in set(ca) & set(cb))
    den = (sum(v * v for v in ca.values()) ** .5) * (sum(v * v for v in cb.values()) ** .5)
    return num / den if den else 0.0


def _char3(t, cap=4000):
    t = t[:cap].lower()
    return Counter(t[i:i + 3] for i in range(len(t) - 2)) if len(t) >= 3 else Counter()


def _suffix_ratio(a, b):
    l = min(len(a), len(b))
    if l == 0:
        return 0.0
    k = 0
    while k < l and a[-1 - k] == b[-1 - k]:
        k += 1
    return k / l


def extract_pair_features(a, b):
    la, lb = len(a), len(b)
    lim = min(la, lb)
    k = 0
    while k < lim and a[k] == b[k]:
        k += 1
    sa, sb = set(_RE_NUMBER.findall(a)), set(_RE_NUMBER.findall(b))
    return {'pr_jac_word': _jaccard(_word_set(a), _word_set(b)),
            'pr_jac_3gram': _jaccard(_word_3grams(a), _word_3grams(b)),
            'pr_len_ratio': (min(la, lb) / max(la, lb)) if max(la, lb) else 1.0,
            'pr_prefix_sim': (k / lim) if lim else 0.0,
            'cos_word': _tf_cos(Counter(w.lower() for w in _RE_WORD.findall(a)),
                                Counter(w.lower() for w in _RE_WORD.findall(b))),
            'cos_char3': _tf_cos(_char3(a), _char3(b)),
            'suffix_sim': _suffix_ratio(a, b),
            'num_jac': len(sa & sb) / len(sa | sb) if sa | sb else 0.0}


# ── 4형 전개표 / four-form table ──────────────────────────────────────────────────────
# A·B 한 쌍의 기저값 (va, vb) 를 mn=min · mx=max · ad=|va-vb| · d=va-vb 로 편다. 원래는
# 기저마다 4형을 다 폈지만(39x4=156), ab224 가 쓰는 133열만 남긴다 — 표에 없는 조합은
# 이름도 배열도 생기지 않는다. user_mention 은 4형이 하나도 안 남았지만 q_user_mention 이
# 계약에 있어 기저 정의 자체는 살아 있다.
# The pair (va, vb) expands to mn/mx/ad/d. Only the 133 combinations ab224 uses are listed;
# anything absent is never named or allocated. user_mention keeps no form here but survives
# in _v3_features because q_user_mention is in the contract.
SYM_FORMS = {
    'char_len':           'mn mx ad d',
    'word_count':         'mn mx ad d',
    'sentence_count':     'mn mx ad d',
    'paragraph_count':    'mn mx ad d',
    'bullet_count':       'mn mx ad d',
    'numbering_count':    'mn mx ad d',
    'heading_count':      'mn mx ad d',
    'has_table':          'd',
    'code_block_count':   'mn mx ad d',
    'avg_sentence_len':   'mn mx ad d',
    'avg_word_len':       'mn mx ad d',
    'type_token_ratio':   'mn mx ad d',
    'unique_3gram_ratio': 'mn mx ad d',
    'rep_top_share':      'mn mx ad d',
    'rep_max_ngram':      'mn mx ad d',
    'number_count':       'mn mx ad d',
    'url_count':          'ad d',
    'example_count':      'mx ad d',
    'citation_count':     'd',
    'formal_words':       'mn mx ad d',
    'uncertainty_words':  'mn mx ad d',
    'politeness':         'mn mx ad d',
    'passive_ratio':      'mn mx ad d',
    'positive_words':     'mn mx ad d',
    'negative_words':     'mn mx ad d',
    'negation_rate':      'mn mx ad d',
    'refusal':            'mn mx ad d',
    'you_rate':           'mn mx ad d',
    'newline_count':      'mn mx ad d',
    'question_marks':     'mn mx ad d',
    'non_ascii_ratio':    'mn mx ad d',
    'prompt_echo':        'mn mx ad d',
    'kw_coverage':        'mn mx ad d',
    'len_vs_prompt':      'mn mx ad d',
    'bold_count':         'mx ad d',
    'emoji_count':        'mx ad d',
}
_FORM = {'mn': np.minimum, 'mx': np.maximum,
         'ad': lambda x, y: np.abs(x - y), 'd': lambda x, y: x - y}

# 프롬프트 자체의 통계 중 계약에 남은 25종 (q_ 접두를 붙이기 전 이름).
# The 25 prompt-side statistics the contract keeps, named before the q_ prefix.
Q_FEATURES = ('char_len', 'word_count', 'sentence_count', 'paragraph_count', 'numbering_count',
              'avg_sentence_len', 'avg_word_len', 'type_token_ratio', 'unique_3gram_ratio',
              'rep_top_share', 'rep_max_ngram', 'number_count', 'citation_count', 'formal_words',
              'uncertainty_words', 'politeness', 'passive_ratio', 'positive_words',
              'negative_words', 'negation_rate', 'you_rate', 'user_mention', 'newline_count',
              'question_marks', 'non_ascii_ratio')


def build_unit_pair_frame(P, A, B):
    # → 166컬럼 = 4형 133 (SYM_FORMS) + 쌍 8 + 프롬프트 q_ 25 (Q_FEATURES)
    fa = extract_unit_frame(A, P, desc='unit A')
    fb = extract_unit_frame(B, P, desc='unit B')
    qf = extract_unit_frame(P, desc='unit Q')[list(Q_FEATURES)].add_prefix('q_')
    out = {}
    for n, forms in SYM_FORMS.items():
        va, vb = fa[n].to_numpy(np.float32), fb[n].to_numpy(np.float32)
        for p in forms.split():
            out[f'{p}_{n}'] = _FORM[p](va, vb)
    sym = pd.DataFrame(out)
    pair = pd.DataFrame([extract_pair_features(a, b) for a, b in _tq(list(zip(A, B)), 'pair 8')])
    dfx = pd.concat([sym, pair, qf], axis=1)
    return pd.DataFrame(np.nan_to_num(dfx.to_numpy(np.float32), posinf=0, neginf=0),
                        columns=dfx.columns)


# ══════════════════════════════════════════════ [정의 2] MiniLM 임베딩 + BERTScore

# One budget for every feature BERT touches: 512 tokens, fp32.
# BERT 가 닿는 모든 피쳐가 같은 예산을 쓴다 — 512토큰·fp32.
#
#   512 tokens — MiniLM's position-embedding limit, so it is the most the model can see.
#     emb_* used to stop at 256, which cut 53% of the responses; at 512 only 21% are cut.
#     512토큰 — MiniLM 위치 임베딩 한계라 모델이 볼 수 있는 최대치다. emb_* 는 256에서
#     끊어 응답의 53%를 잘랐지만, 512 에서는 21%만 잘린다.
#
#   fp32 — NOT a speed choice, a fidelity one. RDNA4 has no matrix core for fp32 so the
#     forward runs ~14x slower than fp16, but the xgb folds were fit on fp32-extracted
#     features and the 17 MiniLM-derived columns all saturate max_bin=256: their split
#     thresholds sit ~1.8e-3 apart in relative terms, only ~3.7x above fp16's 4.9e-4 error.
#     Extraction is cached to npz and runs once per universe, so the time is worth paying.
#     fp32 — 속도가 아니라 충실도 선택이다. RDNA4 는 fp32 행렬 코어가 없어 fp16 대비
#     14배 느리지만, xgb 폴드가 fp32 피쳐 위에서 학습됐고 MiniLM 유래 17열은 전부
#     max_bin=256 을 포화시켜 임계 간격이 상대 1.8e-3 — fp16 오차 4.9e-4 의 3.7배뿐이다.
#     추출은 npz 로 캐시돼 universe 당 한 번만 도니 시간을 치르는 편이 낫다.
MAX_SEQ_LEN = 512

# There is no character guard. emb_* used to pre-cut at 2,000 chars, but no character
# budget is safe: pathological texts tokenize to as few as 3 tokens per 4,096 characters,
# so any guard can bind before the 512-token cap and silently shorten the input. Tokenizing
# in full costs ~7 s over 46,729 rows against an 18-minute fp32 forward, and the BERTScore
# branch never had a guard, so dropping it also gives both branches identical input.
# 문자 상한은 두지 않는다. emb_* 는 2,000자에서 미리 잘랐지만 안전한 문자 예산이란 게
# 없다 — 4,096자에서 3토큰만 나오는 병리 텍스트가 있어, 어떤 상한이든 512토큰보다 먼저
# 걸려 입력을 조용히 줄일 수 있다. 전문 토크나이즈 비용은 46,729행에 ~7초로, 18분짜리
# fp32 순전파 앞에서 무시할 수준이고, BERTScore 갈래는 원래 상한이 없었으니
# 이로써 두 갈래가 같은 입력을 받게 된다.

# Head 60% + tail 40% is the campaign-wide truncation rule: written.ipynb applies it to the
# four encoders at both the character and the token level, and BERTScore has always used it.
# emb_* was the one exception, and only because tok(truncation=True) defaults to head-only —
# never a choice, just an untouched default, so it now follows the same rule as everything else.
# 앞60%+뒤40%는 캠페인 전역의 절단 규약이다 — written.ipynb 가 인코더 4종에 문자·토큰
# 두 층으로 적용하고, BERTScore 도 처음부터 이걸 썼다. emb_* 만 예외였는데 이유는
# tok(truncation=True) 가 앞절단이 기본이어서였다 — 선택한 적 없는 기본값이라, 이제
# 나머지와 같은 규약을 따른다.
HEAD_FRAC = 0.6


def head_tail(ids, k):
    """k 토큰만 남기되 앞 60% + 뒤 40% (V5 Train_ABE.ipynb 와 동일 구현).
    Keep k tokens as head 60% + tail 40% — the same implementation as V5 Train_ABE.ipynb."""
    ids = list(ids)
    if len(ids) <= k:
        return ids
    h = int(k * HEAD_FRAC)
    t = k - h
    return ids[:h] + ids[-t:] if t > 0 else ids[:h]


def pack_head_tail(tok, texts, device, max_len=MAX_SEQ_LEN):
    """Tokenize in full, keep head 60% + tail 40% inside the budget, then pad by hand.
    Both BERT paths call this, so the truncation rule lives in exactly one place.
    전문을 토크나이즈해 예산 안에서 앞60%+뒤40%만 남기고 직접 패딩한다.
    두 BERT 경로가 모두 이걸 부르므로 절단 규약이 한 곳에만 있다.

    An empty text becomes [CLS] [SEP]: the mean-pool branch averages those two, and the
    token-align branch masks them out and falls through to its zero guard.
    빈 텍스트는 [CLS] [SEP] 가 된다 — 평균 풀링 갈래는 둘을 평균하고,
    토큰 정렬 갈래는 마스크에서 빼 0 가드로 떨어진다."""
    import torch
    CLS, SEP, PAD = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
    raw = tok(list(texts), add_special_tokens=False, verbose=False)['input_ids']
    rows = [[CLS] + head_tail(r, max_len - 2) + [SEP] for r in raw]     # [CLS], [SEP] 자리
    L = max(len(r) for r in rows)
    ids = torch.full((len(rows), L), PAD, dtype=torch.long)
    att = torch.zeros((len(rows), L), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        att[i, :len(r)] = 1
    return ids.to(device), att.to(device)


def load_embedder(emb_dir, device=None):
    import torch
    from transformers import AutoTokenizer, AutoModel
    emb_dir = str(emb_dir)
    src = emb_dir if os.path.isdir(emb_dir) else EMB_HF_ID
    hub = {} if src == emb_dir else {'revision': EMB_REVISION}       # the hub copy is read at the pinned commit
    tok = AutoTokenizer.from_pretrained(src, **hub)
    # eager attention, not sdpa — but NOT because it avoids the NaN fault. It does not.
    # eager was chosen on 2026-08-21 when only the sdpa (AOTriton) path had been seen to emit
    # non-deterministic NaN embeddings. A 2026-08-24 campaign showed eager fails at the same
    # rate (~1 batch row per 110 forwards) and ruled out input, precision, attention backend,
    # H2D transfer, tensor shape and mid-forward serialisation; the fault sits below the
    # attention implementation and is suppressed only by reading every submodule's output
    # (see _arm_suppression). eager is kept because it is the path the ab224 checkpoint was
    # extracted and validated on; sdpa costs max|Δ| 1.4e-6 (fp32 noise) and is ~1.14x faster,
    # so switching is a separate decision, not a fix.
    # sdpa 가 아니라 eager 를 쓰지만, NaN 을 피해서가 아니다 — eager 도 똑같이 터진다.
    # 2026-08-21 에는 sdpa 만 터지는 줄 알고 골랐고, 2026-08-24 조사에서 eager 도 같은
    # 비율(순전파 110회당 1행)로 터지며 입력·정밀도·어텐션 구현·H2D·텐서 형상·중간 직렬화가
    # 전부 기각됐다. 결함은 어텐션 구현보다 아래층이고, 서브모듈 출력을 전부 읽어야만
    # 억제된다(_arm_suppression). eager 를 유지하는 이유는 ab224 체크포인트가 이 경로에서
    # 추출·검증됐기 때문이다 — sdpa 는 차이 1.4e-6(fp32 잡음)에 1.14배 빠르지만 별도 결정 사항.
    mod = AutoModel.from_pretrained(src, attn_implementation='eager', **hub)
    if src != emb_dir:
        tok.save_pretrained(emb_dir)
        mod.save_pretrained(emb_dir)
        print(f'saved to {emb_dir}/ for offline use (bundled in the Kaggle Dataset)')
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    mod = mod.to(device).eval()
    # ROCm 기기는 적재 즉시 억제 훅을 무장한다 — 결함이 관측된 스택에서 첫 사건을 기다릴
    # 이유가 없다. CUDA(캐글 T4)·CPU 는 무장하지 않아 비용이 정확히 0 이다.
    # On a ROCm device the suppression hook is armed at load — no reason to wait for the first
    # fault on the very stack where it was observed. CUDA (Kaggle T4) and CPU stay unarmed.
    if device != 'cpu' and getattr(torch.version, 'hip', None):
        _arm_suppression(mod)
        print('  embedder suppression armed at load (ROCm device)', flush=True)
    return tok, mod, device


# batch 64 — bertscore 와 같은 (64, 512) 형상. 예산을 256->512 로 올릴 때 배치 256 을 그대로 둬
# (256, 512) 라는 전례 없는 형상이 생겼었다 — 전 구간에서 완주 실적이 있는 형상 하나로 통일한다.
# batch 64 matches bertscore's proven (64, 512) shape; 256x512 had never run anywhere before.
# Reference check — every REF_EVERY-th batch is recomputed on the CPU (fp32, deterministic, no GPU
# kernel involved) and the hidden states of the valid tokens must agree to REF_TOL. This is the
# only check that catches a GPU kernel returning *finite* garbage; the per-batch finiteness guard
# only catches NaN/inf. Measured GPU-eager vs CPU: 8.6e-6 (fp32 summation order); a wrong kernel
# is off by 1e-1. Cost ≈ 4 s per check on 14 CPU threads, several times that on Kaggle's 4 vCPUs.
# 100 keeps it at ~1 % of batches: the local investigation ran at 10 to net finite corruption
# from the ROCm embedder fault, but that fault was never seen on CUDA and the hook is not armed
# there, so the deploy bundle pays only the 1 % sampling.
# 기준 대조 — REF_EVERY 배치마다 하나를 CPU(fp32·결정적·GPU 커널 무관)로 다시 계산해 유효 토큰의
# 은닉상태가 REF_TOL 안에서 일치해야 한다. GPU 커널이 **유한한 쓰레기값**을 낼 때 잡는 유일한
# 검사다 (유한성 가드는 NaN/inf 만 잡는다). 실측 8.6e-6, 오류 커널은 1e-1 급.
# 로컬 조사 때는 ROCm 결함의 유한 오염을 잡으려 10 으로 내렸지만, 그 결함은 CUDA 에서 관측된 적이
# 없고 훅도 무장되지 않으므로 배포 번들은 1 % 표본(100)만 치른다. 캐글 4 vCPU 에선 검사 1회가
# 로컬의 몇 배라 10 이면 제출 런에 수십 분이 얹힌다.
REF_EVERY = 100
REF_TOL = 1e-4

# 모델 객체를 키로 쓰는 약참조 — id() 는 모델이 해제되면 재사용돼, 나중에 만든 다른 모델이
# 앞선 모델의 CPU 사본을 물려받을 수 있다. 약참조 사전은 모델이 사라질 때 항목도 사라진다.
# Weak keys: id() is recycled once a model is freed, which would hand a later model the earlier
# one's CPU reference copy. A WeakKeyDictionary drops the entry together with the model.
_REF_MODEL = weakref.WeakKeyDictionary()


def _ref_check(mod, ids, att, h_gpu, tag):
    import copy
    import torch
    if mod not in _REF_MODEL:
        _REF_MODEL[mod] = copy.deepcopy(mod).cpu().eval()
    with torch.no_grad():
        h_cpu = _REF_MODEL[mod](input_ids=ids.cpu(), attention_mask=att.cpu()).last_hidden_state
    valid = att.bool().cpu()
    d = float((h_gpu.detach().cpu() - h_cpu)[valid].abs().max())
    if not d < REF_TOL:
        raise AssertionError(f'{tag}: GPU forward disagrees with the CPU reference — '
                             f'max|Δ| {d:.2e} > {REF_TOL:g} (L={int(ids.shape[1])}, batch {int(ids.shape[0])})')
    return d


# ══════════════════════════════════════════════ 임베더 결함 대응 / embedder fault handling
# 2026-08-24 조사 결과 — gfx1201 + ROCm 7.2 에서 임베더 순전파가 비결정적으로 배치의 한 행을
# 통째로 NaN 으로 만든다 (순전파 110회당 1건). 기각된 것: 입력 의존(실패한 그 배치를 다시
# 돌리면 100/100 정상)·정밀도(fp32·fp16 동일)·어텐션 구현(eager·sdpa 동일)·H2D 복사(되읽기
# 대조 전 배치 무결)·텐서 형상(고정 패딩도 실패)·순전파 중간 직렬화(서브모듈마다 synchronize
# 해도 실패). 유일하게 듣는 것은 서브모듈 출력을 **전부 읽는** 것이다 (무장 5,056 순전파 0건).
#
# 대응은 두 겹이다.
#   ① ROCm 기기는 load_embedder 가 적재 즉시 훅을 무장한다 — 결함이 관측된 스택이라
#      첫 사건을 기다리지 않는다 (실측: 무장 후 ~2,500 순전파 0건, 무장 없이면 기대 ~23건)
#   ② 그래도 비유한 값이 나오면 그 배치만 다시 계산한다 — 일과성이라 재계산은 성공하고,
#      혹시 무장이 안 된 경로라면 이때 무장한다
# CUDA(캐글 T4)·CPU 에서는 둘 다 한 번도 실행되지 않아 비용이 정확히 0 이다.
#
# The fault: on gfx1201 + ROCm 7.2 the embedder forward non-deterministically turns one whole row
# of a batch into NaN, about once per 110 forwards. Input dependence, precision, attention
# backend, H2D transfer, tensor shape and mid-forward serialisation were all ruled out; only
# reading every submodule's output suppresses it. The response is two layers: load_embedder arms
# the hook at load on ROCm devices (no waiting for a first fault on the stack where it was
# observed), and any non-finite batch is still recomputed — the transient always clears, and the
# retry arms the hook on any path that somehow was not armed. CUDA (Kaggle T4) and CPU never
# enter either path, so the cost there is exactly zero.
FAULT_RETRIES = 3
FAULTS = {'n': 0}
_SUPPRESS = weakref.WeakKeyDictionary()


class _Fault(Exception):
    """A batch that produced non-finite values — recompute it.
    비유한 값이 나온 배치 — 재계산 대상이다."""


def _arm_suppression(mod):
    """서브모듈마다 출력을 전부 읽는 훅을 단다. 값은 바꾸지 않는 순수 관찰자다.
    Read every submodule's output — a pure observer that changes no value.

    무엇이 듣는지는 400 순전파 예산으로 갈랐다: 같은 크기 텐서 할당만 하면 #44, 1원소 읽기는
    #78, 서브모듈마다 synchronize 는 #126 에서 각각 터졌고, 전 원소 읽기만 무사통과했다.
    Which part matters was separated on a 400-forward budget: allocating a same-sized tensor
    failed at #44, a one-element read at #78, a per-submodule synchronise at #126; only the
    full read survived the budget.
    """
    import torch
    if mod in _SUPPRESS:
        return False

    def hook(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(t) and t.is_floating_point():
            float(t.sum())

    _SUPPRESS[mod] = [m.register_forward_hook(hook) for n, m in mod.named_modules() if n]
    return True


def _on_fault(mod, err, attempt):
    """비유한 배치 하나를 처리한다 — 예산을 다 썼으면 올리고, 아니면 무장 후 재계산한다.
    Handle one non-finite batch: raise when the retry budget is spent, otherwise arm and retry."""
    FAULTS['n'] += 1
    if attempt >= FAULT_RETRIES:
        raise AssertionError(f'{err} — still non-finite after {FAULT_RETRIES} recomputes')
    armed = _arm_suppression(mod)
    print(f'  !! embedder fault #{FAULTS["n"]} — {err}', flush=True)
    print(f'     recomputing this batch (attempt {attempt + 1}/{FAULT_RETRIES})'
          + (' · suppression armed on every submodule' if armed else ''), flush=True)


def emb_frames(eq, ea, eb):
    # ab_emb_sim 은 emb_cos 와 같은 배열이라 카탈로그에서 사용=0 으로 떨어졌다 — 이제 아예
    # 만들지 않는다. 두 번째 반환값은 a/b 두 종뿐이고 build_stat_features 가 내부에서 쓴다.
    # ab_emb_sim was the exact duplicate of emb_cos and scored 0 in the catalog, so it is no
    # longer produced. The second return carries only a/b, consumed by build_stat_features.
    sqa = (eq * ea).sum(1).astype(np.float32)
    sqb = (eq * eb).sum(1).astype(np.float32)
    sab = (ea * eb).sum(1).astype(np.float32)
    emb5 = pd.DataFrame({'emb_cos': sab, 'emb_q_min': np.minimum(sqa, sqb),
                         'emb_q_max': np.maximum(sqa, sqb), 'emb_q_ad': np.abs(sqa - sqb),
                         'emb_q_d': sqa - sqb}).astype(np.float32)
    return emb5, {'a_emb_sim': sqa, 'b_emb_sim': sqb}


def encode_and_align(tok, mod, prompts, texts_a, texts_b, device, batch_size=64,
                     max_len=MAX_SEQ_LEN, desc='embed+bertscore'):
    """Q·A·B 를 배치마다 한 번씩만 인코딩해, 같은 은닉상태에서 두 가지를 모두 뽑는다.

      평균 풀링 -> eq / ea / eb          emb_* 5종의 재료 (문장 하나를 벡터 하나로)
      토큰 정렬 -> bs_* 12종 (BERTScore)  A↔B · Q↔A · Q↔B 세 방향

    두 갈래는 pack_head_tail 의 같은 512토큰 예산·같은 절단을 공유해 은닉상태가 같은
    값이므로, 한 번의 순전파가 둘을 모두 만든다. 차이는 풀링뿐이다.
    Encode Q, A and B once per batch and derive both products from the same hidden states:
    the mean-pooled sentence vectors that feed emb_*, and the token-level alignments that feed
    bs_*. Both branches share pack_head_tail's 512-token budget and truncation, so one forward
    serves both; only the pooling differs.
    """
    import torch
    NEG = -1e4                      # -inf 를 쓰면 뒤의 0 곱에서 nan 이 난다
    CLS, SEP, PAD = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id

    def _encode(texts, tag, bi, i):
        ids, att = pack_head_tail(tok, texts, device, max_len)
        with torch.no_grad():
            h = mod(input_ids=ids, attention_mask=att).last_hidden_state
        assert h.dtype == torch.float32, 'the embedder must run in fp32 — see MAX_SEQ_LEN note'
        # [CLS]/[SEP] stay in the mean — they carry the sentence signal here, and they are what
        # an empty text pools down to. / [CLS]/[SEP] 도 평균에 넣는다 — 문장 신호를 담고,
        # 빈 텍스트가 풀링되는 대상도 이 둘이다.
        mp = att.unsqueeze(-1).to(h.dtype)
        v = (h * mp).sum(1) / mp.sum(1).clamp(min=1e-9)
        v = torch.nn.functional.normalize(v, dim=1)
        v = v.float().cpu().numpy()
        # 유한성 가드가 기준 대조보다 앞에 온다 — 버릴 배치에 CPU 순전파 4초를 쓰지 않는다.
        # The finiteness guard runs before the reference check: no 4 s CPU forward on a batch
        # that is about to be recomputed anyway.
        if not np.isfinite(v).all():
            bad = np.where(~np.isfinite(v).all(1))[0]
            raise _Fault(
                f'{desc} {tag}: non-finite embedding — batch rows {i}..{i + len(v) - 1} '
                f'(padded L={int(ids.shape[1])}), bad in-batch rows {bad[:5].tolist()} '
                f'-> global rows {(i + bad[:5]).tolist()}')
        if bi % REF_EVERY == 0:
            _ref_check(mod, ids, att, h, f'{desc} {tag} batch {bi}')
        # 토큰 정렬 쪽은 토큰마다 정규화하고 특수토큰을 마스크에서 뺀다 — 특수토큰끼리 무조건
        # 짝지어져 점수가 부푼다. / token-level: normalize per token and drop the special tokens,
        # which would otherwise pair with each other and inflate the score.
        hn = torch.nn.functional.normalize(h, dim=-1)
        mt = att.bool()
        for sid in (CLS, SEP, PAD):
            if sid is not None:
                mt &= ids != sid
        return hn, mt, v

    def _align(hx, mx, hy, my):
        """(x 토큰이 y 에서 찾은 최대 코사인의 평균, 그 반대). 행렬 한 번으로 양방향."""
        S = torch.bmm(hx, hy.transpose(1, 2))
        nx, ny = mx.sum(1).float(), my.sum(1).float()
        ok = (nx > 0) & (ny > 0)                                  # 빈 응답 방어
        px = S.masked_fill(~my[:, None, :], NEG).max(2).values
        px = (px * mx.float()).sum(1) / nx.clamp(min=1)
        py = S.masked_fill(~mx[:, :, None], NEG).max(1).values
        py = (py * my.float()).sum(1) / ny.clamp(min=1)
        z = torch.zeros_like(px)
        return torch.where(ok, px, z), torch.where(ok, py, z)

    # 배치 하나를 통째로 계산한다 — 세 번의 인코딩과 세 방향 정렬이 한 단위다. 어디서든
    # 비유한 값이 나오면 _Fault 로 올라가고, 바깥 루프가 그 배치만 다시 돌린다.
    # One batch is one unit: three encodes and three alignments. A non-finite value anywhere
    # raises _Fault and the outer loop recomputes that batch alone.
    ACC = ('ab_p', 'ab_r', 'cov_a', 'cov_b', 'foc_a', 'foc_b')

    def _batch(sl, bi, i):
        hq, mq, vq = _encode(prompts[sl], 'Q', bi, i)
        ha, ma, va = _encode(texts_a[sl], 'A', bi, i)
        hb, mb, vb = _encode(texts_b[sl], 'B', bi, i)
        ab_p, ab_r = _align(ha, ma, hb, mb)       # A->B 정밀도 / B->A 재현율
        foc_a, cov_a = _align(ha, ma, hq, mq)     # A->Q 집중도 / Q->A 커버리지
        foc_b, cov_b = _align(hb, mb, hq, mq)
        got = {}
        for k, val in zip(ACC, (ab_p, ab_r, cov_a, cov_b, foc_a, foc_b)):
            w = val.float().cpu().numpy()
            if not np.isfinite(w).all():
                bad = np.where(~np.isfinite(w))[0]
                raise _Fault(
                    f'{desc}/{k}: non-finite score — batch rows {i}..{i + len(w) - 1}, '
                    f'bad in-batch rows {bad[:5].tolist()} -> global rows {(i + bad[:5]).tolist()}')
            got[k] = w
        return (vq, va, vb), got

    acc = {k: [] for k in ACC}
    emb = {'q': [], 'a': [], 'b': []}
    for bi, i in enumerate(_tq(range(0, len(texts_a), batch_size), desc)):
        sl = slice(i, i + batch_size)
        for attempt in range(FAULT_RETRIES + 1):
            try:
                vs, got = _batch(sl, bi, i)
                break
            except _Fault as e:
                _on_fault(mod, e, attempt)        # 예산을 다 쓰면 여기서 올라간다 / raises when spent
        for k, v in zip('qab', vs):
            emb[k].append(v)
        for k in ACC:
            acc[k].append(got[k])
    v = {k: np.concatenate(x).astype(np.float32) for k, x in acc.items()}
    eq, ea, eb = (np.concatenate(emb[k]).astype(np.float32) for k in 'qab')

    den = v['ab_p'] + v['ab_r']
    out = {'bs_prec': v['ab_p'], 'bs_rec': v['ab_r'],
           'bs_f1': np.where(den > 1e-6, 2 * v['ab_p'] * v['ab_r'],
                             0.0) / np.where(den > 1e-6, den, 1.0),
           'bs_diff': v['ab_p'] - v['ab_r']}
    for kind in ('cov', 'foc'):                   # 4형 — {min, max, d} 가 {A값, B값} 의 완전 복원 기저
        x, y = v[f'{kind}_a'], v[f'{kind}_b']
        out[f'bs_q_{kind}_min'] = np.minimum(x, y)
        out[f'bs_q_{kind}_max'] = np.maximum(x, y)
        out[f'bs_q_{kind}_ad'] = np.abs(x - y)
        out[f'bs_q_{kind}_d'] = x - y
    return eq, ea, eb, pd.DataFrame({k: w.astype(np.float32) for k, w in out.items()})


# ══════════════════════════════════════════════ [정의 3] 토큰·검색·잠재 통계 41종
# (TF-IDF/SVD 만 상태 보유 — 배포는 transform 전용) / (only TF-IDF/SVD hold state — deployment is transform-only)
# emb_sim 은 빠져 있다 — a_emb_sim/b_emb_sim 자체는 계약 열이지만 그 차(diff/abs_diff)는
# ab224 에 없다. / emb_sim is absent: a_/b_emb_sim are contract columns but their difference
# pair is not in ab224.
_STAT_DIFF_BASES = ['token_len', 'char_len', 'len_vs_prompt', 'bm25',
                    'keyword_coverage', 'tfidf_sim', 'unique_ratio']


def _tok_lens(tok, texts, tag):
    out = np.empty(len(texts), np.float32)
    for i in _tq(range(0, len(texts), 1024), f'token_len {tag}'):
        enc = tok([t if t else ' ' for t in texts[i:i + 1024]], truncation=True, max_length=512)
        out[i:i + 1024] = [len(x) for x in enc['input_ids']]
    return out


def _uniq_ratio(texts):
    out = np.empty(len(texts), np.float32)
    for i, t in enumerate(texts):
        w = _RE_WORD.findall(t.lower())
        out[i] = len(set(w)) / len(w) if w else 0.0
    return out


def _kw_cov(P, R, tag):
    out = np.empty(len(R), np.float32)
    for i in _tq(range(len(R)), f'kw_cov {tag}'):
        q = {w for w in _RE_WORD.findall(P[i].lower()) if len(w) > 2}
        out[i] = len(q & set(_RE_WORD.findall(R[i].lower()))) / len(q) if q else 0.0
    return out


def _bm25(P, A, B):
    bm_a = np.empty(len(P), np.float32)
    bm_b = np.empty(len(P), np.float32)
    for i in _tq(range(len(P)), 'bm25'):
        q_words = [w for w in _RE_WORD.findall(P[i].lower()) if len(w) > 2]
        if not q_words:
            bm_a[i] = bm_b[i] = 0.0
            continue
        doc_a = _RE_WORD.findall(A[i].lower())
        doc_b = _RE_WORD.findall(B[i].lower())
        la, lb = len(doc_a), len(doc_b)
        avg = (la + lb) / 2.0 if (la + lb) > 0 else 1.0
        ca, cb2 = Counter(doc_a), Counter(doc_b)
        sa = sb = 0.0
        for word in set(q_words):          # IDF 코퍼스 = 그 행의 응답 2개 (행 내부, 무상태)
            fa_, fb_ = ca[word], cb2[word]
            dfc = (1 if fa_ > 0 else 0) + (1 if fb_ > 0 else 0)
            idf = math.log((2.0 - dfc + 0.5) / (dfc + 0.5) + 1.0)
            da = fa_ + 1.5 * (1.0 - 0.75 + 0.75 * (la / avg))
            db = fb_ + 1.5 * (1.0 - 0.75 + 0.75 * (lb / avg))
            if da > 0:
                sa += idf * (fa_ * 2.5) / da
            if db > 0:
                sb += idf * (fb_ * 2.5) / db
        bm_a[i] = sa
        bm_b[i] = sb
    return bm_a, bm_b


def build_stat_features(P, A, B, emb2, tok, tfidf=None, svd=None, fit_idx=None):
    # fit_idx 를 주면 TF-IDF/SVD fit(학습), 안 주면 transform(추론) — 코드 경로 하나
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    F_ = {}
    F_['prompt_char_len'] = np.array([len(t) for t in P], np.float32)
    F_['a_char_len'] = np.array([len(t) for t in A], np.float32)
    F_['b_char_len'] = np.array([len(t) for t in B], np.float32)
    F_['prompt_token_len'] = _tok_lens(tok, P, 'P')
    F_['a_token_len'] = _tok_lens(tok, A, 'A')
    F_['b_token_len'] = _tok_lens(tok, B, 'B')
    F_['ratio_token_len'] = (F_['a_token_len'] + 1.0) / (F_['b_token_len'] + 1.0)
    F_['a_len_vs_prompt'] = F_['a_token_len'] / (F_['prompt_token_len'] + 1.0)
    F_['b_len_vs_prompt'] = F_['b_token_len'] / (F_['prompt_token_len'] + 1.0)
    F_['a_unique_ratio'] = _uniq_ratio(A)
    F_['b_unique_ratio'] = _uniq_ratio(B)
    if fit_idx is not None:
        tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2,
                                stop_words='english')
        tfidf.fit([P[i] for i in fit_idx] + [A[i] for i in fit_idx] + [B[i] for i in fit_idx])
    assert tfidf is not None, 'TF-IDF artifact not loaded'
    p_mat, a_mat, b_mat = tfidf.transform(P), tfidf.transform(A), tfidf.transform(B)
    F_['a_tfidf_sim'] = np.asarray(p_mat.multiply(a_mat).sum(axis=1)).ravel().astype(np.float32)
    F_['b_tfidf_sim'] = np.asarray(p_mat.multiply(b_mat).sum(axis=1)).ravel().astype(np.float32)
    F_['a_keyword_coverage'] = _kw_cov(P, A, 'A')
    F_['b_keyword_coverage'] = _kw_cov(P, B, 'B')
    diff_mat = a_mat - b_mat
    if fit_idx is not None:
        svd = TruncatedSVD(n_components=8, random_state=SEED)
        svd.fit(diff_mat[fit_idx])
    assert svd is not None
    # 두 아티팩트가 같은 fit 에서 나왔는지 — SVD 는 TF-IDF 출력 위에서 학습됐으므로 폭이 같아야
    # 한다. 짝이 어긋난 joblib 을 실으면 tfidf_svd_* 8열이 조용히 다른 공간의 값이 된다.
    # The two artifacts must come from the same fit: SVD was trained on the TF-IDF output, so
    # the widths have to agree. A mismatched pair would silently put tfidf_svd_* in another basis.
    nvocab = len(tfidf.vocabulary_)
    assert svd.components_.shape == (svd.n_components, nvocab), \
        (f'artifact pair disagrees — SVD components {svd.components_.shape} vs TF-IDF '
         f'vocabulary {nvocab}; tfidf.joblib and svd.joblib are not from the same fit')
    assert svd.n_components == 8, f'SVD n_components {svd.n_components} != 8 (tfidf_svd_0..7)'
    sv = svd.transform(diff_mat).astype(np.float32)
    for k in range(8):
        F_[f'tfidf_svd_{k}'] = sv[:, k]
    F_['a_bm25'], F_['b_bm25'] = _bm25(P, A, B)
    F_['a_emb_sim'] = emb2['a_emb_sim']
    F_['b_emb_sim'] = emb2['b_emb_sim']
    for name in _STAT_DIFF_BASES:
        dd = (F_[f'a_{name}'] - F_[f'b_{name}']).astype(np.float32)
        F_[f'diff_{name}'] = dd
        F_[f'abs_diff_{name}'] = np.abs(dd)
    assert len(F_) == 41, f'{len(F_)} stat features built != 41'
    out = pd.DataFrame(F_)
    out = pd.DataFrame(np.nan_to_num(out.to_numpy(np.float32), posinf=0, neginf=0),
                       columns=out.columns)
    return out, tfidf, svd

# ══════════════════════════════════════════════ 추출 파이프라인 / extraction pipeline
def extract(df, ckpt, emb, fit_idx=None, verbose=True):
    """텍스트 프레임 -> ab224 계약 열 224종. parquet 를 읽지 않는다 — 매번 새로 뽑는다.
    Text frame -> the 224 ab224 contract columns, extracted fresh; never reads a parquet.

    계약은 이 모듈에 박혀 있다 — 다른 피쳐 집합을 뽑는 경로도, 스위치도 없다. 체크포인트
    meta.json 이 있으면 그 열 집합과 대조하고 meta 순서로 맞춘다 (불일치는 assert).
    The contract is baked in: there is no switch and no path that produces another feature set.
    When the checkpoint's meta.json is present the built columns are checked against it and
    reordered to its order; a disagreement is an assertion, not a silent subset.

    ckpt 만 주면 저장 아티팩트로 **transform** 한다 (추론).
    fit_idx 를 주면 그 행으로 TF-IDF/SVD 를 **fit** 하고 (F, tfidf, svd) 를 돌려준다 (재학습).
    Pass ckpt to transform with the stored artifacts; pass fit_idx to fit them instead.
    """
    import gc
    import joblib
    tfidf = svd = None
    if fit_idx is None:
        tfidf = joblib.load(Path(ckpt) / 'artifacts/tfidf.joblib')
        svd = joblib.load(Path(ckpt) / 'artifacts/svd.joblib')

    t0 = time.time()
    P, A, B = parse_columns(df)
    tok, mod, dev = load_embedder(emb)
    if verbose:
        print(f'  embedder device={dev} | {len(df):,} rows | {N_CONTRACT} contract columns')
    eq, ea, eb, bs12 = encode_and_align(tok, mod, P, A, B, dev)   # 한 번의 순전파에서 둘 다
    emb5, emb2 = emb_frames(eq, ea, eb)
    del mod
    gc.collect()
    try:
        import torch
        if dev != 'cpu':
            torch.cuda.empty_cache()
    except ImportError:
        pass
    stat41, tfidf, svd = build_stat_features(P, A, B, emb2, tok, tfidf=tfidf, svd=svd,
                                             fit_idx=fit_idx)
    up = build_unit_pair_frame(P, A, B)

    F = pd.concat([f.reset_index(drop=True) for f in (emb5, bs12, stat41, up)], axis=1)
    assert not F.columns.duplicated().any(), 'duplicated feature columns'
    assert F.shape[1] == N_CONTRACT, \
        f'{F.shape[1]} columns built != {N_CONTRACT} — the definitions and the contract disagree'
    meta = Path(ckpt) / 'meta.json'
    if meta.exists():
        want = list(json.loads(meta.read_text(encoding='utf-8'))['cols'])
        missing = [c for c in want if c not in set(F.columns)]
        extra = sorted(set(F.columns) - set(want))
        assert not missing and not extra, \
            (f'contract drift — {len(missing)} in meta.json but not built {missing[:6]}, '
             f'{len(extra)} built but not in meta.json {extra[:6]}')
        F = F[want]                       # 체크포인트 열 순서로 / in the checkpoint's order
    Xf = F.to_numpy(np.float32)
    if not np.isfinite(Xf).all():
        bad_c = [F.columns[j] for j in np.where(~np.isfinite(Xf).all(0))[0]]
        bad_r = np.where(~np.isfinite(Xf).all(1))[0]
        raise AssertionError(
            f'NaN/inf in the extracted features — columns {bad_c[:8]} '
            f'({len(bad_c)} total), rows {bad_r[:8].tolist()} ({len(bad_r)} total), '
            f'ids {df["id"].to_numpy()[bad_r[:8]].tolist() if "id" in df.columns else "?"}')
    if verbose:
        dt = time.time() - t0
        print(f'  extraction done {F.shape} in {dt / 60:.1f} min ({dt / len(df):.5f} s/row)')
    return (F, tfidf, svd) if fit_idx is not None else F


def preprocess(df, cols, ckpt, emb, verbose=True):
    """**전처리 진입점** — 텍스트 프레임 -> (F 계약열 DataFrame, X 계약열 행렬 float32).
    Preprocessing entry point: text frame -> (contract-column frame, contract-column matrix).

    cols 는 체크포인트 meta 의 계약 열(ab224 224종)이다. F 와 X 는 같은 열·같은 순서라
    골라내는 단계가 없다 — 계약 밖 열은 애초에 만들어지지 않는다.
    F and X hold the same columns in the same order; nothing is selected away afterwards
    because nothing outside the contract is ever built.
    노트북은 이 함수만 부르면 된다 — 추출 로직을 다시 쓸 필요가 없다.
    """
    cols = list(cols)
    F = extract(df, ckpt=ckpt, emb=emb, verbose=verbose)
    assert list(F.columns) == cols, 'extract did not return the contract columns in order'
    X = F.to_numpy(np.float32)
    assert np.isfinite(X).all(), 'NaN/inf in the arith contract matrix'
    return F, X
