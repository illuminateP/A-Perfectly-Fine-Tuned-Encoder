"""lang_check.py — 언어 불일치가 (a) 실재하는가 (b) 인코더의 사각지대인가 검증.

피처로 만들기 전에 근거부터 확인한다. 두 질문에 답한다:
  [1] 분포   — 질문/응답의 언어가 실제로 갈리는 행이 얼마나 되나
  [2] 사각지대 — 그 행들에서 ABE/TB 가 실제로 못 하나, 그리고 불일치가 라벨을 예측하나
      (Decision_TB §3 형식: 인코더 마진을 통제한 뒤의 lift)

언어 판정 — 외부 패키지 없이 2단
  1) 스크립트 히스토그램: 한글/한자/가나/키릴/아랍/데바나가리/태국/히브리/그리스 vs 라틴
  2) 라틴이면 불용어 투표로 en/es/fr/de/pt/it/nl 근사 (짧은 글은 unknown 처리)
정확도보다 재현성·비용이 목적이라 경계 사례는 unknown 으로 흘린다.

사용: python Arith/lang_check.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ARITH = Path('./Arith')
LOGITS = ARITH / 'logits'

SCRIPTS = {
    'hangul': r'[가-힯ᄀ-ᇿ]',
    'han':    r'[一-鿿㐀-䶿]',
    'kana':   r'[぀-ヿ]',
    'cyril':  r'[Ѐ-ӿ]',
    'arab':   r'[؀-ۿ]',
    'deva':   r'[ऀ-ॿ]',
    'thai':   r'[฀-๿]',
    'hebrew': r'[֐-׿]',
    'greek':  r'[Ͱ-Ͽ]',
    'latin':  r'[A-Za-zÀ-ɏ]',
}
SCRIPT_RE = {k: re.compile(v) for k, v in SCRIPTS.items()}

# 라틴 계열 구분용 불용어 (짧고 겹치지 않는 것 위주)
STOP = {
    'en': 'the of and to in is are was were that this it for with you not have be as on at',
    'es': 'de la que el en los las por con una para es un se del no su al como',
    'fr': 'le la les des une dans est pour que qui pas sur avec au ce plus par il',
    'de': 'der die das und ist nicht ein eine den von zu mit sich auf im dem als auch',
    'pt': 'de que os as um uma para com não por mais como você isso ele ser está',
    'it': 'di che il la per una con non sono come del gli nel alla più anche questo',
    'nl': 'de het een van en is in dat te niet op zijn met voor je aan ook',
}
STOP = {k: set(v.split()) for k, v in STOP.items()}
WORD_RE = re.compile(r"[a-zA-ZÀ-ÿ']+")


def detect(text):
    if not isinstance(text, str) or len(text.strip()) < 3:
        return 'empty'
    counts = {k: len(r.findall(text)) for k, r in SCRIPT_RE.items()}
    total = sum(counts.values())
    if total < 5:
        return 'unknown'
    non_latin = {k: v for k, v in counts.items() if k != 'latin'}
    top_nl, top_nl_n = max(non_latin.items(), key=lambda kv: kv[1])
    # 비라틴 문자가 전체 문자의 15% 이상이면 그 스크립트로 판정 (코드/영어 혼용 흡수)
    if top_nl_n / total >= 0.15:
        return 'ja' if top_nl == 'kana' else {'hangul': 'ko', 'han': 'zh', 'cyril': 'ru',
                                              'arab': 'ar', 'deva': 'hi', 'thai': 'th',
                                              'hebrew': 'he', 'greek': 'el'}[top_nl]
    words = [w.lower() for w in WORD_RE.findall(text)[:400]]
    if len(words) < 8:
        return 'unknown'
    hits = {k: sum(w in s for w in words) for k, s in STOP.items()}
    best, n = max(hits.items(), key=lambda kv: kv[1])
    second = sorted(hits.values())[-2]
    if n < 3 or n < 0.02 * len(words) or n <= second:
        return 'unknown'
    return best


def main():
    df = pd.read_csv(ARITH / 'origin' / 'train.csv')
    y = np.select([df['winner_model_a'] == 1, df['winner_model_b'] == 1, df['winner_tie'] == 1],
                  [0, 1, 2], -1).astype(np.int64)

    import json

    def parse_text(x):
        if pd.isna(x):
            return ''
        try:
            t = json.loads(x)
        except (json.JSONDecodeError, TypeError):
            return str(x)
        return ' '.join(s for s in t if isinstance(s, str)) if isinstance(t, list) else str(t)

    Q = df['prompt'].map(parse_text)
    A = df['response_a'].map(parse_text)
    B = df['response_b'].map(parse_text)
    lq, la, lb = Q.map(detect), A.map(detect), B.map(detect)

    print('=== [1] 언어 분포 (train 46,729) ===')
    print('  질문 :', lq.value_counts().head(12).to_dict())
    print('  A    :', la.value_counts().head(8).to_dict())
    n_nonen = int((~lq.isin(['en', 'unknown', 'empty'])).sum())
    print(f'  질문이 영어가 아닌 행: {n_nonen:,} ({n_nonen / len(df) * 100:.1f}%)'
          '   ← DeBERTa-v3 는 영어 전용 백본')

    # 불일치 = 질문 언어가 확정된 행에서 응답 언어가 다름 (unknown 은 판정 보류)
    known = lq.isin(list(STOP) + ['ko', 'zh', 'ja', 'ru', 'ar', 'hi', 'th', 'he', 'el'])
    mm_a = known & la.isin(list(STOP) + ['ko', 'zh', 'ja', 'ru', 'ar', 'hi', 'th', 'he', 'el']) & (la != lq)
    mm_b = known & lb.isin(list(STOP) + ['ko', 'zh', 'ja', 'ru', 'ar', 'hi', 'th', 'he', 'el']) & (lb != lq)
    any_mm = mm_a | mm_b
    xor_mm = mm_a ^ mm_b                                    # 한쪽만 언어를 틀린 행 = 승부 신호 후보
    print(f'\n=== [2] 불일치 규모 ===')
    print(f'  질문 언어 확정: {int(known.sum()):,}행')
    print(f'  A 불일치 {int(mm_a.sum()):,} / B 불일치 {int(mm_b.sum()):,} / '
          f'둘 중 하나 {int(any_mm.sum()):,} ({any_mm.mean() * 100:.2f}%) / '
          f'한쪽만 {int(xor_mm.sum()):,} ({xor_mm.mean() * 100:.2f}%)')

    tie = (y == 2)
    print(f'\n=== [3] 라벨 연관 (전체 tie율 {tie.mean():.3f}) ===')
    for name, m in (('불일치 없음', known & ~any_mm), ('둘 다 불일치', mm_a & mm_b),
                    ('한쪽만 불일치', xor_mm)):
        if m.sum() >= 30:
            print(f'  {name:<12} n={int(m.sum()):>6,}  tie율 {tie[m].mean():.3f}')
    if xor_mm.sum() >= 30:
        # 한쪽만 틀렸다면 그 쪽이 지는가 (A 불일치 -> B 승 기대)
        sub = xor_mm & (y != 2)
        a_bad = mm_a & ~mm_b & (y != 2)
        b_bad = mm_b & ~mm_a & (y != 2)
        print(f'  한쪽만 불일치 & 비-tie: A만 틀림 {int(a_bad.sum())}행 중 B승 '
              f'{(y[a_bad] == 1).mean() * 100:.1f}% / B만 틀림 {int(b_bad.sum())}행 중 A승 '
              f'{(y[b_bad] == 0).mean() * 100:.1f}%   (무신호면 50%)')

    # [4] 인코더 사각지대인가 — 캐시된 ABE/TB train 로짓의 행별 손실
    za, zt = LOGITS / 'abe_train_logits.npz', LOGITS / 'tb_train_logits.npz'
    if za.exists() and zt.exists():
        za, zt = np.load(za), np.load(zt)
        assert (za['ids'] == df['id'].to_numpy()).all()
        z = (za['fwd'] + za['swp']) / 2
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        e_abe = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))
        zb = (zt['fwd'] + zt['swp']) / 2
        pt = np.exp(zb - zb.max(1, keepdims=True))
        pt /= pt.sum(1, keepdims=True)
        pt = pt[:, 1]
        e_tb = -np.where(tie, np.log(np.clip(pt, 1e-15, 1)), np.log(np.clip(1 - pt, 1e-15, 1)))
        print('\n=== [4] 인코더가 이 행에서 못 하나 (train 로짓, in-sample 이라 절대값은 낙관) ===')
        for name, m in (('전체', np.ones(len(y), bool)), ('질문 영어', (lq == 'en').to_numpy()),
                        ('질문 비영어', (known & (lq != 'en')).to_numpy()),
                        ('불일치 있음', any_mm.to_numpy())):
            if m.sum() >= 30:
                print(f'  {name:<12} n={int(m.sum()):>6,}  ABE ll {e_abe[m].mean():.4f}  '
                      f'TB tie ll {e_tb[m].mean():.4f}')
    else:
        print('\n[4] 생략 — 로짓 캐시 없음')


if __name__ == '__main__':
    main()
