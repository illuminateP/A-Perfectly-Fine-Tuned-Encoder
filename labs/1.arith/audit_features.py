"""audit_features.py — 피쳐 추출 전수조사. 재추출 여부를 판단할 근거를 만든다.

무엇을 검사하나
  1. **swap 계약 전수 검증** (가장 중요) — 같은 행을 (P,A,B) 와 (P,B,A) 로 두 번 추출해
     카탈로그의 `swap` 선언과 실제 거동이 일치하는지 259열 전부 확인한다.
     여기가 어긋나면 학습 swap 증강이 **틀린 라벨을 다는 것**이라 조용히 성능을 갉아먹는다.
        불변      f(P,B,A) == f(P,A,B)
        부호반전  f(P,B,A) == −f(P,A,B)
        역수      f(P,B,A) == 1/f(P,A,B)
        교환:X    f(P,B,A)[c] == f(P,A,B)[X]
  2. **수치 건전성** — inf/NaN, 상수열, 완전중복열, 꼬리(왜도·첨도·max|z|).
     꼬리는 트리 소비자에게는 무해하지만(분할이 순위 기반) 선형 소비자에게는 치명적이다.
     그 구분을 표에 명시한다 — 재추출이 필요한 결함과 소비 시점에서 고칠 것을 섞지 않기 위해.
  3. **절단/포화** — `_tok_lens` 가 truncation=True, max_length=512 라 토큰 길이가 검열된다.
     포화 비율을 실측해 정보 손실 크기를 숫자로 낸다.
  4. **정의 대조** — 같은 이름 어간인데 스케일이 다른 쌍(`a_char_len` 원값 vs `mn_char_len` log1p)을
     찾아 표시한다. 이름만 보고 같은 척도라 가정하면 사고가 난다.

사용
  python Arith/audit_features.py               # 300행 표본
  python Arith/audit_features.py --n 800
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ARITH = Path('./Arith')
SEED = 42
TOL = 1e-4


def load_defs():
    """Features.ipynb 의 [정의] 셀만 실행 — 추출 로직의 진실의 원천을 그대로 태운다."""
    nb = json.loads((ARITH / 'Features.ipynb').read_text(encoding='utf-8'))
    ns = {'__name__': 'features_defs'}
    cwd = os.getcwd()
    os.chdir(ARITH)
    try:
        for i in (1, 2, 3, 4):
            src = ''.join(nb['cells'][i]['source'])
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, f'Features.ipynb#cell{i}', 'exec'), ns)
    finally:
        os.chdir(cwd)
    return ns


def build(ns, P, A, B, emb3, tok, tfidf, svd):
    """한 방향의 259열 프레임. Features.ipynb 셀 7 의 조립 순서와 같다."""
    eq, ea, eb = emb3['eq'], emb3['ea'], emb3['eb']
    emb5, e3 = ns['emb_frames'](eq, ea, eb)
    bs12 = ns['bertscore_features'](tok[0], tok[1], P, A, B, tok[2])
    stat44 = ns['build_stat_features'](P, A, B, e3, tok[0], tfidf=tfidf, svd=svd)[0]
    up = ns['build_unit_pair_frame'](P, A, B)
    F = pd.concat([f.reset_index(drop=True) for f in (emb5, bs12, stat44, up)], axis=1)
    assert not F.columns.duplicated().any(), '컬럼 중복'
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300, help='검증 표본 행 수')
    ap.add_argument('--tol', type=float, default=TOL)
    args = ap.parse_args()

    import joblib
    ns = load_defs()
    cat = pd.read_csv(ARITH / 'feature_catalog.csv')
    swap = cat.set_index('feature')['swap']

    rng = np.random.default_rng(SEED)
    df = pd.read_csv(ARITH / 'origin/train.csv')
    idx = rng.choice(len(df), min(args.n, len(df)), replace=False)
    d = df.iloc[idx].reset_index(drop=True)
    P, A, B = ns['parse_columns'](d)
    print(f'표본 {len(P)}행 | 카탈로그 {len(cat)}종')

    os.chdir(ARITH)
    tokm, mod, dev = ns['load_embedder']('emb_model')
    with contextlib.redirect_stdout(io.StringIO()):
        eq = ns['encode_texts'](tokm, mod, P, dev)
        ea = ns['encode_texts'](tokm, mod, A, dev)
        eb = ns['encode_texts'](tokm, mod, B, dev)
    tfidf, svd = joblib.load('artifacts/tfidf.joblib'), joblib.load('artifacts/svd.joblib')
    tk = (tokm, mod, dev)
    with contextlib.redirect_stdout(io.StringIO()):
        F = build(ns, P, A, B, {'eq': eq, 'ea': ea, 'eb': eb}, tk, tfidf, svd)
        # 교환 뷰 — 임베딩도 함께 뒤집어야 stat 블록이 일관된다
        G = build(ns, P, B, A, {'eq': eq, 'ea': eb, 'eb': ea}, tk, tfidf, svd)
    print(f'양방향 추출 완료 — {F.shape[1]}열\n')

    # ── 1) swap 계약 ──
    miss = [c for c in F.columns if c not in swap.index]
    assert not miss, f'카탈로그에 없는 열 {len(miss)}개: {miss[:5]}'
    rows, viol = [], []
    for c in F.columns:
        v, w, rule = F[c].to_numpy(np.float64), G[c].to_numpy(np.float64), swap[c]
        if rule == '불변':
            exp = v
        elif rule == '부호반전':
            exp = -v
        elif rule == '역수':
            exp = 1.0 / np.clip(v, 1e-12, None)
        elif isinstance(rule, str) and rule.startswith('교환:'):
            partner = rule.split(':', 1)[1]
            if partner not in F.columns:
                viol.append((c, rule, np.nan, '파트너 열 없음'))
                continue
            exp = F[partner].to_numpy(np.float64)
        else:
            viol.append((c, str(rule), np.nan, '알 수 없는 규칙'))
            continue
        scale = max(np.abs(v).max(), np.abs(exp).max(), 1.0)
        err = float(np.abs(w - exp).max() / scale)
        rows.append((c, rule, err))
        if err > args.tol:
            viol.append((c, rule, err, f'상대오차 {err:.2e}'))

    print(f'[1] swap 계약 — {len(rows)}열 검사, 위반 {len(viol)}건')
    if viol:
        print(f'  {"열":<26}{"선언":<16}{"상대오차":>12}  비고')
        for c, r, e, note in viol[:40]:
            print(f'  {c:<26}{str(r):<16}{e:>12.2e}  {note}')
    else:
        print(f'  전 열 통과 (최대 상대오차 {max(e for _, _, e in rows):.2e}, 허용 {args.tol:.0e})')
        top = sorted(rows, key=lambda r: -r[2])[:5]
        print('  오차 상위 5: ' + ', '.join(f'{c} {e:.1e}' for c, _, e in top))

    # ── 2) 수치 건전성 ──
    X = F.to_numpy(np.float64)
    print(f'\n[2] 수치 건전성 — inf {int(np.isinf(X).sum())} / NaN {int(np.isnan(X).sum())}')
    const = [c for c in F.columns if F[c].nunique() <= 1]
    print(f'  상수열 {len(const)}개' + (f': {const}' if const else ''))
    z = (X - X.mean(0)) / np.clip(X.std(0), 1e-12, None)
    sk = (z ** 3).mean(0)
    ku = (z ** 4).mean(0) - 3
    mz = np.abs(z).max(0)
    bad = np.argsort(-mz)[:12]
    print('\n  꼬리 상위 12열 — 트리에는 무해(분할이 순위 기반), **선형 소비자에는 치명적**')
    print(f'  {"열":<26}{"max":>12}{"왜도":>9}{"첨도":>11}{"max|z|":>9}')
    for i in bad:
        print(f'  {F.columns[i]:<26}{X[:, i].max():>12.1f}{sk[i]:>9.2f}{ku[i]:>11.1f}{mz[i]:>9.1f}')
    print(f'  * max|z| > 20 인 열 {int((mz > 20).sum())}개 / {len(mz)}. 로짓 신호는 5~7 수준이다.')

    # ── 3) 절단/포화 ──
    print('\n[3] 절단·포화')
    for c in ('a_token_len', 'b_token_len', 'prompt_token_len'):
        v = F[c].to_numpy()
        print(f'  {c:<20} 최댓값 {v.max():>6.0f}  512 도달 {np.mean(v >= 512):>6.2%}  '
              f'(_tok_lens 가 truncation=True, max_length=512)')
    print('  * 도달 비율만큼 길이 정보가 검열된다. a_char_len 은 무절단이라 두 척도가 어긋난다.')

    # ── 4) 같은 어간, 다른 척도 ──
    print('\n[4] 이름 어간은 같은데 척도가 다른 쌍 (log1p 여부)')
    pairs = [('a_char_len', 'mn_char_len'), ('abs_diff_char_len', 'ad_char_len')]
    for a, b in pairs:
        if a in F.columns and b in F.columns:
            print(f'  {a:<20} max {F[a].max():>10.1f}   vs   {b:<16} max {F[b].max():>8.2f}'
                  f'   -> {"다른 척도(원값 vs log1p)" if F[a].max() > 50 * max(F[b].max(), 1e-9) else "유사"}')
    print('  * 통계 44 블록은 원값, 4형 전개 블록은 log1p 다. 이름만 보고 같은 척도라 가정하면 안 된다.')

    # ── 5) 완전중복 ──
    print('\n[5] 완전중복 열 (상관 |ρ| > 0.9999)')
    C = np.corrcoef(np.where(X.std(0) > 0, z, 0).T)
    dup = [(F.columns[i], F.columns[j], C[i, j])
           for i in range(len(F.columns)) for j in range(i + 1, len(F.columns))
           if np.isfinite(C[i, j]) and abs(C[i, j]) > 0.9999]
    print(f'  {len(dup)}쌍' + ('' if not dup else ''))
    for a, b, r in dup[:15]:
        print(f'    {a:<26} ~ {b:<26} {r:+.6f}  (카탈로그 사용 플래그 확인 필요)')


if __name__ == '__main__':
    main()
