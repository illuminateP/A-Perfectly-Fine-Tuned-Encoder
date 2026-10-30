"""bench_arith.py — Arith 255피쳐 **실시간 추출 비용**을 단계별로 잰다. (parquet 미사용)

왜
  Kaggle 비공개 test 는 train 과 분포가 거의 같으므로, train.csv 에서 표본을 뽑아
  피쳐를 처음부터 뽑는 데 드는 s/행 을 재면 test 50k 비용을 외삽할 수 있다.
  지금 이 숫자가 없어서 "임베더 계층(MiniLM 19 + BERTScore, TF-IDF 12, 토크나이저 10 =
  gain 26.84%)을 포기하고 무상태 214개로 재학습할 가치가 있는가" 를 판단할 수 없다.

무엇을 재나 — 단계별로 쪼갠다 (계층별 반사실을 사용자가 직접 조립할 수 있도록)
    parse       parse_columns                      싸다
    encode      encode_texts × 3 (P/A/B)           MiniLM 문장 임베딩
    embframes   emb_frames                         싸다 (emb5 + emb3)
    bertscore   bertscore_features                 MiniLM **토큰 단위** — 보통 여기가 최대
    stat        build_stat_features                혼합: 토크나이저 + TF-IDF transform + BM25
                                                   + emb3 3열 패스스루
    unitpair    build_unit_pair_frame              순수 파이썬 문자열 (3gram·jaccard·char3)
  모델·아티팩트 적재는 **고정 비용**이라 따로 뺀다 (행 수에 무관).

선형성 확인
  두 가지 표본 크기로 재서 s/행 이 일치하는지 본다. 어긋나면 고정 비용이 덜 빠진 것이다.

정합성 확인
  같은 행을 features_57k.parquet 과 대조한다 — 실시간 추출 경로가 저장 피쳐를
  재현하는지. 여기가 어긋나면 위 시간은 다른 계산의 시간이다.

사용 (GPU 필요)
  python final/bench_arith.py 2>&1 | tee final/log_bench_arith.txt
  python final/bench_arith.py --sizes 256,1024 --target 50000
"""
import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

FINAL = Path(__file__).resolve().parent.parent      # 이 파일은 final/temp/ 에 있다
ARITH = FINAL / 'models/ARITH'
ORIGIN = FINAL / 'origin'


@contextlib.contextmanager
def chdir(p):
    """Features.ipynb 는 상대경로로 쓰여 있다 — 그 블록만 CWD 를 옮긴다."""
    old = os.getcwd()
    os.chdir(p)
    try:
        yield
    finally:
        os.chdir(old)


def load_feature_ns():
    """Features.ipynb 의 [정의] 셀만 실행해 추출 함수를 그대로 얻는다 (노트북과 동일)."""
    nb = json.loads((ARITH / 'Features.ipynb').read_text(encoding='utf-8'))
    ns = {'__name__': 'features_defs'}
    with chdir(ARITH):
        for i in (1, 2, 3, 4):
            src = ''.join(nb['cells'][i]['source'])
            assert nb['cells'][i]['cell_type'] == 'code', f'셀 {i} 가 코드가 아니다'
            assert '[정의]' in src or '── 설정 ──' in src, f'셀 {i} 가 정의 셀이 아니다'
            exec(compile(src, f'Features.ipynb#cell{i}', 'exec'), ns)
    need = ('parse_columns', 'load_embedder', 'encode_texts', 'emb_frames',
            'bertscore_features', 'build_stat_features', 'build_unit_pair_frame')
    miss = [f for f in need if f not in ns]
    assert not miss, f'정의 셀에서 함수 누락: {miss}'
    return ns


class Timer(dict):
    @contextlib.contextmanager
    def __call__(self, key):
        t = time.time()
        yield
        self[key] = self.get(key, 0.0) + (time.time() - t)


def run_once(ns, df, emb, tfidf, svd, cols):
    """한 표본에 대해 단계별 시간과 피쳐 행렬을 낸다."""
    tok, mod, dev = emb
    T = Timer()
    with chdir(ARITH):
        with T('parse'):
            P, A, B = ns['parse_columns'](df)
        with T('encode'):
            eq = ns['encode_texts'](tok, mod, P, dev, desc='embed P')
            ea = ns['encode_texts'](tok, mod, A, dev, desc='embed A')
            eb = ns['encode_texts'](tok, mod, B, dev, desc='embed B')
        with T('embframes'):
            emb5, emb3 = ns['emb_frames'](eq, ea, eb)
        with T('bertscore'):
            bs12 = ns['bertscore_features'](tok, mod, P, A, B, dev)
        with T('stat'):
            stat44 = ns['build_stat_features'](P, A, B, emb3, tok, tfidf=tfidf, svd=svd)[0]
        with T('unitpair'):
            up = ns['build_unit_pair_frame'](P, A, B)
    F = pd.concat([f.reset_index(drop=True) for f in (emb5, bs12, stat44, up)], axis=1)
    assert not F.columns.duplicated().any(), '피쳐 컬럼 중복'
    miss = [c for c in cols if c not in F.columns]
    assert not miss, f'모델이 요구하는 피쳐 누락 {len(miss)}종: {miss[:6]}'
    return T, F[cols].to_numpy(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default='256,1024', help='표본 크기 (쉼표) — 선형성 확인용')
    ap.add_argument('--target', type=int, default=50000, help='외삽할 test 행 수')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(',')]

    cols = json.loads((ARITH / 'model/meta.json').read_text(encoding='utf-8'))['cols']
    tr = pd.read_csv(ORIGIN / 'train.csv')
    print(f'train {len(tr):,}행 | 피쳐 {len(cols)} | 표본 {sizes} | 외삽 목표 {args.target:,}행')

    ln_all = (tr['prompt'].str.len() + tr['response_a'].str.len()
              + tr['response_b'].str.len()).to_numpy()
    rng = np.random.default_rng(args.seed)

    import joblib
    t0 = time.time()
    ns = load_feature_ns()
    t_defs = time.time() - t0
    with chdir(ARITH):
        t0 = time.time()
        emb = ns['load_embedder']('emb_model')
        t_emb = time.time() - t0
        t0 = time.time()
        tfidf = joblib.load('artifacts/tfidf.joblib')
        svd = joblib.load('artifacts/svd.joblib')
        t_art = time.time() - t0
    print(f'\n고정 비용: 정의셀 exec {t_defs:.1f}s | 임베더 적재 {t_emb:.1f}s '
          f'| TF-IDF/SVD 적재 {t_art:.1f}s  (합 {t_defs + t_emb + t_art:.1f}s)')
    print(f'  device: {emb[2]}')

    STAGES = ('parse', 'encode', 'embframes', 'bertscore', 'stat', 'unitpair')
    res = {}
    for n in sizes:
        idx = rng.choice(len(tr), n, replace=False)
        sub = tr.iloc[idx].reset_index(drop=True)
        ln = ln_all[idx]
        print(f'\n── 표본 {n:,}행 (문자수 평균 {ln.mean():,.0f} / 중앙 '
              f'{np.median(ln):,.0f}  vs 전체 평균 {ln_all.mean():,.0f} / 중앙 '
              f'{np.median(ln_all):,.0f}) ──')
        T, X = run_once(ns, sub, emb, tfidf, svd, cols)
        tot = sum(T[s] for s in STAGES)
        res[n] = (T, tot, sub['id'].to_numpy(), X)
        print(f'    {"단계":10s} {"초":>9s} {"s/행":>10s} {"비중":>7s}')
        for s in STAGES:
            print(f'    {s:10s} {T[s]:9.2f} {T[s] / n:10.5f} {T[s] / tot * 100:6.1f}%')
        print(f'    {"합계":10s} {tot:9.2f} {tot / n:10.5f}')

    # ── 선형성 ──
    if len(sizes) > 1:
        a, b = sizes[0], sizes[-1]
        ra, rb = res[a][1] / a, res[b][1] / b
        print(f'\n선형성: {a}행 {ra:.5f} s/행  vs  {b}행 {rb:.5f} s/행  '
              f'(비 {rb / ra:.3f})' + ('  — 선형' if 0.8 < rb / ra < 1.25
                                       else '  ** 고정 비용이 덜 빠졌다'))

    # ── 외삽 ──
    n_big = sizes[-1]
    T, tot = res[n_big][0], res[n_big][1]
    print(f'\n── test {args.target:,}행 외삽 (표본 {n_big:,} 기준) ──')
    per = {s: T[s] / n_big for s in STAGES}
    full = sum(per.values()) * args.target
    embed_side = (per['encode'] + per['bertscore']) * args.target
    print(f'    {"단계":10s} {"시":>8s}')
    for s in STAGES:
        print(f'    {s:10s} {per[s] * args.target / 3600:8.2f}')
    print(f'    {"합계":10s} {full / 3600:8.2f} 시간')
    print(f'\n  임베더 계층(encode + bertscore) 만 빼면 {(full - embed_side) / 3600:.2f}시간 '
          f'({embed_side / 3600:.2f}시간 절감, 전체의 {embed_side / full * 100:.0f}%)')
    print('  * stat 단계에는 토크나이저·TF-IDF 도 섞여 있어, 무상태-only 는 이보다 더 준다.')
    print(f'  * Kaggle GPU 세션 9시간 기준 Arith 만으로 {full / 3600 / 9 * 100:.1f}% 소진.')

    # ── 정합성: 저장 parquet 재현 ──
    ids, X = res[n_big][2], res[n_big][3]
    F = pd.read_parquet(ARITH / 'features_57k.parquet').set_index('id')
    have = [i for i in ids if i in F.index]
    if len(have) < len(ids):
        print(f'\n[주의] parquet 에 없는 id {len(ids) - len(have)}개 — 그 행은 대조에서 제외')
    pos = [j for j, i in enumerate(ids) if i in F.index]
    R = F.loc[[ids[j] for j in pos], cols].to_numpy(np.float32)
    d = np.abs(X[pos] - R)
    worst = np.nanmax(d, axis=0)
    order = np.argsort(-worst)[:8]
    print(f'\n── parquet 정합성 ({len(pos):,}행 × {len(cols)}열) ──')
    print(f'    max|Δ| 전체 {np.nanmax(d):.3e} | 평균 {np.nanmean(d):.3e}')
    print('    열별 최대 오차 상위 8:')
    for j in order:
        print(f'      {cols[j]:26s} {worst[j]:.3e}')
    print('    (여기가 크면 실시간 추출이 저장 피쳐와 다른 계산이라는 뜻 — 시간 해석 전에 확인)')


if __name__ == '__main__':
    main()
