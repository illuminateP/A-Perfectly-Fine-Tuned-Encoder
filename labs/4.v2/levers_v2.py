"""levers_v2.py — B(층1 열 5시드 평균 → 층2 1회) 위에서 남은 레버 두 개와 그 조합을 재고 저장한다 (사용자 결정 2026-10-28).

레버
  L6    TTA 뷰 불일치 4열 d = |fwd − swp| — 전문가별 tie 성분(tb2·tb1: s = L1 − L0 · abe2·abe1: t = L2 − logaddexp(L0, L1)).
        A↔B 교환에 짝이라 tie 결합기에만 넣는다. 벌점은 층1 헤드 열과 같은 묶음(λ_g)을 공유한다 — tie 층2 입력 [로짓 5 | d 4 · g 2].
        층1 과 승부 축은 그대로, 결합기만 다시 적합.
  K     tie 헤드 SVD 랭크 K_TIE 512 → 1024 (층1 5시드 재학습 = lab.run, GPU). B 기준(fold seed 5개 OOF 평균)으로 격자 끝(가장 큰 K)이
        이기면 ×2 로 늘린다 (최대 2048 — randomized SVD 는 4096 전랭크를 못 쓴다).
  K+L6  둘 다.
λ 격자 — 노트북 Layer2 그대로 {0, 1, 10, …, 1e4}. 고른 값이 그 축의 위끝이면 ×10 자동 확장(LAM_CEIL 1e12). fold 별 선택과 위끝 도달을 기록.
측정 단위 (전부 시드 42 를 포함한 5개)
  단일  층1 시드 42·1·2·3·4 각각 층1+층2 (fold seed = 층1 시드) — 기준 팔 results/v2
  B     층1 열 5시드 평균 → 층2, fold seed 42·1·2·3·4 (배포 = 42) — 기준 팔 results/v2_B. 'seed' 자리에 fold seed 를 넣은 lab 형식.
  비교  lab.compare (holdout OOF paired · 행 부트스트랩 95% CI · 부호). B 팔에서 compare 의 '시드 z 평균' 줄은 fold seed 평균이다.
층2 적합만 BLAS 스레드 2개로 제한한다(7열 행렬 — 16스레드면 수십 배 느림, 결과 차이 1e-14). SVD 는 다중 스레드 그대로.

출력  results/<팔>/ (lab 형식 summary.json · s<key>.npz — 배포 계수 포함) · results/_compare/levers_v2_report.txt · levers_v2.json
실행  cd labs/4.v2
      python -u levers_v2.py l6              (CPU 1분)
      python -u levers_v2.py auto            (K 1024 → 이기면 2048 → K+L6 → 요약, GPU 15~40분)
      python -u levers_v2.py summary         (저장된 팔로 요약표만)
"""
import sys
sys.dont_write_bytecode = True
import io
import json
import time
import argparse
import contextlib
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import lab                                          # noqa: E402
N = lab.N

SEEDS = [42, 1, 2, 3, 4]
FOLDS = [42, 1, 2, 3, 4]
TAGS = ('tb2', 'tb1', 'abe2', 'abe1')               # tie_base 의 앞 4열(s_tb2 s_tb1 t_abe2 t_abe1)과 같은 순서
R = lab.RESULTS
REPORT = R / '_compare' / 'levers_v2_report.txt'
JSON = R / '_compare' / 'levers_v2.json'
KMAX = 2048

_fit_orig = N.Layer2.fit.__func__


def _fit_limited(cls, S2, g, w):
    with threadpool_limits(limits=2):
        return _fit_orig(cls, S2, g, w)


N.Layer2.fit = classmethod(_fit_limited)           # lab.run 안의 층2 도 이 경로를 탄다

_S2 = {}


def stage2():
    """v2 stage2 (풀링 제외) — 층2 와 d 열에 필요한 것만, 한 번만 읽는다."""
    if 'S' not in _S2:
        N.use(lab.bundle_path('v2'))
        S = N.build('stage2')
        S.pop('pool')
        _S2['S'] = S
    return _S2['S']


def disagree():
    """stage2 행(valid → holdout)의 뷰 불일치 4열 — 번들 로짓 npz 의 fwd·swp(스왑본은 A/B 순서로 되돌린 것)."""
    if 'd' not in _S2:
        S = stage2()
        cols = []
        for tag in TAGS:
            z = np.load(N.OUTPUT_DIR / N.NPZ[tag]['logits_stage2'])
            assert (np.concatenate([z['valid_ids'], z['hold_ids']]) == S['ids']).all(), f'{tag}: ids disagree!!'
            f = np.vstack([z['valid_fwd'], z['hold_fwd']]).astype(np.float64)
            s = np.vstack([z['valid_swp'], z['hold_swp']]).astype(np.float64)
            t = (lambda L: L[:, 1] - L[:, 0]) if f.shape[1] == 2 else (lambda L: L[:, 2] - np.logaddexp(L[:, 0], L[:, 1]))
            cols.append(np.abs(t(f) - t(s)))
        _S2['d'] = np.column_stack(cols)
    return _S2['d']


def cols(arm, seeds=SEEDS):
    Z = {s: np.load(R / arm / f's{s}.npz') for s in seeds}
    S = stage2()
    for s in seeds:
        assert (Z[s]['ids'] == S['ids']).all(), f'{arm} s{s}: rows != stage2!!'
    return {s: (Z[s]['g'], Z[s]['w']) for s in seeds}


def fit_and_save(name, items, config):
    """items = [(key, g 열(층2 tie 블록 λ_g 쪽 전부), w 열, fold seed)] → lab 형식 팔 results/<name>/."""
    S = stage2()
    d = R / name
    d.mkdir(parents=True, exist_ok=True)
    summ = dict(config=config, seeds={})
    for key, gc, wc, fs in items:
        N.SEED = fs
        t0 = time.time()
        L2 = N.Layer2.fit(S, gc, wc)
        zt_d, zw_d = L2.transform(S['tie_base'], gc, S['win_base'], wc)
        o = lab.scores(S['y'], S['nz'], L2.z_tie, L2.z_win, S['hold'])
        dp = lab.scores(S['y'], S['nz'], zt_d, zw_d, S['hold'])
        top = max(N.Layer2.LAMS)
        edge = dict(tie=any(max(q) >= top for q in L2.dep_t['picked']),
                    win=any(max(q) >= top for q in L2.dep_w['picked']))
        np.savez(d / f's{key}.npz', ids=S['ids'], y=S['y'], hold=S['hold'], zt_oof=L2.z_tie, zw_oof=L2.z_win,
                 zt_dep=zt_d, zw_dep=zw_d, g=gc, w=wc,
                 **{f't_{k}': np.asarray(L2.dep_t[k]) for k in N.Layer2.KEYS},
                 **{f'w_{k}': np.asarray(L2.dep_w[k]) for k in N.Layer2.KEYS})
        summ['seeds'][str(key)] = dict(oof=o, dep=dp, fold_seed=fs, lam_tie=list(L2.dep_t['lam']),
                                       lam_win=list(L2.dep_w['lam']), picked_tie=L2.dep_t['picked'],
                                       picked_win=L2.dep_w['picked'], grid_top_hit=edge, sec=round(time.time() - t0, 1))
        print(f'  [{name} {key}] OOF {o["tot"]:.5f} (tie {o["tie"]:.5f} win {o["win"]:.5f}) | 배포적합 {dp["tot"]:.5f} | '
              f'λ tie {L2.dep_t["lam"]} win {L2.dep_w["lam"]}{" · 위끝 도달" if edge["tie"] or edge["win"] else ""}', flush=True)
    (d / 'summary.json').write_text(json.dumps(summ, indent=1), encoding='utf-8')
    return summ


def single(name, src, use_d, note):
    """층1 시드별 (src 팔의 열) → 층2. fold seed = 층1 시드 (기준 팔 v2 와 같은 fold)."""
    C, dd = cols(src), (disagree() if use_d else None)
    items = [(s, np.hstack([dd, C[s][0]]) if use_d else C[s][0], C[s][1], s) for s in SEEDS]
    return fit_and_save(name, items, dict(kind='single', layer1=src, L6=use_d, note=note, seeds=SEEDS))


def ens_b(name, src, use_d, note):
    """B — src 팔의 층1 열 5시드 평균 → 층2, fold seed 5개."""
    C, dd = cols(src), (disagree() if use_d else None)
    g = np.mean([C[s][0] for s in SEEDS], 0)
    w = np.mean([C[s][1] for s in SEEDS], 0)
    gc = np.hstack([dd, g]) if use_d else g
    return fit_and_save(name, [(fs, gc, w, fs) for fs in FOLDS],
                        dict(kind='B', layer1=src, layer1_seeds=SEEDS, fold_seeds=FOLDS, L6=use_d, note=note))


def b_mean(name):
    s = json.loads((R / name / 'summary.json').read_text(encoding='utf-8'))
    return float(np.mean([s['seeds'][str(f)]['oof']['tot'] for f in FOLDS]))


def ensure_base():
    """기준 B 팔 v2_B — ensemble_v2.py 의 B 결과를 같은 경로로 다시 만들고 일치를 확인한다."""
    if (R / 'v2_B' / 'summary.json').exists():
        return
    print('[기준 B] v2_B ← v2 층1 열 5시드 평균 (K512, d 없음)', flush=True)
    ens_b('v2_B', 'v2', False, 'baseline B (K512, no L6)')
    for fs in FOLDS:
        a, b = np.load(R / 'v2_B' / f's{fs}.npz'), np.load(R / '_ens' / f'v2__colavg_f{fs}.npz')
        dmax = max(float(np.nanmax(np.abs(a['zt_oof'] - b['zt_oof']))), float(np.nanmax(np.abs(a['zw_oof'] - b['zw_oof']))))
        assert dmax < 1e-9, f'v2_B fold {fs} != ensemble_v2 result ({dmax:.1e})!!'
    print('  = ensemble_v2.py 의 B 결과와 일치 (fold seed 5개, max|Δ| < 1e-9)', flush=True)


def run_l6():
    ensure_base()
    print('[L6] 단일 5시드 — v2 층1 열 + d 4열 (λ_g 공유)', flush=True)
    single('v2_L6', 'v2', True, 'L6: tie combiner [logits5 | d4 + g2], shared lambda_g')
    print('[L6] B — v2 층1 열 5시드 평균 + d 4열', flush=True)
    ens_b('v2_B_L6', 'v2', True, 'L6 on B')


def run_k(k):
    arm = f'v2_K{k}'
    print(f'[K{k}] 층1 5시드 재학습 (lab.run · K_TIE={k} · GPU)', flush=True)
    lab.run('v2', arm, SEEDS, sets=[f'Layer1.K_TIE={k}'], wiring=False)
    print(f'[K{k}] B — 층1 열 5시드 평균', flush=True)
    ens_b(f'v2_B_K{k}', arm, False, f'K_TIE={k} on B')


def run_combo(k):
    print(f'[K{k}+L6] 단일 5시드 · B', flush=True)
    single(f'v2_K{k}_L6', f'v2_K{k}', True, f'K_TIE={k} + L6')
    ens_b(f'v2_B_K{k}_L6', f'v2_K{k}', True, f'K_TIE={k} + L6 on B')


def auto():
    ensure_base()
    if not (R / 'v2_B_L6' / 'summary.json').exists():
        run_l6()
    ks, best, prev = [], 512, 'v2_B'
    k = 1024
    while k <= KMAX:
        if not (R / f'v2_B_K{k}' / 'summary.json').exists():
            run_k(k)
        ks.append(k)
        m_k, m_prev = b_mean(f'v2_B_K{k}'), b_mean(prev)
        win = m_k < m_prev
        print(f'[K 격자] K{k} B 평균 {m_k:.5f} vs {prev} {m_prev:.5f} → {"격자 끝이 이김 — ×2 로 늘린다" if win and k < KMAX else ("이김 (상한 도달)" if win else "늘리지 않음")}', flush=True)
        if not win:
            break
        best, prev, k = k, f'v2_B_K{k}', k * 2
    for kk in sorted({1024, best} - {512}):
        if not (R / f'v2_B_K{kk}_L6' / 'summary.json').exists():
            run_combo(kk)
    summary()


ARMS = [('L6', 'v2_L6', 'v2_B_L6')] + [(f'K{k}', f'v2_K{k}', f'v2_B_K{k}') for k in (1024, 2048)] \
       + [(f'K{k}+L6', f'v2_K{k}_L6', f'v2_B_K{k}_L6') for k in (1024, 2048)]


def summary():
    lines = []

    def say(t=''):
        print(t, flush=True)
        lines.append(t)

    def cap(fn, *a):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = fn(*a)
        return out, buf.getvalue()

    say(f'# levers_v2.py summary · {time.strftime("%Y-%m-%d %H:%M:%S")} · 층1 시드 {SEEDS} · B fold seed {FOLDS}')
    sv = json.loads((R / 'v2' / 'summary.json').read_text(encoding='utf-8'))
    sb = json.loads((R / 'v2_B' / 'summary.json').read_text(encoding='utf-8'))
    base_single = float(np.mean([sv['seeds'][str(s)]['oof']['tot'] for s in SEEDS]))
    base_b42, base_bm = sb['seeds']['42']['oof']['tot'], b_mean('v2_B')
    say('\n## 요약 (holdout OOF · Δ = 레버 − 기준, 음수 = 나음 · 괄호는 개선 수/5 와 행 부트스트랩 95% CI)')
    say(f'  {"구성":10s} {"단일 5시드 평균":>34s} {"B fold 42":>18s} {"B fold seed 5개 평균":>40s}  위끝')
    say(f'  {"기준":10s} {base_single:34.5f} {base_b42:18.5f} {base_bm:40.5f}')
    res, texts = dict(base=dict(single=base_single, b42=base_b42, bmean=base_bm), arms={}), []
    for label, a_single, a_b in ARMS:
        if not (R / a_b / 'summary.json').exists():
            continue
        cs, ts = cap(lab.compare, 'v2', a_single, SEEDS)
        cb, tb = cap(lab.compare, 'v2_B', a_b, FOLDS)
        texts += [ts, tb]
        s_b = json.loads((R / a_b / 'summary.json').read_text(encoding='utf-8'))
        s_s = json.loads((R / a_single / 'summary.json').read_text(encoding='utf-8'))
        hit = any(v['grid_top_hit']['tie'] or v['grid_top_hit']['win'] for v in s_b['seeds'].values()) or \
            any(v.get('grid_top_hit', {}).get('tie') or v.get('grid_top_hit', {}).get('win') for v in s_s['seeds'].values())
        t1, t2 = cs['oof']['tot'], cb['oof']['tot']
        b42 = s_b['seeds']['42']['oof']['tot']
        say(f'  {label:10s} {t1["b"]:.5f} ({t1["mean"]:+.5f}, {t1["better"]}/5, [{t1["lo"]:+.5f},{t1["hi"]:+.5f}])'
            f' {b42:.5f} ({b42 - base_b42:+.5f}) {t2["b"]:.5f} ({t2["mean"]:+.5f}, {t2["better"]}/5, [{t2["lo"]:+.5f},{t2["hi"]:+.5f}])'
            f'  {"예" if hit else "아니오"}')
        res['arms'][label] = dict(single=t1, b=t2, b42=b42, b42_delta=b42 - base_b42, grid_top_hit=hit,
                                  single_arm=a_single, b_arm=a_b)
    say('\n  위끝 = λ 가 격자 위끝(1e4)에 닿아 자동 확장이 돌았는가(어느 fold·축이든). 상세 선택은 각 팔의 summary.json picked_*')
    say('\n## 상세 (lab.compare 원문 — B 팔의 seed 열은 fold seed)')
    for t in texts:
        for line in t.rstrip('\n').split('\n'):
            say(line)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    JSON.write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
    print(f'\n저장: {REPORT} · {JSON}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('l6')
    sub.add_parser('k').add_argument('--k', type=int, default=1024)
    sub.add_parser('combo').add_argument('--k', type=int, default=1024)
    sub.add_parser('auto')
    sub.add_parser('summary')
    a = ap.parse_args()
    print(f'# levers_v2.py {" ".join(sys.argv[1:])} · {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    if a.cmd == 'l6':
        run_l6()
        summary()
    elif a.cmd == 'k':
        ensure_base()
        run_k(a.k)
    elif a.cmd == 'combo':
        run_combo(a.k)
    elif a.cmd == 'auto':
        auto()
    else:
        summary()


if __name__ == '__main__':
    main()
