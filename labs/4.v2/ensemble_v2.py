"""ensemble_v2.py — 시드 앙상블 두 판(A 출력 z 평균 · B 층1 열 평균)을 5시드로 재고 기록한다. CPU 만 쓴다.

A 출력 z 평균 (lever.md L1 의 측정판)
    시드마다 층1+층2 를 통째로 학습한 결과(results/<arm>/s<seed>.npz)의 z_tie·z_win 을 시드 평균한다. 추가 적합 없음.
    OOF = 시드별 OOF z 의 평균 · 배포적합 = 시드별 배포 z 의 평균. 배포 = 파이프라인 5벌.
B 층1 열 평균 (lever.md 가 배포용으로 권한 판 · 구 colavg.py (c))
    층1 열 g(2)·w(4) 를 시드 평균한 뒤 층2 를 한 번 교차적합한다 = 평균 뒤 보정(pool-then-calibrate). 배포 = 층1 5벌 + 층2 1벌.
    본 수치는 배포 계약의 fold seed 42. 층2 의 fold 추첨 잡음(v1 기록 ±0.0004) 폭을 보려고 fold seed 1·2·3·4 도 적합한다.
검산  v1 기준선(results/v1_sweep)으로 B 를 fold seed 42 에서 다시 적합해 colavg.py 기록(OOF 0.95709 · 배포적합 0.95487)과,
      A 를 seed_sweep 기록(OOF 0.95705 · 배포적합 0.95533)과 대조한다 — 다르면 멈춘다.
비교  행 부트스트랩 95% CI(B=10,000)는 holdout 5,748행 paired. Δ 음수 = 뒤 판이 나음.

출력 (덮어씀)
    results/_compare/ensemble_v2_report.txt · results/_compare/ensemble_v2.json
    results/_ens/<arm>__outavg.npz · <arm>__colavg_f<fs>.npz  — z(OOF·배포)와 평균 열, B 는 배포 계수까지 (번들 재료)
실행: cd labs/4.v2 && python ensemble_v2.py      (층2 적합 6개 병렬, 약 5분)
"""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'          # CPU 만
os.environ['HIP_VISIBLE_DEVICES'] = '-1'
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '2')                 # 층2 적합 6개를 병렬로 돌리므로 프로세스당 BLAS 스레드를 줄인다
sys.dont_write_bytecode = True
import json
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import lab                                          # noqa: E402
N = lab.N

SEEDS = [42, 1, 2, 3, 4]
FOLD_SEEDS = [42, 1, 2, 3, 4]
ARMS = {'v2': 'v2', 'v1_sweep': 'v1'}               # 결과 폴더 → 번들
REC = {'v1_sweep': dict(outavg=(0.95705, 0.95533), colavg_f42=(0.95709, 0.95487))}   # log_sweep_K512 · log_colavg
OUT = lab.RESULTS / '_compare'
ENS = lab.RESULTS / '_ens'
LINES = []
_G = {}                                             # fork 로 자식 프로세스에 넘길 (S2, g 평균, w 평균)


def say(text=''):
    print(text, flush=True)
    LINES.append(text)


def load_seeds(arm):
    s = json.loads((lab.RESULTS / arm / 'summary.json').read_text(encoding='utf-8'))
    miss = [sd for sd in SEEDS if str(sd) not in s['seeds'] or not (lab.RESULTS / arm / f's{sd}.npz').exists()]
    assert not miss, f'{arm}: seeds {miss} are not finished yet — wait for `lab.py run` to end!!'
    Z = {sd: dict(np.load(lab.RESULTS / arm / f's{sd}.npz')) for sd in SEEDS}
    for sd in SEEDS:
        assert (Z[sd]['ids'] == Z[SEEDS[0]]['ids']).all(), f'{arm} seed {sd}: row order differs!!'
    return Z, s


def tot3(d, kind='oof'):
    return lab.scores(d['y'], d['y'] != 2, d[f'zt_{kind}'], d[f'zw_{kind}'], d['hold'])


def rows(d, ids0=None):
    ids, y, r = lab.holdout_rows(d, 'oof')
    if ids0 is None:
        return ids, r
    ix = pd.Index(ids).get_indexer(ids0)
    assert (ix >= 0).all() and len(ix) == len(ids), 'holdout rows disagree!!'
    return ids0, {k: v[ix] for k, v in r.items()}


def paired(label, a, b):
    """a → b 의 holdout OOF paired Δ (tot·tie·win) + 행 부트스트랩 95% CI."""
    ids0, ra = rows(a)
    _, rb = rows(b, ids0)
    out = {}
    for k in ('tot', 'tie', 'win'):
        d = rb[k] - ra[k]
        lo, hi = lab.boot(d)
        out[k] = dict(mean=float(np.nanmean(d)), lo=lo, hi=hi)
    t = out['tot']
    say(f'  {label:44s} Δ {t["mean"]:+.5f} [{t["lo"]:+.5f},{t["hi"]:+.5f}]  CI {"0 배제" if t["lo"] > 0 or t["hi"] < 0 else "0 포함"}'
        f'  (tie {out["tie"]["mean"]:+.5f} · win {out["win"]["mean"]:+.5f})')
    return out


def _fit_b(task):
    arm, fs = task
    S2, g, w = _G[arm]
    N.SEED = fs                                     # Layer2 의 바깥·안쪽 fold 추첨이 이 값을 읽는다
    t0 = time.time()
    L2 = N.Layer2.fit(S2, g, w)
    zt_d, zw_d = L2.transform(S2['tie_base'], g, S2['win_base'], w)
    return dict(arm=arm, fs=fs, zt_oof=L2.z_tie, zw_oof=L2.z_win, zt_dep=zt_d, zw_dep=zw_d,
                dep_t=L2.dep_t, dep_w=L2.dep_w, sec=round(time.time() - t0, 1))


def main():
    t_start = time.time()
    say(f'# ensemble_v2.py · {time.strftime("%Y-%m-%d %H:%M:%S")} · seeds {SEEDS} · fold seeds(B) {FOLD_SEEDS}')
    Z, SUMM, S2 = {}, {}, {}
    for arm, b in ARMS.items():
        Z[arm], SUMM[arm] = load_seeds(arm)
        N.use(lab.bundle_path(b))
        S2[arm] = N.build('stage2')
        S2[arm].pop('pool')                         # 층2 는 풀링을 안 쓴다 — 병렬 자식 프로세스의 메모리를 줄인다
        assert (S2[arm]['ids'] == Z[arm][SEEDS[0]]['ids']).all(), f'{arm}: stage2 rows != saved rows!!'
        g = np.mean([Z[arm][sd]['g'] for sd in SEEDS], 0)
        w = np.mean([Z[arm][sd]['w'] for sd in SEEDS], 0)
        _G[arm] = (S2[arm], g, w)

    # B — 층2 적합을 병렬로 (v2 fold seed 5개 + v1 검산 fold seed 42)
    tasks = [('v2', fs) for fs in FOLD_SEEDS] + [('v1_sweep', 42)]
    with mp.get_context('fork').Pool(len(tasks)) as pool:
        fitted = pool.map(_fit_b, tasks)
    B = {(r['arm'], r['fs']): r for r in fitted}

    ENS.mkdir(parents=True, exist_ok=True)
    res = dict(seeds=SEEDS, fold_seeds=FOLD_SEEDS, arms={})
    for arm in ARMS:
        base = {k: Z[arm][SEEDS[0]][k] for k in ('ids', 'y', 'hold')}
        A = dict(base, **{f'{z}_{k}': np.mean([Z[arm][sd][f'{z}_{k}'] for sd in SEEDS], 0)
                          for z in ('zt', 'zw') for k in ('oof', 'dep')})
        np.savez(ENS / f'{arm}__outavg.npz', **A)
        single = [SUMM[arm]['seeds'][str(sd)] for sd in SEEDS]
        r = dict(single_oof=float(np.mean([x['oof']['tot'] for x in single])),
                 single_dep=float(np.mean([x['dep']['tot'] for x in single])),
                 outavg=dict(oof=tot3(A, 'oof'), dep=tot3(A, 'dep')), colavg={})
        for (a_, fs), x in sorted(B.items()):
            if a_ != arm:
                continue
            d = dict(base, zt_oof=x['zt_oof'], zw_oof=x['zw_oof'], zt_dep=x['zt_dep'], zw_dep=x['zw_dep'])
            np.savez(ENS / f'{arm}__colavg_f{fs}.npz', **d, g=_G[arm][1], w=_G[arm][2],
                     **{f't_{k}': np.asarray(x['dep_t'][k]) for k in N.Layer2.KEYS},
                     **{f'w_{k}': np.asarray(x['dep_w'][k]) for k in N.Layer2.KEYS})
            r['colavg'][str(fs)] = dict(oof=tot3(d, 'oof'), dep=tot3(d, 'dep'), lam_tie=list(x['dep_t']['lam']),
                                        lam_win=list(x['dep_w']['lam']), sec=x['sec'])
            r.setdefault('_rows', {})[f'colavg_f{fs}'] = d
        r['_rows']['outavg'] = A
        res['arms'][arm] = r

    # 검산 — 기록과 다르면 멈춘다
    v1 = res['arms']['v1_sweep']
    got = dict(outavg=(v1['outavg']['oof']['tot'], v1['outavg']['dep']['tot']),
               colavg_f42=(v1['colavg']['42']['oof']['tot'], v1['colavg']['42']['dep']['tot']))
    for k, (o, d) in REC['v1_sweep'].items():
        assert abs(got[k][0] - o) <= 5.01e-6 and abs(got[k][1] - d) <= 5.01e-6, \
            f'v1 {k} {got[k]} != record {(o, d)} (records are rounded to 5 decimals)!!'
    say(f'[검산] v1 A {got["outavg"][0]:.5f}/{got["outavg"][1]:.5f} = seed_sweep 기록 · '
        f'v1 B(fold 42) {got["colavg_f42"][0]:.5f}/{got["colavg_f42"][1]:.5f} = colavg 기록')

    for arm in ('v1_sweep', 'v2'):
        r = res['arms'][arm]
        say(f'\n## {arm} — 시드 {len(SEEDS)}개 (OOF = 판정 기준 · 배포적합 = in-sample)')
        say(f'  단일 시드 평균            OOF {r["single_oof"]:.5f}   배포적합 {r["single_dep"]:.5f}')
        a = r['outavg']
        say(f'  A 출력 z 평균             OOF {a["oof"]["tot"]:.5f} (tie {a["oof"]["tie"]:.5f} win {a["oof"]["win"]:.5f})'
            f'   배포적합 {a["dep"]["tot"]:.5f}   Δ vs 단일 {a["oof"]["tot"] - r["single_oof"]:+.5f}')
        for fs, c in r['colavg'].items():
            say(f'  B 층1 열 평균 (fold {fs:>2s})   OOF {c["oof"]["tot"]:.5f} (tie {c["oof"]["tie"]:.5f} win {c["oof"]["win"]:.5f})'
                f'   배포적합 {c["dep"]["tot"]:.5f}   Δ vs 단일 {c["oof"]["tot"] - r["single_oof"]:+.5f}'
                f'   λ tie {tuple(c["lam_tie"])} win {tuple(c["lam_win"])}')
        if len(r['colavg']) > 1:
            o = [c['oof']['tot'] for c in r['colavg'].values()]
            say(f'  B fold seed {len(o)}개           OOF 평균 {np.mean(o):.5f} (범위 {min(o):.5f}~{max(o):.5f}, 폭 {max(o) - min(o):.5f})')

    say('\n## paired 비교 (holdout OOF, 행 부트스트랩 95% CI)')
    R = {arm: res['arms'][arm].pop('_rows') for arm in ARMS}
    res['paired'] = {
        'v2: A → B(fold 42)': paired('v2: A 출력 평균 → B 열 평균(fold 42)', R['v2']['outavg'], R['v2']['colavg_f42']),
        'v1→v2 A': paired('A 출력 평균: v1 → v2', R['v1_sweep']['outavg'], R['v2']['outavg']),
        'v1→v2 B(fold 42)': paired('B 열 평균(fold 42): v1 → v2', R['v1_sweep']['colavg_f42'], R['v2']['colavg_f42']),
    }
    say(f'\n총 {time.time() - t_start:.0f}초 · 층2 적합 {[(a, fs, B[(a, fs)]["sec"]) for a, fs in tasks]}')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'ensemble_v2.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
    (OUT / 'ensemble_v2_report.txt').write_text('\n'.join(LINES) + '\n', encoding='utf-8')
    print(f'저장: {OUT / "ensemble_v2_report.txt"} · {OUT / "ensemble_v2.json"} · {ENS}/')


if __name__ == '__main__':
    main()
