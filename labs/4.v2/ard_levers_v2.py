"""ard_levers_v2.py — K1024(채택) 위에 결합기 ARD 를 얹으면? 그리고 K2048 에서 릿지가 헤드 열을 통째로 끈 문제를 ARD 가 푸는가?

가설 (사용자, 2026-10-28)
    현행 2묶음 릿지는 층1 헤드 열 g1·g2 가 λ_g 하나를 공유해 함께 줄거나 함께 꺼진다 — K2048 단일 시드에서 λ_g 가 1e8~1e9 로
    자동 확장돼 두 열이 통째로 꺼졌다. ARD 는 열마다 벌점이 따로라 나쁜 열 하나만 끄고 나머지를 살릴 수 있다.
    (앞의 ARD 5회는 7열이 전부 쓸모 있는 열이던 자리에서 희소 사전이 공선 쌍둥이를 지워 졌다 — 나쁜 열이 섞인 자리는 처음이다.)
규칙
    ARD = ard_v2.py 와 같은 penfold BR 의 1/w² 재가중(5회, q0 ∈ {0,1,…,1e4} 안쪽 3-fold, 배포 = 중앙값 q0) — 단 **tie 축에만** 건다.
    K 는 tie 헤드만 바꾸므로 승부 축은 같은 구성의 현행 blk2 팔의 z 를 그대로 쓴다 → Δ 는 전부 tie 축 변화다.
팔 — 층1 열은 저장된 것(재학습 없음), 결합기만 다시 적합
    K1024 · K1024+L6 · K2048 각각 단일(층1 시드 42·1·2·3·4, fold seed = 층1 시드)과 B(층1 열 5시드 평균, fold seed 42·1·2·3·4)
    비교: 같은 구성의 blk2 팔 → ARD 팔. 그리고 K2048+ARD 를 K1024 blk2 와도 비교(구조 대 구조).
출력  results/<blk2 팔>_ard/ · results/_compare/ard_levers_v2_report.txt
실행  cd labs/4.v2 && python -u ard_levers_v2.py      (CPU 2분 안쪽)
"""
import sys
sys.dont_write_bytecode = True
import io
import json
import time
import contextlib
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import levers_v2 as LV                              # noqa: E402  stage2 · d 열 · 층1 열 읽기
import ard_v2 as AV                                 # noqa: E402  ard_combiner (penfold BR 를 노트북 Layer2 연산자로)
lab, N = LV.lab, LV.N

R = lab.RESULTS
SEEDS, FOLDS = LV.SEEDS, LV.FOLDS
NAMES5, NAMES_D, NAMES_G = ['s_tb2', 's_tb1', 't_abe2', 't_abe1', 't_ar'], ['d_tb2', 'd_tb1', 'd_abe2', 'd_abe1'], ['g1', 'g2']
CONF = [  # 이름, 층1 열 출처, d 사용, blk2 단일 팔, blk2 B 팔
    ('K1024', 'v2_K1024', False, 'v2_K1024', 'v2_B_K1024'),
    ('K1024+L6', 'v2_K1024', True, 'v2_K1024_L6', 'v2_B_K1024_L6'),
    ('K2048', 'v2_K2048', False, 'v2_K2048', 'v2_B_K2048'),
]
REPORT = R / '_compare' / 'ard_levers_v2_report.txt'
LINES = []


def say(t=''):
    print(t, flush=True)
    LINES.append(t)


def ard_tie(gc, fs):
    S = LV.stage2()
    y = S['y']
    N.SEED = fs                                     # 바깥·안쪽 fold = blk2 팔과 같은 추첨
    fold = N.Layer2._outer_fold(y)
    with threadpool_limits(limits=2):
        return AV.ard_combiner(np.hstack([S['tie_base'], gc]), (y == 2).astype(np.float64), fold,
                               np.ones(len(y), bool), True)


def make_arm(name, items, base_arm, config):
    """items = [(key, tie 층2 의 λ_g 쪽 열, fold seed)] — 승부 z 는 base_arm 의 같은 key 에서 가져온다."""
    S = LV.stage2()
    d = R / name
    d.mkdir(parents=True, exist_ok=True)
    summ = dict(config=config, seeds={})
    for key, gc, fs in items:
        b = np.load(R / base_arm / f's{key}.npz')
        assert (b['ids'] == S['ids']).all(), f'{base_arm} s{key}: rows != stage2!!'
        zt, zt_d, info = ard_tie(gc, fs)
        arr = dict(ids=S['ids'], y=S['y'], hold=S['hold'], zt_oof=zt, zw_oof=b['zw_oof'], zt_dep=zt_d,
                   zw_dep=b['zw_dep'], g=gc, w=b['w'])
        np.savez(d / f's{key}.npz', **arr)
        o = lab.scores(S['y'], S['nz'], zt, b['zw_oof'], S['hold'])
        dp = lab.scores(S['y'], S['nz'], zt_d, b['zw_dep'], S['hold'])
        summ['seeds'][str(key)] = dict(oof=o, dep=dp, fold_seed=fs, q0=info['q0'], q0_dep=info['q0_dep'],
                                       pen_dep=info['pen_dep'].tolist())
    (d / 'summary.json').write_text(json.dumps(summ, indent=1), encoding='utf-8')
    return summ


def shape(pen):
    m = float(np.mean(pen))
    return ' '.join(f'{(x / m if m > 0 else 0):6.2g}' for x in pen)


def cap(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a)
    return out, buf.getvalue()


def main():
    t0 = time.time()
    say(f'# ard_levers_v2.py · {time.strftime("%Y-%m-%d %H:%M:%S")} · tie 축 ARD (penfold BR) · 층1 시드 {SEEDS} · B fold seed {FOLDS}')
    rows, texts = [], []
    for label, src, use_d, a_single, a_b in CONF:
        C = LV.cols(src)
        dd = LV.disagree() if use_d else None
        names = NAMES5 + (NAMES_D if use_d else []) + NAMES_G
        g_avg = np.mean([C[s][0] for s in SEEDS], 0)
        gc_b = np.hstack([dd, g_avg]) if use_d else g_avg
        n_s, n_b = f'{a_single}_ard', f'{a_b}_ard'
        ss = make_arm(n_s, [(s, np.hstack([dd, C[s][0]]) if use_d else C[s][0], s) for s in SEEDS], a_single,
                      dict(kind='single', layer1=src, L6=use_d, combiner='tie ARD (penfold BR) · win = ' + a_single))
        sb = make_arm(n_b, [(fs, gc_b, fs) for fs in FOLDS], a_b,
                      dict(kind='B', layer1=src, L6=use_d, combiner='tie ARD (penfold BR) · win = ' + a_b))
        say(f'\n## {label} — ARD 가 고른 것 (tie 열: {" ".join(names)})')
        for kind, s_ in (('단일', ss), ('B', sb)):
            for key, v in s_['seeds'].items():
                say(f'  {kind} {key:>2s}: q0 fold {[int(q) for q in v["q0"]]} → {v["q0_dep"]:g} · 배포 벌점 모양 [{shape(v["pen_dep"])}]'
                    f' · OOF {v["oof"]["tot"]:.5f}')
        c1, t1 = cap(lab.compare, a_single, n_s, SEEDS)
        c2, t2 = cap(lab.compare, a_b, n_b, FOLDS)
        texts += [t1, t2]
        rows.append((label, 'blk2 → ARD', c1['oof']['tot'], c2['oof']['tot']))
        if label == 'K2048':
            c3, t3 = cap(lab.compare, 'v2_K1024', n_s, SEEDS)
            c4, t4 = cap(lab.compare, 'v2_B_K1024', n_b, FOLDS)
            texts += [t3, t4]
            rows.append(('K2048+ARD', 'K1024 blk2 → K2048 ARD', c3['oof']['tot'], c4['oof']['tot']))

    say('\n## 요약 (holdout OOF · Δ 음수 = 뒤쪽이 나음 · 괄호 = 개선 수/5 · 95% CI)')
    say(f'  {"구성":10s} {"비교":24s} {"단일 5시드 평균":>48s} {"B fold seed 5개 평균":>48s}')
    for label, what, a, b in rows:
        say(f'  {label:10s} {what:24s} {a["a"]:.5f} → {a["b"]:.5f} ({a["mean"]:+.5f}, {a["better"]}/5, [{a["lo"]:+.5f},{a["hi"]:+.5f}])'
            f'   {b["a"]:.5f} → {b["b"]:.5f} ({b["mean"]:+.5f}, {b["better"]}/5, [{b["lo"]:+.5f},{b["hi"]:+.5f}])')
    say('\n## 상세 (lab.compare 원문 — B 팔의 seed 열은 fold seed)')
    for t in texts:
        for line in t.rstrip('\n').split('\n'):
            say(line)
    say(f'\n총 {time.time() - t0:.0f}초')
    REPORT.write_text('\n'.join(LINES) + '\n', encoding='utf-8')
    print(f'저장: {REPORT}', flush=True)


if __name__ == '__main__':
    main()
