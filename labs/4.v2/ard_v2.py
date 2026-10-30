"""ard_v2.py — 결합기(층2) ARD 대 현행 2묶음 릿지(blk2). penlab·penfold 의 ard 를 v2 5시드에 다시 건다. CPU 만, 층1 재학습 없음.

기록  penlab (08-12 · 구 헤드 · clean 7,748 · 3시드) tie ard +0.00007 (1/3) · win +0.00013 — 노이즈
      penfold BR (08-21 · 최종 헤드 구조 · 5시드 · 양축 ard) +0.00051, 5/5 악화
이번에 바뀐 것  결합기 학습 행 10,748 (+39%) · 새 ABE1 (tie 로짓 공선성 비 22.1 → 17.6) · 기준선 = 현행 노트북 Layer2 (양축 blk2)

팔
  v2      기준선 = results/v2 그대로 (노트북 Layer2: [λ_L × 로짓 | λ_g × 층1 열], λ 격자 {0,1,…,1e4}+자동확장, 안쪽 3-fold,
          배포 = fold 선택의 성분별 중앙값)
  v2_ard  penfold BR 규칙 그대로 (penfold.py `_ard_fit` · `combiner(kind='ard')`):
          전 열을 균일 q0 로 시작해 pen_j = q0·(1/(w_j²+ε)) / 평균 으로 5회 재가중(희소 사전, ε=1e-8). q0 ∈ {0,1,10,…,1e4} 를
          안쪽 3-fold 손실로 고른다(큰 q0 부터, 안쪽 fold 마다 웜스타트). 배포 = fold 선택 q0 의 중앙값으로 전 행에서 재가중.
          양축 모두 — tie 절편 O, win 절편 X·tie 행 제외 (현행과 같은 축 계약).
같은 것  층1 열(g·w, results/v2/s<seed>.npz) · 바깥 fold(Layer2._outer_fold, SEED=시드) · 안쪽 fold 추첨 · 정규화(zstats) · 최적화기
         (Layer2._fit_one = penfold `_fit` 과 같은 식·설정). 다른 것은 벌점 규칙 하나뿐이다.
검산  기준선을 같은 자리에서 다시 적합해 저장된 v2 OOF z 와 대조한다 (BLAS 스레드 수가 달라 1e-9 안의 차이는 허용).

출력  results/v2_ard/ (lab 형식 summary.json · s<seed>.npz) · results/_compare/ard_v2_report.txt · lab.compare 의 판정 json
판정  lab.compare v2 → v2_ard — 시드 42·1·2 (규칙: Δ ≤ −0.0005 · CI 0 배제 · 3/3) 와 5시드. 음수 Δ = ARD 가 나음.
실행  cd labs/4.v2 && python ard_v2.py      (시드 5개 병렬, 1분 안쪽)
"""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'          # CPU 만
os.environ['HIP_VISIBLE_DEVICES'] = '-1'
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '2')                 # 작은 행렬 — BLAS 스레드가 많으면 오히려 수십 배 느리다
sys.dont_write_bytecode = True
import io
import json
import time
import contextlib
import multiprocessing as mp
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import lab                                          # noqa: E402
N = lab.N

SEEDS = [42, 1, 2, 3, 4]
GRID = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]     # penfold.py:57
ARD_IT, EPS = 5, 1e-8                               # penfold.py:98 — penlab 'ard' 규약
BASE, ARM = 'v2', 'v2_ard'
TIE_NAMES = ['s_tb2', 's_tb1', 't_abe2', 't_abe1', 't_ar', 'g1', 'g2']
WIN_NAMES = ['m_abe2', 'm_abe1', 'm_ar', 'W_abe2', 'W_tb2', 'W_abe1', 'W_tb1']
REPORT = lab.RESULTS / '_compare' / 'ard_v2_report.txt'
LINES = []
_S2 = {}                                            # fork 로 자식에 넘기는 stage2 (풀링 제외)


def say(text=''):
    print(text, flush=True)
    LINES.append(text)


def ard_fit(A, y, q0, bias, w0=None):
    """penfold `_ard_fit`: 균일 q0 에서 시작해 pen_j ∝ q0 / w_j² 로 5회 재가중 (평균이 q0 가 되게 정규화)."""
    p = A.shape[1]
    pen = np.full(p, float(q0))
    v = w0
    for _ in range(ARD_IT):
        v, _ = N.Layer2._fit_one(A, y, pen, bias, v)
        inv = 1.0 / (v[:p] ** 2 + EPS)
        pen = q0 * inv / inv.mean()
    return v, pen


def ard_combiner(X, y01, fold, rows, even):
    """penfold `combiner(kind='ard')` 를 노트북 Layer2 의 연산자로 옮긴 것 — 바깥 5-fold 교차적합 + 안쪽 3-fold q0 선택."""
    z = np.full(len(y01), np.nan)
    q0s, pens = [], []
    for k in range(N.Layer2.K):
        tr, te = (fold != k) & rows, (fold == k) & rows
        Xtr, ytr = X[tr], y01[tr]
        inner = np.zeros(len(ytr), int)
        for j, (_, iv) in enumerate(StratifiedKFold(N.Layer2.INNER_K, shuffle=True, random_state=N.SEED)
                                    .split(np.zeros((len(ytr), 1)), ytr)):
            inner[iv] = j
        Zi = [N.Layer2._z(Xtr[inner != j], Xtr[inner == j], even) for j in range(N.Layer2.INNER_K)]
        warm, cache = [None] * N.Layer2.INNER_K, {}

        def loss(q0):
            if q0 not in cache:
                tot = 0.0
                for j in range(N.Layer2.INNER_K):
                    A, B = Zi[j]
                    v, _ = ard_fit(A, ytr[inner != j], q0, even, warm[j])
                    warm[j] = v
                    tot += float(N.rowloss(N.Layer2._score(v, B, even), ytr[inner == j]).sum())
                cache[q0] = tot
            return cache[q0]

        q0 = min(sorted(GRID, reverse=True), key=loss)
        A, B = N.Layer2._z(Xtr, X[te], even)
        v, pen = ard_fit(A, ytr, q0, even)
        z[te] = N.Layer2._score(v, B, even)
        q0s.append(float(q0))
        pens.append(pen)
    mu, sc = N.zstats(X[rows], even)
    q_dep = float(np.median(q0s))
    v, pen_dep = ard_fit((X[rows] - mu) / sc, y01[rows], q_dep, even)
    z_dep = N.Layer2._score(v, (X - mu) / sc, even)
    return z, z_dep, dict(q0=q0s, q0_dep=q_dep, pen_dep=pen_dep, pens=pens)


def _one(sd):
    S2 = _S2['v2']
    Z = dict(np.load(lab.RESULTS / BASE / f's{sd}.npz'))
    assert (Z['ids'] == S2['ids']).all(), f'seed {sd}: saved rows != stage2 rows!!'
    g, w, y = Z['g'], Z['w'], S2['y']
    N.SEED = sd
    t0 = time.time()
    L2 = N.Layer2.fit(S2, g, w)                                       # 검산 — 저장된 기준선 재현
    rc = (float(np.nanmax(np.abs(L2.z_tie - Z['zt_oof']))), float(np.nanmax(np.abs(L2.z_win - Z['zw_oof']))))
    fold = N.Layer2._outer_fold(y)
    zt, zt_d, it = ard_combiner(np.hstack([S2['tie_base'], g]), (y == 2).astype(np.float64), fold,
                                np.ones(len(y), bool), True)
    zw, zw_d, iw = ard_combiner(np.hstack([S2['win_base'], w]), (y == 0).astype(np.float64), fold, S2['nz'], False)
    return dict(sd=sd, zt=zt, zw=zw, zt_d=zt_d, zw_d=zw_d, it=it, iw=iw, rc=rc, sec=round(time.time() - t0, 1))


def captured(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a)
    for line in buf.getvalue().rstrip('\n').split('\n'):
        say(line)
    return out


def shape(pen):
    """배포 벌점 벡터의 모양 — 평균 1 로 나눈 상대값 (q0 = 0 이면 전부 0)."""
    m = float(np.mean(pen))
    return '[' + ' '.join(f'{x / m:7.2g}' if m > 0 else f'{0:7.2g}' for x in pen) + ']'


def main():
    t_start = time.time()
    say(f'# ard_v2.py · {time.strftime("%Y-%m-%d %H:%M:%S")} · seeds {SEEDS} · q0 grid {GRID} · 재가중 {ARD_IT}회')
    s_base = json.loads((lab.RESULTS / BASE / 'summary.json').read_text(encoding='utf-8'))
    miss = [sd for sd in SEEDS if str(sd) not in s_base['seeds']]
    assert not miss, f'{BASE}: seeds {miss} are missing!!'
    N.use(lab.bundle_path('v2'))
    S2 = N.build('stage2')
    S2.pop('pool')                                                    # 층2 는 풀링을 안 쓴다
    _S2['v2'] = S2
    with mp.get_context('fork').Pool(len(SEEDS)) as pool:
        out = {r['sd']: r for r in pool.map(_one, SEEDS)}

    d = lab.RESULTS / ARM
    d.mkdir(parents=True, exist_ok=True)
    summ = dict(config=dict(base=BASE, rule='penfold BR (_ard_fit, combiner kind=ard)', grid=GRID, ard_it=ARD_IT, eps=EPS,
                            layer1=f'results/{BASE}/s<seed>.npz', bundle=s_base['config']['bundle'], made_by='ard_v2.py'),
                seeds={})
    say('\n[검산] 기준선(현행 Layer2) 재적합 vs 저장된 v2 OOF z — max|Δ| tie / win')
    for sd in SEEDS:
        r = out[sd]
        say(f'  seed {sd:>2d}: {r["rc"][0]:.1e} / {r["rc"][1]:.1e}')
        assert max(r['rc']) < 1e-9, f'seed {sd}: the baseline refit does not reproduce results/{BASE}!!'
        Zb = np.load(lab.RESULTS / BASE / f's{sd}.npz')
        arr = dict(ids=Zb['ids'], y=Zb['y'], hold=Zb['hold'], zt_oof=r['zt'], zw_oof=r['zw'],
                   zt_dep=r['zt_d'], zw_dep=r['zw_d'], g=Zb['g'], w=Zb['w'])
        np.savez(d / f's{sd}.npz', **arr)
        y, nz, hold = arr['y'], arr['y'] != 2, arr['hold']
        summ['seeds'][str(sd)] = dict(oof=lab.scores(y, nz, r['zt'], r['zw'], hold),
                                      dep=lab.scores(y, nz, r['zt_d'], r['zw_d'], hold),
                                      q0_tie=r['it']['q0'], q0_win=r['iw']['q0'],
                                      q0_dep=[r['it']['q0_dep'], r['iw']['q0_dep']],
                                      pen_dep_tie=r['it']['pen_dep'].tolist(), pen_dep_win=r['iw']['pen_dep'].tolist(),
                                      sec=r['sec'])
    (d / 'summary.json').write_text(json.dumps(summ, indent=1), encoding='utf-8')

    say('\n[ARD 가 고른 것] fold 별 q0 (바깥 5 fold) → 배포 q0 · 배포 열별 벌점의 상대 모양(평균 = 1)')
    say(f'  tie 열 {TIE_NAMES}')
    for sd in SEEDS:
        s = summ['seeds'][str(sd)]
        say(f'  seed {sd:>2d} tie q0 {s["q0_tie"]} → {s["q0_dep"][0]:g} · 모양 {shape(s["pen_dep_tie"])}')
    say(f'  win 열 {WIN_NAMES}')
    for sd in SEEDS:
        s = summ['seeds'][str(sd)]
        say(f'  seed {sd:>2d} win q0 {s["q0_win"]} → {s["q0_dep"][1]:g} · 모양 {shape(s["pen_dep_win"])}')

    say('\n[판정] 기준선 v2(현행 blk2) → v2_ard')
    captured(lab.compare, BASE, ARM, [42, 1, 2])
    captured(lab.compare, BASE, ARM, SEEDS)
    say(f'\n총 {time.time() - t_start:.0f}초 · 시드별 {[(sd, out[sd]["sec"]) for sd in SEEDS]}')
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(LINES) + '\n', encoding='utf-8')
    print(f'저장: {REPORT} · {d}/')


if __name__ == '__main__':
    main()
