"""finlab.py — clean7k OOF 판정장: 5실험 (λ오라클 · 게이트온도 · LOO · 마이닝 · 부스터).

우주 계약 (모든 실험 공통)
    clean7k 고정 — stage2 에서 오염 3,000 을 뺀 7,748행. holdout 5,748 채점.
    λ 동결 기본값 tie [100,10] / win 10 (lam_ceiling 실측: 오라클=배포, 간격 0).
    fold 추첨은 사전 등록 시드 42..(42+reps−1) 로 반복 — 모든 Δ 는 같은 추첨의
    베이스라인과 paired 다. 채택선: 전 추첨 같은 부호 & 평균 Δ ≤ −1e-4.

실험 1 — λ 오라클: 미세 격자 + 열별(가변) 벌점
    총 ll = tie ll + R_NZ × win ll 로 **정확히** 분해되고 벌점이 축을 넘는 경로가 없으므로,
    축별 오라클의 합이 곧 전체-ll 오라클이다. 구현은 축별 병렬 탐색.
    1a 미세 격자 (그룹 λ 그대로, 간격만 촘촘히) — 추첨 42 천장 + 최적점의 전 추첨 안정성
    1b 열별 **무제한** 개별 벌점 (0 ~ 1e6 격자 + 로그축 연속 정밀화 · m3 무벌점 계약 해제)
       — λ=0 = 그대로 쓴다 · λ≥1e5 = 사실상 제외. 이종 전문가는 빼지 않고 릿지가 정한다
    ! 오라클은 holdout 으로 골랐다 — **상한 측정**이다. 채택하려면 안쪽 CV 정직 선택이
      같은 방향을 가리켜야 한다.

실험 2 — 불일치 게이트 온도
    z' = exp(a + b·(d − d̄)) · z,  d = 표준화 전문가 로짓의 행별 표준편차(불일치).
    d 는 A/B 교환에 짝(even)이라 승부축 홀함수성을 깨지 않는다 (합법 온도).
    oracle(holdout 적합) = 천장 / honest(비-holdout clean 2,000 적합) = 채택 가능치.

실험 a — 전문가 한계 기여 (결합기 수준 LOO)
    전문가의 로짓 열 + (승부축) 소스별 W 열을 빼고 재적합 → holdout Δ.
    ! tie 헤드 열 g1·g2 안의 풀링 기여는 남는다 — 측정치는 완전 제거 손실의 **하한**.
      하한이 이미 크면 그 전문가는 확실히 필요하고, ~0 이면 완전 제거(헤드 재학습) 후보다.

실험 b — 손실 집중 마이닝 (추첨 42 · 분석 전용)
    holdout 행손실 상위 1% 의 손실 점유율과 그 행들의 관측 특성(라벨·길이·전문가
    불일치·TTA fwd/swp 격차)을 나머지와 대조 — 남은 정확도가 사는 곳의 지도.

실험 c — 부스터 (계약 청정형만 — 1전문가 1모델)
    로짓 셀: t_ar/m_ar 를 margins/{mix,sub}_{lgbm,cat}.npz 마진으로 **대체** (열 수 불변)
    풀링 셀: **풀링추가 xgb(자기)** — 기존 checkpoint-ab224 의 궤적 (u,v) 132 를 승부축
             5번째 GATE 체인으로. 모델 추가 0 (인코더가 로짓+풀링 내는 것과 동형)
    both 셀: arith→{lgbm,cat} **통째 대체** — 같은 1모델이 로짓+궤적 풀링을 다 낸다
    (구 pool_lgbm 류 '타 모델 풀링만 추가'는 xgb+lgbm 2모델 앙상블 = 계약 위반 — 제거.
     tie축 풀링 주입은 trajlab 실측 기각 +0.0008~10 — 재시도하지 않는다)

전제 자산 (없으면 해당 실험만 건너뛴다 · 헤드 재학습 없음 — 풀링 셀의 소형 GATE 제외)
    temp/dest_universe.npz         Gt_s42/W_s42        temp/dest_universe_arith.npz  L_ar(ab224)
    models/logits/*.npz            인코더 stage2 로짓   models/ARITH/checkpoint-ab224/oof_train.npz
    temp/arith/margins/{mix,sub}_{lgbm,cat}.npz         temp/arith/rep_traj_{xgb,lgbm,cat}224.npz

실험 e — 무라벨 공선성 (기하 전제 검사 · 라벨 불요)
    열정리의 전제는 통계가 아니라 기하다 — "잉여 열이 생존 열과 공선". 행집합(clean7k /
    holdout / 비-hold2k / train46k)을 바꿔도 VIF·R² 가 같으면 표본 특이하지 않다는 뜻이고,
    **제출 시점에 test 행에서 그대로 재확인**할 수 있다 (라벨이 필요 없으므로).

채택 관문 두 개 (Collect 가 함께 출력)
    추첨 잡음 — 5추첨 부호 일치 & 평균 Δ ≤ −1e-4
    표본 잡음 — 추첨평균 행별 Δ 의 홀드아웃 paired t ≤ −1.96
    ! 5추첨 부호 일치는 표본 잡음을 전혀 막지 못한다 (다섯 추첨이 같은 5,748행을 공유).

실행:  conda activate ML2 && cd final && python temp/finlab.py            # 전부 (~15–25분, GPU)
       python temp/finlab.py --exps 1,2,a,b --reps 5                      # 부분 실행
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold, train_test_split

FINAL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FINAL / 'temp'))
import destnation as D                      # noqa: E402  — 결합기/헤드 기계를 그대로 빌린다

DATA_DIR, MODELS = FINAL / 'origin', FINAL / 'models'
CKPT_AB224 = MODELS / 'ARITH/checkpoint-ab224'
ADIR = FINAL / 'temp/arith'
SEED, HALF, K = 42, 5000, 5
ABE1_STEP, TB1_STEP = 3109, 5000
R_NZ = 3972 / 5748
# plan.md §3 「확정 구조 목표」 + arith=ab224 (temp/arith/rung_ab224.json · 시드 42 실측)
PLAN_TIE, PLAN_WIN = 0.5683157036097410, 0.5638882907650488
PLAN_8SEED = 0.95809                       # plan.md 8시드 평균
FROZEN_TIE, FROZEN_WIN = (100.0, 10.0), 10.0
GATE_D, GATE_SIGN = 1e-4, None              # 채택선: 평균 Δ ≤ −1e-4 & 전 추첨 같은 부호

TIE_NAMES = ['s_tb2', 's_tb1', 't_abe2', 't_abe1', 't_ar', 'g1', 'g2']
WIN_NAMES = ['m_abe2', 'm_abe1', 'm_ar', 'W_abe2', 'W_tb2', 'W_abe1', 'W_tb1']
sd3 = lambda B: (B[:, 0] - B[:, 1], B[:, 2] - np.logaddexp(B[:, 0], B[:, 1]))
rowloss = D.rowloss


# ══════════════════════════════════════════ 자산
def build():
    """clean7k 우주 — 로짓 기저(ab224)·헤드 열·행 마스크. dest_universe 와 같은 원천."""
    za = np.load(MODELS / 'logits/abe2__checkpoint-3000.npz')
    ids = np.concatenate([za['valid_ids'], za['hold_ids']])
    y = np.concatenate([za['valid_labels'], za['hold_labels']]).astype(np.int64)
    n = len(y)
    assert n == 10748, f'stage2 {n}행'

    def tta(rel, nc):
        z = np.load(MODELS / rel)
        assert (np.concatenate([z['valid_ids'], z['hold_ids']]) == ids).all(), f'{rel} id'
        L = np.vstack([(z['valid_fwd'] + z['valid_swp']) / 2,
                       (z['hold_fwd'] + z['hold_swp']) / 2]).astype(np.float64)
        assert L.shape[1] == nc
        return L, z

    L_abe2 = ((np.vstack([za['valid_fwd'], za['hold_fwd']])
               + np.vstack([za['valid_swp'], za['hold_swp']])) / 2).astype(np.float64)
    L_tb2, _ = tta('logits/tb2__checkpoint-4000.npz', 2)
    L_abe1, _ = tta(f'logits/abe1__checkpoint-{ABE1_STEP}.npz', 3)
    L_tb1, ztb1 = tta(f'logits/tb1__checkpoint-{TB1_STEP}.npz', 2)

    zc = np.load(FINAL / 'temp/dest_universe_arith.npz', allow_pickle=False)
    assert (zc['ids'] == ids).all() and str(zc['ckpt']) == 'checkpoint-ab224', \
        'arith 캐시 불일치 — temp/dest_universe.py 를 먼저 돌려라'
    L_ar = zc['L_ar'].astype(np.float64)

    m2, t2 = sd3(L_abe2)
    m1, t1 = sd3(L_abe1)
    mr, tr = sd3(L_ar)
    tie_base = np.column_stack([L_tb2[:, 1] - L_tb2[:, 0], L_tb1[:, 1] - L_tb1[:, 0],
                                t2, t1, tr])
    win_base = np.column_stack([m2, m1, mr])

    zd = np.load(FINAL / 'temp/dest_universe.npz')
    Gt, W = zd['Gt_s42'].astype(np.float64), zd['W_s42'].astype(np.float64)

    full = pd.read_csv(DATA_DIR / 'train_full.csv',
                       usecols=['id', 'winner_model_a', 'winner_model_b'])
    lab = np.select([full['winner_model_a'] == 1, full['winner_model_b'] == 1], [0, 1], 2)
    tr51, _ = train_test_split(full.assign(label=lab), test_size=0.10, stratify=lab,
                               random_state=42, shuffle=True)
    _, va2 = train_test_split(tr51, test_size=2000, stratify=tr51['label'],
                              random_state=42, shuffle=True)
    cu = np.ones(n, bool)
    cu[:HALF] = pd.Index(va2['id']).get_indexer(ids[:HALF]) >= 0
    assert int(cu.sum()) == 7748
    hold, nz = np.arange(n) >= HALF, y != 2
    assert cu[hold].all()
    print(f'  clean7k {int(cu.sum()):,} · holdout {int(hold.sum()):,} · '
          f'비-holdout 보정셋 {int(cu.sum() - hold.sum()):,}')
    return dict(ids=ids, y=y, cu=cu, hold=hold, nz=nz, n=n,
                tie_base=tie_base, win_base=win_base, Gt=Gt, W=W,
                abe2_npz=za, tb1_npz=ztb1)


def fold_of(y_, fseed):
    f = np.zeros(len(y_), int)
    for k, (_, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=fseed)
                                .split(np.zeros((len(y_), 1)), y_)):
        f[te] = k
    return f


def oof_pen(X, y01, fold, rows, pen, even):
    """벌점 벡터를 전 fold 에 고정한 교차적합 OOF. even=tie(절편·짝 표준화)."""
    n_even = X.shape[1] if even else 0
    z = np.full(len(y01), np.nan)
    for k in range(K):
        trm, tem = (fold != k) & rows, (fold == k) & rows
        A, B = D._z(X[trm], X[tem], n_even)
        v, ok = D._fit(A, y01[trm], pen, even)
        assert ok, f'fold{k} 미수렴'
        z[tem] = D._scores(v, B, n_even, even)
    return z


def hold_ll(z, y01, seg):
    m = seg & np.isfinite(z)
    return float(rowloss(z[m], y01[m]).mean())


def hold_rows(zt, zw, yT, yA, hold, nz):
    """holdout 행별 3-class 손실 — score() 와 같은 회계. mean() 이 총점과 항등이다."""
    row = rowloss(zt[hold], yT[hold]).copy()
    hw = hold & nz
    row[nz[hold]] += rowloss(zw[hw], yA[hw])
    return row


# ══════════════════════════════════════════ 공통 러너
class Frame:
    """clean7k 슬라이스 + 추첨별 베이스라인 z 를 들고 다니는 틀."""

    def __init__(self, U):
        cu = U['cu']
        self.XT = np.hstack([U['tie_base'], U['Gt']])[cu]
        self.XW = np.hstack([U['win_base'], U['W']])[cu]
        self.y = U['y'][cu]
        self.nz = U['nz'][cu]
        self.hold = U['hold'][cu]
        self.calib = ~self.hold                       # 비-holdout clean 2,000
        self.yT = (self.y == 2).astype(np.float64)
        self.yA = (self.y == 0).astype(np.float64)
        self.rows_t = np.ones(len(self.y), bool)
        self.rows_w = self.nz
        self.ids7 = U['ids'][cu]

    def pen_tie(self, lam, p=7, nl=5):
        return np.concatenate([np.full(nl, lam[0]), np.full(p - nl, lam[1])])

    def pen_win(self, lam, p=7, nl=3):
        pen = np.full(p, float(lam))
        pen[:nl] = 0.0
        return pen

    def run(self, fold, XT=None, XW=None, pt=None, pw=None, nl_t=5, nl_w=3):
        XT = self.XT if XT is None else XT
        XW = self.XW if XW is None else XW
        pt = self.pen_tie(FROZEN_TIE, XT.shape[1], nl_t) if pt is None else pt
        pw = self.pen_win(FROZEN_WIN, XW.shape[1], nl_w) if pw is None else pw
        zt = oof_pen(XT, self.yT, fold, self.rows_t, pt, True)
        zw = oof_pen(XW, self.yA, fold, self.rows_w, pw, False)
        tie = hold_ll(zt, self.yT, self.hold)
        win = hold_ll(zw, self.yA, self.hold & self.nz)
        row = hold_rows(zt, zw, self.yT, self.yA, self.hold, self.nz)
        assert abs(float(row.mean()) - (tie + R_NZ * win)) < 1e-12, '행손실 회계 항등 위반'
        return dict(tie=tie, win=win, tot=tie + R_NZ * win, row=row, zt=zt, zw=zw)


class Collect:
    """셀별 (총점 · holdout 행손실) 을 모아 추첨 paired 표 + **홀드아웃 paired t** 를 낸다.

    두 잡음원을 갈라 본다.
      추첨 잡음  — 5추첨 부호 일치. fold 분할이 바뀌어도 같은 방향인가
      표본 잡음  — 추첨 평균한 행별 Δ 로 낸 paired t. **홀드아웃 5,748 표본이 바뀌어도
                   버티는가.** 5추첨 부호 일치는 이걸 전혀 막지 못한다 (같은 행을 공유).
    채택은 두 관문 모두 통과해야 한다.
    """

    def __init__(self, fold_seeds):
        self.fs = list(fold_seeds)
        self.tot, self.row, self.btot, self.brow = {}, {}, {}, {}

    def base(self, fs, r):
        self.btot[fs], self.brow[fs] = r['tot'], r['row']
        return r

    def add(self, name, fs, r):
        self.tot.setdefault(name, {})[fs] = r['tot']
        self.row.setdefault(name, {})[fs] = r['row']
        return r

    def table(self, gate=None):
        gate = GATE_D if gate is None else gate
        print(f'\n  {"셀":18s}' + ''.join(f'{"f" + str(x):>10s}' for x in self.fs)
              + f'{"평균Δ":>11s}{"부호":>6s}{"t(홀드아웃)":>12s}{"판정":>8s}')
        out = {}
        for cn in self.tot:
            ds = [self.tot[cn][x] - self.btot[x] for x in self.fs]
            sign = ('전부−' if all(d < 0 for d in ds)
                    else '전부+' if all(d > 0 for d in ds) else '혼합')
            dr = np.mean([self.row[cn][x] - self.brow[x] for x in self.fs], axis=0)
            se = float(dr.std(ddof=1) / np.sqrt(len(dr)))
            t = float(dr.mean() / se) if se > 0 else 0.0
            ok = (sign == '전부−') and (np.mean(ds) <= -gate) and (t <= -1.96)
            print(f'  {cn:18s}' + ''.join(f'{d:+10.5f}' for d in ds)
                  + f'{np.mean(ds):+11.6f}{sign:>6s}{t:>12.2f}'
                  + f'{("채택" if ok else "미달"):>8s}')
            out[cn] = dict(d_mean=float(np.mean(ds)), sign=sign, t_hold=t, se_hold=se,
                           adopt=bool(ok),
                           d_by_rep={str(x): float(d) for x, d in zip(self.fs, ds)})
        print('  ! t 는 추첨평균 행별 Δ 의 paired t (n=5,748) — |t|>1.96 이 표본 잡음 관문이다.')
        return out


# ══════════════════════════════════════════ 실험 1 — λ 오라클
def exp1(F, folds, res):
    print('\n' + '=' * 78)
    print('[실험1] λ 오라클 — 미세 격자(1a) + 열별 좌표하강(1b) · 추첨 42 천장')
    print('=' * 78)
    f42 = folds[SEED]
    base = F.run(f42)
    print(f'  기준(동결 λ) tie {base["tie"]:.5f} · win {base["win"]:.5f} → {base["tot"]:.5f}')

    # 1a 미세 격자 — 그룹 λ, 촘촘한 간격 (100·10 포함)
    t0 = time.time()
    GL = [30, 50, 70, 100, 150, 220, 320, 470, 680, 1000]
    GG = [0.3, 1, 3, 10, 30]
    GW = [0.3, 0.5, 1, 2, 3, 5, 7, 10, 15, 22, 30]
    st = {}
    for a in GL:
        for b in GG:
            zt = oof_pen(F.XT, F.yT, f42, F.rows_t, F.pen_tie((a, b)), True)
            st[(a, b)] = hold_ll(zt, F.yT, F.hold)
    sw = {}
    for g in GW:
        zw = oof_pen(F.XW, F.yA, f42, F.rows_w, F.pen_win(g), False)
        sw[g] = hold_ll(zw, F.yA, F.hold & F.nz)
    ot, ow = min(st, key=st.get), min(sw, key=sw.get)
    print(f'  1a 오라클  tie λ={ot} {st[ot]:.5f} (Δ{st[ot] - base["tie"]:+.6f}) · '
          f'win λ={ow:g} {sw[ow]:.5f} (Δ{sw[ow] - base["win"]:+.6f})  [{time.time() - t0:.0f}s]')
    print(f'     총점 천장 {st[ot] + R_NZ * sw[ow]:.5f} '
          f'(Δ{st[ot] + R_NZ * sw[ow] - base["tot"]:+.6f})')

    # 1b 열별 λ — **무제한 개별 벌점**. 이종 전문가를 빼는 게 아니라 릿지가 정하게 한다:
    #    λ=0 = 무벌점(그대로 쓴다) · λ≥1e5 = 사실상 제외 — 삭제를 벌점의 극한으로 구현.
    #    확장 격자(0 ~ 1e6) 좌표하강 → 로그축 황금분할 연속 정밀화 = 진짜 바닥 추정.
    CAND = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0,
            1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]
    PHI = (5 ** 0.5 - 1) / 2

    def descend(X, y01, rows, seg, even, pen0, names):
        cache = {}

        def ll(pen):
            key = tuple(round(float(p), 6) for p in pen)
            if key not in cache:
                z = oof_pen(X, y01, f42, rows, np.array(pen, float), even)
                cache[key] = hold_ll(z, y01, seg)
            return cache[key]

        pen, best = [float(p) for p in pen0], ll(pen0)
        for _pass in range(2):                        # ① 확장 격자 좌표하강
            moved = False
            for j in range(len(pen)):
                for c in CAND:
                    if c == pen[j]:
                        continue
                    trial = pen.copy()
                    trial[j] = c
                    v = ll(trial)
                    if v < best - 1e-7:
                        pen, best, moved = trial, v, True
            if not moved:
                break
        for j in range(len(pen)):                     # ② 로그축 연속 정밀화 (0·경계는 그대로)
            if pen[j] <= 0 or pen[j] >= CAND[-1]:
                continue

            def f1(x, _j=j):
                trial = pen.copy()
                trial[_j] = 10.0 ** x
                return ll(trial)

            a, b = np.log10(pen[j]) - 0.6, np.log10(pen[j]) + 0.6
            c_, d_ = b - PHI * (b - a), a + PHI * (b - a)
            fc, fd = f1(c_), f1(d_)
            while b - a > 0.05:
                if fc <= fd:
                    b, d_, fd = d_, c_, fc
                    c_ = b - PHI * (b - a)
                    fc = f1(c_)
                else:
                    a, c_, fc = c_, d_, fd
                    d_ = a + PHI * (b - a)
                    fd = f1(d_)
            x = (a + b) / 2
            if f1(x) < best - 1e-7:
                pen[j], best = 10.0 ** x, f1(x)

        tag = lambda p: '무벌점' if p == 0 else ('제외급' if p >= 1e5 else f'{p:.3g}')
        print('     ' + ' · '.join(f'{nm}={tag(p)}' for nm, p in zip(names, pen))
              + f'  ({len(cache)}회 적합)')
        return pen, best

    t0 = time.time()
    print('  1b tie 열별 λ:')
    pt_o, tie_o = descend(F.XT, F.yT, F.rows_t, F.hold, True,
                          list(F.pen_tie(ot)), TIE_NAMES)
    print('  1b win 열별 λ:')
    pw_o, win_o = descend(F.XW, F.yA, F.rows_w, F.hold & F.nz, False,
                          list(F.pen_win(ow)), WIN_NAMES)
    tot_o = tie_o + R_NZ * win_o
    print(f'  1b 오라클  tie {tie_o:.5f} (Δ{tie_o - base["tie"]:+.6f}) · '
          f'win {win_o:.5f} (Δ{win_o - base["win"]:+.6f}) → 총점 천장 {tot_o:.5f} '
          f'(Δ{tot_o - base["tot"]:+.6f})  [{(time.time() - t0) / 60:.1f}분]')

    # 오라클 λ* 를 다른 추첨에 이식 — 천장이 추첨 복권인지 구조인지
    C = Collect(folds)
    for fs, fd in folds.items():
        C.base(fs, F.run(fd))
        C.add('1a λ* 이식', fs, F.run(fd, pt=np.array(F.pen_tie(ot)), pw=np.array(F.pen_win(ow))))
        C.add('1b 열별λ* 이식', fs, F.run(fd, pt=np.array(pt_o, float), pw=np.array(pw_o, float)))
    res['exp1'] = dict(grid_tie={str(k): v for k, v in st.items()},
                       grid_win={str(k): v for k, v in sw.items()},
                       lam_1a=dict(tie=list(ot), win=float(ow)),
                       pen_1b=dict(tie=pt_o, win=pw_o),
                       pairs=C.table())
    print('  ! 오라클은 holdout 으로 골랐다 — 위 Δ 는 채택 근거가 아니라 **천장**이다.')


def score_of(F, zt, zw):
    """수동으로 만든 z 한 쌍을 Frame.run() 과 같은 형식으로 채점한다."""
    tie = hold_ll(zt, F.yT, F.hold)
    win = hold_ll(zw, F.yA, F.hold & F.nz)
    return dict(tie=tie, win=win, tot=tie + R_NZ * win,
                row=hold_rows(zt, zw, F.yT, F.yA, F.hold, F.nz), zt=zt, zw=zw)


# ══════════════════════════════════════════ 실험 1c — 열별 유계 벌점 · 정직 선택
EXCL = 1e5                       # 사실상 제외 — `_z` 가 열별 정규화라 계수만 소멸한다
BASE_TIE = [100.0] * 5 + [10.0] * 2
BASE_WIN = [0.0] * 3 + [10.0] * 4


def cand_of(scheme, base):
    """열별 이산 후보. 자유도는 '어느 열이 어느 값' 뿐 — 연속 λ 보다 선택 낙관이 작다."""
    if scheme == 'S1':
        return [[0.0, EXCL] for _ in base], [0.0] * len(base)
    if scheme == 'S2':
        return [[b, EXCL] for b in base], list(base)
    return [sorted({0.0, b, EXCL}) for b in base], list(base)          # S3


def descend_disc(ev, cands, pen0, passes=3):
    """이산 좌표하강 — 캐시로 중복 평가 차단."""
    cache = {}

    def f(p):
        key = tuple(p)
        if key not in cache:
            cache[key] = ev(list(p))
        return cache[key]

    pen, best = list(pen0), f(pen0)
    for _ in range(passes):
        moved = False
        for j, cs in enumerate(cands):
            for c in cs:
                if c == pen[j]:
                    continue
                t = pen.copy()
                t[j] = c
                v = f(t)
                if v < best - 1e-9:
                    pen, best, moved = t, v, True
        if not moved:
            break
    return pen, best, len(cache)


def _inner_ev(X, y01, tr_mask, even, seed=SEED):
    """tr_mask 행만으로 안쪽 3-fold CV 손실을 내는 평가자 (웜스타트 재사용)."""
    Xtr, ytr = X[tr_mask], y01[tr_mask]
    inner = np.zeros(len(ytr), int)
    for j, (_, iv) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed)
                                .split(np.zeros((len(ytr), 1)), ytr)):
        inner[iv] = j
    n_even = X.shape[1] if even else 0
    Zi = [D._z(Xtr[inner != j], Xtr[inner == j], n_even) for j in range(3)]
    warm = [None] * 3

    def ev(pen):
        tot = 0.0
        pv = np.array(pen, float)
        for j in range(3):
            A, B = Zi[j]
            v, _ok = D._fit(A, ytr[inner != j], pv, even, warm[j])
            warm[j] = v
            s = D._scores(v, B, n_even, even)
            tot += float(rowloss(s, ytr[inner == j]).sum())
        return tot

    return ev


def oof_nested(X, y01, fold, rows, cands, pen0, even):
    """바깥 fold 마다 안쪽 3-fold 로 벌점을 고른다 — fold k 는 자기 행을 보지 않는다.
    현행 그룹 λ 와 **같은 기제**(nested CV)이고 파라미터만 열별로 늘린 것이다."""
    n_even = X.shape[1] if even else 0
    z = np.full(len(y01), np.nan)
    picked, nfit = [], 0
    for k in range(K):
        trm, tem = (fold != k) & rows, (fold == k) & rows
        pen, _, nc = descend_disc(_inner_ev(X, y01, trm, even), cands, pen0)
        picked.append(pen)
        nfit += nc * 3
        A, B = D._z(X[trm], X[tem], n_even)
        v, ok = D._fit(A, y01[trm], np.array(pen, float), even)
        assert ok, f'fold{k} 미수렴'
        z[tem] = D._scores(v, B, n_even, even)
    return z, picked, nfit


def exp_1c(F, folds, res):
    """유계(이산) 열별 벌점 — 체계별 천장 → 통과 체계만 정직 선택으로 확정."""
    print('\n' + '=' * 78)
    print('[실험1c] 열별 유계 벌점 — S1{0,1e5} · S2{계약λ,1e5} · S3{0,계약λ,1e5}')
    print('=' * 78)
    f42 = folds[SEED]
    b42 = F.run(f42)
    print(f'  기준(그룹 동결 λ) tie {b42["tie"]:.5f} · win {b42["win"]:.5f} → {b42["tot"]:.5f}')

    AX = (('tie', F.XT, F.yT, F.rows_t, F.hold, True, BASE_TIE, TIE_NAMES),
          ('win', F.XW, F.yA, F.rows_w, F.hold & F.nz, False, BASE_WIN, WIN_NAMES))
    tag = lambda p: '0' if p == 0 else ('제외' if p >= EXCL else f'{p:g}')

    # ── ① 체계별 오라클 천장 (추첨 42 · holdout 선택) ──
    ceil, pens = {}, {}
    for scheme in ('S1', 'S2', 'S3'):
        parts = {}
        for nm, X, y01, rows, seg, even, base, names in AX:
            cands, pen0 = cand_of(scheme, base)
            t0 = time.time()
            pen, ll, nc = descend_disc(
                lambda p: hold_ll(oof_pen(X, y01, f42, rows, np.array(p, float), even),
                                  y01, seg), cands, pen0)
            parts[nm] = (pen, ll)
            print(f'  {scheme} {nm} 천장 {ll:.5f} (Δ{ll - b42[nm]:+.6f})  '
                  + ' · '.join(f'{n}={tag(p)}' for n, p in zip(names, pen))
                  + f'  [{nc}구성 {time.time() - t0:.0f}s]')
        tot = parts['tie'][1] + R_NZ * parts['win'][1]
        ceil[scheme] = tot - b42['tot']
        pens[scheme] = {k2: v[0] for k2, v in parts.items()}
        print(f'  {scheme} 총점 천장 {tot:.5f} (Δ{ceil[scheme]:+.6f})')

    print(f'\n  {"체계":6s}{"천장 Δ":>12s}   판정 (1b 연속 천장 −0.000733 · 이산삭제 실측 −0.000493)')
    for s_, d_ in ceil.items():
        v = '체계가 갑갑함 — 정직 선택 무의미' if d_ > -GATE_D else '천장 통과 → 정직 선택으로'
        print(f'  {s_:6s}{d_:+12.6f}   {v}')

    # ── ② 천장 통과 체계만 정직 선택 (nested CV) + 5추첨 paired ──
    C = Collect(folds)
    for fs, fd in folds.items():
        C.base(fs, F.run(fd))
    for scheme in [s_ for s_, d_ in ceil.items() if d_ <= -GATE_D]:
        for fs, fd in folds.items():
            zs, pk, nf = {}, {}, 0
            for nm, X, y01, rows, seg, even, base, names in AX:
                cands, pen0 = cand_of(scheme, base)
                z, picked, n_ = oof_nested(X, y01, fd, rows, cands, pen0, even)
                zs[nm], pk[nm], nf = z, picked, nf + n_
            C.add(f'{scheme} 정직', fs, score_of(F, zs['tie'], zs['win']))
            if fs == SEED:
                for nm in ('tie', 'win'):
                    ex = [sum(p >= EXCL for p in pen) for pen in pk[nm]]
                    print(f'  {scheme} 정직 f{fs} {nm}: fold 별 제외 열 수 {ex} '
                          f'({nf} 적합)')
    if C.tot:
        res['exp_1c'] = dict(ceil=ceil, pens={k2: {a: list(map(float, p))
                                                  for a, p in v.items()}
                                              for k2, v in pens.items()},
                             pairs=C.table())
    else:
        res['exp_1c'] = dict(ceil=ceil, pens={}, pairs={})
        print('  → 천장을 넘은 체계가 없다. 유계 이산 벌점으로는 도달 불가.')
    print('  ! 정직 선택은 nested CV — 각 holdout 행의 예측은 그 행을 보지 않고 고른 벌점에서 나온다.')


# ══════════════════════════════════════════ 실험 2 — 불일치 게이트 온도
def gate_fit(z, d, y01, mask):
    dc = d - d[mask].mean()

    def fg(v):
        T = np.exp(v[0] + v[1] * dc[mask])
        s = T * z[mask]
        r = expit(s) - y01[mask]
        return (float(np.mean(np.logaddexp(0, s) - y01[mask] * s)),
                np.array([float(np.mean(r * s)), float(np.mean(r * s * dc[mask]))]))

    res = minimize(fg, np.zeros(2), jac=True, method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-14})
    return res.x, dc


def exp2(F, folds, res):
    print('\n' + '=' * 78)
    print('[실험2] 불일치 게이트 온도 — z\' = exp(a + b·(d−d̄))·z · oracle vs honest')
    print('=' * 78)
    rms_w = np.sqrt((F.XW[:, :3][F.rows_w] ** 2).mean(0))
    d_win = (F.XW[:, :3] / rms_w).std(1)
    mu_t, sd_t = F.XT[:, :5].mean(0), F.XT[:, :5].std(0)
    d_tie = ((F.XT[:, :5] - mu_t) / sd_t).std(1)

    C, params = Collect(folds), {}
    for fs, fd in folds.items():
        b = C.base(fs, F.run(fd))
        for tag, fit_w, fit_t in (('oracle', F.hold & F.nz, F.hold),
                                  ('honest', F.calib & F.nz, F.calib)):
            (aw, bw), dcw = gate_fit(b['zw'], d_win, F.yA, fit_w)
            (at, bt), dct = gate_fit(b['zt'], d_tie, F.yT, fit_t)
            zt2 = np.exp(at + bt * dct) * b['zt']
            zw2 = np.exp(aw + bw * dcw) * b['zw']
            C.add(f'gate·{tag}', fs, score_of(F, zt2, zw2))
            params.setdefault(tag, []).append((at, bt, aw, bw))
    for tag, ps in params.items():
        a_ = np.array(ps).mean(0)
        print(f'  {tag} 평균 파라미터 — tie (a {a_[0]:+.3f}, b {a_[1]:+.3f}) · '
              f'win (a {a_[2]:+.3f}, b {a_[3]:+.3f})   (b<0 = 불일치 클수록 누른다)')
    res['exp2'] = dict(pairs=C.table(),
                       params={t: np.array(p).mean(0).tolist() for t, p in params.items()})
    print('  ! oracle 은 천장 · honest(2,000행 적합)만 채택 가능. honest 가 채택선을 넘는지 보라.')


# ══════════════════════════════════════════ 실험 a — LOO
LOO = {'abe2': ([2], [0], [3]), 'tb2': ([0], [], [4]), 'abe1': ([3], [1], [5]),
       'tb1': ([1], [], [6]), 'arith': ([4], [2], [])}


def exp_a(F, folds, res):
    print('\n' + '=' * 78)
    print('[실험a] 전문가 한계 기여 (결합기 LOO) — g1·g2 의 풀링 기여는 남는다 = 완전 제거의 하한')
    print('=' * 78)
    C = Collect(folds)
    for fs, fd in folds.items():
        C.base(fs, F.run(fd))
        for ex, (t_drop, m_drop, w_drop) in LOO.items():
            kt = [j for j in range(7) if j not in t_drop]
            kw = [j for j in range(7) if j not in m_drop + w_drop]
            r = F.run(fd, XT=F.XT[:, kt], XW=F.XW[:, kw],
                      nl_t=5 - len(t_drop), nl_w=3 - len(m_drop))
            C.add(f'−{ex}', fs, r)
    res['exp_a'] = dict(pairs=C.table())
    print('  ! Δ>0 = 그 전문가를 빼면 나빠진다(=밥값). Δ~0 전문가는 완전 제거(헤드 재학습) 후보.')


# ══════════════════════════════════════════ 실험 b — 손실 집중 마이닝
def exp_b(F, U, folds, res):
    print('\n' + '=' * 78)
    print('[실험b] 손실 집중 마이닝 — 추첨 42 · holdout 5,748')
    print('=' * 78)
    b = F.run(folds[SEED])
    lt = rowloss(b['zt'][F.hold], F.yT[F.hold])
    row = lt.copy()
    hw = F.nz[F.hold]
    row[hw] += rowloss(b['zw'][F.hold & F.nz], F.yA[F.hold & F.nz])

    order = np.argsort(-row)
    n_top = max(int(len(row) * 0.01), 1)
    top = np.zeros(len(row), bool)
    top[order[:n_top]] = True
    print(f'  상위 1% ({n_top}행) 손실 점유 {row[top].sum() / row.sum():.1%} · '
          f'상위 5% 점유 {row[order[:len(row) // 20]].sum() / row.sum():.1%}')

    ids_h = F.ids7[F.hold]
    full = pd.read_csv(DATA_DIR / 'train_full.csv',
                       usecols=['id', 'prompt', 'response_a', 'response_b'])
    pos = pd.Index(full['id']).get_indexer(ids_h)
    assert (pos >= 0).all()
    ln_p = full['prompt'].str.len().to_numpy()[pos]
    ln_ab = ((full['response_a'].str.len() + full['response_b'].str.len()) / 2).to_numpy()[pos]

    rms_w = np.sqrt((F.XW[:, :3][F.rows_w] ** 2).mean(0))
    d_win = (F.XW[:, :3] / rms_w).std(1)[F.hold]
    za = U['abe2_npz']
    m_f = za['hold_fwd'][:, 0] - za['hold_fwd'][:, 1]
    m_s = za['hold_swp'][:, 0] - za['hold_swp'][:, 1]
    tta_gap = np.abs(m_f - m_s)
    y_h = F.y[F.hold]

    print(f'\n  {"":14s}{"상위1%":>10s}{"나머지":>10s}')
    for nm, v in (('행손실 평균', row), ('tie 라벨 비율', (y_h == 2).astype(float)),
                  ('프롬프트 길이', ln_p), ('응답 길이 평균', ln_ab),
                  ('승부 불일치 d', d_win), ('TTA |fwd−swp|', tta_gap)):
        print(f'  {nm:14s}{v[top].mean():>10.3f}{v[~top].mean():>10.3f}')
    lab_top = {int(k): int(c) for k, c in zip(*np.unique(y_h[top], return_counts=True))}
    print(f'  상위1% 라벨 분포 (0=A,1=B,2=tie): {lab_top}')
    res['exp_b'] = dict(top1_share=float(row[top].sum() / row.sum()),
                        top5_share=float(row[order[:len(row) // 20]].sum() / row.sum()),
                        top_label=lab_top)
    print('  ! 상위행이 tie 편중 + 불일치 大 면 실험2 게이트의 과녁이 실재한다는 뜻이다.')


# ══════════════════════════════════════════ 실험 c — 부스터
def _load_margin(name, ids):
    p = ADIR / f'margins/{name}.npz'
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    assert (np.concatenate([z['ids_valid'], z['ids_hold']]) == ids).all(), f'{name} id'
    L = np.vstack([z['raw_valid__xgb'], z['raw_hold__xgb']]).astype(np.float64)
    return sd3(L)                                     # (m_bo, t_bo)


def _w5_column(fam, U, seed=46):
    """승부축 5번째 GATE 체인 — rep_traj_{fam}224 (u,v) 132. 학습 1회 캐시."""
    cache = FINAL / 'temp/finlab_w5.npz'
    key = f'W5_{fam}' if seed == 46 else f'W5_{fam}_s{seed - 4}'
    if cache.exists():
        zc = np.load(cache)
        if key in zc.files:
            return zc[key].astype(np.float64), dict(zc)
    rp = ADIR / f'rep_traj_{fam}224.npz'
    if not rp.exists():
        return None, {}
    z = np.load(rp, allow_pickle=True)
    assert (z['ids_s2'] == U['ids']).all(), f'{fam} rep id'

    za = np.load(MODELS / 'logits/abe2__checkpoint-3000__train.npz')
    ids_tr, y_tr = za['train_ids'], za['train_labels'].astype(np.int64)
    assert (z['ids_tr'] == ids_tr).all()

    def get(rel, nc):
        zz = np.load(MODELS / rel)
        assert (zz['train_ids'] == ids_tr).all()
        return ((zz['train_fwd'] + zz['train_swp']) / 2).astype(np.float64)

    L2 = ((za['train_fwd'] + za['train_swp']) / 2).astype(np.float64)
    Lt2 = get('logits/tb2__checkpoint-4000__train.npz', 2)
    L1 = get(f'logits/abe1__checkpoint-{ABE1_STEP}__train.npz', 3)
    Lt1 = get(f'logits/tb1__checkpoint-{TB1_STEP}__train.npz', 2)
    zar = np.load(CKPT_AB224 / 'oof_train.npz')
    assert (zar['ids_train'] == ids_tr).all() and (zar['y_train'] == y_tr).all()
    m2, t2 = sd3(L2)
    m1, t1 = sd3(L1)
    mr, tr_ = sd3(zar['raw_oof_train__xgb'].astype(np.float64))
    tie_tr = np.column_stack([Lt2[:, 1] - Lt2[:, 0], Lt1[:, 1] - Lt1[:, 0], t2, t1, tr_])
    win_tr = np.column_stack([m2, m1, mr])
    mu_u, sc_u = D.zstats(tie_tr, True)
    _, sc_v = D.zstats(win_tr, False)
    Lu_tr, Lv_tr = (tie_tr - mu_u) / sc_u, win_tr / sc_v
    Lu_s2 = (U['tie_base'] - mu_u) / sc_u
    Lv_s2 = U['win_base'] / sc_v

    ut, vt = z['emb_u_tr'].astype(np.float64), z['emb_v_tr'].astype(np.float64)
    us, vs = z['emb_u_s2'].astype(np.float64), z['emb_v_s2'].astype(np.float64)
    mu_e, sd_e = ut.mean(0), np.clip(ut.std(0), 1e-9, None)
    rms_e = np.clip(np.sqrt((vt ** 2).mean(0)), 1e-9, None)
    ut, us = (ut - mu_e) / sd_e, (us - mu_e) / sd_e
    vt, vs = vt / rms_e, vs / rms_e

    du, dv, nu, nv = 132, 132, 5, 3
    m = y_tr != 2
    y01w = (y_tr == 0).astype(np.float64)[m]
    sk = list(range(du, du + nu)) + list(range(du + nu + dv, du + nu + dv + nv))
    pool = list(range(0, du)) + list(range(du + nu, du + nu + dv))
    cu_t = np.ascontiguousarray(np.hstack([ut[m], Lu_tr[m]]), np.float32)
    cv_t = np.ascontiguousarray(np.hstack([vt[m], Lv_tr[m]]), np.float32)
    cu_s = np.ascontiguousarray(np.hstack([us, Lu_s2]), np.float32)
    cv_s = np.ascontiguousarray(np.hstack([vs, Lv_s2]), np.float32)
    net = lambda d, h: D.WinNet(d, h, sk, pool, du)
    t0 = time.time()
    r_s2, aux = D.fit_head(cu_t, cv_t, y01w, None, False, seed, cu_s, cv_s, net)
    print(f'    W5({fam}) seed{seed} — es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]} '
          f'[{time.time() - t0:.0f}s]')
    prev = dict(np.load(cache)) if cache.exists() else {}
    prev[key] = r_s2.astype(np.float32)
    np.savez_compressed(cache, **prev)
    return r_s2.astype(np.float64), prev


def exp_c(F, U, folds, res):
    print('\n' + '=' * 78)
    print('[실험c] 부스터 — 로짓 4셀(대체) + 풀링 추가(승부축 5번째 GATE) + both')
    print('=' * 78)
    cu = U['cu']
    subs = {nm: _load_margin(nm, U['ids'])
            for nm in ('sub_lgbm', 'sub_cat', 'mix_lgbm', 'mix_cat')}
    w5 = {}
    for fam in ('xgb', 'lgbm', 'cat'):
        col, _ = _w5_column(fam, U)
        if col is not None:
            w5[fam] = col[cu]

    C = Collect(folds)
    for fs, fd in folds.items():
        C.base(fs, F.run(fd))
        for nm, mt in subs.items():                    # 로짓 대체 (열 수 불변)
            if mt is None:
                continue
            m_bo, t_bo = mt[0][cu], mt[1][cu]
            XT2, XW2 = F.XT.copy(), F.XW.copy()
            XT2[:, 4], XW2[:, 2] = t_bo, m_bo
            C.add(f'로짓대체 {nm}', fs, F.run(fd, XT=XT2, XW=XW2))
        if 'xgb' in w5:                                # 자기 궤적 풀링 — 모델 추가 0 · 계약 청정
            XW3 = np.hstack([F.XW, w5['xgb'][:, None]])
            C.add('풀링추가 xgb(자기)', fs, F.run(fd, XW=XW3))
        for fam in ('lgbm', 'cat'):                    # both = arith→부스터 통째 대체 (1모델)
            if fam in w5 and subs.get(f'sub_{fam}') is not None:
                m_bo, t_bo = subs[f'sub_{fam}'][0][cu], subs[f'sub_{fam}'][1][cu]
                XT2 = F.XT.copy()
                XT2[:, 4] = t_bo
                XW4 = np.hstack([F.XW, w5[fam][:, None]])
                XW4[:, 2] = m_bo
                C.add(f'both {fam}', fs, F.run(fd, XT=XT2, XW=XW4))
    res['exp_c'] = dict(pairs=C.table())
    print('  ! 계약 — 풀링추가 xgb(자기)·both 는 1모델 청정 · 구 pool_lgbm 류(xgb 로짓+lgbm 풀링)는 '
          '2모델 앙상블이라 제거')
    print('  ! 참고(구 하네스·s42): both_win −0.0008 · sub_logit +0.0005 · mix_* 는 앙상블 마진(참고치)')


# ══════════════════════════════════════════ 실험 d — 결합기 열 정리 + 3전문가
def exp_d(F, U, folds, res):
    """1b 오라클·LOO 가 합의한 기제(잉여 로짓 열)를 **이산 삭제**로 굳혀 확인한다.

    연속 λ 튜닝(자유도 14·홀드아웃 선택)과 달리 열 삭제는 선택 낙관이 셀 단위 하나뿐이다.
    ! 결합기 수준 — g1·g2 속 abe1/tb1 풀링 기여는 남는다 (완전 제거의 하한).
    """
    print('\n' + '=' * 78)
    print('[실험d] 결합기 열 정리 — 계약 청정 조합 (pool_xgb 자기 궤적 · both_lgbm 1모델)')
    print('=' * 78)
    cu = U['cu']
    w5 = {}
    for fam in ('xgb', 'lgbm'):
        col, _ = _w5_column(fam, U)
        if col is not None:
            w5[fam] = col[cu]
    sub_l = _load_margin('sub_lgbm', U['ids'])

    # (이름, tie 삭제, win 로짓 삭제, W 삭제, W5 계열, sub_lgbm 대체)
    # tie [s_tb2 0, s_tb1 1, t_abe2 2, t_abe1 3, t_ar 4] · win [m_abe2 0, m_abe1 1, m_ar 2,
    # W_abe2 3, W_tb2 4, W_abe1 5, W_tb1 6]
    CELLS = [
        ('열정리',                  [1, 3], [0], [],  None,   False),
        ('열정리+W_tb1삭제',         [1, 3], [0], [6], None,   False),
        ('열정리+W_tb1+pool_xgb',   [1, 3], [0], [6], 'xgb',  False),
        ('열정리+W_tb1+both_lgbm',  [1, 3], [0], [6], 'lgbm', True),
    ]
    C = Collect(folds)
    for fs, fd in folds.items():
        C.base(fs, F.run(fd))
        for nm, t_drop, m_drop, w_drop, fam, sub in CELLS:
            if (fam is not None and fam not in w5) or (sub and sub_l is None):
                continue
            XT2, XW2 = F.XT, F.XW
            if sub:                                    # arith→lgbm 로짓 대체 (열 선택 전, 원 좌표)
                XT2, XW2 = F.XT.copy(), F.XW.copy()
                XT2[:, 4], XW2[:, 2] = sub_l[1][cu], sub_l[0][cu]
            kt = [j for j in range(7) if j not in t_drop]
            kw = [j for j in range(7) if j not in m_drop + w_drop]
            XW3 = XW2[:, kw]
            if fam is not None:
                XW3 = np.hstack([XW3, w5[fam][:, None]])
            r = F.run(fd, XT=XT2[:, kt], XW=XW3,
                      nl_t=5 - len(t_drop), nl_w=3 - len(m_drop))
            C.add(nm, fs, r)
    res['exp_d'] = dict(pairs=C.table())
    print('  ! 삭제 셀은 이산 확인용 — 채택 형태는 열별 벌점(λ 大 = 사실상 제외 · 구조 보존)이 원칙.')
    print('  ! pool_xgb = 자기 궤적(모델 추가 0) · both_lgbm = arith→lgbm 1모델 — 둘 다 계약 청정.')


# ══════════════════════════════════════════ 실험 f — 헤드 시드 평균 both 비교
HS_DEFAULT = (42, 1, 2, 5, 6)
# 벌점 벡터 — 제외 열 인덱스. tie [s_tb2 0,s_tb1 1,t_abe2 2,t_abe1 3,t_ar 4,g1 5,g2 6]
#                            win [m_abe2 0,m_abe1 1,m_ar 2,W_abe2 3,W_tb2 4,W_abe1 5,W_tb1 6]
#   B = 무라벨 사전 등록 — train46k R² 로만 고름 (t_abe1 .751 / m_abe2 .696 ≫ s_tb1 .405)
#   H = 홀드아웃 유래 — 1b 오라클·LOO 가 지목한 집합 (참고 대조)
#   T = 열정리 (s_tb1·t_abe1·m_abe2) · H = T + W_tb1 · L_* = 전문가 LOO
#   LOO 도 삭제가 아니라 λ=1e5 로 낸다 — 이종 전문가는 빼지 않고 릿지가 정한다는 원칙 유지
VEC = {'B': (frozenset({3}), frozenset({0})),
       'T': (frozenset({1, 3}), frozenset({0})),
       'H': (frozenset({1, 3}), frozenset({0, 6})),
       'L_abe2': (frozenset({2}), frozenset({0, 3})),
       'L_tb2': (frozenset({0}), frozenset({4})),
       'L_abe1': (frozenset({3}), frozenset({1, 5})),
       'L_tb1': (frozenset({1}), frozenset({6})),
       'L_arith': (frozenset({4}), frozenset({2}))}

# 셀 묶음 — --fcells 로 고른다. 전부 = 14셀 (헤드시드당 ~5분)
FCELLS = {
    'vec': [('열정리만 T', 'T', None), ('H(홀드아웃)', 'H', None), ('B(무라벨)', 'B', None)],
    'boost': [('both_xgb', None, 'xgb'), ('both_lgbm', None, 'lgbm')],
    'combo': [('B+both_xgb', 'B', 'xgb'), ('B+both_lgbm', 'B', 'lgbm'),
              ('H+both_xgb', 'H', 'xgb'), ('H+both_lgbm', 'H', 'lgbm')],
    'loo': [(f'−{e}', f'L_{e}', None) for e in ('abe2', 'tb2', 'abe1', 'tb1', 'arith')],
}


def pen_vec(excl_t, excl_w, add5):
    """계약 λ 기준 벌점 벡터 + 제외 열은 1e5. W5 추가 시 헤드 열 λ 를 덧붙인다."""
    pt = np.array([EXCL if j in excl_t else b for j, b in enumerate(BASE_TIE)], float)
    pw = [EXCL if j in excl_w else b for j, b in enumerate(BASE_WIN)]
    if add5:
        pw = pw + [BASE_WIN[-1]]                      # W5 = 헤드 열 → 계약 λ_g
    return pt, np.array(pw, float)


def _w5_seeded(fam, U, hs):
    """W5 GATE 를 헤드 시드 hs 로 학습/캐시. 승부 5번째 소스이므로 시드는 hs+4 (si=4)."""
    cache = FINAL / 'temp/finlab_w5.npz'
    key = f'W5_{fam}_s{hs}'
    if cache.exists():
        zc = np.load(cache)
        if key in zc.files:
            return zc[key].astype(np.float64)
    col, _ = _w5_column(fam, U, seed=hs + 4)
    if col is None:
        return None
    prev = dict(np.load(cache)) if cache.exists() else {}
    prev[key] = col.astype(np.float32)
    np.savez_compressed(cache, **prev)
    return col


def _heads_ab224(U, head_seeds):
    """ab224 계보 tie·win 헤드를 시드별로 학습/캐시. **구 arith 계보는 쓰지 않는다.**

    dest_universe 런의 s42 는 이미 ab224 계보라 그대로 씨앗으로 재사용한다 (검증: 그 런이
    plan.md 0.95798 재현 대조를 통과했다). 나머지 시드만 새로 학습한다 (시드당 ~3.5분).
    """
    cache = FINAL / 'temp/finlab_heads_ab224.npz'
    have = dict(np.load(cache)) if cache.exists() else {}
    if 'Gt_s42' not in have:                          # dest_universe 의 ab224 s42 를 씨앗으로
        zd = np.load(FINAL / 'temp/dest_universe.npz')
        have['Gt_s42'], have['W_s42'] = zd['Gt_s42'], zd['W_s42']
        np.savez_compressed(cache, **have)
        print('    s42 = dest_universe 의 ab224 헤드 재사용')
    need = [h for h in head_seeds if f'Gt_s{h}' not in have]
    if need:
        print(f'    ab224 헤드 학습 필요: {need} (시드당 ~3.5분)')
        Z = D.load_svd_cache()
        D.load_train_logits = _train_logits_ab224     # ★ arith OOF 를 ab224 로 교체
        A = D.head_assets(U, Z)
        RAW = D.raw_uv(Z)
        for h in need:
            t0 = time.time()
            Gt, _ = D.tie_head(A, h)
            Wc, _ = D.win_head(A, RAW, h)
            have[f'Gt_s{h}'] = Gt.astype(np.float32)
            have[f'W_s{h}'] = Wc.astype(np.float32)
            np.savez_compressed(cache, **have)        # 시드마다 즉시 저장 (중단 안전)
            print(f'    헤드 s{h} 완료 [{(time.time() - t0) / 60:.1f}분]')
        del A, RAW, Z
        gc.collect()
    return {h: (have[f'Gt_s{h}'].astype(np.float64), have[f'W_s{h}'].astype(np.float64))
            for h in head_seeds}


def exp_f(F, U, folds, res, head_seeds=HS_DEFAULT, groups=('vec', 'boost', 'combo', 'loo')):
    """both_xgb vs both_lgbm 동등 비교 + 무라벨 벡터 B — 헤드 시드 × fold 추첨 이중 평균.

    'both' 배선 = 같은 **1모델**이 로짓과 궤적 풀링을 다 낸다 (계약 청정).
        both_xgb  = 현행 로짓(arith=xgb) + xgb 자기 궤적    ← 로짓 교체 없음
        both_lgbm = lgbm 로짓 대체       + lgbm 궤적
    프로토콜 동일(224피쳐·5-fold OOF·swap·stage1 valid 5,000 ES) · xgb 궤적 계보는
    배포 checkpoint-ab224 와 비트 동일(assert) — 계열만 다른 대조다.

    왜 시드 평균인가: 행별 Δ 분산에 헤드 초기화 잡음이 섞여 SE 를 부풀린다. 시드 5개를
    평균하면 그 성분이 1/5 로 줄어 **표본 잡음 t 관문이 실제로 날카로워진다.**
    """
    print('\n' + '=' * 78)
    print(f'[실험f] ab224 계보 이중 평균 판정 — 헤드 시드 {list(head_seeds)} × 추첨 {list(folds)}')
    print('=' * 78)
    za, zb = (np.load(CKPT_AB224 / 'oof_train.npz'),
              np.load(ADIR / 'ckpts/ck4000-tj224x/oof_train.npz'))
    assert (za['ids_train'] == zb['ids_train']).all() and \
        float(np.abs(za['raw_oof_train__xgb'].astype(np.float64)
                     - zb['raw_oof_train__xgb'].astype(np.float64)).max()) == 0.0, \
        'xgb 궤적 계보가 배포 체크포인트와 다르다 — 동등 비교 전제 붕괴'
    print('  계보 가드 OK — xgb 궤적 = 배포 checkpoint-ab224 (비트 동일)')

    cu = U['cu']
    HD = _heads_ab224(U, head_seeds)                  # ab224 계보 — 구 arith 계보 폐기
    hs_ok = list(head_seeds)
    sub_l = _load_margin('sub_lgbm', U['ids'])

    CELLS = [c for g in groups for c in FCELLS[g]]
    print(f'  셀 {len(CELLS)}개 ({", ".join(groups)}) × 헤드시드 {len(hs_ok)} × 추첨 {len(folds)}')
    acc = {c[0]: {} for c in CELLS}
    tots = {c[0]: {} for c in CELLS}
    base, btots = {}, {}
    for hs in hs_ok:
        Gt, Wc = HD[hs]
        XT0 = np.hstack([U['tie_base'], Gt])[cu]
        XW0 = np.hstack([U['win_base'], Wc])[cu]
        w5h = {}
        for fam in ('xgb', 'lgbm'):
            col = _w5_seeded(fam, U, hs)
            if col is not None:
                w5h[fam] = col[cu]
        t0 = time.time()
        for fs, fd in folds.items():
            pt0, pw0 = pen_vec(frozenset(), frozenset(), False)
            b = F.run(fd, XT=XT0, XW=XW0, pt=pt0, pw=pw0)
            base[(hs, fs)], btots[(hs, fs)] = b['row'], b['tot']
            for nm, vk, fam in CELLS:
                if (fam is not None and fam not in w5h) or (fam == 'lgbm' and sub_l is None):
                    continue
                et, ew = VEC[vk] if vk else (frozenset(), frozenset())
                XT, XW = XT0, XW0
                if fam == 'lgbm':
                    XT, XW = XT0.copy(), XW0.copy()
                    XT[:, 4], XW[:, 2] = sub_l[1][cu], sub_l[0][cu]
                if fam is not None:
                    XW = np.hstack([XW, w5h[fam][:, None]])
                pt, pw = pen_vec(et, ew, fam is not None)
                r = F.run(fd, XT=XT, XW=XW, pt=pt, pw=pw)
                acc[nm][(hs, fs)], tots[nm][(hs, fs)] = r['row'], r['tot']
        print(f'  헤드 시드 {hs}: 기준 {np.mean([btots[(hs, x)] for x in folds]):.5f} '
              f'[{time.time() - t0:.0f}s]')

    keys = sorted(base)
    bg = float(np.mean([btots[k2] for k2 in keys]))
    per_hs_b = [float(np.mean([btots[(h, x)] for x in folds])) for h in hs_ok]
    print(f'\n  기준 그랜드 평균 {bg:.5f} (헤드시드별 '
          + ' · '.join(f'{v:.5f}' for v in per_hs_b)
          + f' · sd {np.std(per_hs_b, ddof=1):.5f})')
    print(f'  참고 plan.md — 8시드 평균 {PLAN_8SEED:.5f} · s42 단독 {PLAN_TIE + R_NZ * PLAN_WIN:.5f}')
    print(f'\n  {"셀":18s}' + ''.join(f'{"s" + str(h):>10s}' for h in hs_ok)
          + f'{"평균Δ":>11s}{"시드부호":>9s}{"t(홀드아웃)":>12s}{"판정":>8s}')
    out = {}
    for nm, _vk, _fam in CELLS:
        if not acc[nm]:
            continue
        per_hs = [float(np.mean([tots[nm][(h, x)] - btots[(h, x)] for x in folds]))
                  for h in hs_ok]
        dr = np.mean([acc[nm][k2] - base[k2] for k2 in keys], axis=0)
        se = float(dr.std(ddof=1) / np.sqrt(len(dr)))
        t = float(dr.mean() / se) if se > 0 else 0.0
        sign = ('전부−' if all(d < 0 for d in per_hs)
                else '전부+' if all(d > 0 for d in per_hs) else '혼합')
        ok = (sign == '전부−') and (np.mean(per_hs) <= -GATE_D) and (t <= -1.96)
        print(f'  {nm:18s}' + ''.join(f'{d:+10.5f}' for d in per_hs)
              + f'{np.mean(per_hs):+11.6f}{sign:>9s}{t:>12.2f}'
              + f'{("채택" if ok else "미달"):>8s}')
        out[nm] = dict(d_mean=float(np.mean(per_hs)), t_hold=t, se_hold=se, sign=sign,
                       adopt=bool(ok), d_by_hs=dict(zip(map(str, hs_ok), per_hs)))
    res['exp_f'] = dict(head_seeds=list(hs_ok), cells=out)
    print(f'  ! Δ 는 {len(keys)}개 (헤드시드 × 추첨) 조합 평균 · t 는 그 평균 행별 Δ 의 paired t')
    print('  ! 헤드는 **ab224 계보**로 시드별 재학습 — 노트북 목표값(plan.md 0.95798)과 같은 계보')
    print('  ! B 는 무라벨 R² 로만 고른 벡터 · H 는 홀드아웃 유래 (대조용)')


# ══════════════════════════════════════════ 실험 g — 선택 밖 행 전이 검정
def _seg_rows(zt, zw, yT, yA, seg, nz):
    """seg 행별 3-class 손실. mean() = tie ll + (seg 내 비-tie 비율) × win ll."""
    row = rowloss(zt[seg], yT[seg]).copy()
    sw = seg & nz
    row[nz[seg]] += rowloss(zw[sw], yA[sw])
    return row


def _run_full(U, fold7, XT, XW, pt, pw, y, nz, cu):
    """cu(7,748) 위에서 교차적합 OOF + **cu 밖 행은 fold 평균**으로 예측.

    clean7k 계약 — fold 는 청정 7,748 위에서 추첨한다. cu 밖 3,000행은 어느 fold 의
    적합에도 없으므로 5개 fold 모델 예측의 평균으로 낸다 (각 모델이 cu 의 4/5 를 봤다).
    """
    yT, yA = (y == 2).astype(np.float64), (y == 0).astype(np.float64)
    idx7 = np.where(cu)[0]
    out_i = np.where(~cu)[0]
    res = {}
    for tag, X, y01, pen, even, rows7 in (('t', XT, yT, pt, True, np.ones(len(idx7), bool)),
                                          ('w', XW, yA, pw, False, nz[cu])):
        n_even = X.shape[1] if even else 0
        z = np.full(len(y), np.nan)
        acc = np.zeros(len(out_i))
        for k in range(K):
            tr7, te7 = (fold7 != k) & rows7, (fold7 == k) & rows7
            tri, tei = idx7[tr7], idx7[te7]
            A, Bm = D._z(X[tri], X[np.concatenate([tei, out_i])], n_even)
            v, ok = D._fit(A, y01[tri], pen, even)
            assert ok, f'{tag} fold{k} 미수렴'
            sc = D._scores(v, Bm, n_even, even)
            z[tei] = sc[:len(tei)]
            acc += sc[len(tei):]
        z[out_i] = acc / K                        # fold 평균 — cu 밖 행은 전부 out-of-fit
        res[tag] = z
    return res['t'], res['w']


def _run_adaptive(U, fold7, XT, XW, y, nz, cu):
    """**fold별 적응 λ** — destnation 계약 원본(안쪽 3-fold 가 fold 마다 λ 재선택).
    cu 밖 행은 fold 평균으로 예측한다 (동결 경로 `_run_full` 과 같은 규약).

    안쪽 CV 격자·자동확장은 destnation 을 그대로 쓴다: tie = LAMS0 × LAMS 30점 2-D,
    win = LAMS 1-D · m3 무벌점. 배포 λ 는 fold 선택의 (성분별) 중앙값이지만 여기서는
    **채점에 쓰이는 OOF z 만** 필요하므로 fold 별 선택을 그대로 쓴다.
    """
    yT, yA = (y == 2).astype(np.float64), (y == 0).astype(np.float64)
    idx7 = np.where(cu)[0]
    out_i = np.where(~cu)[0]
    res, picks = {}, {}
    for tag, X, y01, even, rows7, nl in (('t', XT, yT, True, np.ones(len(idx7), bool), 5),
                                         ('w', XW, yA, False, nz[cu], 3)):
        n_even = X.shape[1] if even else 0
        z = np.full(len(y), np.nan)
        acc = np.zeros(len(out_i))
        for k in range(K):
            tr7, te7 = (fold7 != k) & rows7, (fold7 == k) & rows7
            tri, tei = idx7[tr7], idx7[te7]
            Xtr, ytr = X[tri], y01[tri]
            inner = np.zeros(len(ytr), int)
            for j, (_, iv) in enumerate(StratifiedKFold(D.INNER, shuffle=True,
                                                        random_state=SEED)
                                        .split(np.zeros((len(ytr), 1)), ytr)):
                inner[iv] = j
            Zi = [D._z(Xtr[inner != j], Xtr[inner == j], n_even) for j in range(D.INNER)]
            sc_map, warm = {}, [None] * D.INNER

            def ev(q, _Zi=Zi, _ytr=ytr, _inner=inner, _warm=warm, _sc=sc_map):
                if q in _sc:
                    return
                tot = 0.0
                for j in range(D.INNER):
                    A, Bm = _Zi[j]
                    pen = (np.concatenate([np.full(nl, q[0]),
                                           np.full(A.shape[1] - nl, q[1])]) if even
                           else np.where(np.arange(A.shape[1]) < nl, 0.0, q))
                    v, _ok = D._fit(A, _ytr[_inner != j], pen, even, _warm[j])
                    _warm[j] = v
                    sc = D._scores(v, Bm, n_even, even)
                    yy = _ytr[_inner == j]
                    tot += float((np.logaddexp(0, sc) - yy * sc).sum())
                _sc[q] = tot

            if even:                                   # tie — 2-D 격자 + 축별 자동확장
                gl, gg = list(D.LAMS0), list(D.LAMS)
                for a in sorted(gl, reverse=True):
                    for b in sorted(gg, reverse=True):
                        ev((a, b))
                while True:
                    q = min(sc_map, key=sc_map.get)
                    ext = False
                    if q[0] == max(gl) and max(gl) < D.LAM_CEIL:
                        gl.append(max(gl) * 10.0)
                        for b in gg:
                            ev((gl[-1], b))
                        ext = True
                    if q[1] == max(gg) and max(gg) < D.LAM_CEIL:
                        gg.append(max(gg) * 10.0)
                        for a in gl:
                            ev((a, gg[-1]))
                        ext = True
                    if not ext:
                        break
                q = min(sc_map, key=sc_map.get)
                pen = np.concatenate([np.full(nl, q[0]), np.full(X.shape[1] - nl, q[1])])
            else:                                      # win — 1-D 격자 + 자동확장
                for lam in sorted(set(D.LAMS), reverse=True):
                    ev(lam)
                while min(sc_map, key=sc_map.get) == max(sc_map) and max(sc_map) < D.LAM_CEIL:
                    warm[:] = [None] * D.INNER
                    ev(max(sc_map) * 10)
                q = min(sc_map, key=sc_map.get)
                pen = np.where(np.arange(X.shape[1]) < nl, 0.0, float(q))

            A, Bm = D._z(Xtr, X[np.concatenate([tei, out_i])], n_even)
            v, ok = D._fit(A, ytr, pen, even)
            assert ok, f'{tag} fold{k} 배포적합 미수렴'
            sc = D._scores(v, Bm, n_even, even)
            z[tei] = sc[:len(tei)]
            acc += sc[len(tei):]
            picks.setdefault(tag, []).append(pen.tolist())
        z[out_i] = acc / K
        res[tag] = z
    return res['t'], res['w'], picks


def exp_g(F, U, folds, res, head_seeds=HS_DEFAULT, groups=('vec',)):
    """벌점벡터가 **선택에 쓰이지 않은 행**으로 전이되는가 — 비순환 검정.

    벌점벡터 선택(1b 오라클 · S2 오라클 · LOO · 채택선 t)은 전부 holdout 5,748 에서만 했다.
    나머지 5,000행은 선택에 한 번도 쓰이지 않았으므로 전이 검정이 성립한다.

        hold      5,748  선택셋 — 참조용 (여기 Δ 는 낙관을 포함한다)
        calib2k   2,000  cu 안 · 비-holdout. **전 전문가 청정.** OOF z 로 채점
        dirty3k   3,000  cu 밖. V3 49,729 에 3,000/3,000 포함 → **ABE1 만 in-sample**
                         (현행 train.csv 46,729 에는 0개 — ABE2·TB1·TB2·Arith 청정)
        out5k     5,000  calib2k + dirty3k — 최대 검정력

    ! dirty3k 의 편향 방향: t_abe1·m_abe1·W_abe1·g1·g2 가 과대평가된다. 따라서 ABE1 열을
      버리는 B 에게 **불리한 보수적 검정**이다 — 거기서도 음수면 강한 증거다.
      반대로 TB1 열 제외(H − B)는 dirty3k 에서도 청정하게 검정된다.
    """
    print('\n' + '=' * 78)
    print('[실험g] 선택 밖 행 전이 검정 — calib2k(청정) · dirty3k(ABE1 오염) · out5k')
    print('=' * 78)
    cu, y, nz = U['cu'], U['y'], U['nz']
    hold = U['hold']
    SEG = {'hold': hold, 'calib2k': cu & ~hold, 'dirty3k': ~cu, 'out5k': ~hold}
    for k2, v2 in SEG.items():
        print(f'  {k2:9s} {int(v2.sum()):>6,}행 · 비-tie {int((v2 & nz).sum()):>5,}')
    HD = _heads_ab224(U, head_seeds)
    CELLS = [c for g in groups for c in FCELLS[g]]
    CELLS = [c for c in CELLS if c[2] is None]        # 부스터 셀은 제외 (풀링 자산 불필요)
    CELLS = [('cv-adaptive λ', None, None)] + CELLS   # 계약 원본 — 기준(동결 λ) 대비 Δ
    print(f'  셀 {len(CELLS)}개 × 헤드시드 {len(head_seeds)} × 추첨 {len(folds)}')

    yT, yA = (y == 2).astype(np.float64), (y == 0).astype(np.float64)
    acc = {c[0]: {sg: {} for sg in SEG} for c in CELLS}
    base = {sg: {} for sg in SEG}
    for hs in head_seeds:
        Gt, Wc = HD[hs]
        XT = np.hstack([U['tie_base'], Gt])
        XW = np.hstack([U['win_base'], Wc])
        t0 = time.time()
        for fs, fd7 in folds.items():
            pt0, pw0 = pen_vec(frozenset(), frozenset(), False)
            zt, zw = _run_full(U, fd7, XT, XW, pt0, pw0, y, nz, cu)
            for sg, m in SEG.items():
                base[sg][(hs, fs)] = _seg_rows(zt, zw, yT, yA, m, nz)
            for nm, vk, _f in CELLS:
                if nm == 'cv-adaptive λ':              # 계약 원본 — fold 마다 λ 재선택
                    zt, zw, _pk = _run_adaptive(U, fd7, XT, XW, y, nz, cu)
                else:
                    et, ew = VEC[vk] if vk else (frozenset(), frozenset())
                    pt, pw = pen_vec(et, ew, False)
                    zt, zw = _run_full(U, fd7, XT, XW, pt, pw, y, nz, cu)
                for sg, m in SEG.items():
                    acc[nm][sg][(hs, fs)] = _seg_rows(zt, zw, yT, yA, m, nz)
        print(f'  헤드 시드 {hs} 완료 [{time.time() - t0:.0f}s]')

    keys = sorted(base['hold'])
    out = {}
    print(f'\n  {"셀":14s}' + ''.join(f'{sg:>22s}' for sg in SEG))
    print(f'  {"":14s}' + ''.join(f'{"Δ":>11s}{"t":>11s}' for _ in SEG))
    for nm, _v, _f in CELLS:
        row, o = f'  {nm:14s}', {}
        for sg in SEG:
            dr = np.mean([acc[nm][sg][k2] - base[sg][k2] for k2 in keys], axis=0)
            se = float(dr.std(ddof=1) / np.sqrt(len(dr)))
            t = float(dr.mean() / se) if se > 0 else 0.0
            row += f'{dr.mean():+11.6f}{t:>11.2f}'
            o[sg] = dict(d=float(dr.mean()), t=t, se=se, n=int(len(dr)))
        print(row)
        out[nm] = o
    res['exp_g'] = dict(head_seeds=list(head_seeds), segs={k2: int(v2.sum())
                                                          for k2, v2 in SEG.items()},
                        cells=out)
    print('  ! hold 는 선택셋이라 낙관 포함 · calib2k/dirty3k/out5k 는 선택에 안 쓰인 행')
    print('  ! dirty3k 는 ABE1 만 오염 → ABE1 열을 버리는 안에 **불리한** 보수적 검정')
    print('  ! H − B 차이(TB1 두 열)는 dirty3k 에서도 청정하게 읽힌다')


# ══════════════════════════════════════════ 실험 e — 무라벨 공선성 (기하 전제 검사)
def _norm(X, even):
    """결합기 `_z` 와 같은 정규화 — 짝은 중심화+표준화, 홀은 무중심 RMS(반대칭 보존)."""
    A = np.asarray(X, np.float64).copy()
    if even:
        A -= A.mean(0)
    return A / np.clip(np.sqrt((A ** 2).mean(0)), 1e-12, None)


def _vif(X, even):
    """열별 VIF = diag(inv(2차 모멘트)). 클수록 다른 열들이 그 열을 설명한다."""
    A = _norm(X, even)
    M = A.T @ A / len(A)
    return np.diag(np.linalg.pinv(M)), float(np.linalg.cond(M))


def _r2_on(X, j, keep, even):
    """열 j 를 keep 열들로 회귀한 R² — 1 에 가까우면 그 열은 잉여다."""
    A = _norm(X, even)
    yj, B = A[:, j], A[:, keep]
    coef, *_ = np.linalg.lstsq(B, yj, rcond=None)
    r = yj - B @ coef
    return 1.0 - float((r ** 2).sum()) / max(float((yj ** 2).sum()), 1e-30)


def _train_logits_ab224():
    """train 46,729 로짓 기저 — **arith 는 ab224 OOF**. `D.load_train_logits` 를 이걸로 갈아
    끼워 헤드가 배포 계보(ab224)로 학습되게 한다. 구 arith 계보는 폐기.
    반환 서명은 destnation.load_train_logits 와 동일: (ids, y, tie_tr, win_tr)."""
    za = np.load(MODELS / 'logits/abe2__checkpoint-3000__train.npz')
    ids, y_tr = za['train_ids'], za['train_labels'].astype(np.int64)

    def g(rel):
        z = np.load(MODELS / rel)
        assert (z['train_ids'] == ids).all(), f'{rel} id'
        return ((z['train_fwd'] + z['train_swp']) / 2).astype(np.float64)

    L2 = ((za['train_fwd'] + za['train_swp']) / 2).astype(np.float64)
    Lt2 = g('logits/tb2__checkpoint-4000__train.npz')
    L1 = g(f'logits/abe1__checkpoint-{ABE1_STEP}__train.npz')
    Lt1 = g(f'logits/tb1__checkpoint-{TB1_STEP}__train.npz')
    zar = np.load(CKPT_AB224 / 'oof_train.npz')
    assert (zar['ids_train'] == ids).all() and (zar['y_train'] == y_tr).all(), 'arith OOF 정렬'
    m2, t2 = sd3(L2)
    m1, t1 = sd3(L1)
    mr, tr = sd3(zar['raw_oof_train__xgb'].astype(np.float64))
    return (ids, y_tr,
            np.column_stack([Lt2[:, 1] - Lt2[:, 0], Lt1[:, 1] - Lt1[:, 0], t2, t1, tr]),
            np.column_stack([m2, m1, mr]))


def exp_e(F, res):
    """무라벨 공선성 — 열정리의 전제(잉여 열이 생존 열과 공선)가 기하로 성립하는가.

    라벨이 필요 없으므로 **제출 시점에 test 행에서 그대로 재확인**할 수 있다. 열 삭제의
    전이 논거는 통계가 아니라 기하다 — 공선성이 test 에서도 같으면 논거가 데이터로 선다.
    """
    print('\n' + '=' * 78)
    print('[실험e] 무라벨 공선성 — 라벨 불요 · 제출 시 test 행에서 재확인 가능')
    print('=' * 78)
    _, _, tr_t, tr_w = _train_logits_ab224()
    SETS_full = (('clean7k', np.ones(len(F.y), bool)), ('holdout', F.hold),
                 ('비-hold2k', F.calib))
    out = {}

    for nm, XT, XW, names_t, names_w, nl in (
            ('전 7열 (로짓+헤드)', F.XT, F.XW, TIE_NAMES, WIN_NAMES, 7),
            ('로짓만', F.XT[:, :5], F.XW[:, :3], TIE_NAMES[:5], WIN_NAMES[:3], 5)):
        print(f'\n  ── {nm} ──')
        for ax, X, nms, even in (('tie', XT, names_t, True), ('win', XW, names_w, False)):
            print(f'  {ax} VIF' + ''.join(f'{n:>10s}' for n in nms) + f'{"cond":>9s}')
            rows = list(SETS_full)
            if nm == '로짓만':                      # train 은 헤드 열이 없어 로짓만 가능
                rows = rows + [('train46k', None)]
            for rn, msk in rows:
                Xs = (tr_t[:, :5] if ax == 'tie' else tr_w[:, :3]) if msk is None else X[msk]
                v, cd = _vif(Xs, even)
                print(f'  {rn:>8s}' + ''.join(f'{x:>10.2f}' for x in v) + f'{cd:>9.1f}')
                out[f'{nm}|{ax}|{rn}'] = dict(vif=[float(x) for x in v], cond=cd)

    print('\n  ── 잉여 후보의 생존열 설명력 R² (1 에 가까울수록 잉여) ──')
    CHK = (('tie', 's_tb1', 1, [0, 2, 3, 4, 5, 6], True),
           ('tie', 't_abe1', 3, [0, 1, 2, 4, 5, 6], True),
           ('win', 'm_abe2', 0, [1, 2, 3, 4, 5, 6], False),
           ('win', 'W_tb1', 6, [0, 1, 2, 3, 4, 5], False))
    for ax, cn, j, keep, even in CHK:
        X = F.XT if ax == 'tie' else F.XW
        vals = {rn: _r2_on(X[msk], j, keep, even) for rn, msk in SETS_full}
        print(f'  {cn:>8s} | 생존 6열  ' + ' · '.join(f'{k}={v:.4f}' for k, v in vals.items()))
        out[f'r2|{cn}'] = {k: float(v) for k, v in vals.items()}
    for ax, cn, j, keep in (('tie', 's_tb1', 1, [0, 2, 3, 4]),
                            ('tie', 't_abe1', 3, [0, 1, 2, 4]),
                            ('win', 'm_abe2', 0, [1, 2])):
        even = ax == 'tie'
        Xs = F.XT[:, :5] if ax == 'tie' else F.XW[:, :3]
        Xt_ = tr_t[:, :5] if ax == 'tie' else tr_w[:, :3]
        v7 = {rn: _r2_on(Xs[msk], j, keep, even) for rn, msk in SETS_full}
        v7['train46k'] = _r2_on(Xt_, j, keep, even)
        print(f'  {cn:>8s} | 로짓 생존만 ' + ' · '.join(f'{k}={x:.4f}' for k, x in v7.items()))
        out[f'r2logit|{cn}'] = {k: float(x) for k, x in v7.items()}
    res['exp_e'] = out
    print('  ! 행집합이 바뀌어도 VIF·R² 가 같으면 기하 전제가 표본 특이하지 않다는 뜻이다.')
    print('  ! train46k 는 결합기가 한 번도 안 본 46,729행 — 홀드아웃과 독립인 유일한 확인처.')


# ══════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--exps', default='1,1c,2,a,b,c,d,e,f,g')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--fcells', default='vec,boost,combo,loo')
    ap.add_argument('--out', default='temp/finlab')
    args = ap.parse_args()
    want = set(args.exps.split(','))
    t_all = time.time()

    print('[적재]')
    U = build()
    F = Frame(U)
    fold_seeds = list(range(42, 42 + args.reps))
    folds = {fs: fold_of(F.y, fs) for fs in fold_seeds}

    print(f'\n[베이스라인] clean7k · λ 동결 {FROZEN_TIE}/{FROZEN_WIN:g} · 추첨 {fold_seeds}')
    base_by = {}
    for fs in fold_seeds:
        b = F.run(folds[fs])
        base_by[fs] = b
        print(f'  f{fs}  tie {b["tie"]:.5f} · win {b["win"]:.5f} → {b["tot"]:.5f}')
    tots = np.array([base_by[fs]['tot'] for fs in fold_seeds])
    print(f'  평균 {tots.mean():.5f} · sd {tots.std(ddof=1):.5f}' if len(tots) > 1 else '')

    res = {'baseline': {str(fs): {k: float(base_by[fs][k]) for k in ('tie', 'win', 'tot')}
                        for fs in fold_seeds},
           'frozen_lam': dict(tie=list(FROZEN_TIE), win=FROZEN_WIN),
           'fold_seeds': fold_seeds}
    if '1' in want:
        exp1(F, folds, res)
    if '1c' in want:
        exp_1c(F, folds, res)
    if '2' in want:
        exp2(F, folds, res)
    if 'a' in want:
        exp_a(F, folds, res)
    if 'b' in want:
        exp_b(F, U, folds, res)
    if 'c' in want:
        exp_c(F, U, folds, res)
    if 'd' in want:
        exp_d(F, U, folds, res)
    if 'e' in want:
        exp_e(F, res)
    if 'f' in want:
        exp_f(F, U, folds, res,
              groups=tuple(x for x in args.fcells.split(',') if x))
    if 'g' in want:
        exp_g(F, U, folds, res,
              groups=tuple(x for x in args.fcells.split(',') if x))

    op = FINAL / args.out
    with open(op.with_suffix('.json'), 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print(f'\n저장: {op}.json')
    print(f'채택선: 전 추첨 같은 부호 & 평균 Δ ≤ −{GATE_D:g} (오라클 천장은 채택 근거 아님)')
    print(f'총 {(time.time() - t_all) / 60:.1f}분')


if __name__ == '__main__':
    main()
