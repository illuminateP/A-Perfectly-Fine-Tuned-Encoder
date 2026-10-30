"""blend.py — 세 전문가의 '확률'을 scipy 로 섞는다: 산술 풀(혼합) vs 기하 풀(로짓 가중).

배경
  로짓 공간의 선형 결합은 이미 있다 — Davidson κ=1 (0.9705) 이 정확히 그것이다.
  따라서 여기서 새로 잴 것은 하나다: **산술 풀(확률 공간의 혼합)**.
      기하 풀  p* = σ(a₀ + Σ aᵢ tᵢ)         — 불일치 행에서 서로를 곱으로 깎는다 (교집합)
      산술 풀  p* = Σ wᵢ σ(cᵢ tᵢ + bᵢ)      — 양쪽 의견을 다 살린다 (혼합)
  산술 풀은 기하 점수의 단조 변환이 아니라 **행 순위 자체가 바뀐다**. 그래서 '보정기
  8계열 기각·낙관 상한 +0.0001'(캘리브레이션 폐쇄) 결론에 덮이지 않는 자유도다.
  합의 행에선 두 풀이 거의 같으니 승부는 전문가가 갈리는 행에서 난다.

입력 계약 — reprhead.py 의 W0/T0 과 동일 (log_repr.txt 와 나란히 읽기 위함)
  L_ABE  model_ens/boost_meta.json 의 steps 로짓 평균(TTA)      L_TB  동 meta (tb3)
  L_ARI  arith_experts_raw.npz 부스터 3종 raw 로짓 평균 (S4 팔만 3종 분리)
  좌표:  m = z_A−z_B (홀)   t = z_tie−logaddexp(z_A,z_B) (짝)   s_tie = z_tie−z_notie (짝)
  승부축 입력 [m3(ABE), m_ar(Arith)] — TB 는 승부 출력이 없다.
  tie축  입력 [s_tie(TB), t3(ABE), t_ar(Arith)]

팔 (각 축 독립 — 파라미터가 안 겹치므로 축별 최적화 = 3-class 최적화와 동치)
  G0(기하 풀)           로짓 가중. tie 절편 허용, 승부 절편 금지(반대칭). W0/T0 재현이 나와야 한다.
  A1(산술 풀·원확률)    p* = Σ wᵢ σ(tᵢ). 전문가가 내뱉은 확률 그대로, 심플렉스 가중만.
                        (온도 미적용 확률이라 보정 손해를 안고 시작한다 — A2 가 그걸 걷어낸 판)
  A2(산술 풀·보정)      p* = Σ wᵢ σ(cᵢtᵢ+bᵢ). 전문가별 온도역수 cᵢ·절편 bᵢ(tie축만)를 fold 안에서 적합.
  C3(볼록결합)          γ·A2 + (1−γ)·G0, γ∈[0,1] 을 fold 안에서 적합 — 데이터가 풀을 고른다.
  S4(산술 풀·Arith분리) A2 에서 Arith 부스터 3종을 따로 (tie 5·승부 4 전문가).

검증 — Davidson.py 와 같은 방식
  stage2 10,748 을 같은 StratifiedKFold(5, shuffle, seed 42) 로 5회 교차적합.
  fold k 의 예측은 나머지 4 fold 파라미터로만 만든다. 지표는 OOF 풀링 손실
  (전체 + holdout 부분집합)과 paired bootstrap Δ (기준 G0, 문턱 0.0005).
  승부축 손실은 **비-tie 행당** — log_repr.txt 와 같은 단위라 표가 그대로 겹쳐 읽힌다.

swap 등변성은 구조로 보장된다: 승부축에 절편·bᵢ 가 없으면
  σ(Σa(−m)) = 1−σ(Σa m),  Σw σ(−cm) = 1−Σw σ(cm).  수치 검산도 찍는다.

사용 (수십 초)
  python Boost/blend.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.model_selection import StratifiedKFold

sys.stdout.reconfigure(line_buffering=True)

LLM = Path('.')
BOOST, ARITH = LLM / 'Boost', LLM / 'Arith'
SEED = 42
N_SPLITS = 5
N_START = 3          # 산술 풀은 비볼록 — 다중 시작 후 학습손실 최선을 취한다
N_BOOT = 2000
THRESH = 0.0005
PEPS = 1e-12
HALF = 5000          # 앞 5,000 = valid, 뒤 5,748 = holdout
ARITH_MODELS = ('lgbm', 'xgb', 'cat')

# log_repr.txt (reprhead.py, 같은 fold·같은 입력 계약) — 나란히 읽을 참조행
#   (전체 OOF 손실, holdout 손실)
REPR_WIN = {'W0(로짓만)': (0.5739, 0.5662), 'W1(TB 표현 홀 추가, 최선)': (0.5730, 0.5650)}
REPR_TIE = {'T0(로짓만)': (0.5740, 0.5734), 'T2(TB 표현 짝 추가, 최선)': (0.5716, 0.5713)}
REPR_TOTAL = {'로짓 기준 W0/T0': 0.9705, '표현 최선 W1/T2': 0.9675}


# ══════════════════════════════════════════════════════ 지표
def bce_rows(p, y01):
    p = np.clip(p, PEPS, 1 - PEPS)
    return -(y01 * np.log(p) + (1 - y01) * np.log(1 - p))


def paired_ci(ll_a, ll_b, n_boot=N_BOOT, seed=SEED):
    """Δ = mean(ll_a − ll_b). 양수면 b 가 좋다."""
    d = np.asarray(ll_a) - np.asarray(ll_b)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def ci_str(m, lo, hi):
    sig = '유의' if (lo > 0 or hi < 0) else '노이즈'
    return f'{m:+.4f} [{lo:+.4f},{hi:+.4f}] {sig}' + (' ★' if sig == '유의' and abs(m) >= THRESH else '')


# ══════════════════════════════════════════════════════ 신호 로드 (reprhead 와 동일 계약)
def load_signals():
    z = np.load(BOOST / 'boost_oof.npz')
    ids, y = z['ids'], z['y']
    meta = json.loads((BOOST / 'model_ens/boost_meta.json').read_text(encoding='utf-8'))
    ex = meta['experts']

    def stack(tag, steps):
        acc = []
        for s in steps:
            w = np.load(BOOST / f'logits/{tag}__checkpoint-{s}.npz')
            assert (np.concatenate([w['valid_ids'], w['hold_ids']]) == ids).all(), f'{tag} id 불일치'
            assert (np.concatenate([w['valid_labels'], w['hold_labels']]) == y).all(), f'{tag} 라벨 불일치'
            acc.append(np.vstack([(w['valid_fwd'] + w['valid_swp']) / 2,
                                  (w['hold_fwd'] + w['hold_swp']) / 2]).astype(np.float64))
        return np.mean(acc, axis=0), steps

    L_ABE, abe_steps = stack('abe', ex['abe'].get('steps') or [ex['abe']['step']])
    L_TB, tb_steps = stack('tb3', ex['tb'].get('steps') or [ex['tb']['step']])

    za = np.load(ARITH / 'outputs/arith_experts_raw.npz')
    assert (np.concatenate([za['ids_valid'], za['ids_hold']]) == ids).all(), 'Arith id 불일치'
    raw = {m: np.vstack([za[f'raw_valid__{m}'], za[f'raw_hold__{m}']]).astype(np.float64)
           for m in ARITH_MODELS}
    L_ARI = np.mean(list(raw.values()), axis=0)

    sd = lambda B: (B[:, 0] - B[:, 1], B[:, 2] - np.logaddexp(B[:, 0], B[:, 1]))
    m3, t3 = sd(L_ABE)
    m_ar, t_ar = sd(L_ARI)
    s_tie = L_TB[:, 1] - L_TB[:, 0]
    m_sep, t_sep = {}, {}
    for m in ARITH_MODELS:
        m_sep[m], t_sep[m] = sd(raw[m])

    WIN = np.column_stack([m3, m_ar])
    TIE = np.column_stack([s_tie, t3, t_ar])
    WIN_SEP = np.column_stack([m3] + [m_sep[m] for m in ARITH_MODELS])
    TIE_SEP = np.column_stack([s_tie, t3] + [t_sep[m] for m in ARITH_MODELS])
    names = {
        'win': ['m3(ABE)', 'm_ar(Arith)'],
        'tie': ['s_tie(TB)', 't3(ABE)', 't_ar(Arith)'],
        'win_sep': ['m3(ABE)'] + [f'm_{m}' for m in ARITH_MODELS],
        'tie_sep': ['s_tie(TB)', 't3(ABE)'] + [f't_{m}' for m in ARITH_MODELS],
    }
    assert all(np.isfinite(A).all() for A in (WIN, TIE, WIN_SEP, TIE_SEP)), '비유한 신호'
    return ids, y, WIN, TIE, WIN_SEP, TIE_SEP, names, abe_steps, tb_steps


# ══════════════════════════════════════════════════════ 풀 (pool)
def sigm(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def fit_geo(Z, y01, bias):
    """기하 풀 = 무규제 로지스틱 (볼록). 해석적 gradient 로 L-BFGS-B."""
    n = len(y01)
    X = np.hstack([np.ones((n, 1)), Z]) if bias else Z

    def f(w):
        z = X @ w
        return float(np.mean(np.logaddexp(0, z) - y01 * z))

    def g(w):
        return X.T @ (sigm(X @ w) - y01) / n

    r = minimize(f, np.zeros(X.shape[1]), jac=g, method='L-BFGS-B')
    assert r.success or r.fun < f(np.zeros(X.shape[1])), f'geo 적합 실패: {r.message}'
    return r.x


def pred_geo(Z, w, bias):
    X = np.hstack([np.ones((len(Z), 1)), Z]) if bias else Z
    return sigm(X @ w)


def amix(theta, Z, mode):
    """산술 풀. mode 'w': 가중만 | 'cw': 온도역수+가중 | 'cbw': 온도역수+절편+가중."""
    K = Z.shape[1]
    u = theta[:K]
    c = theta[K:2 * K] if mode != 'w' else np.ones(K)
    b = theta[2 * K:3 * K] if mode == 'cbw' else np.zeros(K)
    w = np.exp(u - u.max())
    w /= w.sum()
    return sigm(Z * c + b) @ w


def amix_dim(K, mode):
    return K * (1 + (mode != 'w') + (mode == 'cbw'))


def fit_amix(Z, y01, mode, seed):
    """비볼록이라 다중 시작 (균등 1 + 섭동 2, 시드 고정). gradient 는 수치미분 —
    파라미터 ≤15개·행 ≤1만이라 FD 비용이 무시된다."""
    K = Z.shape[1]
    x0 = np.zeros(amix_dim(K, mode))
    if mode != 'w':
        x0[K:2 * K] = 1.0

    def f(th):
        return float(bce_rows(amix(th, Z, mode), y01).mean())

    rng = np.random.default_rng(seed)
    best = None
    for s in range(N_START):
        th0 = x0 if s == 0 else x0 + rng.normal(0, 0.3, len(x0))
        r = minimize(f, th0, method='L-BFGS-B')
        if best is None or r.fun < best.fun:
            best = r
    return best.x


# ══════════════════════════════════════════════════════ 축 하나의 교차적합
def run_axis(axis, Z, Zsep, y01, fold, fitmask):
    """fitmask: 적합·채점에 쓰는 행 (승부축 = 비-tie). 예측은 전 행에 만든다 —
    3-class 조립에는 tie 행의 승부확률도 필요하고, 항등식 덕에 채점엔 안 들어간다."""
    bias = axis == 'tie'
    mode = 'cbw' if bias else 'cw'
    n = len(y01)
    arms = ('G0', 'A1', 'A2', 'C3', 'S4')
    oof = {a: np.zeros(n) for a in arms}
    prm = {a: [] for a in arms}
    t0 = time.time()
    for k in range(N_SPLITS):
        tr = (fold != k) & fitmask
        te = fold == k
        w_g = fit_geo(Z[tr], y01[tr], bias)
        th1 = fit_amix(Z[tr], y01[tr], 'w', seed=SEED + 10 * k)
        th2 = fit_amix(Z[tr], y01[tr], mode, seed=SEED + 10 * k + 1)
        th4 = fit_amix(Zsep[tr], y01[tr], mode, seed=SEED + 10 * k + 2)
        pg_tr, pa_tr = pred_geo(Z[tr], w_g, bias), amix(th2, Z[tr], mode)
        gam = float(minimize_scalar(lambda g: float(bce_rows(g * pa_tr + (1 - g) * pg_tr, y01[tr]).mean()),
                                    bounds=(0.0, 1.0), method='bounded').x)
        oof['G0'][te] = pred_geo(Z[te], w_g, bias)
        oof['A1'][te] = amix(th1, Z[te], 'w')
        oof['A2'][te] = amix(th2, Z[te], mode)
        oof['C3'][te] = gam * oof['A2'][te] + (1 - gam) * oof['G0'][te]
        oof['S4'][te] = amix(th4, Zsep[te], mode)
        for a, p in (('G0', w_g), ('A1', th1), ('A2', th2), ('C3', gam), ('S4', th4)):
            prm[a].append(p)

    # swap 등변 검산 (승부축만 의미) — fold 0 파라미터로 f(−m) = 1−f(m) 을 직접 잰다
    if axis == 'win':
        sub = Z[:1000]
        viol = {
            'G0': np.abs(pred_geo(-sub, prm['G0'][0], False) - (1 - pred_geo(sub, prm['G0'][0], False))).max(),
            'A1': np.abs(amix(prm['A1'][0], -sub, 'w') - (1 - amix(prm['A1'][0], sub, 'w'))).max(),
            'A2': np.abs(amix(prm['A2'][0], -sub, 'cw') - (1 - amix(prm['A2'][0], sub, 'cw'))).max(),
            'S4': np.abs(amix(prm['S4'][0], -Zsep[:1000], 'cw') - (1 - amix(prm['S4'][0], Zsep[:1000], 'cw'))).max(),
        }
        worst = max(viol.values())
        print(f'  [검산] 승부축 swap 등변 max|f(−m)−(1−f(m))| = {worst:.2e}  '
              + ('OK — 절편 없는 구조가 반대칭을 보장' if worst < 1e-12 else '** 위반'))
    print(f'  {("tie축" if bias else "승부축")} 5-fold 교차적합 완료 ({time.time() - t0:.0f}s)')
    return oof, prm


# ══════════════════════════════════════════════════════ main
def main():
    ids, y, WIN, TIE, WIN_SEP, TIE_SEP, names, abe_steps, tb_steps = load_signals()
    n = len(y)
    hold = np.arange(n) >= HALF
    y_tie = (y == 2).astype(np.float64)
    nz = y != 2
    y_win = (y == 0).astype(np.float64)          # 비-tie 행에서만 채점
    tie_rate = float(y_tie.mean())

    fold = np.zeros(n, int)
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    for k, (_, te) in enumerate(skf.split(np.zeros((n, 1)), y)):
        fold[te] = k

    print(f'stage2 {n:,}행 (valid {HALF:,} + hold {n - HALF:,}) | fold {np.bincount(fold)}')
    print(f'ABE steps {list(abe_steps)} 평균 / TB3 steps {list(tb_steps)} | Arith raw 3종 '
          f'(기본 팔은 평균, S4 만 분리)')
    print(f'tie율 {tie_rate:.4f} (상수 기준선 tie축 0.6183 · 승부축 0.6931)')
    print(f'문턱 {THRESH} + paired bootstrap 95% CI | 산술 풀 다중시작 {N_START}회\n')

    ARM_NAME = {
        'G0': 'G0(기하 풀 — 로짓 가중, W0/T0·Davidson κ=1 재현)',
        'A1': 'A1(산술 풀 — 원확률 그대로 심플렉스 혼합)',
        'A2': 'A2(산술 풀 — 전문가별 온도·절편 보정 후 혼합)',
        'C3': 'C3(볼록결합 γ·A2 + (1−γ)·G0, γ는 fold 안 적합)',
        'S4': 'S4(산술 풀 A2 + Arith 부스터 3종 분리)',
    }
    arms = ('G0', 'A1', 'A2', 'C3', 'S4')

    oof_w, prm_w = run_axis('win', WIN, WIN_SEP, y_win, fold, nz)
    oof_t, prm_t = run_axis('tie', TIE, TIE_SEP, y_tie, fold, np.ones(n, bool))

    # ── [1] 승부축 ──
    def axis_table(title, oof, yy, mask, repr_ref):
        print('=' * 104)
        print(title)
        print('=' * 104)
        rows_all = {a: bce_rows(oof[a], yy)[mask] for a in arms}
        hold_m = hold[mask]
        print(f'  {"팔":<52} {"전체":>8} {"holdout":>8}   Δ vs G0 (전체, 양수=이득)')
        print('  ' + '-' * 100)
        for a in arms:
            full = float(rows_all[a].mean())
            ho = float(rows_all[a][hold_m].mean())
            if a == 'G0':
                delta = '기준'
            else:
                delta = ci_str(*paired_ci(rows_all['G0'], rows_all[a]))
            print(f'  {ARM_NAME[a]:<52} {full:8.4f} {ho:8.4f}   {delta}')
        for nm, (full, ho) in repr_ref.items():
            print(f'  {"(참조) repr " + nm:<52} {full:8.4f} {ho:8.4f}   log_repr.txt (ridge, 같은 fold)')
        print()
        return rows_all

    rows_w = axis_table(f'[1] 승부축 (A 가 이겼나) — 비-tie {int(nz.sum()):,}행, 행당 손실. '
                        f'절편·bᵢ 없음(반대칭)', oof_w, y_win, nz, REPR_WIN)
    rows_t = axis_table(f'[2] tie축 (비겼나) — 전 {n:,}행, 행당 손실', oof_t, y_tie,
                        np.ones(n, bool), REPR_TIE)

    # ── [3] 학습된 가중치 — '어떤 전문가를 얼마나 믿나' (fold 평균 ± 표준편차) ──
    print('=' * 104)
    print('[3] 학습된 가중치 (fold 평균 ± std) — 산술 풀 w 는 심플렉스, c 는 온도역수, b 는 절편')
    print('=' * 104)

    def geo_report(axis, prm, nm, bias):
        P = np.array(prm)
        off = 1 if bias else 0
        cols = (['절편'] if bias else []) + nm
        body = '  '.join(f'{cols[i]} {P[:, i].mean():+.3f}±{P[:, i].std(ddof=1):.3f}'
                         for i in range(off + len(nm)))
        print(f'  {axis} G0(기하 풀)   {body}')

    def amix_report(axis, prm, nm, mode):
        TH = np.array(prm)
        K = len(nm)
        W = np.exp(TH[:, :K] - TH[:, :K].max(1, keepdims=True))
        W /= W.sum(1, keepdims=True)
        C = TH[:, K:2 * K] if mode != 'w' else np.ones((len(TH), K))
        B = TH[:, 2 * K:3 * K] if mode == 'cbw' else np.zeros((len(TH), K))
        for i, name in enumerate(nm):
            extra = '' if mode == 'w' else f'  c {C[:, i].mean():+.3f}±{C[:, i].std(ddof=1):.3f}'
            extra += f'  b {B[:, i].mean():+.3f}±{B[:, i].std(ddof=1):.3f}' if mode == 'cbw' else ''
            print(f'      {name:<14} w {W[:, i].mean():.3f}±{W[:, i].std(ddof=1):.3f}{extra}')

    for axis, prm, nm, nm_sep, mode in (('승부축', prm_w, names['win'], names['win_sep'], 'cw'),
                                        ('tie축 ', prm_t, names['tie'], names['tie_sep'], 'cbw')):
        geo_report(axis, prm['G0'], nm, mode == 'cbw')
        print(f'  {axis} A1(원확률 혼합)')
        amix_report(axis, prm['A1'], nm, 'w')
        print(f'  {axis} A2(보정 혼합)')
        amix_report(axis, prm['A2'], nm, mode)
        print(f'  {axis} S4(Arith 분리)')
        amix_report(axis, prm['S4'], nm_sep, mode)
        G = np.array(prm['C3'], dtype=np.float64)
        print(f'  {axis} C3 γ(산술 쪽 비중) {G.mean():.3f} ± {G.std(ddof=1):.3f}   '
              f'(0=기하 풀만, 1=산술 풀만)')
        print()

    # ── [4] 종합 — 3-class 조립 + 항등식 검산 ──
    print('=' * 104)
    print('[4] 종합 — p = [(1−p_tie)·p_A|승부, (1−p_tie)·(1−p_A|승부), p_tie] 로 조립한 총 ll')
    print('=' * 104)
    nz_frac = float(nz.mean())
    print(f'  항등식: 총 ll = tie축 + {nz_frac:.4f}·승부축(비-tie 행당). 매 팔에서 1e-9 로 검산한다.\n')
    P3 = {}
    print(f'  {"구성":<52} {"tie축":>7} {"승부축":>7} {"총(전체)":>9} {"총(hold)":>9}')
    print('  ' + '-' * 100)
    for a in arms:
        pt, pw = oof_t[a], oof_w[a]
        p3 = np.column_stack([(1 - pt) * pw, (1 - pt) * (1 - pw), pt])
        P3[a] = p3
        row3 = -np.log(np.clip(p3[np.arange(n), y], PEPS, 1.0))
        total = float(row3.mean())
        tie_l = float(rows_t[a].mean())
        win_l = float(rows_w[a].mean())
        assert abs(total - tie_l - nz_frac * win_l) < 1e-9, f'{a} 축 분해 검산 실패'
        print(f'  {ARM_NAME[a]:<52} {tie_l:7.4f} {win_l:7.4f} {total:9.4f} '
              f'{float(row3[hold].mean()):9.4f}')
    for nm, tot in REPR_TOTAL.items():
        print(f'  {"(참조) repr " + nm:<52} {"":>7} {"":>7} {tot:9.4f}')

    # 사후 최선 조합 (낙관적 — 채택 판정은 축별 CI 로)
    best_t = min(arms, key=lambda a: rows_t[a].mean())
    best_w = min(arms, key=lambda a: rows_w[a].mean())
    print(f'\n  축별 사후 최선: tie축 {best_t} + 승부축 {best_w} -> 총 '
          f'{rows_t[best_t].mean() + nz_frac * rows_w[best_w].mean():.4f}  (사후 선택이라 낙관적)')

    # 저장된 Davidson κ=1 OOF 와의 앵커 대조 (같은 행·같은 fold 구조)
    zdav = np.load(BOOST / 'davidson_oof.npz')
    if (zdav['ids'] == ids).all() and 'oof__DV-분리 (κ=1)' in zdav:
        pdv = np.asarray(zdav['oof__DV-분리 (κ=1)'], np.float64)
        rdv = -np.log(np.clip(pdv[np.arange(n), y] / pdv.sum(1), PEPS, 1.0))
        rg0 = -np.log(np.clip(P3['G0'][np.arange(n), y], PEPS, 1.0))
        print(f'  (검산) 저장된 Davidson κ=1 OOF 총 {float(rdv.mean()):.4f} | '
              f'G0 과의 paired Δ {ci_str(*paired_ci(rdv, rg0))} — 0 부근이어야 정상')

    np.savez_compressed(BOOST / 'blend_oof.npz', ids=ids, y=y,
                        **{f'oof__{a}': P3[a].astype(np.float32) for a in arms})
    print(f'\nsaved: {BOOST}/blend_oof.npz (팔별 3-class OOF — 이후 paired 비교용)')
    print('다음 단계: 유의 팔이 나오면 stage2 전량 재적합 가중치를 배포 계약(JSON)으로 굳힌다.')


if __name__ == '__main__':
    main()
