"""lam_ceiling.py — λ 격자의 **오라클 천장**. λ 탐색을 얼려도 되는가를 판정한다.

전제: λ 가 곧 온도다
    층2 결합기에서 각 전문가를 얼마나 믿을지는 λ 가 정한다 — tie 는 [λ_L(원 로짓 블록),
    λ_g(헤드 열 블록)], 승부는 [m3 무벌점 | W4 λ]. 사후 스칼라 온도는 `T·z = X @ (T·v)`
    로 v 의 노름에 흡수되지만, λ 는 축소 방향까지 바꾸므로 **엄격히 더 표현력이 크다.**
    그러니 보정 여지는 λ 격자 위에서 재는 것이 맞다.

무엇을 재나 (우주별 · 축별로 독립)
    총 손실이 tie ll + R_NZ × win ll 로 정확히 분해되므로 두 축을 따로 쓸어도 최적이 같다.

    cv-adaptive   현행 — fold 마다 안쪽 3-fold 가 λ 를 따로 고름 (= 보고된 숫자)
    cv-median     배포와 같은 조건 — fold 선택의 중앙값 λ 를 전 fold 에 고정
    oracle        격자 전체를 쓸어 **홀드아웃 손실을 최소화하는 λ**. 낙관 상한

    판정            oracle − cv-median  =  λ 선택이 남긴 여지의 상한
                    격자 표면의 평탄도  =  λ 를 얼려도 되는가의 직접 증거

전제 자산 (새 학습·새 추출 없음)
    temp/dest_universe.npz          Gt_s42 / W_s42 (헤드 열)
    temp/dest_universe_arith.npz    L_ar (arith stage2 마진 — 재추출본, 플립 4행)
    models/logits/*.npz             인코더 stage2 로짓 4벌

실행:  conda activate ML2 && cd final && python temp/lam_ceiling.py
예상:  1~2분 (train_full.csv 파싱 + 격자 350여 회 적합)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold, train_test_split

FINAL = Path(__file__).resolve().parent.parent
DATA_DIR, MODELS = FINAL / 'origin', FINAL / 'models'

SEED, HALF, K, MAXIT = 42, 5000, 5, 1000
ABE1_STEP, TB1_STEP = 3109, 5000
R_NZ = 3972 / 5748
PLAN_TIE, PLAN_WIN = 0.5683157036097410, 0.5638882907650488
PLAN_TOT = PLAN_TIE + R_NZ * PLAN_WIN

# 계약 격자(destnation) = LAMS0 × LAMS. 오라클이 경계 밖을 원하는지 보려고 한 칸씩 넓힌다.
LAMS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
CONTRACT_L, CONTRACT_G = [0.0] + LAMS, LAMS            # tie: λ_L, λ_g
CONTRACT_W = LAMS                                      # win: λ
GRID_L = [0.0] + LAMS + [100000.0]
GRID_G = [0.1] + LAMS + [100000.0]
GRID_W = [0.0, 0.1] + LAMS + [100000.0]

sd3 = lambda B: (B[:, 0] - B[:, 1], B[:, 2] - np.logaddexp(B[:, 0], B[:, 1]))


# ══════════════════════════════════════ 결합기 원본 (destnation.py 그대로)
def rowloss(z, y01):
    return np.logaddexp(0, z) - y01 * z


def _fit(X, y01, pen, bias, w0=None):
    n, p = X.shape
    pen = np.asarray(pen, np.float64)

    def fg(v):
        w = v[:p]
        z = X @ w
        if bias:
            z = z + v[p]
        val = float(np.mean(np.logaddexp(0, z) - y01 * z) + np.dot(pen * w, w) / n)
        r = (expit(z) - y01) / n
        gw = X.T @ r + 2.0 * pen * w / n
        return val, (np.concatenate([gw, [r.sum()]]) if bias else gw)

    v0 = np.zeros(p + (1 if bias else 0)) if w0 is None else w0
    res = minimize(fg, v0, jac=True, method='L-BFGS-B',
                   options={'maxiter': MAXIT, 'maxcor': 20, 'ftol': 1e-12, 'gtol': 1e-7})
    return res.x, bool(res.success)


def _z(Xtr, Xte, n_even):
    A, B = Xtr.copy(), Xte.copy()
    if n_even:
        mu = A[:, :n_even].mean(0)
        sd = np.clip(A[:, :n_even].std(0), 1e-9, None)
        A[:, :n_even] = (A[:, :n_even] - mu) / sd
        B[:, :n_even] = (B[:, :n_even] - mu) / sd
    if n_even < A.shape[1]:
        rms = np.clip(np.sqrt((A[:, n_even:] ** 2).mean(0)), 1e-9, None)
        A[:, n_even:] /= rms
        B[:, n_even:] /= rms
    return A, B


def _scores(v, B, n_even, bias):
    p = B.shape[1]
    w = v[:p]
    b = float(v[p]) if bias else 0.0
    ze = B[:, :n_even] @ w[:n_even] if n_even else 0.0
    zo = B[:, n_even:] @ w[n_even:] if n_even < p else 0.0
    return ze + zo + b


def oof_forced(X, y01, fold, rows, lam, even, n_logit):
    """λ 를 전 fold 에 **고정**한 교차적합 OOF. 안쪽 3-fold 없음 (그게 이 런의 요점)."""
    n_even = X.shape[1] if even else 0
    z = np.full(len(y01), np.nan)
    bad = 0
    for k in range(K):
        tr, te = (fold != k) & rows, (fold == k) & rows
        A, B = _z(X[tr], X[te], n_even)
        if even:
            pen = np.concatenate([np.full(n_logit, lam[0]),
                                  np.full(X.shape[1] - n_logit, lam[1])])
        else:
            pen = np.full(X.shape[1], float(lam))
            pen[:n_logit] = 0.0
        v, ok = _fit(A, y01[tr], pen, even)
        bad += (not ok)
        z[te] = _scores(v, B, n_even, even)
    return z, bad


# ══════════════════════════════════════ 자산 재구성
def _tta(rel, n_class):
    z = np.load(MODELS / rel)
    L = np.vstack([(z['valid_fwd'] + z['valid_swp']) / 2,
                   (z['hold_fwd'] + z['hold_swp']) / 2]).astype(np.float64)
    assert L.shape[1] == n_class, f'{rel} 열 {L.shape[1]}'
    return L, z


def build():
    """stage2 로짓 기저 + 행 우주 — dest_universe.build() 와 같은 원천·같은 규약."""
    L_abe2, za = _tta('logits/abe2__checkpoint-3000.npz', 3)
    ids = np.concatenate([za['valid_ids'], za['hold_ids']])
    y = np.concatenate([za['valid_labels'], za['hold_labels']]).astype(np.int64)
    n = len(y)
    assert n == 10748, f'stage2 {n}행'
    L_tb2, _ = _tta('logits/tb2__checkpoint-4000.npz', 2)
    L_abe1, _ = _tta(f'logits/abe1__checkpoint-{ABE1_STEP}.npz', 3)
    L_tb1, _ = _tta(f'logits/tb1__checkpoint-{TB1_STEP}.npz', 2)

    zc = np.load(FINAL / 'temp/dest_universe_arith.npz', allow_pickle=False)
    assert (zc['ids'] == ids).all(), 'arith 캐시 id 정렬'
    L_ar = zc['L_ar'].astype(np.float64)
    print(f'  arith 캐시 재사용 — {str(zc["ckpt"])} · 판정본 대비 플립 {int(zc["n_flip"])}행')

    m_abe2, t_abe2 = sd3(L_abe2)
    m_abe1, t_abe1 = sd3(L_abe1)
    m_ar, t_ar = sd3(L_ar)
    tie_base = np.column_stack([L_tb2[:, 1] - L_tb2[:, 0], L_tb1[:, 1] - L_tb1[:, 0],
                                t_abe2, t_abe1, t_ar])
    win_base = np.column_stack([m_abe2, m_abe1, m_ar])

    full = pd.read_csv(DATA_DIR / 'train_full.csv',
                       usecols=['id', 'winner_model_a', 'winner_model_b'])
    assert len(full) == 57477, f'train_full.csv {len(full)}행'
    lab = np.select([full['winner_model_a'] == 1, full['winner_model_b'] == 1], [0, 1], 2)
    tr51, _ = train_test_split(full.assign(label=lab), test_size=0.10, stratify=lab,
                               random_state=42, shuffle=True)
    _, va2 = train_test_split(tr51, test_size=2000, stratify=tr51['label'],
                              random_state=42, shuffle=True)
    cu = np.ones(n, bool)
    cu[:HALF] = pd.Index(va2['id']).get_indexer(ids[:HALF]) >= 0
    assert int(cu.sum()) == 7748, f'clean {int(cu.sum())}'
    hold, nz = np.arange(n) >= HALF, y != 2
    assert cu[hold].all(), 'hold ⊄ cu'
    return dict(n=n, y=y, cu=cu, hold=hold, nz=nz,
                tie_base=tie_base, win_base=win_base)


def fold_of(y_, fseed=SEED):
    f = np.zeros(len(y_), int)
    for k, (_, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=fseed)
                                .split(np.zeros((len(y_), 1)), y_)):
        f[te] = k
    return f


# ══════════════════════════════════════ 스윕
def sweep(X, y01, fold, rows, hold_, grid, even, n_logit, label):
    """격자 전체를 쓸어 holdout 손실을 낸다. 반환 {λ: ll}."""
    out, n_bad = {}, 0
    for g in grid:
        z, bad = oof_forced(X, y01, fold, rows, g, even, n_logit)
        n_bad += bad
        m = hold_ & np.isfinite(z)
        out[g] = float(rowloss(z[m], y01[m]).mean())
    if n_bad:
        print(f'    ! {label}: 미수렴 {n_bad}회 (해석 주의)')
    return out


def main():
    print('[적재]')
    U = build()
    y, cu, hold, nz = U['y'], U['cu'], U['hold'], U['nz']
    Z = np.load(FINAL / 'temp/dest_universe.npz')
    Xt = np.hstack([U['tie_base'], Z['Gt_s42'].astype(np.float64)])
    Xw = np.hstack([U['win_base'], Z['W_s42'].astype(np.float64)])
    yT, yA = (y == 2).astype(np.float64), (y == 0).astype(np.float64)
    print(f'  Xt {Xt.shape} · Xw {Xw.shape} · holdout {int(hold.sum()):,}')

    # 현행 CV 가 고른 λ (dest_universe 런 결과 · 배포 단일적합에 쓰이는 중앙값)
    CV_TIE, CV_WIN = (100.0, 10.0), 10.0
    grid_t = [(a, b) for a in GRID_L for b in GRID_G]
    W_ = 78

    for name in ('mixed', 'clean7k'):
        if name == 'mixed':
            xt, xw, yy, yt, ya = Xt, Xw, y, yT, yA
            fold, rt, rw, hh, nn = fold_of(y), cu, cu & nz, hold, nz
        else:
            xt, xw, yy = Xt[cu], Xw[cu], y[cu]
            yt, ya, nn = yT[cu], yA[cu], nz[cu]
            fold, hh = fold_of(y[cu]), hold[cu]
            rt, rw = np.ones(len(yy), bool), nn

        print('\n' + '=' * W_)
        print(f'[{name}] 적합 행 tie {int(rt.sum()):,} / win {int(rw.sum()):,} · '
              f'holdout {int(hh.sum()):,}')
        print('=' * W_)

        st = sweep(xt, yt, fold, rt, hh, grid_t, True, 5, 'tie')
        sw = sweep(xw, ya, fold, rw, hh & nn, GRID_W, False, 3, 'win')

        print('  tie 축 — holdout ll  (행 λ_L × 열 λ_g · * = 계약 격자 밖)')
        print('      λ_L\\λ_g ' + ''.join(f'{g:>10g}' + ('*' if g not in CONTRACT_G else ' ')
                                          for g in GRID_G))
        for a in GRID_L:
            mark = '' if a in CONTRACT_L else '*'
            print(f'  {a:>10g}{mark:1s} ' + ''.join(f'{st[(a, b)]:>10.5f} ' for b in GRID_G))
        print('  win 축 — holdout ll')
        print('        λ    ' + ''.join(f'{g:>10g}' + ('*' if g not in CONTRACT_W else ' ')
                                        for g in GRID_W))
        print('             ' + ''.join(f'{sw[g]:>10.5f} ' for g in GRID_W))

        ot = min(st, key=st.get)
        ow = min(sw, key=sw.get)
        base_t, base_w = st[CV_TIE], sw[CV_WIN]
        base_tot, ora_tot = base_t + R_NZ * base_w, st[ot] + R_NZ * sw[ow]
        flat_t = sum(v < st[ot] + 1e-4 for v in st.values())
        flat_w = sum(v < sw[ow] + 1e-4 for v in sw.values())

        print(f'\n  {"":14s}{"λ":>18s}{"tie ll":>10s}{"win ll":>10s}{"총점":>10s}')
        print(f'  {"cv-median":14s}{str(CV_TIE) + " / " + f"{CV_WIN:g}":>18s}'
              f'{base_t:10.5f}{base_w:10.5f}{base_tot:10.5f}')
        print(f'  {"oracle":14s}{str(ot) + " / " + f"{ow:g}":>18s}'
              f'{st[ot]:10.5f}{sw[ow]:10.5f}{ora_tot:10.5f}')
        print(f'  {"간격":14s}{"":18s}{st[ot] - base_t:+10.6f}{sw[ow] - base_w:+10.6f}'
              f'{ora_tot - base_tot:+10.6f}   ← λ 선택이 남긴 여지의 **상한**')
        print(f'  평탄도 — 최적 대비 1e-4 이내: tie {flat_t}/{len(st)}점 · win {flat_w}/{len(sw)}점')
        if ot[0] in (GRID_L[0], GRID_L[-1]) or ot[1] in (GRID_G[0], GRID_G[-1]) \
                or ow in (GRID_W[0], GRID_W[-1]):
            print('  ! 오라클이 격자 경계에 있다 — 격자를 더 넓혀야 한다')
        if name == 'mixed':
            print(f'  참고 cv-adaptive(현행 보고) tie {PLAN_TIE:.5f} · win {PLAN_WIN:.5f} '
                  f'→ {PLAN_TOT:.5f}   (fold 마다 λ 를 따로 고른 값)')

    print('\n' + '=' * W_)
    print('읽는 법')
    print('  · 「간격」이 −1e-4 보다 얕으면 λ 탐색은 이미 정답을 찾고 있다 → **얼려도 된다.**')
    print('    그러면 안쪽 3-fold × 30점 격자를 제출 경로에서 통째로 뺄 수 있다.')
    print('  · 평탄도가 넓으면(다수 격자점이 1e-4 이내) λ 자체가 둔감하다는 뜻 —')
    print('    우주가 바뀌어도 같은 λ 를 써도 손해가 없다는 직접 증거다.')
    print('  · mixed 와 clean7k 의 오라클 λ 가 같으면 λ 는 우주를 건너 이식 가능하다.')
    print('    (단 λ 는 n 에 스케일 의존한다 — 두 우주는 적합 행이 7,748 로 같아 비교가 성립)')


if __name__ == '__main__':
    main()
