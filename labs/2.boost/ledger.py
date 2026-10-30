"""ledger.py — 결합 원장. "전문가 단독은 이만큼인데 합쳐서 왜 이것밖에 안 나오나"를 정면으로 잰다.

왜 필요한가
  단독 숫자와 결합 숫자가 서로 다른 집합·다른 프로토콜로 재져 있어 비교가 성립하지 않는다.
      단독 보고치   holdout 5,748, 인코더 자체 온도(boost_meta 의 T)          <- 재적합 없음
      결합 보고치   stage2 10,748, 5-fold 교차적합 가중치                     <- 재적합 있음
  게다가 axis_report 의 '승부축' 은 **전체 행 수로 나눈** 값이다. 비-tie 행당으로 고치려면
  (1 - tie율) 로 나눠야 하고, 그래야 ABE 의 'win binary logloss' 와 같은 척도가 된다.
  이 파일은 전부 같은 집합·같은 fold·같은 척도로 다시 깔고, 증분을 축별로 분해한다.

무엇을 잰다
  [1] 대조   사용자 보고 단독 숫자를 holdout 슬라이스에서 재현 — 척도가 같은지 먼저 확인한다.
  [2] 원장   stage2 교차적합. 축별로 단독 -> 쌍 -> 전부. 증분(마지막에 넣었을 때 버는 양)까지.
             여기서 "합쳐도 안 는다" 가 사실인지, 는데 우리가 못 쓰는 건지 갈린다.
  [3] 구조   결합기가 받는 건 **이미 평균된** 신호다:
                 ABE   체크포인트 7개 -> 로짓 등가중 평균 -> (m, t) 한 쌍
                 Arith 모델 3종      -> raw 등가중 평균  -> (m, t) 한 쌍
                 TB    체크포인트 1개                     -> t 하나
             원재료 자유도 21 을 5 로 눌러서 넘기는 셈이다. 등가중이 최적이 아니면 그 차이가
             곧 **구조 손실**이다. 펼쳐서(각 체크포인트/모델을 따로 열로) 능형으로 받아 비교한다.
             λ 는 바깥 fold 안쪽 3-fold 로 다시 고른다 — λ->inf 면 열이 죽으므로 질 수 없다.

읽는 법
  tie축   = 1{y=tie} 에 대한 이진 logloss, **전체 행 평균**. 기준선은 tie율의 엔트로피.
  승부축  = 1{y=A} 에 대한 이진 logloss, **비-tie 행 평균**. 기준선은 A 승률의 엔트로피.
  총 ll   = tie축 + (1-tie율)*승부축  — 축 분해 항등식. 검산으로 확인한다.
  승부축 적합에는 절편을 두지 않는다. 절편은 A/B 슬롯 편향이라 swap 반대칭을 깨기 때문이다.

사용
  python Boost/ledger.py
  python Boost/ledger.py --no-unfold      # [3] 생략 (가장 무거운 절)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

BOOST = Path('./Boost')
ARITH = Path('./Arith')
SEED, K, INNER = 42, 5, 3
EPS = 1e-15
MONO_EPS = 0.01          # 등순위 회귀는 끝단에서 0/1 로 붙는다 — 로그오즈가 발산하지 않게 자른다
LAMS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
ARITH_MODELS = ('lgbm', 'xgb', 'cat')


# ══════════════════════════════════════════════════════ 출력 유틸 (한글 폭 보정)
def _w(s):
    return sum(2 if ord(c) > 0x1100 and not 0x2000 <= ord(c) <= 0x206F else 1 for c in str(s))


def _pad(s, n, right=False):
    s = str(s)
    g = ' ' * max(n - _w(s), 0)
    return g + s if right else s + g


def table(headers, rows, align=None):
    align = align or ['l'] * len(headers)
    cols = [max(_w(h), *(_w(r[i]) for r in rows)) if rows else _w(h)
            for i, h in enumerate(headers)]
    print('  ' + '  '.join(_pad(h, cols[i], align[i] == 'r') for i, h in enumerate(headers)))
    print('  ' + '-' * (sum(cols) + 2 * (len(cols) - 1)))
    for r in rows:
        print('  ' + '  '.join(_pad(v, cols[i], align[i] == 'r') for i, v in enumerate(r)))


def head(tag, text):
    print('\n' + '=' * 100)
    print(f'[{tag}] {text}')
    print('=' * 100)


def ci_str(m, lo, hi):
    sig = '유의 ★' if lo > 0 or hi < 0 else '노이즈'
    return f'{m:+.4f} [{lo:+.4f},{hi:+.4f}] {sig}'


def paired_ci(a, b, n_boot=2000, seed=SEED):
    """a - b 의 부트스트랩 95% CI. 양수면 b 가 낫다(손실이므로)."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bs = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ══════════════════════════════════════════════════════ 축 적합 (교차적합 + 중첩 능형)
def _nll(p, Z, y01, lam, bias):
    w = p[:Z.shape[1]]
    z = Z @ w + (p[-1] if bias else 0.0)
    return float(np.mean(np.logaddexp(0, z) - y01 * z) + lam * np.dot(w, w) / len(y01))


def _fit(Z, y01, lam, bias):
    n = Z.shape[1] + (1 if bias else 0)
    r = minimize(_nll, np.zeros(n), args=(Z, y01, lam, bias), method='L-BFGS-B',
                 options={'maxiter': 500})
    return r.x


def _score(p, Z, bias):
    return Z @ p[:Z.shape[1]] + (p[-1] if bias else 0.0)


def _rowloss(z, y01):
    return np.logaddexp(0, z) - y01 * z


def crossfit_axis(Z, y01, fold, bias, ridge=False, rows=None):
    """fold k 의 파라미터는 나머지 4 fold 로만. ridge=True 면 안쪽 3-fold 로 λ 를 다시 고른다.
    rows 는 적합·평가에 쓸 행 마스크 (승부축은 비-tie 행만)."""
    Z = np.atleast_2d(np.asarray(Z, np.float64))
    if Z.shape[0] != len(y01):
        Z = Z.T
    m = np.ones(len(y01), bool) if rows is None else rows
    out = np.full(len(y01), np.nan)
    lam_used, W = [], []
    for k in range(K):
        tr, te = (fold != k) & m, (fold == k) & m
        Ztr, ytr = Z[tr], y01[tr]
        if ridge:
            # 안쪽 3-fold 는 층화분할로 새로 만든다. 바깥 fold id 를 3 으로 나머지 취하면
            # 남은 4 개가 {0,1,0,1} 처럼 두 그룹으로만 떨어져 한 그룹이 비는 fold 가 생긴다.
            inner = np.zeros(len(ytr), int)
            for j, (_, iv) in enumerate(StratifiedKFold(INNER, shuffle=True, random_state=SEED)
                                        .split(np.zeros((len(ytr), 1)), ytr)):
                inner[iv] = j
            assert set(np.unique(inner)) == set(range(INNER)), '안쪽 fold 가 비었다'
            best, bl = LAMS[0], np.inf
            for lam in LAMS:
                tot = 0.0
                for j in range(INNER):
                    it, iv = inner != j, inner == j
                    p = _fit(Ztr[it], ytr[it], lam, bias)
                    tot += _rowloss(_score(p, Ztr[iv], bias), ytr[iv]).sum()
                if tot < bl:
                    bl, best = tot, lam
            lam = best
        else:
            lam = 0.0
        p = _fit(Ztr, ytr, lam, bias)
        out[te] = _rowloss(_score(p, Z[te], bias), y01[te])
        lam_used.append(lam)
        W.append(p)
    return out[m], np.array(W), lam_used


def crossfit_scores(Z, y01, fold, bias, rows=None):
    """crossfit_axis 와 같은 적합인데 손실 대신 **행별 점수(로그오즈)** 를 돌려준다.
    불일치 분석은 확률이 있어야 하므로 별도로 둔다 (기존 호출부를 건드리지 않기 위해)."""
    Z = np.atleast_2d(np.asarray(Z, np.float64))
    if Z.shape[0] != len(y01):
        Z = Z.T
    m = np.ones(len(y01), bool) if rows is None else rows
    out = np.full(len(y01), np.nan)
    for k in range(K):
        tr, te = (fold != k) & m, (fold == k) & m
        p = _fit(Z[tr], y01[tr], 0.0, bias)
        out[te] = _score(p, Z[te], bias)
    return out


def ent(p):
    p = float(np.clip(p, EPS, 1 - EPS))
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def auc_of(v, yb):
    r = pd.Series(v).rank().to_numpy()
    n1 = int(yb.sum())
    n0 = len(yb) - n1
    return float((r[yb].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gauss_ll(auc, pi, grid=8001, lim=12.0):
    """등분산 가우스 등가 모형에서 **그 AUC 로 달성 가능한** logloss.
    양성 ~ N(+d/2, 1), 음성 ~ N(-d/2, 1), d = sqrt(2)*Phi^-1(AUC) 로 두면 순위 성능이 AUC 와 같다.
    그때의 최적 로그오즈는 d*s + log(pi/(1-pi)) 이고, 그 기대 NLL 을 수치적분한다.
    등분산 가정이므로 실제 점수 분포가 비대칭이면 하한이 아니라 **참고 기준선**이다."""
    d = float(np.sqrt(2.0) * norm.ppf(float(np.clip(auc, 1e-6, 1 - 1e-6))))
    x = np.linspace(-lim, lim, grid)
    w = np.exp(-0.5 * x ** 2)
    w = w / w.sum()
    lg = np.log(pi / (1 - pi))
    lp = d * d / 2 + d * x + lg          # 양성: s = d/2 + x
    ln = -d * d / 2 + d * x + lg         # 음성: s = -d/2 + x
    return float(pi * (w * np.logaddexp(0, -lp)).sum() + (1 - pi) * (w * np.logaddexp(0, ln)).sum())


def _iso_fit(v, y01, odd):
    """등순위 회귀로 점수 -> 확률. odd=True 면 (-v, 1-y) 쌍둥이를 함께 먹여
    로그오즈가 홀함수가 되게 강제한다 — 승부축은 반대칭이 깨지면 안 된다."""
    ir = IsotonicRegression(out_of_bounds='clip', y_min=MONO_EPS, y_max=1 - MONO_EPS)
    if odd:
        ir.fit(np.concatenate([v, -v]), np.concatenate([y01, 1.0 - y01]))
    else:
        ir.fit(v, y01)

    def f(x):
        p = np.clip(ir.predict(x), MONO_EPS, 1 - MONO_EPS)
        return np.log(p / (1 - p))
    return f


def crossfit_mono(vs, y01, fold, bias, rows=None, odd=False):
    """단조 재보정 + 선형 결합을 **같은 바깥 fold 안에서 함께** 적합한다.
    변환을 미리 교차적합해 두고 결합기를 따로 적합하면, 결합기의 학습행이 이미
    평가 fold 를 본 변환을 거친 값이라 누수가 된다."""
    m = np.ones(len(y01), bool) if rows is None else rows
    out = np.full(len(y01), np.nan)
    for k in range(K):
        tr, te = (fold != k) & m, (fold == k) & m
        fs = [_iso_fit(v[tr], y01[tr], odd) for v in vs]
        Zt = np.column_stack([f(v[tr]) for f, v in zip(fs, vs)])
        Ze = np.column_stack([f(v[te]) for f, v in zip(fs, vs)])
        p = _fit(Zt, y01[tr], 0.0, bias)
        out[te] = _rowloss(_score(p, Ze, bias), y01[te])
    return out[m]


# ══════════════════════════════════════════════════════ 신호 적재
def load_all():
    z = np.load(BOOST / 'boost_oof.npz')
    ids, y = z['ids'], z['y']
    meta = json.loads((BOOST / 'model_ens/boost_meta.json').read_text(encoding='utf-8'))
    ex = meta['experts']

    def one(tag, step):
        w = np.load(BOOST / f'logits/{tag}__checkpoint-{step}.npz')
        assert (np.concatenate([w['valid_ids'], w['hold_ids']]) == ids).all(), f'{tag} id'
        assert (np.concatenate([w['valid_labels'], w['hold_labels']]) == y).all(), f'{tag} 라벨'
        return np.vstack([(w['valid_fwd'] + w['valid_swp']) / 2,
                          (w['hold_fwd'] + w['hold_swp']) / 2]).astype(np.float64)

    abe_steps = ex['abe'].get('steps') or [ex['abe']['step']]
    tb_steps = ex['tb'].get('steps') or [ex['tb']['step']]
    ABE = [one('abe', s) for s in abe_steps]
    TB = [one('tb3', s) for s in tb_steps]

    za = np.load(ARITH / 'outputs/arith_experts_raw.npz')
    assert (np.concatenate([za['ids_valid'], za['ids_hold']]) == ids).all(), 'Arith id'
    ARI = [np.vstack([za[f'raw_valid__{m}'], za[f'raw_hold__{m}']]).astype(np.float64)
           for m in ARITH_MODELS]

    n_valid = len(np.load(BOOST / f'logits/abe__checkpoint-{abe_steps[0]}.npz')['valid_ids'])
    is_hold = np.zeros(len(ids), bool)
    is_hold[n_valid:] = True
    return dict(ids=ids, y=y, ABE=ABE, TB=TB, ARI=ARI, is_hold=is_hold,
                abe_steps=abe_steps, tb_steps=tb_steps,
                T_abe=ex['abe'].get('T'), T_tb=ex['tb'].get('T'))


def sd(B):
    """3-class 로짓 -> (m, t). 무손실 재매개화 — m 은 홀, t 는 짝."""
    return B[:, 0] - B[:, 1], B[:, 2] - np.logaddexp(B[:, 0], B[:, 1])


# ══════════════════════════════════════════════════════ [6] 타이 신호의 정체 + 불일치
def section6(TIE_SIG, WIN_SIG, ids, y_tie, y_win, nz, fold):
    head('6', '왜 산술 피쳐가 크로스 인코더에 근접하나 — 타이 신호의 정체와 전문가 불일치')
    N = len(y_tie)
    # tie 축 행별 손실 — [2] 와 같은 교차적합이다. --only why 로도 단독 실행되게 여기서 다시 잰다.
    TIE_REF = {k: crossfit_axis(v.reshape(-1, 1), y_tie, fold, True)[0]
               for k, v in TIE_SIG.items()}
    TIE_REF['ALL'] = crossfit_axis(np.column_stack([TIE_SIG[k] for k in TIE_SIG]),
                                   y_tie, fold, True)[0]

    # 6a ─ 순위력 회수 비율
    print('  6a. 순위력 회수 — (AUC - 0.5) 로 재면 Arith 가 인코더의 몇 % 를 되찾나')
    print('      타이가 표면 형태 판단이고 승부가 의미 판단이라면, tie 축에서 비율이 더 높아야 한다.\n')
    rows = []
    for axis, sigs, yy, msk, enc in [('tie', TIE_SIG, y_tie, np.ones(N, bool), 'TB'),
                                     ('win', WIN_SIG, y_win, nz, 'ABE')]:
        a_enc = auc_of(sigs[enc][msk], yy[msk].astype(bool))
        a_ari = auc_of(sigs['ARI'][msk], yy[msk].astype(bool))
        rows.append([f'{axis}축', f'{enc} {a_enc:.4f}', f'ARI {a_ari:.4f}',
                     f'{a_enc - 0.5:.4f}', f'{a_ari - 0.5:.4f}',
                     f'{(a_ari - 0.5) / (a_enc - 0.5):.1%}'])
    table(['축', '인코더 AUC', 'Arith AUC', '인코더 순위력', 'Arith 순위력', '회수 비율'],
          rows, ['l', 'r', 'r', 'r', 'r', 'r'])

    # 6b ─ 개별 피쳐 단독 tie AUC
    print('\n  6b. 개별 피쳐 **하나만**으로 잰 tie AUC — 학습이 전혀 없다, 순위 통계뿐이다')
    print('      값싼 표면 피쳐 하나가 0.6 을 넘으면 위 해석이 그 자리에서 확정된다.')
    cat = pd.read_csv(ARITH / 'feature_catalog.csv')
    use = cat[cat['사용'] == 1] if '사용' in cat.columns else cat
    swap_of = cat.set_index('feature')['swap']
    feat = pd.read_parquet(ARITH / 'features_57k.parquet')
    pos = pd.Index(feat['id']).get_indexer(ids)
    assert (pos >= 0).all(), 'features_57k 에 stage2 id 가 없다'
    cols = [c for c in use['feature'] if c in feat.columns]
    X = feat.iloc[pos][cols].to_numpy(np.float64)
    yb = y_tie.astype(bool)
    aucs = np.array([auc_of(X[:, j], yb) for j in range(X.shape[1])])
    strength = np.abs(aucs - 0.5)
    odd_m = np.array([swap_of.get(c) == '부호반전' for c in cols])
    order = np.argsort(-strength)[:25]
    print()
    table(['#', '피쳐', 'swap', 'tie AUC', '순위력 |AUC-.5|'],
          [[str(i + 1), cols[j], str(swap_of.get(cols[j])), f'{aucs[j]:.4f}', f'{strength[j]:.4f}']
           for i, j in enumerate(order)], ['r', 'l', 'l', 'r', 'r'])
    print(f'    참고 — 인코더 단독: TB {auc_of(TIE_SIG["TB"], yb):.4f} / '
          f'ABE {auc_of(TIE_SIG["ABE"], yb):.4f} / Arith 부스터 {auc_of(TIE_SIG["ARI"], yb):.4f}')
    if odd_m.any():
        print(f'    계약 검산 — 홀(부호반전) 피쳐 {int(odd_m.sum())}개의 tie AUC 는 0.5 여야 한다 '
              f'(타이 라벨이 swap 불변이므로). 실측 평균 {aucs[odd_m].mean():.4f}, '
              f'최대 이탈 {strength[odd_m].max():.4f}')
        print('      이탈이 크면 데이터에 A/B 슬롯 편향이 있다는 뜻이다.')

    # 6c ─ 상위 k 피쳐만의 로지스틱
    print('\n  6c. 상위 k 개 피쳐만으로 로지스틱 — 몇 개면 Arith 부스터(253피쳐 15벌)를 따라잡나')
    print('      **상위 k 선택은 학습 fold 안에서만** 한다. 전체로 고르면 그게 곧 선택 편향이다.')
    print('      표준화 통계도 학습 fold 것만 쓴다.\n')
    rows = []
    for kk in (1, 3, 5, 10, 20, 40):
        loss = np.zeros(N)
        for f_ in range(K):
            tr, te = fold != f_, fold == f_
            a_tr = np.array([auc_of(X[tr, j], yb[tr]) for j in range(X.shape[1])])
            sel = np.argsort(-np.abs(a_tr - 0.5))[:kk]
            mu, sdv = X[tr][:, sel].mean(0), np.clip(X[tr][:, sel].std(0), 1e-9, None)
            Ztr, Zte = (X[tr][:, sel] - mu) / sdv, (X[te][:, sel] - mu) / sdv
            prm = _fit(Ztr, y_tie[tr], 1.0, True)          # 열이 많아질 수 있어 약한 능형 고정
            loss[te] = _rowloss(_score(prm, Zte, True), y_tie[te])
        rows.append([f'상위 {kk}', f'{loss.mean():.4f}',
                     ci_str(*paired_ci(loss, TIE_REF['ARI']))])
    rows.append(['Arith 부스터 (253피쳐×15벌)', f'{TIE_REF["ARI"].mean():.4f}', '기준'])
    rows.append(['TB 인코더', f'{TIE_REF["TB"].mean():.4f}',
                 ci_str(*paired_ci(TIE_REF['ARI'], TIE_REF['TB']))])
    table(['구성', 'tie ll', 'Δ vs Arith 부스터(양수=이득)'], rows, ['l', 'r', 'l'])

    # 6d ─ 층화 맞대결
    print('\n  6d. 층화 맞대결 — 표면이 비슷한 층에서 Arith 가 세다면 해석이 맞다')
    print('      층 기준은 라벨을 안 쓰는 피쳐라 편향이 없다.\n')
    for gname, gcol in [('응답 유사도 pr_jac_word', 'pr_jac_word'),
                        ('길이차 ad_char_len', 'ad_char_len')]:
        if gcol not in feat.columns:
            continue
        g = feat.iloc[pos][gcol].to_numpy(np.float64)
        q = np.digitize(g, np.quantile(g, [.25, .5, .75]))
        rows = []
        for s in range(4):
            m_ = q == s
            r = [f'Q{s + 1}', f'{int(m_.sum()):,}', f'{y_tie[m_].mean():.3f}']
            for kk in ('TB', 'ABE', 'ARI'):
                r.append(f'{TIE_REF[kk][m_].mean():.4f}')
            best = min(('TB', 'ABE', 'ARI'), key=lambda kk: TIE_REF[kk][m_].mean())
            r.append(best)
            rows.append(r)
        print(f'   {gname} 사분위')
        table(['층', 'n', 'tie율', 'TB ll', 'ABE ll', 'ARI ll', '그 층의 최강'],
              rows, ['l', 'r', 'r', 'r', 'r', 'r', 'l'])
        print()

    # 6e ─ 불일치
    print('  6e. 불일치 — 한쪽만 맞히는 행이 얼마나 되나')
    print('      판정 문턱은 **비율 맞춤**이다: 학습 fold 의 tie율만큼만 tie 로 예측하게 자른다.')
    print('      0.5 로 자르면 tie율이 0.309 라 거의 전부 non-tie 로 찍혀 표가 의미를 잃는다.\n')
    P = {}
    for kk, v in TIE_SIG.items():
        z = crossfit_scores(v.reshape(-1, 1), y_tie, fold, True)
        pred = np.zeros(N, bool)
        for f_ in range(K):
            tr, te = fold != f_, fold == f_
            thr = np.quantile(z[tr], 1.0 - y_tie[tr].mean())
            pred[te] = z[te] >= thr
        P[kk] = pred
    for a, b in (('ARI', 'TB'), ('ARI', 'ABE'), ('TB', 'ABE')):
        ca, cb = P[a] == yb, P[b] == yb
        rows = []
        for lab, m_ in (('전체', np.ones(N, bool)), ('참 tie', yb), ('참 non-tie', ~yb)):
            n = int(m_.sum())
            both, only_a = int((ca & cb & m_).sum()), int((ca & ~cb & m_).sum())
            only_b, none = int((~ca & cb & m_).sum()), int((~ca & ~cb & m_).sum())
            rows.append([lab, f'{n:,}', f'{both:,} ({both / n:.1%})',
                         f'{only_a:,} ({only_a / n:.1%})', f'{only_b:,} ({only_b / n:.1%})',
                         f'{none:,} ({none / n:.1%})'])
        print(f'   {a} vs {b}')
        table(['구간', 'n', '둘 다 맞음', f'{a} 만 맞음', f'{b} 만 맞음', '둘 다 틀림'],
              rows, ['l', 'r', 'r', 'r', 'r', 'r'])
        ex = int((ca ^ cb).sum())
        print(f'    한쪽만 맞히는 행 {ex:,} ({ex / N:.1%}) — 이게 0 이면 완전 중복, 클수록 상보적이다')
        print('    다만 상보적이어도 **어느 쪽이 맞는지 알 수 없으면** 못 쓴다. 아래 오라클이 그 상한이다.\n')

    print('  오라클 — 행마다 손실이 더 낮은 전문가를 고를 수 있다면 (라벨을 봐야 하므로 달성 불가,')
    print('  어떤 게이트/MoE 도 이보다 잘할 수 없는 **상한**이다)')
    rows = []
    lin_all = TIE_REF['ALL'].mean()
    for a, b in (('TB', 'ARI'), ('TB', 'ABE'), ('ARI', 'ABE')):
        o = np.minimum(TIE_REF[a], TIE_REF[b])
        rows.append([f'{a} / {b}', f'{TIE_REF[a].mean():.4f}', f'{TIE_REF[b].mean():.4f}',
                     f'{o.mean():.4f}', ci_str(*paired_ci(TIE_REF['ALL'], o))])
    o3 = np.minimum.reduce([TIE_REF['TB'], TIE_REF['ABE'], TIE_REF['ARI']])
    rows.append(['셋 전부', '—', '—', f'{o3.mean():.4f}', ci_str(*paired_ci(TIE_REF['ALL'], o3))])
    table(['조합', 'A 단독', 'B 단독', '오라클 선택', f'Δ vs 선형 결합 {lin_all:.4f}'],
          rows, ['l', 'r', 'r', 'r', 'l'])
    print('    오라클이 선형 결합보다 크게 낮지 않으면 게이트를 만들 여지가 없다 —')
    print('    실제 게이트는 라벨을 못 보므로 이 상한의 일부만 회수한다 (주장 D 는 이미 기각됐다).')

    # 6f ─ 오라클의 널
    print('\n  6f. 오라클의 널 — **행별 전문성이 0 이어도** 오라클은 크게 나온다')
    print('      오라클은 행마다 라벨을 보고 더 나은 쪽을 고르는 통계라 결과에 대한 선택이 들어간다.')
    print('      그래서 널을 깔지 않으면 위 +격차 를 상보성의 증거로 읽을 수 없다.')
    print('      널: 관측 상관으로 1-요인 모형을 맞춘다.   z_i = a_i*u + sqrt(1-a_i^2)*e_i')
    print('      u 는 세 전문가가 공유하는 요인(실측 점수에서 회귀추정), e_i 는 독립 가우스 잡음이다.')
    print('      합성 전문가는 **구조적으로 상보성이 없다** — 잡음은 어떤 게이트도 예측할 수 없다.')
    print('      그런데도 실측과 비슷한 격차가 나오면, 실측 격차도 같은 착시일 뿐이다.\n')
    KS = list(TIE_SIG)
    Zc = np.column_stack([crossfit_scores(TIE_SIG[k].reshape(-1, 1), y_tie, fold, True)
                          for k in KS])
    Zs = (Zc - Zc.mean(0)) / Zc.std(0)
    C3 = np.corrcoef(Zs.T)
    prs = [(1, 2), (0, 2), (0, 1)]
    a = np.array([np.sqrt(np.clip(C3[i, j] * C3[i, kk] / C3[j, kk], 0.0, 1.0))
                  for i, (j, kk) in enumerate(prs)])
    u = Zs @ np.linalg.solve(C3, a)
    u = (u - u.mean()) / u.std()
    print('    1-요인 적재 a: ' + ', '.join(f'{KS[i]} {a[i]:.3f}' for i in range(3))
          + '   (a^2 = 공통분산 비중)')

    REPS = 5
    rng = np.random.default_rng(SEED)
    acc = {'solo': np.zeros(3), 'or3': 0.0, 'lin': 0.0, 'ex': 0.0}
    for _ in range(REPS):
        Zt = np.column_stack([a[i] * u + np.sqrt(max(1 - a[i] ** 2, 0.0))
                              * rng.standard_normal(N) for i in range(3)])
        L = [crossfit_axis(Zt[:, i].reshape(-1, 1), y_tie, fold, True)[0] for i in range(3)]
        acc['solo'] += np.array([x.mean() for x in L])
        acc['or3'] += np.minimum.reduce(L).mean()
        acc['lin'] += crossfit_axis(Zt, y_tie, fold, True)[0].mean()
        ok = []
        for i in range(3):
            z = crossfit_scores(Zt[:, i].reshape(-1, 1), y_tie, fold, True)
            pr = np.zeros(N, bool)
            for f_ in range(K):
                tr, te = fold != f_, fold == f_
                pr[te] = z[te] >= np.quantile(z[tr], 1.0 - y_tie[tr].mean())
            ok.append(pr == yb)
        acc['ex'] += float((ok[KS.index('ARI')] ^ ok[KS.index('TB')]).mean())
    for k_ in acc:
        acc[k_] = acc[k_] / REPS
    r_ex = float(((P['ARI'] == yb) ^ (P['TB'] == yb)).mean())
    print()
    table(['항목', '실측', f'합성 널 ({REPS}회 평균)', '판단'],
          [['단독 손실 ' + '/'.join(KS),
            '/'.join(f'{TIE_REF[k].mean():.4f}' for k in KS),
            '/'.join(f'{x:.4f}' for x in acc['solo']), '널이 실측과 비슷해야 비교가 성립'],
           ['선형 결합 (셋)', f'{TIE_REF["ALL"].mean():.4f}', f'{acc["lin"]:.4f}', ''],
           ['오라클 (셋)', f'{o3.mean():.4f}', f'{acc["or3"]:.4f}', ''],
           ['오라클 격차 (선형−오라클)', f'{TIE_REF["ALL"].mean() - o3.mean():+.4f}',
            f'{acc["lin"] - acc["or3"]:+.4f}',
            '널이 실측만큼 크면 실측 격차는 전부 잡음'],
           ['ARI/TB 한쪽만 맞힘', f'{r_ex:.1%}', f'{acc["ex"]:.1%}', '']],
          ['l', 'r', 'r', 'l'])
    gap_r = TIE_REF['ALL'].mean() - o3.mean()
    gap_n = acc['lin'] - acc['or3']
    print(f'\n    >> 실측 격차의 {gap_n / gap_r:.0%} 가 순수 잡음으로 설명된다.')
    print('       100% 근처면 게이트/MoE 는 닫힌 얘기다 — 고를 근거가 데이터에 없다.')
    print('       크게 남으면 그 잔여분이 실제 행별 전문성이고, 관측 가능한 피쳐로 예측되는지가 다음 질문이다.')


# ══════════════════════════════════════════════════════ 본체
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-unfold', action='store_true', help='[3] 펼치기 검사 생략')
    ap.add_argument('--only', default='all', choices=['all', 'why'],
                    help='why = [6] 만 단독 실행')
    args = ap.parse_args()

    R = load_all()
    y, is_hold = R['y'], R['is_hold']
    N = len(y)
    fold = np.zeros(N, int)
    for k, (_, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=SEED)
                                .split(np.zeros((N, 1)), y)):
        fold[te] = k

    tie = (y == 2)
    nz = ~tie
    y_tie = tie.astype(np.float64)
    y_win = (y == 0).astype(np.float64)
    p_tie, p_A = tie.mean(), y_win[nz].mean()
    print(f'stage2 {N:,}행 (valid {int((~is_hold).sum()):,} + holdout {int(is_hold.sum()):,}) | '
          f'ABE {R["abe_steps"]} | TB3 {R["tb_steps"]} | Arith {list(ARITH_MODELS)}')
    print(f'tie율 {p_tie:.4f} (기준선 {ent(p_tie):.4f}) | '
          f'비-tie {int(nz.sum()):,}행 A승률 {p_A:.4f} (기준선 {ent(p_A):.4f})')

    # 등가중 사전평균 — 현행 결합기가 실제로 받는 신호
    L_ABE = np.mean(R['ABE'], axis=0)
    L_TB = np.mean(R['TB'], axis=0)
    L_ARI = np.mean(R['ARI'], axis=0)
    m3, t3 = sd(L_ABE)
    m_ar, t_ar = sd(L_ARI)
    s_tie = L_TB[:, 1] - L_TB[:, 0]

    TIE_SIG = {'TB': s_tie, 'ABE': t3, 'ARI': t_ar}
    WIN_SIG = {'ABE': m3, 'ARI': m_ar}
    if args.only == 'why':
        section6(TIE_SIG, WIN_SIG, R['ids'], y_tie, y_win, nz, fold)
        return

    # ── [1] 대조 ────────────────────────────────────────────────────────────
    head('1', '대조 — 보고된 단독 숫자를 holdout 슬라이스에서 재현한다 (척도 일치 확인)')
    print('  단독 보고치는 재적합 없이 인코더 자체 출력이다. 온도만 boost_meta 값을 쓴다.')
    print('  여기가 어긋나면 아래 원장 전체가 다른 걸 재고 있는 것이다.\n')
    rows = []
    for nm, zz, yy, msk, T in [
            ('TB 단독 tie', s_tie, y_tie, is_hold, R['T_tb']),
            ('ABE 단독 tie', t3, y_tie, is_hold, R['T_abe']),
            ('Arith 단독 tie', t_ar, y_tie, is_hold, None),
            ('ABE 단독 승부', m3, y_win, is_hold & nz, R['T_abe']),
            ('Arith 단독 승부', m_ar, y_win, is_hold & nz, None)]:
        raw = _rowloss(zz[msk], yy[msk]).mean()
        tmp = _rowloss(zz[msk] / T, yy[msk]).mean() if T else np.nan
        # AUC (순위 통계 — Mann-Whitney)
        v, yb = zz[msk], yy[msk].astype(bool)
        r_ = pd.Series(v).rank().to_numpy()
        auc = (r_[yb].sum() - yb.sum() * (yb.sum() + 1) / 2) / (yb.sum() * (~yb).sum())
        rows.append([nm, f'{int(msk.sum()):,}', f'{raw:.4f}',
                     '—' if T is None else f'{tmp:.4f}', f'{auc:.4f}'])
    table(['신호', 'n', '온도 없음', '온도 적용', 'AUC'], rows,
          ['l', 'r', 'r', 'r', 'r'])
    print(f'  * 온도: ABE T={R["T_abe"]:.4f}, TB T={R["T_tb"]:.4f} (boost_meta). Arith 는 온도 없음.')

    # ── [2] 원장 ────────────────────────────────────────────────────────────
    head('2', '원장 — 같은 집합·같은 fold·교차적합. 단독 -> 쌍 -> 전부, 축별 증분')
    print('  각 구성은 그 열들만으로 축 로지스틱을 교차적합한 결과다. 단독끼리도 여기선 공정하다')
    print('  (스케일 재적합이 들어가므로 [1] 의 숫자보다 좋게 나오는 게 정상이다).\n')

    def axis_run(sigs, names, bias):
        Z = np.column_stack([sigs[n] for n in names]) if names else np.zeros((N, 0))
        rows_m = None if bias else nz
        if not names:                       # 기준선 = 상수만
            if bias:
                pr, _, _ = crossfit_axis(np.zeros((N, 1)), y_tie, fold, True, rows=None)
            else:
                pr = _rowloss(np.zeros(int(nz.sum())), y_win[nz])
            return pr
        pr, _, _ = crossfit_axis(Z, y_tie if bias else y_win, fold, bias, rows=rows_m)
        return pr

    def combos(keys):
        out = [[]]
        for i in range(1, len(keys) + 1):
            from itertools import combinations
            out += [list(c) for c in combinations(keys, i)]
        return out

    store = {}
    for axis, sigs, bias, yy in [('tie', TIE_SIG, True, y_tie), ('win', WIN_SIG, False, y_win)]:
        keys = list(sigs)
        print(f'  ── {axis}축 ' + '─' * 60)
        base = axis_run(sigs, [], bias)
        rows = []
        res = {}
        for c in combos(keys):
            pr = base if not c else axis_run(sigs, c, bias)
            res[tuple(c)] = pr
            nm = '기준선(상수)' if not c else '+'.join(c)
            d = ci_str(*paired_ci(base, pr)) if c else ''
            rows.append([nm, str(len(c) + (1 if bias else 0)), f'{pr.mean():.4f}', d])
        table(['구성', '파라미터', '손실', 'Δ vs 기준선(양수=이득)'], rows, ['l', 'r', 'r', 'l'])
        full = res[tuple(keys)]
        print('\n  증분 — 나머지 전부가 이미 들어간 상태에서 그 하나를 **더 넣었을 때** 버는 양')
        irows = []
        for k in keys:
            rest = tuple(x for x in keys if x != k)
            irows.append([k, f'{res[rest].mean():.4f}', f'{full.mean():.4f}',
                          ci_str(*paired_ci(res[rest], full))])
        table(['신호', '없을 때', '있을 때', 'Δ (양수=그 신호가 값을 함)'], irows,
              ['l', 'r', 'r', 'l'])
        store[axis] = res
        print()

    tie_full = store['tie'][tuple(TIE_SIG)]
    win_full = store['win'][tuple(WIN_SIG)]
    total = tie_full.mean() + (1 - p_tie) * win_full.mean()
    print(f'  축 합산 총 ll = {tie_full.mean():.4f} + {1 - p_tie:.4f}*{win_full.mean():.4f} '
          f'= {total:.4f}   (Davidson κ=1 의 0.9705 와 대조)')
    print('  * 승부축은 **비-tie 행 평균**이다. axis_report 의 승부축은 전체 행으로 나눈 값이라')
    print(f'    {(1 - p_tie) * win_full.mean():.4f} 로 보인다 — 같은 수치의 다른 척도다.')

    # 상관 — 얼마나 겹치나
    print('\n  신호 상관 (겹칠수록 결합 이득의 천장이 낮다)')
    CORR = {}
    for axis, sigs in [('tie', TIE_SIG), ('win', WIN_SIG)]:
        ks = list(sigs)
        C = np.corrcoef(np.column_stack([sigs[k] for k in ks]).T)
        CORR[axis] = (ks, C)
        rows = [[ks[i]] + [f'{C[i, j]:+.3f}' for j in range(len(ks))] for i in range(len(ks))]
        print(f'   {axis}축')
        table([''] + ks, rows, ['l'] + ['r'] * len(ks))

    # ── [2b] 잉여도 회계 ────────────────────────────────────────────────────
    head('2b', '잉여도 회계 — 결합기가 못 뽑는 건가, 애초에 뽑을 게 없는 건가')
    print('  가우스 등가 모형: 예측자 k 개의 개별 설명력이 g_i, 쌍 상관이 Σ 일 때')
    print('      결합 설명력 = r\' Σ⁻¹ r,   r_i = sqrt(g_i)')
    print('  손실 감소량 g_i 를 설명력의 대리값으로 넣는다. 예측치와 실측치가 맞으면')
    print('  **결합기는 관측된 잉여도가 허용하는 만큼을 이미 다 뽑고 있다**는 뜻이다.')
    print('  어긋나면(실측 < 예측) 그 차이가 결합기의 구조 손실이다.\n')
    rows = []
    for axis, sigs, base_ent in [('tie', TIE_SIG, ent(p_tie)), ('win', WIN_SIG, ent(p_A))]:
        ks, C = CORR[axis]
        res = store[axis]
        g = np.array([base_ent - res[(k,)].mean() for k in ks])
        r = np.sqrt(np.clip(g, 0, None))
        pred = float(r @ np.linalg.solve(C, r))
        obs = base_ent - res[tuple(ks)].mean()
        rows.append([f'{axis}축', ' '.join(f'{x:.4f}' for x in g), f'{g.sum():.4f}',
                     f'{pred:.4f}', f'{obs:.4f}', f'{obs / pred:.2f}배',
                     f'{obs / g.sum():.0%}'])
    table(['축', '개별 이득', '단순합', '예측(잉여도 보정)', '실측', '실측/예측', '실측/단순합'],
          rows, ['l', 'l', 'r', 'r', 'r', 'r', 'r'])
    print('  * 실측/예측 이 1.0 근처면 결합기는 무죄다 — 천장이 전문가 다양성이지 결합 방식이 아니다.')
    print('  * 실측/단순합 이 낮은 건 전문가들이 같은 텍스트를 같은 식으로 읽고 있다는 뜻이다.')

    # ── [2c] 새 전문가가 얼마나 달라야 값을 하나 ────────────────────────────
    head('2c', '새 전문가를 붙인다면 — 얼마나 세고 얼마나 달라야 문턱(0.0005)을 넘나')
    print('  현행 3종 결합 점수를 하나의 합성 신호로 보고, 거기에 새 신호 하나를 더할 때의')
    print('  가우스 등가 이득: Δ = (sqrt(g_new) - ρ*sqrt(g_cur))² / (1 - ρ²)')
    print('  ρ 는 새 신호와 **현행 합성 점수**의 상관이다. 같은 텍스트를 읽는 인코더끼리는')
    print('  관측상 0.65~0.79 였다. 백본이 다르면 그보다 낮아야 값을 한다.\n')
    for axis, sigs, base_ent, keys in [('tie', TIE_SIG, ent(p_tie), list(TIE_SIG)),
                                       ('win', WIN_SIG, ent(p_A), list(WIN_SIG))]:
        res = store[axis]
        g_cur = base_ent - res[tuple(keys)].mean()
        g_solo = {k: base_ent - res[(k,)].mean() for k in keys}
        print(f'  ── {axis}축  현행 합성 이득 {g_cur:.4f} '
              f'(단독 최강 {max(g_solo, key=g_solo.get)} {max(g_solo.values()):.4f})')
        GN = [0.5, 0.75, 1.0, 1.25]
        rows = []
        for rho in (0.4, 0.5, 0.6, 0.7, 0.8):
            cells = []
            for f_ in GN:
                gn = g_cur * f_
                num = (np.sqrt(gn) - rho * np.sqrt(g_cur)) ** 2
                d = num / (1 - rho ** 2)
                cells.append(f'{d:+.4f}' + ('★' if d > 0.0005 else ' '))
            rows.append([f'{rho:.1f}'] + cells)
        table(['ρ(새, 현행합성)'] + [f'새 이득 = 현행×{f_}' for f_ in GN], rows,
              ['r'] + ['r'] * len(GN))
        print('    ★ = 채택 문턱 0.0005 초과. 표의 Δ 는 그 축의 손실 감소량이다.')
        if axis == 'win':
            print(f'    (총 ll 로는 (1-tie율)={1 - p_tie:.4f} 를 곱해야 한다)')
        print()

    # ── [4] 형상 손실 ───────────────────────────────────────────────────────
    head('4', '형상 손실 — AUC 는 좋은데 ll 이 나쁜 신호가 있나? 선형 재조정으로 안 잡히는 몫')
    print('  "학습이 덜 됐다" 면 마진이 0 쪽으로 눌릴 뿐이고 그건 스칼라 재조정이 정확히 되돌린다.')
    print('  [2] 는 이미 교차적합 2-파라미터(기울기+절편)를 태운 뒤의 숫자이므로 압축은 흡수돼 있다.')
    print('  또 학습 부족이면 순위(AUC)도 같이 나빠져야 한다. 그러니 여기서 볼 것은 **형상**이다:')
    print('  점수와 참 로그오즈의 관계가 비선형이면 스칼라로는 못 고치고 단조 재보정이 필요하다.')
    print('  Arith 마진은 5 fold × 2 뷰 = 10겹 평균이라 꼬리가 눌릴 소지가 크다 —')
    print('  꼬리 압축은 확신 구간의 logloss 를 직격하면서 순위는 거의 안 건드린다.\n')
    print('  기울기 = 그 신호 단독 적합의 계수. |기울기| 가 1 보다 크게 크면 원 마진이 눌려 있었다는 뜻.')
    print('  가우스 기준선 = 그 AUC 를 가진 등분산 가우스 점수가 낼 수 있는 ll (하한 아님, 참고치).\n')
    for axis, sigs, yy, rows_m, odd, pi in [
            ('tie', TIE_SIG, y_tie, None, False, p_tie),
            ('win', WIN_SIG, y_win, nz, True, p_A)]:
        print(f'  ── {axis}축 ' + '─' * 60)
        rows = []
        for k, v in sigs.items():
            msk = np.ones(N, bool) if rows_m is None else rows_m
            a = auc_of(v[msk], yy[msk].astype(bool))
            lin, W, _ = crossfit_axis(v.reshape(-1, 1), yy, fold, bias=(not odd), rows=rows_m)
            mono = crossfit_mono([v], yy, fold, bias=(not odd), rows=rows_m, odd=odd)
            gl = gauss_ll(a, pi)
            rows.append([k, f'{a:.4f}', f'{W[:, 0].mean():+.3f}', f'{gl:.4f}',
                         f'{lin.mean():.4f}', f'{mono.mean():.4f}',
                         ci_str(*paired_ci(lin, mono))])
        table(['신호', 'AUC', '기울기', '가우스 기준선', '선형 재조정', '단조 재보정',
               'Δ 단조−선형(양수=형상 손실 회수)'], rows, ['l', 'r', 'r', 'r', 'r', 'r', 'l'])
        print()

    # ── [5] 재보정 후 결합 + 축별 전문가 배정 ───────────────────────────────
    head('5', '단조 재보정한 신호로 결합하면 달라지나 + tie 축을 Arith·TB 로만 맡기는 안')
    for axis, sigs, yy, rows_m, odd, keys_list in [
            ('tie', TIE_SIG, y_tie, None, False,
             [['TB', 'ABE', 'ARI'], ['TB', 'ARI'], ['ABE', 'ARI'], ['TB', 'ABE']]),
            ('win', WIN_SIG, y_win, nz, True, [['ABE', 'ARI'], ['ABE']])]:
        print(f'  ── {axis}축 ' + '─' * 60)
        rows = []
        full_lin = store[axis][tuple(sigs)]
        for names in keys_list:
            vs = [sigs[n] for n in names]
            lin, _, _ = crossfit_axis(np.column_stack(vs), yy, fold, bias=(not odd), rows=rows_m)
            mono = crossfit_mono(vs, yy, fold, bias=(not odd), rows=rows_m, odd=odd)
            rows.append(['+'.join(names), f'{lin.mean():.4f}', f'{mono.mean():.4f}',
                         ci_str(*paired_ci(lin, mono)),
                         '기준' if len(names) == len(sigs) else ci_str(*paired_ci(full_lin, mono))])
        table(['구성', '선형', '단조 재보정', 'Δ 단조−선형', 'Δ vs 전체 선형(양수=이득)'],
              rows, ['l', 'r', 'r', 'l', 'l'])
        print()
    print('  * tie 축에서 TB+ARI 는 사용자 제안(타이 판정을 Arith·TB 에만 맡김)이다.')
    print('    ABE 를 빼도 되는지는 마지막 열이 답한다 — 양수 유의여야 뺄 수 있다.')
    print('  * 승부축 변환은 (-m, 1-y) 쌍둥이를 함께 먹여 홀함수로 강제했다.')
    print('    그러지 않으면 A/B 슬롯 비대칭이 새로 들어가 제출 시 순서 뒤집기에 답이 달라진다.')

    section6(TIE_SIG, WIN_SIG, R['ids'], y_tie, y_win, nz, fold)

    # ── [3] 구조 ────────────────────────────────────────────────────────────
    if args.no_unfold:
        return
    head('3', '구조 — 결합기에 **사전평균**을 주는 게 손실인가? 펼쳐서 능형으로 받아 비교한다')
    print(f'  현행:  ABE 체크포인트 {len(R["ABE"])}개 -> 등가중 로짓평균 -> (m,t) 한 쌍')
    print(f'         Arith 모델 {len(R["ARI"])}종 -> 등가중 raw평균 -> (m,t) 한 쌍')
    print(f'         TB 체크포인트 {len(R["TB"])}개 -> t 하나')
    print(f'  원재료 자유도 {2 * len(R["ABE"]) + 2 * len(R["ARI"]) + len(R["TB"])} 를 '
          f'{2 + 2 + len(R["TB"])} 로 눌러서 넘긴다. 등가중이 최적이 아니면 그 차이가 구조 손실이다.')
    print('  펼친 판은 열이 많아 능형이 필수다 — λ 는 바깥 fold 안쪽 3-fold 로 다시 고른다')
    print('  (λ->inf 면 여분 열이 죽어 사전평균판에 수렴하므로, 구조적으로 질 수 없는 비교다).\n')

    ABE_mt = [sd(B) for B in R['ABE']]
    ARI_mt = [sd(B) for B in R['ARI']]
    TB_t = [B[:, 1] - B[:, 0] for B in R['TB']]

    abe_nm = [f'ABE@{s}' for s in R['abe_steps']]
    tb_nm = [f'TB@{s}' for s in R['tb_steps']]
    ari_nm = [f'ARI:{m}' for m in ARITH_MODELS]

    for axis, bias, yy, rows_m, packed, unpacked, unames in [
        ('tie', True, y_tie, None,
         [s_tie, t3, t_ar],
         [*TB_t, *[t for _, t in ABE_mt], *[t for _, t in ARI_mt]],
         tb_nm + abe_nm + ari_nm),
        ('win', False, y_win, nz,
         [m3, m_ar],
         [*[m for m, _ in ABE_mt], *[m for m, _ in ARI_mt]],
         abe_nm + ari_nm),
    ]:
        Zp, Zu = np.column_stack(packed), np.column_stack(unpacked)
        pr_p, _, _ = crossfit_axis(Zp, yy, fold, bias, rows=rows_m)
        pr_u, Wu, lam = crossfit_axis(Zu, yy, fold, bias, ridge=True, rows=rows_m)
        print(f'  ── {axis}축 ' + '─' * 60)
        table(['구성', '열', '손실', 'Δ (양수=펼치기가 이득)'],
              [['사전평균 (현행)', str(Zp.shape[1]), f'{pr_p.mean():.4f}', '기준'],
               ['펼치기 + 능형', str(Zu.shape[1]), f'{pr_u.mean():.4f}',
                ci_str(*paired_ci(pr_p, pr_u))]], ['l', 'r', 'r', 'l'])
        print(f'    선택된 λ (fold 별): {lam}')

        # 가중치 — 능형이 실제로 무엇을 버렸나. 기여도 오름차순이라 위쪽이 회수 후보다.
        msk_r = np.ones(len(yy), bool) if rows_m is None else rows_m
        wm, ws = Wu[:, :Zu.shape[1]].mean(0), Wu[:, :Zu.shape[1]].std(0)
        csd = Zu[msk_r].std(0)
        contrib = np.abs(wm) * csd                      # 1σ 움직일 때 로짓이 흔들리는 폭
        same = (np.sign(Wu[:, :Zu.shape[1]]) == np.sign(wm)).sum(0)
        order = np.argsort(contrib)
        print('\n    펼친 열의 가중치 — 기여도(|w|×열 std, 1σ당 로짓) **오름차순**.')
        print('    위쪽이 능형에 눌려 값을 못 받은 열이고, 곧 회수(제거) 후보다.')
        print('    부호일치는 5 fold 중 평균과 같은 부호가 몇 번인지 — 3 이하면 방향이 흔들린다는 뜻.')
        table(['#', '열', 'w 평균', 'w std', '열 std', '기여도', '부호일치'],
              [[str(i + 1), unames[j], f'{wm[j]:+.4f}', f'{ws[j]:.4f}',
                f'{csd[j]:.3f}', f'{contrib[j]:.4f}', f'{same[j]}/{K}']
               for i, j in enumerate(order)], ['r', 'l', 'r', 'r', 'r', 'r', 'r'])

        # 회수 검사 — 기여도 하위 k 개를 **학습 fold 안에서 골라** 버리고 재적합
        print('\n    회수 검사 — 기여도 하위 k 개를 버리고 재적합. 열 선택은 학습 fold 안에서만 한다')
        print('    (전체를 보고 버리면 그게 곧 선택 편향이다). 손실이 안 오르면 그 열들은 없어도 된다.')
        print('    λ 는 전부 사용 때 고른 fold 별 값을 재사용한다 — 열마다 중첩 CV 를 다시 도는 비용을 아낀다.')
        rows = []
        for dk in range(0, Zu.shape[1] - 1):
            loss = np.full(len(yy), np.nan)
            kept = []
            for f_ in range(K):
                tr, te = (fold != f_) & msk_r, (fold == f_) & msk_r
                p0 = _fit(Zu[tr], yy[tr], lam[f_], bias)
                c0 = np.abs(p0[:Zu.shape[1]]) * Zu[tr].std(0)
                keep = np.sort(np.argsort(c0)[dk:])
                p1 = _fit(Zu[tr][:, keep], yy[tr], lam[f_], bias)
                loss[te] = _rowloss(_score(p1, Zu[te][:, keep], bias), yy[te])
                kept.append(set(keep))
            drop_always = sorted(set(range(Zu.shape[1])) - set.union(*kept))
            note = '—' if dk == 0 else (','.join(unames[j] for j in drop_always)
                                        or '(fold 마다 갈림)')
            rows.append([str(dk), str(Zu.shape[1] - dk), f'{loss[msk_r].mean():.4f}',
                         ci_str(*paired_ci(pr_u, loss[msk_r])), note])
        table(['버린 수', '남은 열', '손실', 'Δ vs 전부 사용(양수=버리는 게 이득)',
               '전 fold 공통으로 버려진 열'], rows, ['r', 'r', 'r', 'l', 'l'])
        print()


if __name__ == '__main__':
    main()
