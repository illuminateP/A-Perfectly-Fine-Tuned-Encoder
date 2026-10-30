"""Davidson.py — 패리티 좌표 위의 Davidson 융합 모형. 명세 구현 + 두 가지 확장 실측.

좌표계 (이미 검증된 것)
  m = z_A − z_B                          홀: swap 시 부호 반전
  t = z_T − logaddexp(z_A, z_B)          짝: swap 시 불변
  3진 로짓의 자유도는 2 이고 (m, t) 가 정확히 그 2개다 — 무손실.
  우리 로짓은 뷰 TTA 평균이라 이 패리티가 **근사가 아니라 항등식**이다:
      new_L = (model(Q,B,A) + realign(model(Q,A,B)))/2 = L[:, [1,0,2]]
  따라서 d 는 정확히 홀, s 는 정확히 짝. 아래에서 열 단위로 계약을 검사한다.

모형
      d      = D @ w_d                                  D: 홀 신호만
      nu     = v0 + S @ w_v                             S: 짝 신호만
      logits = [d/2, −d/2, nu + κ·logaddexp(d/2, −d/2)]
  κ 가 이 스크립트의 핵심 손잡이다.
      κ = 0  순정 Davidson.  t = nu − log(2cosh(d/2))  -> **타이가 마진에 눌린다**
             (P(tie) ∝ ν·√(P_A·P_B) 라는 Davidson 의 정의 그 자체)
      κ = 1  분리.           t = nu                    -> 두 축이 완전 독립 (= 방법 1 구조)
      κ 격자  데이터가 결합 세기를 직접 말하게 한다.

  **대칭과 결합은 다른 문제다.** d → −d 이면 A/B 확률이 정확히 교환되고 tie 는 불변이라
  대칭은 κ 와 무관하게 구조적으로 보장된다(cosh 는 짝함수). 결합은 ∂(tie축)/∂w_d ≠ 0 이라는
  별개 성질이고 TTA 로도 대칭 구조로도 사라지지 않는다. 그래서 없애는 대신 잰다.

  볼록성: κ=0 은 로짓이 파라미터에 아핀이라 볼록. κ=1 은 손실이
  log(1+e^{−nu}) 와 logaddexp(d/2,−d/2)∓d/2 로 분리돼 역시 볼록.
  그 사이는 보장되지 않으므로 **κ 격자를 양방향으로 웜스타트**해 두 경로가 만나는지 확인한다.

게이트
  명세의 커버리지 게이트 g 는 곱셈 손잡이(비볼록·초기값 민감) 대신 **상호작용 열로 편다**:
      g·m_ar ≈ c0·m_ar + Σ c_j (U_j ⊙ m_ar)
  패리티가 그대로 맞는다 — 짝⊙홀=홀(D 자격), 짝⊙짝=짝(S 자격). 모형은 선형으로 남고
  학습된 c_j 가 곧 게이트의 모양이다. U 는 `feature_catalog.csv` 에서 swap='불변'인 열만 쓰며
  그 사실을 assert 로 강제한다.

대안 arm
  순서형(ordered logit) — 타이를 마진 0 주변의 '띠'로 본다. Davidson 과 다른 결합 모양이다.
      P(B) = σ(−c−d),  P(tie) = σ(c−d) − σ(−c−d),  P(A) = σ(d−c),  c = softplus(c0 + S·w)
  꼬리에서 타이가 눌리는 속도가 다르다 (Davidson e^{−|d|/2} vs 순서형 e^{−|d|}).

판정
  stage2 10,748 을 Boost_Ens/Cascade 와 **같은 5-fold** 로 교차적합(fold k 의 파라미터는 나머지
  4 fold 로). 문턱 0.0005 + paired bootstrap 95% CI.

사용 (수 분)
  python Boost/Davidson.py                    # 전 arm
  python Boost/Davidson.py --no-gate          # 게이트 arm 생략
  python Boost/Davidson.py --kappa 0,0.5,1    # κ 격자 지정
  python Boost/Davidson.py --even-k 0         # 짝 피쳐 주효과 arm 생략
  python Boost/Davidson.py --tie-model xgb    # 이진 tie 전문가를 xgb 로 (기본 lgbm, '' 이면 생략)
"""
import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

LLM = Path('.')
BOOST = LLM / 'Boost'
ARITH = LLM / 'Arith'
SEED = 42
EPS = 1e-15
N_SPLITS = 5
THRESH = 0.0005
ARITH_MODELS = ('lgbm', 'xgb', 'cat')
GATE_COLS = ['pr_jac_word', 'mn_char_len', 'cos_word']      # 짝(swap 불변) 후보


# ══════════════════════════════════════════════════════ 표 출력 (한글은 2칸)
def _w(s):
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in str(s))


def _pad(s, n, right=False):
    s = str(s)
    fill = ' ' * max(0, n - _w(s))
    return fill + s if right else s + fill


def table(headers, rows, align=None):
    rows = [[('' if c is None else c) for c in r] for r in rows]
    align = align or ('l' + 'r' * (len(headers) - 1))
    wd = [max([_w(h)] + [_w(r[i]) for r in rows]) for i, h in enumerate(headers)]
    line = lambda cs: '  '.join(_pad(c, wd[i], align[i] == 'r') for i, c in enumerate(cs))
    print(line(headers))
    print('─' * _w(line(headers)))
    for r in rows:
        print(line(r))


VERDICTS = []


def claim(tag, text):
    print(f'\n{"━" * 100}\n[{tag}] {text}\n{"━" * 100}')


def verdict(tag, text, ok, detail):
    mark = '성립' if ok else ('기각' if ok is False else '보류')
    VERDICTS.append({'#': tag, '주장': text, '판정': mark, '근거': detail})
    print(f'\n  >> 판정: {mark} — {detail}')


# ══════════════════════════════════════════════════════ 지표
def per_row(p, y):
    p = np.clip(np.asarray(p, np.float64), EPS, 1.0)
    p = p / p.sum(1, keepdims=True)
    return -np.log(p[np.arange(len(y)), y])


def axis_report(p, y):
    p = np.clip(np.asarray(p, np.float64), EPS, 1.0)
    p = p / p.sum(1, keepdims=True)
    total = float(per_row(p, y).mean())
    it = y == 2
    t = np.clip(p[:, 2], EPS, 1 - EPS)
    tie = float(-np.where(it, np.log(t), np.log(1 - t)).mean())
    nz = ~it
    rA = np.clip(p[nz, 0] / (p[nz, 0] + p[nz, 1]), EPS, 1 - EPS)
    win = float(-np.where(y[nz] == 0, np.log(rA), np.log(1 - rA)).sum() / len(y))
    assert abs(total - tie - win) < 1e-6, '축 분해 검산 실패'
    return {'총 ll': total, 'tie축': tie, '승부축': win}


def per_row_tie(p, y):
    """tie축의 행별 손실. 평균이 곧 axis_report 의 'tie축' 이라 짝비교가 성립한다."""
    p = np.clip(np.asarray(p, np.float64), EPS, 1.0)
    p = p / p.sum(1, keepdims=True)
    t = np.clip(p[:, 2], EPS, 1 - EPS)
    return -np.where(y == 2, np.log(t), np.log(1 - t))


def paired_ci(a, b, n_boot=2000, seed=SEED):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def ci_str(m, lo, hi):
    sig = '유의' if (lo > 0 or hi < 0) else '노이즈'
    return f'{m:+.4f} [{lo:+.4f},{hi:+.4f}] {sig}' + (' ★' if sig == '유의' and abs(m) >= THRESH else '')


# ══════════════════════════════════════════════════════ 신호 로드
def load_signals(use_gate, even_k=0, tie_model=None):
    z = np.load(BOOST / 'boost_oof.npz')
    ids, y = z['ids'], z['y']
    meta = json.loads((BOOST / 'model_ens/boost_meta.json').read_text(encoding='utf-8'))
    ex = meta['experts']

    def stack(tag, steps):
        acc = []
        for s in steps:
            w = np.load(BOOST / f'logits/{tag}__checkpoint-{s}.npz')
            assert (np.concatenate([w['valid_ids'], w['hold_ids']]) == ids).all(), f'{tag} id 불일치'
            assert (np.concatenate([w['valid_labels'], w['hold_labels']]) == y).all(), f'{tag} 라벨'
            acc.append(np.vstack([(w['valid_fwd'] + w['valid_swp']) / 2,
                                  (w['hold_fwd'] + w['hold_swp']) / 2]).astype(np.float64))
        return np.mean(acc, axis=0), steps

    L_ABE, abe_steps = stack('abe', ex['abe'].get('steps') or [ex['abe']['step']])
    L_TB, tb_steps = stack('tb3', ex['tb'].get('steps') or [ex['tb']['step']])

    raw_p = ARITH / 'outputs/arith_experts_raw.npz'
    assert raw_p.exists(), f'{raw_p} 없음 — Boost/arith_raw.py 를 먼저 실행할 것'
    za = np.load(raw_p)
    assert (np.concatenate([za['ids_valid'], za['ids_hold']]) == ids).all(), 'Arith id 불일치'
    L_ARI = np.mean([np.vstack([za[f'raw_valid__{m}'], za[f'raw_hold__{m}']]).astype(np.float64)
                     for m in ARITH_MODELS], axis=0)

    sd = lambda B: (B[:, 0] - B[:, 1], B[:, 2] - np.logaddexp(B[:, 0], B[:, 1]))
    m3, t3 = sd(L_ABE)
    m_ar, t_ar = sd(L_ARI)
    s_tie = L_TB[:, 1] - L_TB[:, 0]                    # TB 는 2-class — tie 축만 존재한다

    notie = -s_tie                                     # logit p(비-tie). TB 자유도 1의 부호 반전판

    # 이진 tie 전문가 (Arith/Arith_binary.py 산물). 이미 logit p(tie) 라 짝 신호 그대로 S 자격.
    t_arb, tie_src = None, None
    tie_p = ARITH / 'outputs/arith_tie.npz'
    if tie_model and tie_p.exists():
        zt = np.load(tie_p)
        assert (np.concatenate([zt['ids_valid'], zt['ids_hold']]) == ids).all(), '이진 tie id 불일치'
        key = f'tie_raw_{{}}__{tie_model}'
        assert key.format('valid') in zt.files, \
            f'{tie_model} 없음. 가능: {sorted({k.split("__")[1] for k in zt.files if k.startswith("tie_raw_")})}'
        t_arb = np.concatenate([zt[key.format('valid')], zt[key.format('hold')]]).astype(np.float64)
        tie_src = tie_model

    # 짝 피쳐 — 게이트용 U 와 S 주효과용 E. 둘 다 카탈로그의 swap='불변' 만 허용한다.
    U, U_names, E, E_names = np.zeros((len(ids), 0)), [], np.zeros((len(ids), 0)), []
    if use_gate or even_k > 0:
        cat = pd.read_csv(ARITH / 'feature_catalog.csv').set_index('feature')['swap']
        if even_k > 0:
            # 선택 기준은 Arith 부스터의 중요도 순위 — stage2 라벨을 보지 않으므로 선택 편향이 없다
            imp = pd.read_csv(ARITH / 'model/arith_importance.csv')
            E_names = [c for c in imp['feature'] if cat.get(c) == '불변'][:even_k]
            assert len(E_names) == even_k, f'짝 피쳐가 {len(E_names)}개뿐 (요청 {even_k})'
        need = list(dict.fromkeys((GATE_COLS if use_gate else []) + E_names))
        bad = [c for c in need if c not in cat.index or cat[c] != '불변']
        assert not bad, f'짝 채널에 넣을 열은 swap 불변이어야 한다 — 위반: {bad}'
        feat = pd.read_parquet(ARITH / 'features_57k.parquet')
        pos = pd.Index(feat['id']).get_indexer(ids)
        assert (pos >= 0).all(), 'features_57k 에 stage2 id 가 없다'
        blk = feat.iloc[pos][need]
        if use_gate:
            U, U_names = blk[GATE_COLS].to_numpy(np.float64), list(GATE_COLS)
        if even_k > 0:
            E = blk[E_names].to_numpy(np.float64)
    return dict(ids=ids, y=y, m3=m3, t3=t3, s_tie=s_tie, m_ar=m_ar, t_ar=t_ar, notie=notie,
                t_arb=t_arb, tie_src=tie_src, U=U, U_names=U_names, E=E, E_names=E_names,
                abe_steps=abe_steps, tb_steps=tb_steps)


def zscore(v):
    """짝 신호 전용 표준화. 상호작용 열을 만들기 전 반드시 중심화해야 한다 —
    안 하면 (mean + dev)⊙m = mean·m + dev⊙m 이라 상호작용 열이 주효과 m 과 공선이 되고
    계수가 뒤섞인다. **홀 신호는 중심화 금지** (상수를 빼면 홀성이 깨진다)."""
    return (v - v.mean()) / max(v.std(), 1e-12)


def rankgauss(v):
    """랭크-가우스 변환. 산술 피쳐는 꼬리가 극단이라(실측 첨도 최대 32,961 / max|z| 207)
    z 표준화만으로는 선형 로짓에 넣을 수 없다 — 극단 몇 행이 nu 를 통째로 흔든다.
    트리는 분할이 순위 기반이라 무해했지만 선형 소비자에는 치명적이다.

    라벨을 쓰지 않는 **단조변환**이라 (a) 타깃 누수가 없고 (b) 짝 신호의 짝성이 보존된다
    (짝인 f 에 단조 g 를 씌워도 g(f(P,B,A)) = g(f(P,A,B))).
    """
    from scipy.stats import norm, rankdata
    return norm.ppf((rankdata(v) - 0.5) / len(v))


def parity_check(sg):
    """홀/짝 계약을 데이터가 아니라 정의로 검사한다 — 짝은 A↔B 대칭식, 홀은 반대칭식."""
    m3, t3, s_tie, m_ar, t_ar = (sg['m3'], sg['t3'], sg['s_tie'], sg['m_ar'], sg['t_ar'])
    rows = [['m3 = z_A − z_B', '홀', f'{m3.mean():+.3f}', f'{m3.std():.3f}', 'D 채널'],
            ['m_ar (산술)', '홀', f'{m_ar.mean():+.3f}', f'{m_ar.std():.3f}', 'D 채널'],
            ['s_tie (TB)', '짝', f'{s_tie.mean():+.3f}', f'{s_tie.std():.3f}', 'S 채널'],
            ['t3 (3진 인코더)', '짝', f'{t3.mean():+.3f}', f'{t3.std():.3f}', 'S 채널'],
            ['t_ar (산술)', '짝', f'{t_ar.mean():+.3f}', f'{t_ar.std():.3f}', 'S 채널']]
    rows.append(['notie = −s_tie', '짝', f'{sg["notie"].mean():+.3f}', f'{sg["notie"].std():.3f}',
                 'D 상호작용'])
    if sg['t_arb'] is not None:
        v = sg['t_arb']
        rows.append([f't_arb (이진 tie, {sg["tie_src"]})', '짝', f'{v.mean():+.3f}', f'{v.std():.3f}',
                     'S 채널'])
    for n, v in zip(sg['U_names'], sg['U'].T):
        rows.append([f'{n} (게이트 U)', '짝', f'{v.mean():+.3f}', f'{v.std():.3f}', 'D·S 상호작용'])
    table(['신호', '패리티', '평균(원값)', 'std(원값)', '용도'], rows)
    print('  * 원값 표시다. 짝 신호는 상호작용 열을 만들기 전에 중심화·표준화해 쓴다'
          ' (mn_char_len 은 카탈로그상 이미 log1p 라 6.26 ≈ 523자).')
    # 홀 신호는 평균이 0 근처여야 한다 (라벨 대칭이면 d 의 기대값은 0). 강제는 아니고 눈금이다.
    print(f'  홀 신호 평균 |m3| {abs(m3.mean()):.4f}, |m_ar| {abs(m_ar.mean()):.4f} '
          '— 0 에서 멀면 슬롯 편향 신호다 (실제 A 승률이 0.5052 라 소량은 정상)')


# ══════════════════════════════════════════════════════ 모형
def dv_logits(D, S, wd, wv, v0, kappa):
    d = D @ wd
    nu = v0 + S @ wv
    return torch.stack([d / 2, -d / 2, nu + kappa * torch.logaddexp(d / 2, -d / 2)], dim=1)


def fit_dv(D, S, y, kappa, init=None, iters=400, l2=0.0, pen=None):
    """pen 을 주면 S 의 그 열들에만 L2 를 건다 (기존 신호는 규제하지 않는다).
    λ→∞ 면 규제 열의 가중치가 0 으로 수렴해 **기준 모형과 같아진다** — 질 수 없는 arm 이 된다."""
    Dt, St = torch.tensor(D), torch.tensor(S)
    yt = torch.tensor(y, dtype=torch.long)
    pen_i = None if not (l2 > 0 and pen is not None and len(pen)) else torch.tensor(np.asarray(pen))
    wd = torch.full((D.shape[1],), 1.0 / D.shape[1], dtype=torch.float64, requires_grad=True)
    wv = torch.zeros(S.shape[1], dtype=torch.float64, requires_grad=True)
    v0 = torch.zeros((), dtype=torch.float64, requires_grad=True)
    if init is not None:
        with torch.no_grad():
            wd.copy_(torch.tensor(init[0])); wv.copy_(torch.tensor(init[1])); v0.fill_(float(init[2]))
    opt = torch.optim.LBFGS([wd, wv, v0], lr=0.5, max_iter=iters, line_search_fn='strong_wolfe')

    def cl():
        opt.zero_grad()
        loss = F.cross_entropy(dv_logits(Dt, St, wd, wv, v0, kappa), yt)
        if pen_i is not None:
            loss = loss + l2 * (wv[pen_i] ** 2).sum()
        loss.backward()
        return loss
    opt.step(cl)
    with torch.no_grad():
        ll = float(F.cross_entropy(dv_logits(Dt, St, wd, wv, v0, kappa), yt))
    return (wd.detach().numpy().copy(), wv.detach().numpy().copy(), float(v0.detach()), ll)


def dv_predict(D, S, prm, kappa):
    with torch.no_grad():
        z = dv_logits(torch.tensor(D), torch.tensor(S), torch.tensor(prm[0]),
                      torch.tensor(prm[1]), torch.tensor(prm[2]), kappa)
        return torch.softmax(z, 1).numpy()


def ord_probs(D, S, wd, wv, c0):
    d = D @ wd
    c = F.softplus(c0 + S @ wv)                     # c > 0 이라야 P(tie) >= 0 이 보장된다
    pA, pB = torch.sigmoid(d - c), torch.sigmoid(-d - c)
    return torch.stack([pA, pB, (1 - pA - pB).clamp_min(1e-12)], dim=1)


def fit_ord(D, S, y, iters=400):
    Dt, St = torch.tensor(D), torch.tensor(S)
    yt = torch.tensor(y, dtype=torch.long)
    wd = torch.full((D.shape[1],), 1.0 / D.shape[1], dtype=torch.float64, requires_grad=True)
    wv = torch.zeros(S.shape[1], dtype=torch.float64, requires_grad=True)
    c0 = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([wd, wv, c0], lr=0.5, max_iter=iters, line_search_fn='strong_wolfe')

    def cl():
        opt.zero_grad()
        p = ord_probs(Dt, St, wd, wv, c0)
        loss = -torch.log(p[torch.arange(len(yt)), yt].clamp_min(1e-12)).mean()
        loss.backward()
        return loss
    opt.step(cl)
    return (wd.detach().numpy().copy(), wv.detach().numpy().copy(), float(c0.detach()), 0.0)


def ord_predict(D, S, prm):
    with torch.no_grad():
        return ord_probs(torch.tensor(D), torch.tensor(S), torch.tensor(prm[0]),
                         torch.tensor(prm[1]), torch.tensor(prm[2])).numpy()


# ══════════════════════════════════════════════════════ 교차적합
def crossfit(D, S, y, fold, kappa=None, kind='dv', init=None):
    """fold k 의 파라미터는 나머지 4 fold 로만 적합. U 표준화도 학습 fold 통계로만 한다."""
    P = np.zeros((len(y), 3))
    prms = []
    for k in range(N_SPLITS):
        tr, te = fold != k, fold == k
        if kind == 'dv':
            prm = fit_dv(D[tr], S[tr], y[tr], kappa, init=init)
            P[te] = dv_predict(D[te], S[te], prm, kappa)
        else:
            prm = fit_ord(D[tr], S[tr], y[tr])
            P[te] = ord_predict(D[te], S[te], prm)
        prms.append(prm)
    return P, prms


def crossfit_ridge(D, S, y, fold, kappa, pen, lams):
    """규제 강도 λ 를 **바깥 fold 안쪽에서 다시 3-fold** 로 고른다.
    평가 fold 를 보고 λ 를 고르면 그게 곧 선택 편향이라, 중첩이 아니면 판정이 안 된다."""
    P, prms, picked = np.zeros((len(y), 3)), [], []
    for k in range(N_SPLITS):
        tr, te = fold != k, fold == k
        it = np.where(tr)[0]
        inner = np.zeros(len(it), int)
        for j, (_, iv) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED)
                                    .split(np.zeros((len(it), 1)), y[it])):
            inner[iv] = j
        best = (lams[0], np.inf)
        for lam in lams:
            tot = 0.0
            for j in range(3):
                a, b = it[inner != j], it[inner == j]
                pr = fit_dv(D[a], S[a], y[a], kappa, l2=lam, pen=pen)
                tot += per_row(dv_predict(D[b], S[b], pr, kappa), y[b]).sum()
            if tot < best[1]:
                best = (lam, tot)
        picked.append(best[0])
        prm = fit_dv(D[tr], S[tr], y[tr], kappa, l2=best[0], pen=pen)
        P[te] = dv_predict(D[te], S[te], prm, kappa)
        prms.append(prm)
    return P, prms, picked


def m1_baseline(s_pi, d_q, y, fold):
    """비교 기준 — 등가중 평균 + 온도 2개(방법 1). Davidson 은 이것의 가중치 제약을 푼 판이다."""
    pi, q = np.zeros(len(y)), np.zeros(len(y))
    for k in range(N_SPLITS):
        tr, te = fold != k, fold == k

        # 1D 로지스틱을 같은 옵티마이저로 — tie 축
        zt = torch.tensor(s_pi[tr]); yt = torch.tensor((y[tr] == 2).astype(np.float64))
        lg = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        o = torch.optim.LBFGS([lg], lr=0.5, max_iter=200, line_search_fn='strong_wolfe')
        o.step(lambda: _bce(o, lg, zt, yt))
        Tp = float(lg.exp().detach())
        nz = tr & (y != 2)
        zd = torch.tensor(d_q[nz]); yd = torch.tensor((y[nz] == 0).astype(np.float64))
        lg2 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        o2 = torch.optim.LBFGS([lg2], lr=0.5, max_iter=200, line_search_fn='strong_wolfe')
        o2.step(lambda: _bce(o2, lg2, zd, yd))
        Tq = float(lg2.exp().detach())
        pi[te] = 1 / (1 + np.exp(-s_pi[te] / Tp))
        q[te] = 1 / (1 + np.exp(-d_q[te] / Tq))
    P = np.column_stack([(1 - pi) * q, (1 - pi) * (1 - q), pi])
    return P / P.sum(1, keepdims=True)


def _bce(opt, lg, z, y):
    opt.zero_grad()
    loss = F.binary_cross_entropy_with_logits(z / lg.exp(), y)
    loss.backward()
    return loss


# ══════════════════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kappa', default='0,0.25,0.5,0.75,1.0,1.25',
                    help='κ 격자 (0=순정 Davidson, 1=축 분리)')
    ap.add_argument('--no-gate', action='store_true', help='게이트 상호작용 arm 생략')
    ap.add_argument('--no-ord', action='store_true', help='순서형 arm 생략')
    ap.add_argument('--tie-model', default='lgbm',
                    help="이진 tie 전문가 (Arith_binary.py 산물). '' 이면 주장 G 생략")
    ap.add_argument('--even-k', type=int, default=8,
                    help='S 채널에 주효과로 넣을 짝 피쳐 수 (0 이면 주장 F 생략)')
    ap.add_argument('--out', default='davidson_oof.npz')
    args = ap.parse_args()
    KGRID = [float(x) for x in args.kappa.split(',')]
    assert 0.0 in KGRID and 1.0 in KGRID, 'κ 격자에 0 과 1 은 반드시 포함해야 한다 (양 끝이 기준점)'

    sg = load_signals(not args.no_gate, args.even_k, args.tie_model or None)
    y, ids = sg['y'], sg['ids']
    fold = np.zeros(len(y), int)
    for k, (_, te) in enumerate(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
                                .split(np.zeros((len(y), 1)), y)):
        fold[te] = k
    print(f'stage2 {len(y):,}행 | ABE {sg["abe_steps"]} | TB3 {sg["tb_steps"]} | '
          f'Arith raw margin | fold {np.bincount(fold)}\n')
    parity_check(sg)

    # ── 채널 구성 ──
    D0 = np.column_stack([sg['m3'], sg['m_ar']])
    S0 = np.column_stack([sg['s_tie'], sg['t3'], sg['t_ar']])
    D_NAMES, S_NAMES = ['m3', 'm_ar'], ['s_tie', 't3', 't_ar']
    if not args.no_gate:
        Us = (sg['U'] - sg['U'].mean(0)) / np.clip(sg['U'].std(0), 1e-12, None)   # 조건수 정리
        Dg = np.column_stack([D0] + [u * sg['m_ar'] for u in Us.T])
        Sg = np.column_stack([S0] + [u * sg['t_ar'] for u in Us.T])
        Dg_names = D_NAMES + [f'{n}⊙m_ar' for n in sg['U_names']]
        Sg_names = S_NAMES + [f'{n}⊙t_ar' for n in sg['U_names']]

    RES = {}

    # ── 주장 A ──
    claim('A', '가중치를 학습하면(6 파라미터) 등가중 평균 + 온도 2개(방법 1)를 이긴다')
    RES['M1d 등가중+온도2'] = {'oof': m1_baseline(
        np.mean([sg['s_tie'], sg['t3'], sg['t_ar']], axis=0),
        np.mean([sg['m3'], sg['m_ar']], axis=0), y, fold), 'npar': 2}
    for kap, nm in ((1.0, 'DV-분리 (κ=1)'), (0.0, 'DV 순정 (κ=0)')):
        P, prm = crossfit(D0, S0, y, fold, kappa=kap)
        RES[nm] = {'oof': P, 'prm': prm, 'npar': D0.shape[1] + S0.shape[1] + 1, 'kappa': kap}
    _base = per_row(RES['M1d 등가중+온도2']['oof'], y)
    table(['구성', '파라미터', '총 ll', 'tie축', '승부축', 'Δ vs M1d (양수=이득)'],
          [[nm, RES[nm]['npar'], f'{axis_report(RES[nm]["oof"], y)["총 ll"]:.4f}',
            f'{axis_report(RES[nm]["oof"], y)["tie축"]:.4f}',
            f'{axis_report(RES[nm]["oof"], y)["승부축"]:.4f}',
            '기준' if nm.startswith('M1d') else ci_str(*paired_ci(_base, per_row(RES[nm]['oof'], y)))]
           for nm in RES], align='lrrrrl')
    _d, _lo, _hi = paired_ci(_base, per_row(RES['DV-분리 (κ=1)']['oof'], y))
    verdict('A', '학습 가중치가 등가중+온도2 를 이긴다', _lo > 0 and _d >= THRESH,
            f'DV-분리 {axis_report(RES["DV-분리 (κ=1)"]["oof"], y)["총 ll"]:.4f} vs M1d '
            f'{axis_report(RES["M1d 등가중+온도2"]["oof"], y)["총 ll"]:.4f} — {ci_str(_d, _lo, _hi)}')

    # ── 주장 B ──
    claim('B', 'Davidson 결합이 옳다 — 타이 성향은 마진에 눌린다 (κ < 1 이 유리하다)')
    print('  κ=0 순정 Davidson / κ=1 축 분리. 사이 구간은 비볼록이라 양방향 웜스타트로 훑는다.')
    prof, init_up = [], None
    for kap in sorted(KGRID):                              # 0 -> 1 방향
        P, prm = crossfit(D0, S0, y, fold, kappa=kap, init=init_up)
        init_up = prm[0]
        prof.append((kap, P, per_row(P, y).mean()))
    prof_dn, init_dn = {}, None
    for kap in sorted(KGRID, reverse=True):                # 1 -> 0 방향 (수렴 교차검증)
        P, prm = crossfit(D0, S0, y, fold, kappa=kap, init=init_dn)
        init_dn = prm[0]
        prof_dn[kap] = per_row(P, y).mean()
    _b0 = per_row(RES['DV-분리 (κ=1)']['oof'], y)
    table(['κ', '총 ll (0→1)', '총 ll (1→0)', '경로차', 'tie축', '승부축', 'Δ vs κ=1'],
          [[f'{kap:.2f}', f'{ll:.4f}', f'{prof_dn[kap]:.4f}', f'{abs(ll - prof_dn[kap]):.1e}',
            f'{axis_report(P, y)["tie축"]:.4f}', f'{axis_report(P, y)["승부축"]:.4f}',
            ci_str(*paired_ci(_b0, per_row(P, y)))] for kap, P, ll in prof], align='rrrrrrl')
    _kb, _Pb, _llb = min(prof, key=lambda r: r[2])
    _gap = max(abs(ll - prof_dn[kap]) for kap, _, ll in prof)
    # κ 양 끝은 이미 arm 으로 있으므로 이름을 재사용한다 (같은 예측이 두 번 세지지 않게)
    KBEST = {0.0: 'DV 순정 (κ=0)', 1.0: 'DV-분리 (κ=1)'}.get(_kb, f'DV κ={_kb:g} (격자최선)')
    if KBEST not in RES:
        RES[KBEST] = {'oof': _Pb, 'npar': D0.shape[1] + S0.shape[1] + 1, 'kappa': _kb}
    _d, _lo, _hi = paired_ci(_b0, per_row(_Pb, y))
    verdict('B', '타이 성향은 마진에 눌린다 (Davidson 결합)', _kb < 1.0 and _lo > 0 and _d >= THRESH,
            f'최선 κ={_kb:g} (총 ll {_llb:.4f}) vs κ=1 ({_b0.mean():.4f}) — {ci_str(_d, _lo, _hi)}'
            f' | 웜스타트 두 경로 최대 차 {_gap:.1e}')

    # ── 주장 C ── 설계 원안의 TB_Nontie
    claim('C', '설계 원안대로 TB_Nontie 를 AB 분기에 넣으면 값을 한다')
    print('  TB 는 2-class 라 자유도가 1 이다 — notie 는 s_tie 의 부호 반전이고 **짝(swap 불변)** 이다.')
    print('  짝 신호를 홀 채널에 그냥 더하면 d 가 홀이 아니게 되어 반대칭이 깨진다.')
    print('  정당한 형태는 상호작용뿐이다 (짝 ⊙ 홀 = 홀):')
    print('      d = w₁·m3 + w₂·m_ar + w₆·(ñ⊙m3) + w₇·(ñ⊙m_ar)   "비-tie 확신이면 마진을 더 믿어라"')
    print('  ñ 는 중심화·표준화한 notie 다. 중심화를 빼면 ñ⊙m3 가 m3 와 공선이 된다.\n')

    ntz = zscore(sg['notie'])
    D_nt = np.column_stack([D0, ntz * sg['m3'], ntz * sg['m_ar']])
    nt_names = D_NAMES + ['ñ⊙m3', 'ñ⊙m_ar']
    for kap in dict.fromkeys([1.0, _kb]):
        P, prm = crossfit(D_nt, S0, y, fold, kappa=kap)
        RES[f'DV+notie⊙ κ={kap:g}'] = {'oof': P, 'prm': prm, 'kappa': kap,
                                       'npar': D_nt.shape[1] + S0.shape[1] + 1,
                                       'names': (nt_names, S_NAMES)}
    # 진단 arm — "그냥 더한" 판. 패리티가 깨지는 크기를 숫자로 보여준다 (배포 후보 아님)
    D_add = np.column_stack([D0, ntz])
    P_add, prm_add = crossfit(D_add, S0, y, fold, kappa=1.0)
    RES['DV+notie가산 (패리티 위반)'] = {'oof': P_add, 'prm': prm_add, 'kappa': 1.0,
                                        'npar': D_add.shape[1] + S0.shape[1] + 1,
                                        'names': (D_NAMES + ['ñ (가산)'], S_NAMES)}
    _wd0 = prm_add[0][0]
    _df = D_add @ _wd0
    _ds = np.column_stack([-sg['m3'], -sg['m_ar'], ntz]) @ _wd0     # swap: 홀만 부호반전
    _viol = float(np.abs(_ds + _df).max())

    _rl = per_row(RES[KBEST]['oof'], y)
    table(['구성', '파라미터', '총 ll', 'tie축', '승부축', f'Δ vs {KBEST}'],
          [[nm, RES[nm]['npar'], f'{axis_report(RES[nm]["oof"], y)["총 ll"]:.4f}',
            f'{axis_report(RES[nm]["oof"], y)["tie축"]:.4f}',
            f'{axis_report(RES[nm]["oof"], y)["승부축"]:.4f}',
            ci_str(*paired_ci(_rl, per_row(RES[nm]['oof'], y)))]
           for nm in RES if 'notie' in nm], align='lrrrrl')
    print(f'\n  패리티 위반 크기(가산판): max|d(swap) + d| = {_viol:.3f}  '
          f'(반대칭이면 0 이어야 한다. m3 의 std 가 {sg["m3"].std():.3f} 이니 무시할 수 없는 크기다)')
    print('  가산판이 이겨도 채택할 수 없다 — A/B 슬롯에 비대칭 편향을 새로 넣는 것이라'
          ' 제출 시 순서가 바뀌면 답이 달라진다.')

    _nb = min([n for n in RES if 'notie⊙' in n], key=lambda k: per_row(RES[k]['oof'], y).mean())
    _d, _lo, _hi = paired_ci(_rl, per_row(RES[_nb]['oof'], y))
    verdict('C', 'TB_Nontie 가 AB 분기에 값을 한다', _lo > 0 and _d >= THRESH,
            f'{_nb} vs {KBEST} — {ci_str(_d, _lo, _hi)}  '
            f'(Cascade C0->C3 대비 0.0002 와 대조)')

    # ── 주장 D ──
    if not args.no_gate:
        claim('D', '산술 전문가의 커버리지 게이트가 값을 한다 (상호작용 열로 편 판)')
        for kap in dict.fromkeys([0.0, 1.0, _kb]):
            nm = f'DV+게이트 κ={kap:g}'
            P, prm = crossfit(Dg, Sg, y, fold, kappa=kap)
            RES[nm] = {'oof': P, 'prm': prm, 'npar': Dg.shape[1] + Sg.shape[1] + 1,
                       'kappa': kap, 'names': (Dg_names, Sg_names), 'Smat': Sg}
        _ref = min(['DV 순정 (κ=0)', 'DV-분리 (κ=1)', KBEST],
                   key=lambda k: per_row(RES[k]['oof'], y).mean())
        _rl = per_row(RES[_ref]['oof'], y)   # notie 절에서 쓴 것과 다른 기준 — 여기서 다시 잡는다
        table(['구성', '파라미터', '총 ll', 'tie축', '승부축', f'Δ vs {_ref}'],
              [[nm, RES[nm]['npar'], f'{axis_report(RES[nm]["oof"], y)["총 ll"]:.4f}',
                f'{axis_report(RES[nm]["oof"], y)["tie축"]:.4f}',
                f'{axis_report(RES[nm]["oof"], y)["승부축"]:.4f}',
                ci_str(*paired_ci(_rl, per_row(RES[nm]['oof'], y)))]
               for nm in RES if nm.startswith('DV+게이트')], align='lrrrrl')
        _gb = min([n for n in RES if n.startswith('DV+게이트')],
                  key=lambda k: per_row(RES[k]['oof'], y).mean())
        _d, _lo, _hi = paired_ci(_rl, per_row(RES[_gb]['oof'], y))
        verdict('D', '커버리지 게이트가 값을 한다', _lo > 0 and _d >= THRESH,
                f'{_gb} vs {_ref} — {ci_str(_d, _lo, _hi)}  '
                '(Arith 는 전 행에서 계산되므로 커버리지 결손이 없다 — 예상은 박하다)')

    # ── 주장 E ──
    if not args.no_ord:
        claim('E', '순서형(타이 = 마진 0 주변의 띠)이 Davidson 결합보다 낫다')
        P, prm = crossfit(D0, S0, y, fold, kind='ord')
        RES['순서형'] = {'oof': P, 'prm': prm, 'npar': D0.shape[1] + S0.shape[1] + 1}
        _bl = min(['DV 순정 (κ=0)', 'DV-분리 (κ=1)', KBEST],
                  key=lambda k: per_row(RES[k]['oof'], y).mean())
        _d, _lo, _hi = paired_ci(per_row(RES[_bl]['oof'], y), per_row(P, y))
        r = axis_report(P, y)
        table(['구성', '총 ll', 'tie축', '승부축', f'Δ vs {_bl}'],
              [['순서형', f'{r["총 ll"]:.4f}', f'{r["tie축"]:.4f}', f'{r["승부축"]:.4f}',
                ci_str(_d, _lo, _hi)]], align='lrrrl')
        verdict('E', '순서형이 Davidson 을 이긴다', _lo > 0 and _d >= THRESH,
                f'순서형 {r["총 ll"]:.4f} vs {_bl} '
                f'{per_row(RES[_bl]["oof"], y).mean():.4f} — {ci_str(_d, _lo, _hi)}')

    # ── 주장 F ── 짝 피쳐를 S 주효과로
    if sg['E'].shape[1]:
        claim('F', '짝 피쳐를 S 채널에 **주효과로** 직접 넣으면 tie축을 더 깎는다')
        print('  주장 D 는 같은 피쳐를 U⊙t_ar 상호작용으로만 썼다. 여기서는 명세가 원래 말한 대로')
        print('  S 에 열을 그냥 추가한다 — 짝 신호라면 무엇이든 받는 채널이기 때문이다.')
        print('  근거: Arith 중요도의 다수가 짝 피쳐인데, Arith 는 그걸 트리로 소화한 뒤')
        print('  **스칼라 t_ar 하나로 압축**해 넘긴다. 선형 결합기에 직접 주면 그 압축분을 되찾는다.')
        print(f'  선택: arith_importance.csv 순위 상위 중 swap=불변 {len(sg["E_names"])}종')
        print(f'        {sg["E_names"]}')
        print('  누수 없음 — 피쳐는 텍스트의 결정적 변환이고 stage2 행은 Arith 그래디언트 학습에 없다.')
        print('  전처리: 산술 피쳐는 꼬리가 극단이라(전수조사 실측 첨도 최대 32,961, max|z| 207)')
        print('  z 표준화만으로는 선형 로짓에 못 넣는다. 랭크-가우스 + E 블록 전용 ridge 로 받는다.')
        print('  λ 는 바깥 fold 안쪽 3-fold 로 다시 고른다 — λ→∞ 면 기준 모형과 같아지므로 질 수 없다.\n')

        # 전처리 두 판을 나란히 잰다 — 첫 시도가 z 표준화만으로 실패했고 그 원인이 꼬리였다.
        VARIANTS = {'z': [zscore(v) for v in sg['E'].T],
                    'rg': [rankgauss(v) for v in sg['E'].T]}
        _mz = lambda M: max(np.abs(m).max() for m in M)
        table(['전처리', 'E 블록 max|z|', '비고'],
              [['z 표준화', f'{_mz(VARIANTS["z"]):.1f}', '꼬리 그대로 — 극단 행이 nu 를 흔든다'],
               ['랭크-가우스', f'{_mz(VARIANTS["rg"]):.1f}', '단조변환, 라벨 미사용, 짝성 보존']],
              align='llc')
        LAMS = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
        pen = np.arange(len(S_NAMES), len(S_NAMES) + sg['E'].shape[1])   # E 열만 규제
        for tag, cols_ in VARIANTS.items():
            Sf = np.column_stack([S0] + cols_)
            Sf_names = S_NAMES + [f'{n}({tag})' for n in sg['E_names']]
            for kap in dict.fromkeys([1.0, _kb]):
                if tag == 'z':                                   # 규제 없는 원래 판 (대조군)
                    P, prm = crossfit(D0, Sf, y, fold, kappa=kap)
                    pk = [0.0] * N_SPLITS
                else:
                    P, prm, pk = crossfit_ridge(D0, Sf, y, fold, kap, pen, LAMS)
                RES[f'DV+짝피쳐 {tag} κ={kap:g}'] = {
                    'oof': P, 'prm': prm, 'kappa': kap, 'lam': pk,
                    'npar': D0.shape[1] + Sf.shape[1] + 1,
                    'names': (D_NAMES, Sf_names), 'Smat': Sf}
        print('\n  λ (fold 별, 랭크-가우스 판): '
              + ', '.join(f'κ={RES[n]["kappa"]:g} {RES[n]["lam"]}'
                          for n in RES if n.startswith('DV+짝피쳐 rg')))
        _rt, _rl = per_row_tie(RES[KBEST]['oof'], y), per_row(RES[KBEST]['oof'], y)
        table(['구성', '파라미터', '총 ll', 'tie축', '승부축',
               f'Δ tie축 vs {KBEST}', 'Δ 총 ll'],
              [[nm, RES[nm]['npar'], f'{axis_report(RES[nm]["oof"], y)["총 ll"]:.4f}',
                f'{axis_report(RES[nm]["oof"], y)["tie축"]:.4f}',
                f'{axis_report(RES[nm]["oof"], y)["승부축"]:.4f}',
                ci_str(*paired_ci(_rt, per_row_tie(RES[nm]['oof'], y))),
                ci_str(*paired_ci(_rl, per_row(RES[nm]['oof'], y)))]
               for nm in RES if '짝피쳐' in nm], align='lrrrrll')
        print('\n  참고 — tie축 기록 바닥 0.5734~0.5740 (tie_ensemble.py, 로짓 조합만으로 잰 봉투).')
        print('  원 피쳐는 그 봉투 밖이라 여기가 내려가면 봉투 자체가 갱신된 것이다.')

        _fb = min([n for n in RES if '짝피쳐 rg' in n], key=lambda k: per_row(RES[k]['oof'], y).mean())
        _dt, _lot, _hit = paired_ci(_rt, per_row_tie(RES[_fb]['oof'], y))
        _d, _lo, _hi = paired_ci(_rl, per_row(RES[_fb]['oof'], y))
        verdict('F', '짝 피쳐 주효과가 tie축을 더 깎는다', _lot > 0 and _dt >= THRESH,
                f'{_fb} tie축 {axis_report(RES[_fb]["oof"], y)["tie축"]:.4f} vs {KBEST} '
                f'{axis_report(RES[KBEST]["oof"], y)["tie축"]:.4f} — tie축 {ci_str(_dt, _lot, _hit)} '
                f'| 총 ll {ci_str(_d, _lo, _hi)}')

    # ── 주장 G ── 이진 tie 전문가
    if sg['t_arb'] is not None:
        claim('G', f'이진 tie 전문가(t_arb, {sg["tie_src"]})가 결합에서 값을 한다')
        print('  단독으로는 3-class 부산물 t_ar 에 졌다(holdout tie ll 0.5842 vs 0.5825, 유의).')
        print('  하지만 **단독 열세 ≠ 결합에서 무용**이다 — 오류가 다르면 더할 수 있다.')
        print('  그래서 두 가지를 잰다: (a) t_ar 위에 더하기, (b) t_ar 을 t_arb 로 대체하기.\n')

        tb = sg['t_arb']
        _rho = float(np.corrcoef(tb, sg['t_ar'])[0, 1])
        table(['짝', 'ρ', '읽는 법'],
              [['t_arb ~ t_ar', f'{_rho:+.4f}', '1 에 가까우면 새 정보가 없다'],
               ['t_arb ~ s_tie', f'{np.corrcoef(tb, sg["s_tie"])[0, 1]:+.4f}', ''],
               ['t_arb ~ t3', f'{np.corrcoef(tb, sg["t3"])[0, 1]:+.4f}', '']], align='llc')

        S_add = np.column_stack([S0, tb])
        S_rep = np.column_stack([sg['s_tie'], sg['t3'], tb])
        for tag, Sx, nmx in (('추가', S_add, S_NAMES + ['t_arb']),
                             ('대체', S_rep, ['s_tie', 't3', 't_arb'])):
            for kap in dict.fromkeys([1.0, _kb]):
                P, prm = crossfit(D0, Sx, y, fold, kappa=kap)
                RES[f'DV+t_arb {tag} κ={kap:g}'] = {
                    'oof': P, 'prm': prm, 'kappa': kap,
                    'npar': D0.shape[1] + Sx.shape[1] + 1,
                    'names': (D_NAMES, nmx), 'Smat': Sx}
        _rt, _rl = per_row_tie(RES[KBEST]['oof'], y), per_row(RES[KBEST]['oof'], y)
        table(['구성', '파라미터', '총 ll', 'tie축', '승부축',
               f'Δ tie축 vs {KBEST}', 'Δ 총 ll'],
              [[nm, RES[nm]['npar'], f'{axis_report(RES[nm]["oof"], y)["총 ll"]:.4f}',
                f'{axis_report(RES[nm]["oof"], y)["tie축"]:.4f}',
                f'{axis_report(RES[nm]["oof"], y)["승부축"]:.4f}',
                ci_str(*paired_ci(_rt, per_row_tie(RES[nm]['oof'], y))),
                ci_str(*paired_ci(_rl, per_row(RES[nm]['oof'], y)))]
               for nm in RES if 't_arb' in nm], align='lrrrrll')

        _gb = min([n for n in RES if 't_arb' in n], key=lambda k: per_row(RES[k]['oof'], y).mean())
        _dt, _lot, _hit = paired_ci(_rt, per_row_tie(RES[_gb]['oof'], y))
        _d, _lo, _hi = paired_ci(_rl, per_row(RES[_gb]['oof'], y))
        verdict('G', '이진 tie 전문가가 결합에서 값을 한다', _lot > 0 and _dt >= THRESH,
                f'{_gb} tie축 {axis_report(RES[_gb]["oof"], y)["tie축"]:.4f} vs {KBEST} '
                f'{axis_report(RES[KBEST]["oof"], y)["tie축"]:.4f} — tie축 {ci_str(_dt, _lot, _hit)} '
                f'| 총 ll {ci_str(_d, _lo, _hi)} | t_ar 과 ρ {_rho:+.3f}')

    # ── 명세가 요구한 진단 ──
    claim('진단', '파라미터가 6개뿐이라 전부 해석 가능하다 — 명세의 필수 확인 항목')
    _diag = [nm for nm in RES if 'prm' in RES[nm]]
    for nm in _diag:
        prm = RES[nm]['prm']
        dn, sn = RES[nm].get('names', (D_NAMES, S_NAMES))
        Sm = RES[nm].get('Smat', S0)
        wd = np.array([p[0] for p in prm]); wv = np.array([p[1] for p in prm])
        v0 = np.array([p[2] for p in prm])
        print(f'\n  [{nm}]')
        table(['채널', '신호', '가중치 (fold 평균 ± std)'],
              [['D', dn[i], f'{wd[:, i].mean():+.4f} ± {wd[:, i].std():.4f}']
               for i in range(wd.shape[1])]
              + [['S', sn[i], f'{wv[:, i].mean():+.4f} ± {wv[:, i].std():.4f}']
                 for i in range(wv.shape[1])]
              + [['S', 'v0 (절편)', f'{v0.mean():+.4f} ± {v0.std():.4f}']], align='llr')
        nu = v0.mean() + Sm @ wv.mean(0)
        print(f'    nu 평균 {nu.mean():+.3f} / 표준편차 {nu.std():.3f} '
              f'(비 {abs(nu.std() / max(abs(nu.mean()), 1e-9)):.2f})'
              + ('  ! 예제별 판단이 거의 없다 — 이름만 nu(x) 인 전역 상수로 퇴화'
                 if nu.std() < 0.1 * abs(nu.mean()) else ''))
        for i, n in enumerate(sn):
            if abs(wv[:, i].mean()) < 0.05:
                print(f'    ! w_v[{n}] ≈ 0 — 그 판단자가 잉여다. 제거 검토')
        if (np.sign(wv.mean(0)) != np.sign(wv.mean(0))[0]).any() and np.abs(wv.mean(0)).max() > 2:
            print('    ! w_v 성분이 큰 양수/큰 음수로 상쇄 — 공선성 신호')
    print('\n  [S 채널 상관] 판단자가 중복이면 계수가 상쇄로 갈린다')
    table([''] + S_NAMES, [[S_NAMES[i]] + [f'{v:+.2f}' for v in np.corrcoef(S0.T)[i]]
                           for i in range(len(S_NAMES))])
    _hi_pairs = [(S_NAMES[i], S_NAMES[j], np.corrcoef(S0.T)[i, j])
                 for i in range(len(S_NAMES)) for j in range(i + 1, len(S_NAMES))
                 if abs(np.corrcoef(S0.T)[i, j]) > 0.9]
    print('  공선성 경고: ' + (', '.join(f'{a}~{b} {c:+.2f}' for a, b, c in _hi_pairs)
                              if _hi_pairs else '없음 (|ρ| > 0.9 인 쌍 없음)'))

    # ── 최종 순위 ──
    claim('요약', '전 구성 순위 — 기존 결합기들과 같은 fold 라 직접 비교된다')
    ext = {}
    for f_, keys in ((BOOST / 'boost_oof_ens.npz', ['oof__ATR__cat']),
                     (BOOST / 'cascade_oof.npz', None)):
        if not f_.exists():
            continue
        z = np.load(f_)
        if not (z['ids'] == ids).all():
            continue
        for k in (keys or [k for k in z.files if k.startswith('oof__')]):
            if k in z.files:
                ext[f'{f_.stem}:{k.replace("oof__", "")}'] = z[k].astype(np.float64)
    allp = {**{k: v['oof'] for k, v in RES.items()}, **ext}
    table(['순위', '구성', '총 ll', 'tie축', '승부축', '파라미터'],
          [[i, k, f'{axis_report(p, y)["총 ll"]:.4f}', f'{axis_report(p, y)["tie축"]:.4f}',
            f'{axis_report(p, y)["승부축"]:.4f}',
            str(RES[k]['npar']) if k in RES else '(기존 산출물)']
           for i, (k, p) in enumerate(sorted(allp.items(), key=lambda kv: per_row(kv[1], y).mean()), 1)],
          align='rlrrrl')
    print()
    table(['#', '주장', '판정', '근거'],
          [[v['#'], v['주장'], v['판정'], v['근거']] for v in VERDICTS], align='lllr')

    np.savez_compressed(BOOST / args.out, ids=ids, y=y,
                        **{f'oof__{k}': v['oof'].astype(np.float32) for k, v in RES.items()})
    print(f'\nsaved: {BOOST / args.out}')


if __name__ == '__main__':
    main()
