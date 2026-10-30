"""reprhead.py — 인코더 **헤드를 떼고 표현 위에 새 헤드**를 얹으면 값을 하나 (stage2 선행 판정).

무엇을 묻나
  현행 결합기는 세 전문가를 **스칼라 5개**로만 만나게 한다 (m3, m_ar / s_tie, t3, t_ar).
  원재료는 ABE 2,048 + TB3 6,144 + Arith 253 차원이다. 그 압축이 손해인지 잰다.
  판정 질문은 "최종 헤드가 좋은가" 가 아니라 **"표현에 로짓이 안 나르는 신호가 있나"** 다.

최대 가설 — 어느 칸이 비어 있나
  ┌────────┬──────────────────────┬──────────────────────────┐
  │        │ tie 축               │ 승부 축                  │
  ├────────┼──────────────────────┼──────────────────────────┤
  │ ABE    │ t3   (증분 +0.0024)  │ m3   (주력)              │
  │ TB3    │ s_tie(주력)          │ **없음 <- 여기**         │
  │ Arith  │ t_ar (증분 +0.0011)  │ m_ar (증분 +0.0047)      │
  └────────┴──────────────────────┴──────────────────────────┘
  TB 는 2-class 라 출력 자유도가 1 이고 그걸 전부 tie 축에 쓴다. 승부에 대해 한 마디도 못 한다.
  그런데 cross-encoder 다 — "비겼나" 를 판단하려면 두 응답이 어떻게 다른지 인코딩해야 한다.
  그 정보가 표현에는 있는데 **내보낼 자유도가 없어서** 막혀 있을 수 있다. tb_v 가 그 후보다.
  이건 잔차 회수가 아니라 새 신호다.

패리티 — 회전이지 피쳐 공학이 아니다
  (ea, eb) -> ((ea+eb)/2, (ea-eb)/2) 는 가역 선형 변환이다. 선형 헤드에서 두 기저는
  **같은 함수 공간**이다. 정보 차이 0. 쟁점은 회전이 아니라 그 다음에 거는 제약이다.
      tie 축  = 짝 열만        <=> 원 기저에서 w_ea = w_eb
      승부축  = 홀 열만+절편0  <=> 원 기저에서 w_ea = -w_eb, 절편 0
  이 제약은 선택이 아니라 강제다 — tie 라벨이 swap 불변이므로 nu(swap)=nu 를 요구하면
  홀 열의 선형 가중치는 0 이어야 한다. 승부축은 반대칭이라 짝 열과 절편이 0 이어야 한다.

  **단 "홀 열에 tie 정보가 없다" 는 뜻이 아니다.** 홀의 **짝함수**는 완전히 합법이다:
      |v|, v(.)v, ||v||, v_i v_j    <- 전부 짝
  실제로 TB 자기 헤드가 그걸 쓴다: cat([q, a, b, |a-b|, tanh(a)tanh(b)]).
  ABE 헤드에는 |a-b| 가 없다 (a-b 만 있다). 그래서 |v_abe| 를 tie 축에 직접 주는 팔을 넣는다.
  |v_tb| 는 대조군이다 — TB 헤드에 이미 있으니 잉여로 나와야 앞의 해석이 맞다.

세 팔이 무엇을 가르나
  제약 A       tie=짝만 / 승부=홀만+절편0.  함수공간 등변만, 파라미터 절반
  비제약 U     두 축 모두 전 블록 + 절편.   함수공간 전체, 순서 의존 발생
  U-TTA        비제약을 예측 시 대칭화.     함수공간 등변만, 파라미터 전체
  A ~= U-TTA 면 제약은 공짜다. U > A 면 순서 의존성이 실제로 값을 한다 —
  그때는 패리티 위반량을 보고 채택 여부를 정한다 (제출 시 A/B 순서가 바뀌면 답이 달라진다).

  "원 벡터를 일직선으로" 는 비제약 U 와 **같은 모형**이다 (회전이 가역이므로).
  그래서 원 벡터 경로를 따로 두지 않는다. 홀 블록의 부호를 뒤집으면 그게 swap 된 입력이다.

규율
  · stage2 10,748 을 기존과 **같은 5-fold** 로 교차적합. 파라미터도 lambda 도 fold 밖에서만.
  · lambda 는 바깥 fold 안쪽 3-fold 로 다시 고른다. lambda->inf 면 표현 블록이 죽어
    기준 모형에 수렴하므로 **구조적으로 질 수 없는 비교**다.
  · 로짓 5열은 사실상 무벌점 (사전 스케일). 표현 블록에만 벌점이 걸린다.
  · holdout 5,748 슬라이스를 따로 찍는다 — stage2 valid 5,000 은 두 인코더의 체크포인트
    선택 집합이라 표현이 낙관적이다. 기준선과 처리군이 같은 행을 쓰므로 증분은 대체로
    공정하지만 확인은 해야 한다.
  · 판정은 비대칭이다. **양성이면 확정**(8,600행에서 보였으면 46,729행에선 더 보인다).
    음성이면 검정력 부족일 수 있어 확정이 아니다.

선행
  python Boost/extract_logits.py --pooled --only abe,tb3 --steps abe:3000,tb3:4000

사용
  python Boost/reprhead.py                    # core 팔 (권장, 15~30분)
  python Boost/reprhead.py --arms all         # 전 팔 (비제약 대형 포함, 1시간+)
  python Boost/reprhead.py --arms uncon       # 비제약 팔만
  python Boost/reprhead.py --lams 0.3,3,30    # lambda 격자 축소로 시간 단축
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

LLM = Path('.')
BOOST, ARITH = LLM / 'Boost', LLM / 'Arith'
POOL = BOOST / 'pooled'
SEED, K, INNER = 42, 5, 3
EPS = 1e-15
MAXIT = 1000              # 람다 경로 웜스타트가 있어 실제로는 훨씬 적게 쓴다
ARITH_MODELS = ('lgbm', 'xgb', 'cat')
DUP_DROP = ['d_kw_coverage', 'ad_kw_coverage']    # 전수조사에서 잡은 완전중복 (audit_features)

# 팔 이름 — 표에 그대로 찍힌다. 코드(한글 설명) 형식이고, 코드는 대화에서 짚기 위해 남긴다.
# 다른 절에서 참조하는 것만 상수로 뺀다 (문자열이 어긋나면 조용히 KeyError 가 난다).
NM_W0 = 'W0(로짓만 — ABE·Arith 의 승부로짓 z_A−z_B, 2열)'
NM_W1 = 'W1(TB 표현 홀 성분 A벡터−B벡터 추가) ← 최대 가설'
NM_W5 = 'W5(로짓 + 표현 홀 전부)'
NM_W6 = 'W6(표현 홀만 헤드 입력으로 사용 — 로짓 제외)'
NM_T0 = 'T0(로짓만 — TB·ABE·Arith 의 타이로짓, 3열)'
NM_T5 = 'T5(|ABE 홀| 절댓값 추가 — ABE 헤드에 없는 항)'
NM_T6 = 'T6(|TB 홀| 절댓값 추가 — TB 헤드엔 이미 있음, 대조군)'
NM_T8 = 'T8(로짓 + 표현 짝 전부 + 짝함수)'
NM_T9 = 'T9(표현 짝만 헤드 입력으로 사용 — 로짓 제외)'


# ══════════════════════════════════════════════════════ 출력 유틸 (한글 폭 보정)
def _w(s):
    return sum(2 if ord(c) > 0x1100 and not 0x2000 <= ord(c) <= 0x206F else 1 for c in str(s))


def _pad(s, n, right=False):
    s = str(s)
    g = ' ' * max(n - _w(s), 0)
    return g + s if right else s + g


def table(headers, rows, align=None):
    if not rows:
        return
    align = align or ['l'] * len(headers)
    cols = [max(_w(h), *(_w(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print('  ' + '  '.join(_pad(h, cols[i], align[i] == 'r') for i, h in enumerate(headers)))
    print('  ' + '-' * (sum(cols) + 2 * (len(cols) - 1)))
    for r in rows:
        print('  ' + '  '.join(_pad(v, cols[i], align[i] == 'r') for i, v in enumerate(r)))


def head(tag, text):
    print('\n' + '=' * 104)
    print(f'[{tag}] {text}')
    print('=' * 104)


def ci_str(m, lo, hi):
    return f'{m:+.4f} [{lo:+.4f},{hi:+.4f}] ' + ('유의 ★' if lo > 0 or hi < 0 else '노이즈')


def paired_ci(a, b, n_boot=2000, seed=SEED):
    """a - b 의 부트스트랩 95% CI. 손실이므로 양수면 b 가 낫다."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def rowloss(z, y01):
    return np.logaddexp(0, z) - y01 * z


def ent(p):
    p = float(np.clip(p, EPS, 1 - EPS))
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


# ══════════════════════════════════════════════════════ 블록 적재
def parity_split_arith(feat, pos, cat):
    """253 피쳐를 카탈로그 swap 선언대로 짝/홀로 정확히 가른다. 무손실이다.
        불변      -> 짝
        부호반전  -> 홀
        교환:X    -> (a+b)/2 짝, (a-b)/2 홀   (쌍을 한 번만 소비한다)
        역수      -> log v   (역수는 log 를 취하면 부호반전이 된다) -> 홀
    """
    swap = cat.set_index('feature')['swap']
    use = cat[cat['사용'] == 1]['feature'] if '사용' in cat.columns else cat['feature']
    cols = [c for c in use if c in feat.columns and c not in DUP_DROP]
    ev, evn, od, odn, done = [], [], [], [], set()
    for c in cols:
        if c in done:
            continue
        r = swap.get(c)
        v = feat.iloc[pos][c].to_numpy(np.float64)
        if r == '불변':
            ev.append(v); evn.append(c); done.add(c)
        elif r == '부호반전':
            od.append(v); odn.append(c); done.add(c)
        elif r == '역수':
            od.append(np.log(np.clip(v, 1e-12, None))); odn.append(f'log {c}'); done.add(c)
        elif isinstance(r, str) and r.startswith('교환:'):
            p = r.split(':', 1)[1]
            if p not in cols:
                continue
            w = feat.iloc[pos][p].to_numpy(np.float64)
            ev.append((v + w) / 2); evn.append(f'({c}+{p})/2')
            od.append((v - w) / 2); odn.append(f'({c}-{p})/2')
            done.update((c, p))
    E = np.column_stack(ev) if ev else np.zeros((len(pos), 0))
    O = np.column_stack(od) if od else np.zeros((len(pos), 0))
    return E, evn, O, odn


def load_blocks(abe_step, tb_step):
    z = np.load(BOOST / 'boost_oof.npz')
    ids, y = z['ids'], z['y']
    n = len(ids)

    pa = POOL / f'abe__checkpoint-{abe_step}.npz'
    pt = POOL / f'tb3__checkpoint-{tb_step}.npz'
    for p in (pa, pt):
        assert p.exists(), f'{p} 없음 — extract_logits.py --pooled 를 먼저 실행할 것'
    za, zt = np.load(pa), np.load(pt)
    assert (np.concatenate([za['valid_ids'], za['hold_ids']]) == ids).all(), 'ABE 풀링 id 불일치'
    assert (np.concatenate([zt['valid_ids'], zt['hold_ids']]) == ids).all(), 'TB 풀링 id 불일치'
    cat_ = lambda w, k: np.vstack([w[f'valid_{k}'], w[f'hold_{k}']]).astype(np.float64)

    ea, eb = cat_(za, 'ea'), cat_(za, 'eb')
    q_f, a_f, b_f = cat_(zt, 'q_f'), cat_(zt, 'a_f'), cat_(zt, 'b_f')
    q_s, a_s, b_s = cat_(zt, 'q_s'), cat_(zt, 'a_s'), cat_(zt, 'b_s')

    # swap 은 fwd <-> swp 통째 교환이다: q_f<->q_s, a_f<->a_s, b_f<->b_s.
    # 의미 단위는 구간이 아니라 응답이다 — swp 뷰의 a 구간에는 B 가 들어 있다.
    A_rep, B_rep = (a_f + b_s) / 2, (b_f + a_s) / 2          # 두 뷰에서 본 A / B (A<->B 로 교환)
    D_a, D_b = (a_f - b_s) / 2, (b_f - a_s) / 2              # 뷰 불일치 (D_a <-> -D_b)

    EV = {                                                    # 짝 블록
        'abe_u': (ea + eb) / 2,
        'tb_u': (A_rep + B_rep) / 2,
        'tbq_u': (q_f + q_s) / 2,
        'tbd_u': (D_a - D_b) / 2,
    }
    OD = {                                                    # 홀 블록
        'abe_v': (ea - eb) / 2,
        'tb_v': (A_rep - B_rep) / 2,
        'tbq_v': (q_f - q_s) / 2,
        'tbd_v': (D_a + D_b) / 2,
    }

    # 로짓 (기준선). 기존 결합기가 실제로 받는 5 스칼라.
    meta = json.loads((BOOST / 'model_ens/boost_meta.json').read_text(encoding='utf-8'))
    ex = meta['experts']

    def stack(tag, steps):
        acc = []
        for s in steps:
            w = np.load(BOOST / f'logits/{tag}__checkpoint-{s}.npz')
            acc.append(np.vstack([(w['valid_fwd'] + w['valid_swp']) / 2,
                                  (w['hold_fwd'] + w['hold_swp']) / 2]).astype(np.float64))
        return np.mean(acc, axis=0)

    L_ABE = stack('abe', ex['abe'].get('steps') or [ex['abe']['step']])
    L_TB = stack('tb3', ex['tb'].get('steps') or [ex['tb']['step']])
    za2 = np.load(ARITH / 'outputs/arith_experts_raw.npz')
    L_ARI = np.mean([np.vstack([za2[f'raw_valid__{m}'], za2[f'raw_hold__{m}']]).astype(np.float64)
                     for m in ARITH_MODELS], axis=0)
    sd = lambda B: (B[:, 0] - B[:, 1], B[:, 2] - np.logaddexp(B[:, 0], B[:, 1]))
    m3, t3 = sd(L_ABE)
    m_ar, t_ar = sd(L_ARI)
    s_tie = L_TB[:, 1] - L_TB[:, 0]
    EV['logit'] = np.column_stack([s_tie, t3, t_ar])
    OD['logit'] = np.column_stack([m3, m_ar])

    # Arith 원 피쳐
    cat = pd.read_csv(ARITH / 'feature_catalog.csv')
    feat = pd.read_parquet(ARITH / 'features_57k.parquet')
    pos = pd.Index(feat['id']).get_indexer(ids)
    assert (pos >= 0).all(), 'features_57k 에 stage2 id 가 없다'
    E, en, O, on = parity_split_arith(feat, pos, cat)
    EV['ari'], OD['ari'] = E, O

    # 홀의 짝함수 — tie 축 합법. TB 헤드의 |a-b| 와 같은 구성이다.
    EV['abs_abe'] = np.abs(OD['abe_v'])
    EV['abs_tb'] = np.abs(OD['tb_v'])
    EV['norm'] = np.column_stack([np.linalg.norm(OD[k], axis=1)
                                  for k in ('abe_v', 'tb_v', 'tbq_v', 'tbd_v')])

    n_valid = len(za['valid_ids'])
    is_hold = np.zeros(n, bool)
    is_hold[n_valid:] = True
    return dict(ids=ids, y=y, EV=EV, OD=OD, is_hold=is_hold,
                n_ari_even=len(en), n_ari_odd=len(on))


# ══════════════════════════════════════════════════════ 적합
def _fit(X, y01, pen, bias, w0=None):
    """목적함수 = mean(logistic loss) + (1/n)·Σ pen_j·w_j².

    **열별 벌점 벡터**를 직접 쓴다. 앞서 로짓 열만 ×50 으로 키워 무벌점을 흉내 냈는데,
    그건 블록 간 조건수를 50 배로 벌려서 lbfgs 가 300 회에 못 붙는 원인이었다.
    pen_j = 0 이면 그 열은 조건수를 건드리지 않고 정확히 무벌점이 된다.

    반환: (파라미터 벡터, 수렴 성공 여부). bias 면 마지막 원소가 절편이다.
    """
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
    """학습 fold 통계로 정규화. **홀 열은 중심화하지 않는다.**

    홀 열에서 표본평균을 빼면 w·(x-mu) = w·x - w·mu 가 되어 상수항이 생긴다. 절편을 금지한
    승부축에서 이건 절편을 몰래 넣는 것과 같고, 그 순간 반대칭이 깨진다. 게다가 swap 이
    표준화 공간에서 더 이상 단순 부호반전이 아니게 되어 TTA·위반량 계산도 틀어진다.
    그래서 홀 열은 중심화 없이 RMS 로만 나눈다 (holl 의 이론적 평균은 0 이다).

    열 배치 규약: [0:n_even) 짝, [n_even:) 홀. 로짓 열은 항상 맨 앞이고 그 축의 패리티를 따른다.
    """
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


def _scores(v, B, n_even, bias, want_swap):
    """점수와, 원하면 swap 된 입력의 점수. 선형이므로 행렬을 새로 만들 필요가 없다 —
    swap 은 홀 블록의 부호반전이고 (정규화가 홀을 중심화하지 않으므로 정확하다),
        z(x)      = w_e·x_e + w_o·x_o + b
        z(swap x) = w_e·x_e − w_o·x_o + b
    이라 부분 내적 두 개면 끝난다. 8,000열짜리 사본을 하나 덜 만든다."""
    p = B.shape[1]
    w = v[:p]
    b = float(v[p]) if bias else 0.0
    ze = B[:, :n_even] @ w[:n_even] if n_even else 0.0
    zo = B[:, n_even:] @ w[n_even:] if n_even < p else 0.0
    return (ze + zo + b), ((ze - zo + b) if want_swap else None)


def _penvec(p, n_logit, lam):
    """로짓 열은 무벌점(0), 표현 블록만 lam."""
    pen = np.full(p, float(lam))
    if n_logit:
        pen[:n_logit] = 0.0
    return pen


LAM_CEIL = 1e12           # 자동 확장의 수치 안전판 (무한 루프 방지용이지 탐색 상한이 아니다)


def crossfit(X, y01, fold, bias, lams, rows=None, n_even=0, n_logit=0, want_swap=False,
             blocks=None):
    """바깥 5-fold 교차적합 + 안쪽 3-fold lambda 선택. 점수(로그오즈)를 돌려준다.

    lambda 는 큰 값부터 훑으며 **웜스타트**한다 — 규제가 센 해가 다음 해의 좋은 출발점이라
    반복수가 크게 준다.

    **격자 위쪽 자동 확장(상한 없음)**: 고른 lambda 가 격자 최댓값이면 판정이 성립하지 않는다.
    이 설계의 보장이 "lambda->inf 면 표현 블록이 죽어 기준 모형에 수렴하므로 질 수 없다" 인데,
    최적점이 격자 밖이면 그 극한에 닿은 적이 없는 것이라 "표현이 해롭다" 가 아니라
    "규제를 덜 걸었다" 가 된다. 그래서 최댓값이 뽑히는 한 x10 씩 **계속** 늘린다.
    (실측: 격자를 30 에서 자르니 1,000~6,000 열 팔이 전멸했다. 적정값이 그보다 훨씬 위다.)
    LAM_CEIL 은 무한 루프를 막는 수치 안전판이지 탐색 상한이 아니다.

    blocks 를 주면 (이름, 시작열, 끝열) 목록으로 **블록별 기여**를 같이 낸다:
        z = Σ_b (X_b · w_b) + 절편  이므로  cov(기여_b, z) / var(z) 를 더하면 정확히 1 이다.
        블록끼리 상관이 있어도 성립하는 분해다. 음수면 억제자(다른 블록의 과대추정을 깎는 역할).
    """
    m = np.ones(len(y01), bool) if rows is None else rows
    z = np.full(len(y01), np.nan)
    zs = np.full(len(y01), np.nan) if want_swap else None
    contrib = {b[0]: np.full(len(y01), np.nan) for b in (blocks or [])}
    logit_w = []
    picked, n_bad, n_ext = [], 0, 0
    for k in range(K):
        tr, te = (fold != k) & m, (fold == k) & m
        Xtr, ytr = X[tr], y01[tr]
        if len(lams) > 1:
            inner = np.zeros(len(ytr), int)
            for j, (_, iv) in enumerate(StratifiedKFold(INNER, shuffle=True, random_state=SEED)
                                        .split(np.zeros((len(ytr), 1)), ytr)):
                inner[iv] = j
            Zi = [_z(Xtr[inner != j], Xtr[inner == j], n_even) for j in range(INNER)]
            score = {}

            def eval_lam(lam, warm):
                nb, tot = 0, 0.0
                for j in range(INNER):
                    A, B = Zi[j]
                    v, ok = _fit(A, ytr[inner != j], _penvec(A.shape[1], n_logit, lam), bias,
                                 warm[j])
                    warm[j] = v
                    nb += (not ok)
                    tot += rowloss(_scores(v, B, n_even, bias, False)[0], ytr[inner == j]).sum()
                score[lam] = tot
                return nb

            warm = [None] * INNER
            for lam in sorted(set(lams), reverse=True):     # 큰 값부터 = 웜스타트에 유리
                n_bad += eval_lam(lam, warm)
            e = 0
            while min(score, key=score.get) == max(score) and max(score) < LAM_CEIL:
                warm = [None] * INNER                       # 새 최댓값은 처음부터 (해가 0 근처)
                n_bad += eval_lam(max(score) * 10, warm)
                e += 1
            n_ext = max(n_ext, e)
            lam = min(score, key=score.get)
        else:
            lam = lams[0]
        picked.append(lam)
        A, B = _z(Xtr, X[te], n_even)
        v, ok = _fit(A, ytr, _penvec(A.shape[1], n_logit, lam), bias)
        n_bad += (not ok)
        zf, zw = _scores(v, B, n_even, bias, want_swap)
        z[te] = zf
        if want_swap:
            zs[te] = zw
        if n_logit:
            logit_w.append(v[:n_logit].copy())
        for nm, s, e_ in (blocks or []):
            contrib[nm][te] = B[:, s:e_] @ v[s:e_]
    return z, zs, picked, n_bad, n_ext, contrib, np.array(logit_w)


def auc_of(v, yb):
    r = pd.Series(v).rank().to_numpy()
    n1 = int(yb.sum())
    if n1 == 0 or n1 == len(yb):
        return float('nan')
    return float((r[yb].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yb) - n1)))


def source_table(contrib, y, m, odd_axis):
    """결과별(A승/B승/tie) 블록 기여 평균 — **헤드가 어디서 신호를 받는가**.

    out-of-fold 기여라 적합에 쓰인 적이 없다. 라벨로 층을 갈라도 누수가 아니고,
    순수한 기술 통계다. 인과가 아니라 '어디서 오나' 만 말한다 —
    블록끼리 상관이 있으면 한 블록이 다른 블록 몫을 대신 지고 있을 수 있다
    (그건 기여 비중 표가 답한다).

    패리티 검산이 공짜로 딸려온다:
        승부 점수는 홀 -> A승 평균과 B승 평균이 **부호만 반대**여야 한다
        tie  점수는 짝 -> A승 평균과 B승 평균이 **같아야** 한다
    """
    g = {'A승': m & (y == 0), 'B승': m & (y == 1), 'tie': m & (y == 2)}
    yb = (y == 0) if odd_axis else (y == 2)
    rows, chk = [], []
    for nm, c in contrib.items():
        mu = {k: (float(np.mean(c[v])) if v.sum() else float('nan')) for k, v in g.items()}
        sep = (mu['A승'] - mu['B승']) if odd_axis else \
              (mu['tie'] - float(np.mean(c[m & (y != 2)])))
        rows.append([nm, f'{mu["A승"]:+.3f}', f'{mu["B승"]:+.3f}',
                     '—' if odd_axis else f'{mu["tie"]:+.3f}',
                     f'{sep:+.3f}', f'{auc_of(c[m], yb[m]):.4f}'])
        chk.append(abs(mu['A승'] + mu['B승']) if odd_axis else abs(mu['A승'] - mu['B승']))
    rows.sort(key=lambda r: -abs(float(r[4])))
    return rows, (max(chk) if chk else 0.0)


def share_table(contrib, z, m):
    """블록별 기여 분해. cov(기여_b, z)/var(z) 는 상관된 블록에서도 합이 정확히 1 이다."""
    zz = z[m]
    vz = float(np.var(zz))
    rows = []
    for nm, c in contrib.items():
        cc = c[m]
        rows.append((nm, float(np.std(cc)), float(np.cov(cc, zz)[0, 1] / vz) if vz > 0 else 0.0))
    return sorted(rows, key=lambda r: -abs(r[2]))


def build(blocks, names, n_logit_first=True):
    """블록 이름 목록 -> 설계행렬. 'logit' 이 있으면 맨 앞에 둔다.
    반환: (X, 로짓 열 수, 순서, [(이름, 시작열, 끝열)]) — 마지막이 기여 분해용 구간이다."""
    order = ([n for n in names if n == 'logit'] + [n for n in names if n != 'logit']) \
        if n_logit_first else list(names)
    mats = [blocks[n] for n in order]
    nl = blocks['logit'].shape[1] if 'logit' in order else 0
    rng, c = [], 0
    for n, mm in zip(order, mats):
        rng.append((n, c, c + mm.shape[1]))
        c += mm.shape[1]
    return np.column_stack(mats).astype(np.float64), nl, order, rng


# ══════════════════════════════════════════════════════ 본체
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--abe-step', type=int, default=3000)
    ap.add_argument('--tb-step', type=int, default=4000)
    ap.add_argument('--arms', default='core', help='core | all | uncon')
    ap.add_argument('--lams', default='1,10,100,1000,10000',
                    help='표현 블록 능형 lambda 격자 (로그 간격). 고른 값이 최댓값이면 '
                         'x10 씩 자동 확장하므로 시작 격자가 좁아도 된다. '
                         '열이 1,000~6,000 개면 적정값이 1e3~1e4 대다')
    args = ap.parse_args()
    LAMS = [float(s) for s in args.lams.split(',') if s.strip()]
    # 파이프(| tee)로 넘기면 stdout 이 블록 버퍼링돼 수십 분간 아무것도 안 보인다.
    sys.stdout.reconfigure(line_buffering=True)
    T_START = time.time()

    print('블록 적재 중 — 풀링 npz 352MB + features 57k parquet ...', flush=True)
    R = load_blocks(args.abe_step, args.tb_step)
    print(f'  적재 완료 {time.time() - T_START:.0f}s')
    y, is_hold, EV, OD = R['y'], R['is_hold'], R['EV'], R['OD']
    N = len(y)
    fold = np.zeros(N, int)
    for k, (_, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=SEED)
                                .split(np.zeros((N, 1)), y)):
        fold[te] = k
    tie = y == 2
    nz = ~tie
    y_tie, y_win = tie.astype(np.float64), (y == 0).astype(np.float64)
    p_tie, p_A = tie.mean(), y_win[nz].mean()

    print(f'stage2 {N:,}행 (valid {int((~is_hold).sum()):,} + holdout {int(is_hold.sum()):,}) | '
          f'ABE ckpt-{args.abe_step} / TB3 ckpt-{args.tb_step} | fold {np.bincount(fold)}')
    print(f'tie율 {p_tie:.4f} (기준선 {ent(p_tie):.4f}) | 비-tie {int(nz.sum()):,}행 '
          f'A승률 {p_A:.4f} (기준선 {ent(p_A):.4f}) | lambda 격자 {LAMS}')
    print(f'슬롯 편향 검정: A승률 {p_A:.4f} vs 0.5, SE {np.sqrt(.25 / nz.sum()):.4f} -> '
          f'{abs(p_A - .5) / np.sqrt(.25 / nz.sum()):.1f}σ '
          f'(유의하지 않으면 패리티 제약이 잃을 게 없다)')

    print("""
[기호 설명] 벡터와 스칼라를 헷갈리지 말 것 — 이 구분이 이번 실험의 전부다.

    (Q, A) --인코더--> ea  (1024 차원 풀링 벡터)  ┐
                                                  ├--헤드 MLP--> (z_A, z_B, z_tie)  스칼라 3 개
    (Q, B) --인코더--> eb  (1024 차원 풀링 벡터)  ┘

  ea, eb  = encode(Q,A), encode(Q,B). Boost/pooled/ 에 뽑아 둔 것이 이것이다.
  z_A     = "**A 가 이긴다**" 는 클래스의 로짓. 숫자 하나다.
            아래첨자 A 는 '응답 A' 가 아니라 '**A 승리 클래스**' 다 — 글자가 겹치니 주의.
            P(A 승) = exp(z_A) / (exp(z_A)+exp(z_B)+exp(z_tie))

  즉 헤드는 2,048 차원을 3 개(실질 자유도 2)로 압축한다. **그 압축이 손해인지가 이 파일의 질문이다.**

[좌표 설명] 표 전체가 이 두 축 위에 있다.
  3-class 로짓 (z_A, z_B, z_tie) 는 softmax 가 상수 이동에 불변이라 **자유도가 2** 다.
  그 2 개를 패리티(A<->B 교환 시 거동)로 잡는다:

      승부로짓 m = z_A − z_B                      swap 시 부호 반전  -> 홀
      타이로짓 t = z_tie − logaddexp(z_A, z_B)    swap 시 불변      -> 짝

      σ(m) = P(A) / (P(A) + P(B))     "A 와 B 중 A 가 이길 확률"
      σ(t) = P(tie)                    "비길 확률"

  무손실 가역이다. 부스터의 'raw margin'(softmax 이전 점수)과는 다른 뜻이니 혼동하지 말 것.
  TB 는 2-class 라 자유도가 1 이고 그걸 전부 타이로짓(s_tie = z_tie − z_notie)에 쓴다 —
  **승부축 출력이 아예 없다.** W1 이 최대 가설인 이유가 그것이다: 출력은 없지만 표현에는 있을 것.
""")
    print('블록 구성 — 짝/홀 분해는 가역 회전이라 원 벡터와 같은 함수 공간이다')
    table(['채널', '블록', '차원', '의미'],
          [['짝', 'abe_u', str(EV['abe_u'].shape[1]), '(ea+eb)/2'],
           ['짝', 'tb_u', str(EV['tb_u'].shape[1]), '(A_rep+B_rep)/2'],
           ['짝', 'tbq_u', str(EV['tbq_u'].shape[1]), '(q_f+q_s)/2  프롬프트'],
           ['짝', 'tbd_u', str(EV['tbd_u'].shape[1]), '(D_a-D_b)/2  뷰 불일치'],
           ['짝', 'ari', str(EV['ari'].shape[1]), 'Arith 짝 피쳐'],
           ['짝', 'abs_abe', str(EV['abs_abe'].shape[1]), '|abe_v|  ABE 헤드에 없는 항'],
           ['짝', 'abs_tb', str(EV['abs_tb'].shape[1]), '|tb_v|   TB 헤드엔 이미 있음(대조군)'],
           ['짝', 'norm', str(EV['norm'].shape[1]), '홀 블록 노름 4개'],
           ['홀', 'abe_v', str(OD['abe_v'].shape[1]), '(ea-eb)/2'],
           ['홀', 'tb_v', str(OD['tb_v'].shape[1]), '(A_rep-B_rep)/2  ← 최대 가설'],
           ['홀', 'tbq_v', str(OD['tbq_v'].shape[1]), '(q_f-q_s)/2'],
           ['홀', 'tbd_v', str(OD['tbd_v'].shape[1]), '(D_a+D_b)/2'],
           ['홀', 'ari', str(OD['ari'].shape[1]), 'Arith 홀 피쳐']], ['l', 'l', 'r', 'l'])

    # ── 팔 정의 ──────────────────────────────────────────────────────────────
    # 이름이 표에 그대로 찍힌다. 코드(한글 설명) 형식 — 코드는 대화에서 짚기 위해 남긴다.
    WIN = [(NM_W0, ['logit']),
           (NM_W1, ['logit', 'tb_v']),
           ('W2(ABE 표현 홀 성분 ea−eb 추가)', ['logit', 'abe_v']),
           ('W3(Arith 홀 피쳐 추가 — 부호반전·차이형)', ['logit', 'ari']),
           ('W4(TB 프롬프트·뷰차 홀 성분 추가)', ['logit', 'tbq_v', 'tbd_v']),
           (NM_W5, ['logit', 'abe_v', 'tb_v', 'tbq_v', 'tbd_v', 'ari']),
           (NM_W6, ['abe_v', 'tb_v', 'tbq_v', 'tbd_v', 'ari'])]
    TIE = [(NM_T0, ['logit']),
           ('T1(ABE 표현 짝 성분 ea+eb 추가)', ['logit', 'abe_u']),
           ('T2(TB 표현 짝 성분 A벡터+B벡터 추가)', ['logit', 'tb_u']),
           ('T3(TB 프롬프트 표현 q_fwd+q_swp 추가)', ['logit', 'tbq_u']),
           ('T4(Arith 짝 피쳐 추가 — 불변·절대차형)', ['logit', 'ari']),
           (NM_T5, ['logit', 'abs_abe']),
           (NM_T6, ['logit', 'abs_tb']),
           ('T7(홀 블록 노름 4개 추가 — 전체 차이 크기)', ['logit', 'norm']),
           (NM_T8, ['logit', 'abe_u', 'tb_u', 'tbq_u', 'tbd_u', 'ari',
                    'abs_abe', 'abs_tb', 'norm']),
           (NM_T9, ['abe_u', 'tb_u', 'tbq_u', 'tbd_u', 'ari'])]
    CORE_W = {NM_W0, NM_W1, 'W2(ABE 표현 홀 성분 ea−eb 추가)', NM_W5, NM_W6}
    CORE_T = {NM_T0, 'T1(ABE 표현 짝 성분 ea+eb 추가)', 'T2(TB 표현 짝 성분 A벡터+B벡터 추가)',
              NM_T5, NM_T6, 'T7(홀 블록 노름 4개 추가 — 전체 차이 크기)', NM_T8, NM_T9}
    if args.arms == 'core':
        WIN = [a for a in WIN if a[0] in CORE_W]
        TIE = [a for a in TIE if a[0] in CORE_T]
    elif args.arms == 'uncon':
        WIN, TIE = WIN[:1], TIE[:1]

    RES, SHARE, CONTRIB = {}, {}, {}

    def run_axis(label, arms, blocks, y01, rows, bias, base_ent, all_even, logit_names):
        # all_even: 이 축의 열이 전부 짝인가(tie축) 전부 홀인가(승부축). 정규화 규약을 가른다.
        # logit_names: 로짓 열 이름 — 계수를 그대로 찍어 "헤드가 무엇을 얼마나 믿나" 를 본다.
        head(label[0], label[1])
        base, rows_out = None, []
        print(f'  팔 {len(arms)}개를 차례로 적합한다. 끝나는 대로 한 줄씩 찍고, 마지막에 표로 다시 낸다.\n')
        for i, (nm, names) in enumerate(arms, 1):
            t0 = time.time()
            X, nl, _, rng = build(blocks, names)
            n_fit = K * (1 + (len(LAMS) * INNER if X.shape[1] > nl else 0))
            print(f'  [{i}/{len(arms)}] {nm}  — {X.shape[1]:,}열, 적합 {n_fit}회 이상 '
                  f'(λ 자동확장분 추가) ...', flush=True)
            z, _, lam, bad, ext, contrib, lw = crossfit(
                X, y01, fold, bias, LAMS if X.shape[1] > nl else [0.0],
                rows=rows, n_even=X.shape[1] if all_even else 0, n_logit=nl, blocks=rng)
            sl = rows if rows is not None else slice(None)
            pr = rowloss(z[sl], y01[sl])
            RES[nm] = (z, pr)
            if base is None:
                base = pr
            hm = (is_hold[rows] if rows is not None else is_hold)
            d = '기준' if pr is base else ci_str(*paired_ci(base, pr))
            hit = max(lam) >= LAM_CEIL          # 수치 안전판까지 갔다 = 병적인 경우
            warn = ('' if not bad else f'미수렴 {bad}') + \
                   ('' if not hit else ('  ' if bad else '') + 'λ한계도달')
            print(f'        손실 {pr.mean():.4f} (holdout {pr[hm].mean():.4f})  {d}  '
                  f'| λ중앙 {np.median(lam):g}{"" if not ext else f" (자동확장 x10 {ext}회)"} '
                  f'| {time.time() - t0:.0f}s{"  ** " + warn if warn else ""}  '
                  f'[누적 {(time.time() - T_START) / 60:.0f}분]')
            # 헤드가 무엇을 보나 — 블록별 기여 비중. 상관된 블록에서도 합이 정확히 1 이다.
            sh = share_table(contrib, z, sl if rows is not None else slice(None))
            print('        기여 비중  ' + ' | '.join(f'{b} {s:+.2f}' for b, _, s in sh))
            if nl and len(lw):
                print('        로짓 계수  ' + '  '.join(
                    f'{c} {lw[:, j].mean():+.3f}±{lw[:, j].std():.3f}'
                    for j, c in enumerate(logit_names)))
            rows_out.append([nm, f'{X.shape[1]:,}', f'{pr.mean():.4f}', f'{pr[hm].mean():.4f}',
                             d, f'{np.median(lam):g}', f'{time.time() - t0:.0f}s', warn])
            SHARE[nm] = sh
            CONTRIB[nm] = (contrib, np.ones(len(y), bool) if rows is None else rows, not all_even)
        print()
        table(['구성 (기준 위에 무엇을 더했나)', '열', '손실(전체)', '손실(holdout)',
               'Δ vs 기준(양수=이득)', 'λ중앙', '시간', '경고'],
              rows_out, ['l', 'r', 'r', 'r', 'l', 'r', 'r', 'l'])
        print(f'  아무 신호도 안 쓴 상수 모형 {base_ent:.4f} — 거기서 얼마나 내려왔나가 정보량이다.')
        sat = [r[0] for r in rows_out if 'λ한계도달' in r[7]]
        if sat:
            print(f'  ! λ 가 수치 안전판 {LAM_CEIL:g} 까지 간 팔: {", ".join(sat)}')
            print('    표현 블록이 그 정도로 눌려야 했다는 뜻이라 사실상 전부 잉여였다는 신호다.')
        print('  기여 비중: cov(블록 기여, 점수)/var(점수). 블록끼리 상관이 있어도 합이 정확히 1 이다.')
        print('  음수면 억제자 — 다른 블록의 과대추정을 깎는 역할이라 단독으로는 정보가 아니다.')
        return base

    run_axis(('1', '승부축 (A 가 이겼나) — 홀 채널만. 절편 없음, 중심화 없음'
                   '(둘 다 상수항이라 반대칭을 깬다). 비-tie 행만'),
             WIN, OD, y_win, nz, False, ent(p_A), all_even=False,
             logit_names=['m3(ABE)', 'm_ar(Arith)'])
    run_axis(('2', 'tie축 (비겼나) — 짝 채널만. 홀 성분은 짝함수(절댓값·노름)로만 들어온다'),
             TIE, EV, y_tie, None, True, ent(p_tie), all_even=True,
             logit_names=['s_tie(TB)', 't3(ABE)', 't_ar(Arith)'])

    # ── 어디서 신호가 오나 ───────────────────────────────────────────────────
    head('3', '어디서 신호가 오나 — 결과별(A승/B승/tie) 블록 기여 평균')
    print('  out-of-fold 기여라 적합에 쓰인 적이 없다. 라벨로 층을 갈라도 누수가 아니고 기술 통계다.')
    print('  "어디서 오나" 만 말한다 — 블록끼리 상관이 있으면 한 블록이 다른 블록 몫을 대신')
    print('  지고 있을 수 있고, 그건 위의 기여 비중 표가 답한다.')
    print('  패리티 검산이 공짜로 딸려온다: 승부 점수는 홀이라 A승·B승 평균이 부호만 반대여야 하고,')
    print('  tie 점수는 짝이라 A승·B승 평균이 같아야 한다. 어긋나면 배선이 잘못된 것이다.')
    for axis_nm, base_nm, full_nm, odd in (('승부축', NM_W0, NM_W5, True),
                                           ('tie축', NM_T0, NM_T8, False)):
        pool = [k for k in CONTRIB if CONTRIB[k][2] == odd]
        if not pool:
            continue
        best = min(pool, key=lambda k: RES[k][1].mean())
        show = [k for k in dict.fromkeys([base_nm, best, full_nm]) if k in CONTRIB]
        for k in show:
            contrib, m, _ = CONTRIB[k]
            rows, chk = source_table(contrib, y, m, odd)
            print(f'\n  ── {axis_nm}  {k}')
            table(['블록', 'A승 평균', 'B승 평균', 'tie 평균',
                   'A승−B승' if odd else 'tie−비tie', '블록 단독 AUC'],
                  rows, ['l', 'r', 'r', 'r', 'r', 'r'])
            lab = 'max|A승+B승| (홀이면 0)' if odd else 'max|A승−B승| (짝이면 0)'
            print(f'    패리티 검산 {lab} = {chk:.4f}')

    # ── 비제약 팔 ────────────────────────────────────────────────────────────
    if args.arms in ('all', 'uncon'):
        head('4', '비제약 — 두 축 모두 전 블록 + 절편. "원 벡터 일직선" 과 같은 모형이다')
        print('  회전이 가역이므로 원 벡터를 그냥 넣는 것과 짝·홀 블록 전량을 제약 없이 넣는 것은')
        print('  같은 함수 공간이다. 차이는 열이 아니라 제약뿐이라 여기서 그것만 푼다.')
        print('  swap 된 입력 = 홀 블록의 부호 반전. 그걸로 TTA 와 패리티 위반량을 함께 낸다.\n')
        ALL_E = ['logit', 'abe_u', 'tb_u', 'tbq_u', 'tbd_u', 'ari']
        ALL_O = ['abe_v', 'tb_v', 'tbq_v', 'tbd_v', 'ari']
        Xe, nl, _, _ = build(EV, ALL_E)
        Xo, _, _, _ = build(OD, ALL_O, n_logit_first=False)
        # 승부축 로짓(m3, m_ar)은 홀이라 짝 블록에 못 넣는다. 두 축 공용 행렬을 쓰되
        # 무벌점 대상은 짝 로짓 3열뿐이고, 홀 로짓은 벌점을 받는다 — 기준선보다 불리한 쪽이라
        # 비제약 팔에 유리하게 기울지 않는다.
        X = np.hstack([Xe, Xo, OD['logit']])
        n_even = Xe.shape[1]
        print(f'  설계행렬 {X.shape[0]:,} x {X.shape[1]:,} '
              f'(짝 {n_even:,} + 홀 {X.shape[1] - n_even:,}) | float64 '
              f'{X.nbytes / 1e9:.2f} GB, fold 사본 포함 peak 약 {X.nbytes * 3 / 1e9:.1f} GB\n')

        out = []
        for axis, y01, rows, base_nm, con_nm in (
                ('승부축', y_win, nz, NM_W0, NM_W5),
                ('tie축', y_tie, None, NM_T0, NM_T8)):
            t0 = time.time()
            z, zs, lam, bad, ext, _, _ = crossfit(X, y01, fold, True, LAMS, rows=rows,
                                                  n_even=n_even, n_logit=nl, want_swap=True)
            sl = rows if rows is not None else slice(None)
            odd_axis = axis == '승부축'
            # TTA = 반대칭화(승부) / 대칭화(tie). 제약과 **같은 함수**를 사후로 만든 것.
            z_tta = (z - zs) / 2 if odd_axis else (z + zs) / 2
            v = (z + zs) if odd_axis else (z - zs)
            viol = float(np.abs(v[sl]).max())
            pr_u, pr_t = rowloss(z[sl], y01[sl]), rowloss(z_tta[sl], y01[sl])
            ref = RES[base_nm][1]
            con = RES[con_nm][1] if con_nm in RES else None
            for nm, pr in (('비제약 U', pr_u), ('비제약 + TTA', pr_t)):
                out.append([axis, nm, f'{pr.mean():.4f}', ci_str(*paired_ci(ref, pr)),
                            '—' if con is None else ci_str(*paired_ci(con, pr)),
                            f'{viol:.3f}' if nm == '비제약 U' else '0 (구조)',
                            f'{np.median(lam):g}', f'{time.time() - t0:.0f}s',
                            '' if not bad else f'미수렴 {bad}'])
        table(['축', '구성', '손실', 'Δ vs 로짓기준', 'Δ vs 제약 전량', '패리티 위반 max|·|',
               'λ중앙', '시간', '경고'], out, ['l', 'l', 'r', 'l', 'l', 'r', 'r', 'r', 'l'])
        print('  패리티 위반 = 승부축 max|d(x)+d(swap x)|, tie축 max|ν(x)−ν(swap x)|.')
        print('  0 이어야 응답 순서를 바꿔도 답이 같다. 비제약 U 가 이겨도 이 값이 크면')
        print('  채택할 수 없다 — 정확도가 아니라 제출 정합성 문제다.')

    # ── 종합 ─────────────────────────────────────────────────────────────────
    head('5', '종합 — 축 점수를 3-class 로 합쳐 총 ll 로 본다')
    print('  logits = [d/2, −d/2, ν + logaddexp(d/2,−d/2)] 이면 p_tie = σ(ν), '
          'p_A/(p_A+p_B) = σ(d) 가 정확히 성립한다 (축 분리).')
    print('  따라서 총 ll = tie축 + (1−tie율)·승부축 이고, 두 축을 따로 고른 최선을 합치면 된다.\n')
    bw = min((k for k in RES if k.startswith('W')), key=lambda k: RES[k][1].mean())
    bt = min((k for k in RES if k.startswith('T')), key=lambda k: RES[k][1].mean())
    w0, t0_ = RES.get(NM_W0), RES.get(NM_T0)
    rows = []
    for nm, wk, tk in (('현행 로짓 기준', NM_W0, NM_T0), ('표현 최선 조합', bw, bt)):
        if wk not in RES or tk not in RES:
            continue
        tot = RES[tk][1].mean() + (1 - p_tie) * RES[wk][1].mean()
        rows.append([nm, f'{RES[tk][1].mean():.4f}', f'{RES[wk][1].mean():.4f}', f'{tot:.4f}',
                     wk.split('(')[0] + ' / ' + tk.split('(')[0]])
    table(['구성', 'tie축', '승부축(비-tie 행당)', '총 ll', '고른 팔'],
          rows, ['l', 'r', 'r', 'r', 'l'])
    if w0 and t0_:
        base_tot = t0_[1].mean() + (1 - p_tie) * w0[1].mean()
        best_tot = RES[bt][1].mean() + (1 - p_tie) * RES[bw][1].mean()
        print(f'\n  총 ll {base_tot:.4f} -> {best_tot:.4f}  (Davidson κ=1 의 0.9705 와 대조)')
        print('  * 두 축의 최선을 사후로 고른 값이라 낙관적이다 — 채택 판정은 각 축의 Δ CI 로 한다.')
    print('\n  판정은 비대칭이다: 양성이면 확정(8,600행에서 보였으면 46,729행에선 더 보인다),')
    print('  음성이면 검정력 부족일 수 있어 확정이 아니다.')


if __name__ == '__main__':
    main()
