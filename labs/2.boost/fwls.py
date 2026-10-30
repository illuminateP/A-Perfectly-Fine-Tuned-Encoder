"""fwls.py — FWLS(Feature-Weighted Linear Stacking): Arith 피쳐가 전문가 **가중치**를 국면별로 바꾸나.

모형 (Sill et al. 2009, Netflix Prize)
  보통 스태킹  p̂ = σ(Σᵢ wᵢ·gᵢ)                 wᵢ 상수
  FWLS        wᵢ(x) = wᵢ + Σⱼ vᵢⱼ·Uⱼ(x)   ⇒   p̂ = σ(Σᵢ wᵢgᵢ + Σᵢⱼ vᵢⱼ·Uⱼ⊙gᵢ)
  메타피쳐×로짓 곱을 열로 갖는 **선형모형**이라 reprhead 의 릿지 기계(기저 로짓 pen=0 +
  상호작용만 λ, 안쪽 3-fold λ 선택, 상한 없는 자동확장)를 그대로 쓴다.
  λ→∞ 면 정확히 기준 모형을 회복하므로 음성 = "게이트 신호 없음" 으로 읽힌다.

패리티 배선 (전부 구조로 강제)
  짝피쳐 U × 승부로짓 m  = 홀  -> 승부축 ✓   (U 주효과는 반대칭 위반이라 금지)
  짝피쳐 U × 타이로짓 t  = 짝  -> tie축 ✓
  홀피쳐 V × 승부로짓 m  = 짝  -> tie축 ✓   ("피쳐가 가리키는 승자와 전문가 승자의 합의" 채널)
  주의: 짝 U 는 rankgauss(순위→정규분위) — swap 불변이라 정확성 유지.
        홀 V 는 sign·log1p(|v|/s) — 홀 변환은 기함수여야 반대칭이 깨지지 않는다 (rankgauss 금지).

재료 계약 (2026-08-05 변경) — 기본은 **전문가당 단일 최적 체크포인트, 온도 피팅 전무**
  사전 평균(ABE 7ckpt·Arith 부스터 3종)이 상한을 깎는지 보기 위해 기본 계약을 단일로 바꿨다.
  선택은 valid 5,000 반쪽의 **raw ll** — 파이프라인 어디에도 온도 피팅이 없다 (재료 탐색
  중엔 "부스터가 못 줄인 것"과 "보정이 흡수한 것"의 귀속이 섞이면 안 된다는 지시).
  --ens 를 주면 기존 앙상블 계약으로 복귀한다. **단일 계약에서는 0.5739/0.5740/0.9705 앵커와
  (참조) repr/blend 행이 직접 비교되지 않는다** — 그것들은 앙상블 계약 수치다.

훈련 데이터 — stage2 10,748 만 쓴다 (train 46,729 는 안 쓴다)
  FWLS 의 학습 신호는 "어느 국면에서 어느 전문가가 맞나" 인데 train 로짓은 in-sample 이라
  그 '맞음' 자체가 암기로 부풀려져 있다 (newhead 실측 갭 +0.07~0.16). 인코더별 암기량이
  달라 상대 신뢰도가 계통적으로 비틀리므로, 게이트 학습에는 out-of-sample 로짓만 자격이 있다.

이미 아는 것 (이 실험의 사전 확률)
  · Davidson 3-피쳐 게이트(m_ar 만 변조) = FWLS 최소판: −0.0002 노이즈
  · 짝피쳐 주효과: z 판 −0.0022 유의 악화, rankgauss 판 노이즈 -> 주효과는 전 팔에서 제외
  · 피쳐의 신호 내용은 이미 GBDT 3종이 m_ar/t_ar 로 증류 — 남은 것은 신뢰도 지표뿐
  · blend: tie축만 행 의존 결합 선호(γ=0.81), 승부축은 ABE 독주(γ=0.03) -> 희망은 tie축

팔 (축별 독립. F0 이 기준, Δ 는 전부 F0 대비)
  F0   로짓만 — W0/T0 재현 (0.5739 / 0.5740 이 나와야 하네스가 맞다)
  F1   선별 짝피쳐 14종 × 전 로짓 게이트 (국면 기술자 — Netflix 정신)
  F5   버려진 공통성분 게이트: c_tb=(z_tie+z_notie)/2, c_abe=mean(z) — CE 기울기가 정확히 0 인
       방향이라 init 동결 무작위 사영. 실측: std 1/22, tie AUC 0.46, 길이 corr ±0.41
       (tie축 F5a 는 주효과 2열 포함 — 로짓 파생이라 '피쳐 주효과 기각'에 안 덮인다)
  F5b  (tie축) 로짓 합의항 [m3⊙m_ar, |m3|, |m_ar|, |m3−m_ar|] — Davidson κ 의 일반화
  F4   (tie축) 홀피쳐 67 × 승부로짓 = 합의 채널 134열
  F2   가족별 게이트 13팔 (유형별 국소화, Bonferroni 주의)
  F3   짝피쳐 전부 × 전 로짓 kitchen-sink (승부 374열 / tie 561열) — 참조팔

검증 — Davidson/reprhead/blend 와 같은 5-fold(StratifiedKFold seed 42) 교차적합.
  지표: OOF 풀링 손실(전체·holdout), 승부축은 비-tie 행당 — log_repr.txt 와 같은 단위.
  paired bootstrap 95% CI, 채택 문턱 0.0005. 팔이 ~34개라 가족별 판정은 Bonferroni 로 본다.

재료 팔 ([0]) — 새 전문가를 기저에 추가한다 (게이트 없음, 로짓 전열 pen=0 = 무규제 적합)
  TB1   ModernBERT cross-encoder 2-class (V6/TB1_old ckpt-5000). stage2 전량 무접촉.
  ABE1  V3 DeBERTa bi-encoder 3-class (ckpt-3109). 승부·tie 양축 출력을 다 갖는 두 번째
        신경망이다 — 현행 승부축의 신경망 재료는 ABE2 하나뿐이었다(TB3·TB1 은 2-class).
        자산 우선순위: abe1__ckpt(전 행 재추출) > tt__ckpt(clean 7,748행 변환본, 자동 강등).

  ABE1 행 계층 — **stage2 10,748 전 행을 재료로 쓴다** (사용자 결정 2026-08-05):
      holdout 5,748  V3 test set = 현행 holdout. ABE1·프로토콜 모두 무접촉 (가장 깨끗한 읽기)
      valid  2,000   V3 valid. 체크포인트 선택 노출 (ABE2/TB3 의 stage2-valid 노출과 같은 종류)
      valid  3,000   V3 train. **in-sample** — 다른 전문가에겐 out-of-sample 인 행이다
  전 행을 쓰면 3,000행에서 ABE1 만 유리해지므로, 각 ABE1 팔에 **계층별 Δ** 를 함께 낸다.
  이득이 in-sample 계층에 몰리면 암기이고 holdout 에서도 같이 나오면 실신호다 — 판정은 사용자가.

사용 (전체 약 30~60분. 빠른 검산: --only F0 / 재료만: --only R)
  python Boost/fwls.py 2>&1 | tee Boost/log_fwls.txt
  python Boost/fwls.py --only F0            # 앵커 재현만 (수 초)
  python Boost/fwls.py --only R             # 재료 팔만 (수 분)
  python Boost/fwls.py --no-families        # F2 26팔 생략
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri
from sklearn.model_selection import StratifiedKFold

import reprhead as RH        # crossfit·_z·_fit·parity_split_arith·paired_ci·ci_str·rowloss·table 재사용

sys.stdout.reconfigure(line_buffering=True)

LLM = Path('.')
BOOST, ARITH = LLM / 'Boost', LLM / 'Arith'
SEED, K = RH.SEED, RH.K
HALF = 5000
THRESH = 0.0005
LAMS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
ARITH_MODELS = ('lgbm', 'xgb', 'cat')
TB1_STEP = 5000            # V6/TB1_old val_state 선택 ckpt (stage2 무접촉 선택)
ABE1_STEP = 3109           # V3 epoch-best 단일 ckpt (abe1 = 전 행 재추출, tt = clean 변환본)

# F1 선별 — 국면 기술자 성격의 짝피쳐 (존재를 로드 시 assert)
F1_SEL = ['mn_char_len', 'ad_char_len', 'prompt_char_len',            # 길이·프롬프트 규모
          'cos_word', 'bs_f1', 'emb_cos',                             # A/B 유사도 (합의 국면)
          'emb_q_min', 'emb_q_ad', 'bs_q_cov_min',                    # 질문 적합도 (둘 다 헛다리 국면)
          'abs_diff_keyword_coverage',
          'mn_code_block_count', 'mx_code_block_count',               # 코드 국면
          'mx_refusal', 'mx_non_ascii_ratio']                         # 거부·비영어 국면

# log_repr.txt / log_blend.txt — 같은 fold·같은 단위의 참조치
REPR_WIN = {'W0(로짓만)': (0.5739, 0.5662), 'W1(TB 표현 홀, 최선)': (0.5730, 0.5650)}
REPR_TIE = {'T0(로짓만)': (0.5740, 0.5734), 'T2(TB 표현 짝, 최선)': (0.5716, 0.5713)}
REF_TOTAL = {'로짓 기준(W0/T0=Davidson κ=1)': 0.9705, 'repr 표현 최선(W1/T2)': 0.9675,
             'blend 최선(C3)': 0.9703}


# ══════════════════════════════════════════════════════ 변환
def rankgauss(v):
    """순위 -> 정규분위. 짝피쳐 전용 — swap 불변값이라 임의 단조변환이 정확성을 안 깨진다."""
    r = pd.Series(v).rank(method='average').to_numpy()
    return ndtri((r - 0.5) / len(r))


def signed_log(v):
    """홀피쳐 전용 기함수 변환 sign·log1p(|v|/s). 홀에 rankgauss/중심화를 쓰면
    swap 이 부호반전으로 안 닫혀 반대칭이 깨진다 — 기함수만 허용."""
    s = np.clip(np.median(np.abs(v)), 1e-9, None)
    return np.sign(v) * np.log1p(np.abs(v) / s)


# ══════════════════════════════════════════════════════ 적재
def _raw_ll(z, y_idx):
    """무보정 valid CE — **온도 피팅 없음** (사용자 지시 2026-08-05: 선택 단계에 피팅이
    끼면 '부스터가 못 줄인 것'과 '보정이 흡수한 것'의 귀속이 섞인다). 로짓도 raw 그대로."""
    zs = z - z.max(1, keepdims=True)
    p = np.exp(zs)
    p /= p.sum(1, keepdims=True)
    return float(-np.log(np.clip(p[np.arange(len(y_idx)), y_idx], 1e-15, 1)).mean())


def load_all(single_best=True):
    """single_best=True(기본): 전문가마다 **최적 체크포인트 1개**만 쓴다 — 사전 평균(ABE 7ckpt,
    Arith 3종)이 상한을 깎는지 보기 위한 계약 (사용자 지시 2026-08-05). 선택은 valid 5,000 의
    **raw ll — 온도 피팅 전무** (§1.6: 선택은 valid, holdout 무접촉). --ens 로 앙상블 복귀."""
    z = np.load(BOOST / 'boost_oof.npz')
    ids, y = z['ids'], z['y']
    meta = json.loads((BOOST / 'model_ens/boost_meta.json').read_text(encoding='utf-8'))
    ex = meta['experts']

    def load_one(tag, s):
        w = np.load(BOOST / f'logits/{tag}__checkpoint-{s}.npz')
        assert (np.concatenate([w['valid_ids'], w['hold_ids']]) == ids).all(), f'{tag} id 불일치'
        assert (np.concatenate([w['valid_labels'], w['hold_labels']]) == y).all(), f'{tag} 라벨 불일치'
        return np.vstack([(w['valid_fwd'] + w['valid_swp']) / 2,
                          (w['hold_fwd'] + w['hold_swp']) / 2]).astype(np.float64)

    def stack(tag, steps):
        return np.mean([load_one(tag, s) for s in steps], axis=0)

    def pick_best(tag, steps, n_class):
        """valid 반쪽의 **raw ll** 로 고른다 — 온도 피팅 없음, 로짓도 raw 그대로."""
        rows, best = [], None
        for s in steps:
            L = load_one(tag, s)
            yv = y[:HALF] if n_class == 3 else (y[:HALF] == 2).astype(np.int64)
            ll = _raw_ll(L[:HALF], yv)
            rows.append((s, ll))
            if best is None or ll < best[1]:
                best = (s, ll, L)
        print(f'  [{tag}] valid raw ll: ' + '  '.join(f'{s}:{r:.4f}' for s, r in rows)
              + f'  -> 선택 {best[0]}')
        return best

    za = np.load(ARITH / 'outputs/arith_experts_raw.npz')
    assert (np.concatenate([za['ids_valid'], za['ids_hold']]) == ids).all(), 'Arith id 불일치'
    ari_raw = {m: np.vstack([za[f'raw_valid__{m}'], za[f'raw_hold__{m}']]).astype(np.float64)
               for m in ARITH_MODELS}

    abe_steps = ex['abe'].get('steps') or [ex['abe']['step']]
    tb_steps = ex['tb'].get('steps') or [ex['tb']['step']]
    if single_best:
        import glob as _g
        all_of = lambda t: sorted(int(p.rsplit('-', 1)[1][:-4])
                                  for p in _g.glob(str(BOOST / f'logits/{t}__checkpoint-*.npz'))
                                  if '__train' not in p)
        sa, lla, L_ABE = pick_best('abe', all_of('abe'), 3)
        st, llt, L_TB = pick_best('tb3', all_of('tb3'), 2)
        ari_ll = {m: _raw_ll(ari_raw[m][:HALF], y[:HALF]) for m in ARITH_MODELS}
        ma = min(ari_ll, key=ari_ll.get)
        print('  [arith] valid raw ll: '
              + '  '.join(f'{m}:{ari_ll[m]:.4f}' for m in ARITH_MODELS) + f'  -> 선택 {ma}')
        L_ARI = ari_raw[ma]
        contract = (f'단일 최적 ckpt·raw 선택(온도 피팅 없음) — ABE2 {sa}(vll {lla:.4f}) | '
                    f'TB3 {st}(vll {llt:.4f}) | Arith {ma}(vll {ari_ll[ma]:.4f})')
    else:
        L_ABE = stack('abe', abe_steps)
        L_TB = stack('tb3', tb_steps)
        L_ARI = np.mean(list(ari_raw.values()), axis=0)
        contract = f'앙상블 — ABE2 {len(abe_steps)}ckpt 평균 {abe_steps} | TB3 {tb_steps} | Arith 3종 평균'

    sd = lambda B: (B[:, 0] - B[:, 1], B[:, 2] - np.logaddexp(B[:, 0], B[:, 1]))
    m3, t3 = sd(L_ABE)
    m_ar, t_ar = sd(L_ARI)
    s_tie = L_TB[:, 1] - L_TB[:, 0]
    c_tb = (L_TB[:, 1] + L_TB[:, 0]) / 2          # CE 기울기 0 방향 — 버려지던 공통 성분 (짝)
    c_abe = L_ABE.mean(1)

    cat = pd.read_csv(ARITH / 'feature_catalog.csv')
    feat = pd.read_parquet(ARITH / 'features_57k.parquet')
    pos = pd.Index(feat['id']).get_indexer(ids)
    assert (pos >= 0).all(), '피쳐 id 불일치'
    E, evn, O, odn = RH.parity_split_arith(feat, pos, cat)

    fam_of = cat.set_index('feature')['유형']

    def col_family(name):
        if name.startswith('log '):
            name = name[4:]
        if name.startswith('('):                   # '(a+b)/2' / '(a-b)/2'
            name = name[1:].replace('-', '+').split('+')[0]
        return str(fam_of.get(name, '기타'))

    fam_e = [col_family(c) for c in evn]
    fam_o = [col_family(c) for c in odn]

    Uz = np.column_stack([rankgauss(E[:, j]) for j in range(E.shape[1])])
    Vz = np.column_stack([signed_log(O[:, j]) for j in range(O.shape[1])])
    Cz = np.column_stack([rankgauss(c_tb), rankgauss(c_abe)])

    M = np.column_stack([m3, m_ar])                                  # 승부축 기저 (홀)
    T = np.column_stack([s_tie, t3, t_ar])                           # tie축 기저 (짝)

    # ── 새 재료 — TB1(ModernBERT) / ABE1(V3 bi-encoder) ──
    def load_tta(p, tag, n_class):
        z = np.load(p)
        assert (np.concatenate([z['valid_ids'], z['hold_ids']]) == ids).all(), f'{tag} id 불일치'
        assert (np.concatenate([z['valid_labels'], z['hold_labels']]) == y).all(), f'{tag} 라벨 불일치'
        L = np.vstack([(z['valid_fwd'] + z['valid_swp']) / 2,
                       (z['hold_fwd'] + z['hold_swp']) / 2]).astype(np.float64)
        assert L.shape == (len(ids), n_class), f'{tag} 형상 {L.shape}'
        return L, z

    s_tb1 = np.diff(load_tta(BOOST / f'logits/tb1__checkpoint-{TB1_STEP}.npz', 'TB1', 2)[0]).ravel()
    assert np.isfinite(s_tb1).all(), 'TB1 비유한값'

    # ABE1 — 전 행 재추출본이 우선. 없으면 clean 변환본(tt)으로 자동 강등한다.
    p_full = BOOST / f'logits/abe1__checkpoint-{ABE1_STEP}.npz'
    p_clean = BOOST / f'logits/tt__checkpoint-{ABE1_STEP}.npz'
    m_a1 = t_a1 = a1_rows = None
    a1_full, a1_src = False, None
    if p_full.exists():
        La1, _ = load_tta(p_full, 'ABE1', 3)
        assert np.isfinite(La1).all(), 'ABE1 비유한값'
        m_a1, t_a1 = sd(La1)
        a1_rows = np.ones(len(ids), bool)
        a1_full, a1_src = True, p_full.name
    elif p_clean.exists():
        La1, z = load_tta(p_clean, 'ABE1(tt)', 3)
        m_a1, t_a1 = sd(La1)
        a1_rows = np.concatenate([z['valid_clean'].astype(bool),
                                  np.ones(len(ids) - len(z['valid_clean']), bool)])
        assert np.isfinite(m_a1[a1_rows]).all() and np.isfinite(t_a1[a1_rows]).all(), \
            'ABE1 clean 행에 비유한값'
        a1_src = p_clean.name

    # V3 in-sample 행(현행 valid 5,000 중 3,000) — 계층별 읽기용. seed 42 분할 재현.
    a1_ins = np.zeros(len(ids), bool)
    if a1_src is not None:
        from sklearn.model_selection import train_test_split
        v3 = pd.read_csv(LLM / 'V3/origin/train.csv')
        v3l = np.select([v3['winner_model_a'] == 1, v3['winner_model_b'] == 1], [0, 1], 2)
        tr51, te = train_test_split(v3.assign(label=v3l), test_size=0.10, stratify=v3l,
                                    random_state=42, shuffle=True)
        tr49, va2 = train_test_split(tr51, test_size=2000, stratify=tr51['label'],
                                     random_state=42, shuffle=True)
        a1_ins = pd.Index(tr49['id']).get_indexer(ids) >= 0
        assert (pd.Index(te['id']).get_indexer(ids) >= 0).sum() == len(ids) - HALF, 'V3 hold 재현 실패'
        assert int(a1_ins.sum()) == 3000, f'V3 in-sample {int(a1_ins.sum())} != 3000'
        if p_clean.exists():         # 변환본의 clean 마스크와 교차 검증 — 두 경로가 일치해야 한다
            zc = np.load(p_clean)['valid_clean'].astype(bool)
            assert (zc == ~a1_ins[:HALF]).all(), 'V3 분할 재현이 tt 변환본 마스크와 다르다'

    return dict(ids=ids, y=y, M=M, T=T, Uz=Uz, Vz=Vz, Cz=Cz,
                evn=evn, odn=odn, fam_e=fam_e, fam_o=fam_o,
                m3=m3, m_ar=m_ar, s_tb1=s_tb1, m_a1=m_a1, t_a1=t_a1,
                a1_rows=a1_rows, a1_full=a1_full, a1_ins=a1_ins, a1_src=a1_src,
                contract=contract)


# ══════════════════════════════════════════════════════ 상호작용 조립
def build_inter(U, unames, groups, L, lnames):
    """U 열마다 L 전 열과의 곱. 같은 그룹이 연속 열이 되도록 정렬해 블록 분해가 성립하게 한다.
    반환: X(상호작용), blocks [(그룹, 시작, 끝)] — 시작/끝은 상호작용 내부 기준."""
    order = sorted(range(U.shape[1]), key=lambda j: groups[j])
    cols, blocks, cur, start = [], [], None, 0
    for j in order:
        if groups[j] != cur:
            if cur is not None:
                blocks.append((cur, start, len(cols)))
            cur, start = groups[j], len(cols)
        for i in range(L.shape[1]):
            cols.append(U[:, j] * L[:, i])
    if cur is not None:
        blocks.append((cur, start, len(cols)))
    X = np.column_stack(cols) if cols else np.zeros((U.shape[0], 0))
    return X, blocks


# ══════════════════════════════════════════════════════ 팔 실행
def run_arm(spec, y01, fold, rows, hold, base_ll, bias=None):
    """spec: (code, label, Xinter, inter_blocks, main_even) — main_even 은 로짓 뒤에 붙는
    짝 주효과 열 (tie축 F5a 의 c_tb·c_abe 뿐이다).
    bias 를 명시하지 않으면 rows 유무로 축을 추정한다 (재료 팔처럼 tie축에 행 마스크를
    걸 때는 반드시 명시할 것 — 절편은 축 패리티의 문제지 행 우주의 문제가 아니다)."""
    code, label, Lbase, Xinter, iblocks, main = spec
    nl = Lbase.shape[1]
    parts = [Lbase] + ([main] if main is not None else []) + ([Xinter] if Xinter.shape[1] else [])
    X = np.hstack(parts)
    off = nl + (main.shape[1] if main is not None else 0)
    blocks = [('logit', 0, nl)]
    if main is not None:
        blocks.append(('c주효과', nl, off))
    blocks += [(nm, off + s, off + e) for nm, s, e in iblocks]
    if bias is None:
        bias = rows is None                        # tie축만 절편 (짝) — 승부축은 반대칭이라 금지
    n_even = X.shape[1] if bias else 0
    lams = [0.0] if X.shape[1] == nl else LAMS

    t0 = time.time()
    z, _, picked, n_bad, n_ext, contrib, logit_w = RH.crossfit(
        X, y01, fold, bias, lams, rows=rows, n_even=n_even, n_logit=nl, blocks=blocks)
    secs = time.time() - t0

    m = np.ones(len(y01), bool) if rows is None else rows
    ll = RH.rowloss(z[m], y01[m])
    hm = hold[m]
    full, ho = float(ll.mean()), float(ll[hm].mean())
    delta = RH.paired_ci(base_ll, ll) if base_ll is not None else None

    zc = z[m]
    shares = {nm: float(np.cov(contrib[nm][m], zc)[0, 1] / np.var(zc)) for nm, _, _ in blocks}
    return dict(code=code, label=label, cols=X.shape[1], full=full, hold=ho, ll=ll,
                delta=delta, lam=float(np.median(picked)), n_bad=n_bad, n_ext=n_ext,
                secs=secs, shares=shares, logit_w=logit_w, z=z)


def arm_line(r, t_start):
    d = '기준' if r['delta'] is None else RH.ci_str(*r['delta'])
    warn = (f' ** 미수렴 {r["n_bad"]}' if r['n_bad'] else '') + \
           (f' (자동확장 x10 {r["n_ext"]}회)' if r['n_ext'] else '')
    print(f"        손실 {r['full']:.4f} (holdout {r['hold']:.4f})  {d}  | "
          f"λ중앙 {r['lam']:g}{warn} | {r['secs']:.0f}s  [누적 {(time.time() - t_start) / 60:.0f}분]")
    top = sorted(((nm, v) for nm, v in r['shares'].items() if nm != 'logit'),
                 key=lambda x: -abs(x[1]))
    top = [x for x in top[:10] if abs(x[1]) >= 0.001] or top[:4]
    if top:
        s = ' | '.join(f'{nm} {v:+.3f}' for nm, v in top)
        print(f"        기여 비중  logit {r['shares']['logit']:+.2f} | {s}")


# ══════════════════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='', help='쉼표로 팔 코드 부분일치 필터 (예: F0,F5)')
    ap.add_argument('--no-families', action='store_true', help='가족별 F2 26팔 생략')
    ap.add_argument('--ens', action='store_true',
                    help='기존 앙상블 계약(ABE 7ckpt·Arith 3종 평균)으로 복귀. 기본은 단일 최적 ckpt')
    args = ap.parse_args()
    only = [s.strip() for s in args.only.split(',') if s.strip()]

    t_start = time.time()
    print('자산 적재 중 (로짓 + Arith 피쳐 + 패리티 분해, ckpt 선택 포함) ...')
    D = load_all(single_best=not args.ens)
    y, M, T, Uz, Vz, Cz = D['y'], D['M'], D['T'], D['Uz'], D['Vz'], D['Cz']
    n = len(y)
    assert n == 10748, f'stage2 행수 {n}'
    hold = np.arange(n) >= HALF
    nz = y != 2
    y_tie = (y == 2).astype(np.float64)
    y_win = (y == 0).astype(np.float64)

    fold = np.zeros(n, int)
    for k, (_, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=SEED)
                                .split(np.zeros((n, 1)), y)):
        fold[te] = k

    sel = [D['evn'].index(c) for c in F1_SEL if c in D['evn']]
    assert len(sel) == len(F1_SEL), f'F1 선별 피쳐 누락: {set(F1_SEL) - set(D["evn"])}'
    Usel = Uz[:, sel]
    sel_names = [D['evn'][j] for j in sel]

    agree = np.column_stack([D['m3'] * D['m_ar'], np.abs(D['m3']), np.abs(D['m_ar']),
                             np.abs(D['m3'] - D['m_ar'])])          # 홀⊙홀·|홀| — 전부 짝

    print(f'stage2 {n:,}행 (valid {HALF:,} + hold {n - HALF:,}) | fold {np.bincount(fold)}')
    print(f'재료 계약: {D["contract"]}')
    print(f'짝피쳐 {Uz.shape[1]} (rankgauss) | 홀피쳐 {Vz.shape[1]} (signed-log) | '
          f'선별 {len(sel)} | 공통성분 2 | λ격자 {LAMS} + 상한 없는 자동확장')
    print(f'승부축 기저 [m3, m_ar] (비-tie {int(nz.sum()):,}행) | tie축 기저 [s_tie, t3, t_ar]')
    print('전 팔에서 로짓 열은 pen=0 — λ→∞ 는 정확히 F0 회복이라 음성이 "게이트 없음"으로 읽힌다')
    if not args.ens:
        print('! 단일 ckpt 계약 — (참조) repr/blend 행과 0.5739/0.5740/0.9705 앵커는 앙상블 계약'
              ' 수치라 직접 비교하지 말 것. F0 재현 검산은 --ens 로만 성립한다.\n')
    else:
        print()

    fams = sorted(set(D['fam_e']))

    def specs_for(axis):
        """(code, label, Lbase, Xinter, iblocks, main_even) 목록."""
        L, lnames = (M, ['m3', 'm_ar']) if axis == 'win' else (T, ['s_tie', 't3', 't_ar'])
        Z0 = np.zeros((n, 0))
        out = [(f'{axis[0].upper()}-F0', '로짓만 — 기준 재현', L, Z0, [], None)]
        Xi, bi = build_inter(Usel, sel_names, sel_names, L, lnames)
        out.append((f'{axis[0].upper()}-F1', f'선별 짝피쳐 {len(sel)}종 게이트', L, Xi, bi, None))
        Xc, bc = build_inter(Cz, ['c_tb', 'c_abe'], ['c_tb', 'c_abe'], L, lnames)
        if axis == 'win':
            out.append(('W-F5', '공통성분(c_tb·c_abe) 게이트', L, Xc, bc, None))
            # 짝→홀 방향: 타이로짓이 승부 로짓의 이득을 제어 ("tie 같으면 확신을 줄여라").
            # Davidson κ 는 반대 방향(|d|→tie축)만 쟀다 — 이 방향은 최초 측정이다.
            Gz = np.column_stack([rankgauss(T[:, j]) for j in range(T.shape[1])])
            Xg, bg = build_inter(Gz, ['s_tie', 't3', 't_ar'], ['s_tie', 't3', 't_ar'], L, lnames)
            out.append(('W-F6', '타이로짓 게이트 [s_tie,t3,t_ar] × 승부로짓', L, Xg, bg, None))
        else:
            out.append(('T-F5a', '공통성분 주효과+게이트', L, Xc, bc, Cz))
            out.append(('T-F5b', '로짓 합의항 [m3⊙m_ar, |m3|, |m_ar|, |m3−m_ar|]', L, agree,
                        [('합의항', 0, agree.shape[1])], None))
            Xv, bv = build_inter(Vz, D['odn'], D['fam_o'], M, ['m3', 'm_ar'])
            out.append(('T-F4', f'홀피쳐 {Vz.shape[1]} × 승부로짓 = 합의 채널', L, Xv, bv, None))
        if not args.no_families:
            for f in fams:
                idx = [j for j, g in enumerate(D['fam_e']) if g == f]
                fn = [D['evn'][j] for j in idx]
                # 블록을 피쳐 낱개로 — "가족 안에서 어떤 피쳐가 결정자로 적합한가"를 기여 비중으로 분해.
                # 기여는 OOF 기술통계다(적합에 쓰인 적 없음) — 판정은 여전히 팔 단위 Δ CI 로만.
                Xf, bf = build_inter(Uz[:, idx], fn, fn, L, lnames)
                out.append((f'{axis[0].upper()}-F2:{f}', f'가족 게이트 [{f}] {len(idx)}피쳐',
                            L, Xf, bf, None))
        Xa, ba = build_inter(Uz, D['evn'], D['fam_e'], L, lnames)
        out.append((f'{axis[0].upper()}-F3', f'짝피쳐 전부 게이트 (kitchen-sink {Xa.shape[1]}열)',
                    L, Xa, ba, None))
        return out

    # ── [0] 재료 팔 — 새 전문가(TB1·ABE1) 기저 추가. 게이트 없음, 전열 pen=0(무규제 적합) ──
    matres = {}
    run_mat = (not only) or any(o.upper().startswith('R') for o in only)
    if run_mat:
        RH.head('0', '재료 팔 — TB1(ModernBERT)·ABE1(V3 bi-encoder) 기저 추가. 전 10,748행')
        s_tb1, m_a1, t_a1 = D['s_tb1'], D['m_a1'], D['t_a1']
        a1_rows, a1_full, a1_ins = D['a1_rows'], D['a1_full'], D['a1_ins']
        print(f'  단독 신호 — TB1 tie AUC {RH.auc_of(s_tb1, y == 2):.4f} (기록 0.5983 근방이어야 정상)')
        if a1_rows is None:
            print('  ! ABE1 자산 없음 — extract_logits.py --only abe1 (전 행) 실행 후 ABE1 팔이 열린다')
            ar = None
        else:
            ar = a1_rows if not a1_full else np.ones(n, bool)
            aw = ar & nz
            print(f"  단독 신호 — ABE1 tie AUC {RH.auc_of(t_a1[ar], (y == 2)[ar]):.4f} / "
                  f"승부 AUC {RH.auc_of(m_a1[aw], (y == 0)[aw]):.4f}  "
                  f"({D['a1_src']}, {int(ar.sum()):,}행)")
            if a1_full:
                print(f'  ABE1 행 계층: holdout {int(hold.sum()):,} 무접촉 | '
                      f'valid {int((~hold & ~a1_ins).sum()):,} V3 선택노출 | '
                      f'valid {int(a1_ins.sum()):,} V3 학습(in-sample)')
                print('  -> 사용자 결정(2026-08-05)에 따라 전 행을 재료로 쓴다. 계층별 Δ 를 아래에 같이 낸다 —')
                print('     이득이 in-sample 계층에 몰리면 암기, holdout 에서도 같이 나오면 실신호다.')
            else:
                print(f'  ! 전 행 재추출본이 없어 clean {int(ar.sum()):,}행으로 강등 — '
                      f'본표(10,748행)와 손실 수준을 직접 비교하지 말 것')
        Z0 = np.zeros((n, 0))
        rw, rt = (ar & nz if ar is not None else nz), (ar if ar is not None else None)
        groups = [
            (f'tie축 · {int(rt.sum()) if rt is not None else n:,}행', y_tie, rt, True, [
                ('RT0', '로짓만 — 기준', T),
                ('RT1', '+TB1', np.column_stack([T[:, 0], s_tb1, T[:, 1], T[:, 2]])),
            ]),
        ]
        if ar is not None:
            groups[0][4].extend([
                ('RT2', '+ABE1', np.column_stack([T[:, 0], T[:, 1], t_a1, T[:, 2]])),
                ('RT3', '+TB1+ABE1', np.column_stack([T[:, 0], s_tb1, T[:, 1], t_a1, T[:, 2]])),
            ])
            groups.append((f'승부축 · 비-tie {int(rw.sum()):,}행', y_win, rw, False, [
                ('RW0', '로짓만 — 기준', M),
                ('RW1', '+ABE1 승부로짓 [m3, m_a1, m_ar]',
                 np.column_stack([M[:, 0], m_a1, M[:, 1]])),
            ]))
        # 계층 정의 — ABE1 오염 경계. 승부축은 각 계층 안에서 다시 비-tie 로 좁힌다.
        strata = [('holdout(무접촉)', hold), ('valid·V3선택', ~hold & ~a1_ins),
                  ('valid·V3학습', ~hold & a1_ins)]
        for gname, y01g, rowsg, biasg, arms_g in groups:
            print(f'\n  ── {gname} ──')
            base_ll, mg = None, (np.ones(n, bool) if rowsg is None else rowsg)
            for code, label, Lb in arms_g:
                r = run_arm((code, label, Lb, Z0, [], None), y01g, fold, rowsg, hold,
                            base_ll, bias=biasg)
                if base_ll is None:
                    base_ll = r['ll']
                print(f'  [{code}] {label}  ({Lb.shape[1]}열)')
                arm_line(r, t_start)
                if a1_full and code in ('RT2', 'RT3', 'RW1'):
                    cells = []
                    for snm, sm in strata:
                        k = sm[mg]
                        if k.sum() < 50:
                            continue
                        d = RH.paired_ci(base_ll[k], r['ll'][k])
                        cells.append(f'{snm} {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]')
                    print('        계층별 Δ  ' + ' | '.join(cells))
                matres[code] = r
        if ar is not None and 'RW1' in matres:
            uni = '전 10,748행' if a1_full else f'clean {int(ar.sum()):,}행'
            nzf_u = float(nz[ar].mean())
            hb_w, hb_t = hold[rw], hold[rt]
            nzf_h = float(nz[hold].mean())
            print(f'\n  ── 종합 ({uni}) — 총 ll = tie축 + (1−tie율)·승부축. holdout 열은 본표와 비교 가능 ──')
            for nm, wc, tc in [('기준        RW0/RT0', 'RW0', 'RT0'), ('+TB1        RW0/RT1', 'RW0', 'RT1'),
                               ('+ABE1       RW1/RT2', 'RW1', 'RT2'), ('둘 다       RW1/RT3', 'RW1', 'RT3')]:
                lw, lt = matres[wc]['ll'], matres[tc]['ll']
                print(f'    {nm:<20} 총 {float(lt.mean() + nzf_u * lw.mean()):.4f} '
                      f'(holdout {float(lt[hb_t].mean() + nzf_h * lw[hb_w].mean()):.4f})')
        print()

    results = {'win': [], 'tie': []}
    for axis in ('win', 'tie'):
        y01 = y_win if axis == 'win' else y_tie
        rows = nz if axis == 'win' else None
        title = ('[1] 승부축 (A 가 이겼나) — 비-tie 행당 손실, 절편·주효과 금지(반대칭)'
                 if axis == 'win' else '[2] tie축 (비겼나) — 전 행당 손실')
        RH.head('1' if axis == 'win' else '2', title.split('] ')[1])
        specs = specs_for(axis)
        if only:
            specs = [s for s in specs if any(o in s[0] for o in only)]
            if not any(s[0].endswith('F0') for s in specs):     # 빈 축도 F0 은 돌린다 (종합 계산용)
                specs = [x for x in specs_for(axis) if x[0].endswith('F0')] + specs
        base_ll = None
        for i, spec in enumerate(specs):
            print(f'  [{i + 1}/{len(specs)}] {spec[0]}({spec[1]})  — 기저 {spec[2].shape[1]}'
                  f'+{spec[3].shape[1] + (spec[5].shape[1] if spec[5] is not None else 0)}열, 적합 중 ...')
            r = run_arm(spec, y01, fold, rows, hold, base_ll)
            if base_ll is None:
                base_ll = r['ll']
            arm_line(r, t_start)
            results[axis].append(r)

        ref = REPR_WIN if axis == 'win' else REPR_TIE
        rows_tbl = [[f"{r['code']}({r['label']})", f"{r['cols']:,}", f"{r['full']:.4f}",
                     f"{r['hold']:.4f}", '기준' if r['delta'] is None else RH.ci_str(*r['delta']),
                     f"{r['lam']:g}", f"{r['secs']:.0f}s",
                     f"미수렴 {r['n_bad']}" if r['n_bad'] else '']
                    for r in results[axis]]
        rows_tbl += [[f'(참조) repr {nm}', '', f'{a:.4f}', f'{b:.4f}', 'log_repr.txt', '', '', '']
                     for nm, (a, b) in ref.items()]
        print()
        RH.table(['구성', '열', '손실(전체)', '손실(holdout)', 'Δ vs F0(양수=이득)', 'λ중앙', '시간', '경고'],
                 rows_tbl, align=['l', 'r', 'r', 'r', 'l', 'r', 'r', 'l'])
        print()

    # ── [3] 종합 ──
    RH.head('3', '종합 — 총 ll = tie축 + (1−tie율)·승부축(비-tie 행당)')
    nz_frac = float(nz.mean())
    nz_frac_h = float(nz[hold].mean())

    def total_of(rw, rt):
        lw, lt = rw['ll'], rt['ll']
        hw, ht = hold[nz], hold
        full = float(lt.mean() + nz_frac * lw.mean())
        ho = float(lt[ht].mean() + nz_frac_h * lw[hw].mean())
        return full, ho

    w0, t0_ = results['win'][0], results['tie'][0]
    bw = min(results['win'], key=lambda r: r['full'])
    bt = min(results['tie'], key=lambda r: r['full'])
    rows_tot = []
    for nm, rw, rt in [('F0/F0 (기준)', w0, t0_),
                       (f"최선 {bw['code']} / {bt['code']} (사후 선택 — 낙관적)", bw, bt)]:
        full, ho = total_of(rw, rt)
        rows_tot.append([nm, f'{full:.4f}', f'{ho:.4f}'])
    for nm, v in REF_TOTAL.items():
        rows_tot.append([f'(참조) {nm}', f'{v:.4f}', ''])
    RH.table(['구성', '총(전체)', '총(holdout)'], rows_tot, align=['l', 'r', 'r'])

    n_arms = sum(len(v) for v in results.values()) - 2
    print(f'\n  판정 규율: 채택은 축별 Δ CI + 문턱 {THRESH}. 기준 외 팔 {n_arms}개 — 가족별(F2) 발견은')
    print('  Bonferroni 로 본다 (reprhead 전례: T2 는 3.9σ 로 8팔 보정을 살아남았다).')
    print('  음성은 검정력 부족일 수 있어 "없음"의 증명이 아니라 "이 자로는 안 보임"이다.')

    out_p = BOOST / 'fwls_oof.npz'
    data = {}
    if out_p.exists():                              # --only 재실행이 이전 팔 결과를 지우지 않게 병합
        with np.load(out_p) as old:
            data = {k: old[k] for k in old.files}
    data['ids'], data['y'] = D['ids'], y
    data.update({f"z{ax}__{r['code']}": r['z'].astype(np.float32)
                 for ax in results for r in results[ax]})
    # 재료 팔도 저장한다 — 이후 조합 비교가 같은 행에서 paired 로 성립하려면 점수가 남아야 한다.
    # 축 표시는 코드가 들고 있다 (RW*/RT*). 미사용 행은 crossfit 이 NaN 으로 남긴다.
    data.update({f"zmat__{c}": r['z'].astype(np.float32) for c, r in matres.items()})
    np.savez_compressed(out_p, **data)
    print(f'\nsaved: {out_p} (팔별 축 점수 OOF, 병합 저장)')


if __name__ == '__main__':
    main()
