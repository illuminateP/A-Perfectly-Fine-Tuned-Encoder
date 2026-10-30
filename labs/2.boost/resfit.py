"""resfit.py — 잔차-앵커 train 적합: 표현 판독 β 를 train 46,729행에서 배우면
stage2-적합 β(현행 T-L/W-L)를 이기는가.

왜 이 설계인가 (train 그대로 적합은 왜 안 하나)
  train 행의 풀링·로짓은 인코더의 in-sample 산출물이다. 로짓 가중을 train 에 재적합하면
  −0.0399(fwls_transfer), 표현 헤드를 train 에 적합하면 −0.031~−0.093(newhead) 실측.
  여기서는 로짓 결합을 stage2-clean 에서 적합해 **방향 하나(s0)로 동결**하고, train 에서는
  s0 의 스케일 1개(무벌점 자유)와 β 만 배운다. 인코더가 암기로 이미 맞히는 행은 s0 가
  흡수하므로 β 의 학습 신호에서 암기 성분이 빠진다 — 잔차 논리.
  실패 경로도 명확하다: 잔차 자체가 in-sample 왜곡을 물려받으면 β 는 전이되지 않는다.
  어느 쪽인지 가르는 게 이 측정이다.

계약
  train 로짓 (단일 ckpt — train 추출이 이것뿐):
      abe-3000 / tb3-4000 / abe1-3109 / tb1-5000(선택)
      + Arith 는 **단일 최적 부스터** — valid raw ll 선택(load_all 과 같은 규칙, 실측 xgb).
      train 쪽은 그 부스터의 raw_oof_train (OOF 라 clean). 인코더 열은 in-sample — s0 전용.
  s0 앵커 가중은 stage2 에서도 **같은 단일-ckpt 열**로 적합한다 (계약 혼합 금지).
  평가 팔의 기저(M3/T5)는 **단일 최적 ckpt 계약** — valid raw ll 로 고른 ckpt 1개/전문가
  (load_all(single_best=True); 앙상블 7-ckpt 는 추론 예산 대비 지나침 — 사용자 지시).
  주의 1: 앵커의 ckpt(abe-3000/tb3-4000)와 평가 기저의 선택 ckpt 는 다를 수 있다 —
          앵커는 train 에 존재하는 추출본 고정이라 제약이지 선택이 아니다.
  주의 2: 이 파일의 R0/L 절대값은 앙상블 기저의 본표·repr5 와 직접 비교 금지 —
          파일 안 paired 만 유효하다.
  풀링: tie tb_u(tb3-4000) / 승부 a1_v(abe1-3109) — repr5 에서 유의했던 두 블록 (추출본 고정).
  정규화 통계는 train 에서 내고(라벨 비접촉) stage2 에 같은 값을 적용한다.
  홀 블록은 RMS-only — 중심화하면 반대칭이 깨진다.

앵커 유/무 — "로짓 베껴 쓰기" 가설의 직접 검증 (사용자 가설 2026-08-05)
  stage2 릿지에서 pen=0 로짓 열은 지름길이다: 판별 부담을 로짓이 다 지고 풀링은 작은
  보정으로 수축된다. 이 지름길이 풀링이 배울 수 있던 것을 **억눌렀다**는 가설이 맞다면,
  train 4.3배 행에서 로짓 없이 배운 판독(무앵커 β)이 더 많은 전이 신호를 담아야 한다.
  반대로 앵커의 존재 이유(잔차 논리: in-sample 암기를 s0 가 흡수)가 맞다면, 무앵커 β 는
  train 에서 암기 방향(linear probe)을 배워 stage2 전이가 무너져야 한다 — newhead 전례.
  B1(앵커) vs N1(무앵커)이 두 가설을 가른다. 두 β 의 stage2 상관·무적합 AUC 도 찍는다.

v2 (2026-08-05 저녁) — 1차 런 후 확장. 1차 실측: B1 종합 0.9636(최선) / N1 0.9661 / L 0.9654,
  B2·N2 는 L 대비 +0.0001 로 실질 0 이라 팔에서 뺐다(기록: log_resfit.txt). 이번 판이 더 묻는 것:
  · **소스 4종 확장**: train 풀링이 4종 전부 생겼다(abe/tb3/abe1/tb1, 전부 bf16).
    소스별로 앵커-잔차 β 를 독립 적합해 r 4열을 만들고, M1(전 소스)로 합산 가치를 잰다.
  · **부호 진단**: 1차 런에서 승부축 r(앵커)의 전이 AUC 0.380 — train 에서 배운 방향의
    **반대**가 stage2 에서 유용했다. 기준 팔(R0)의 잔차와 r 의 상관을 fold 별로 찍어
    부호가 5-fold 일관인지(계통적 레짐 차 -> 부호만 뒤집으면 사용 가능) 본다.
  · **경로 통일**: 1차 런의 a1 stage2 풀링은 V3 fp16 임베딩 매핑이었는데 train 은 자체
    bf16 추출이었다 — β 좌표계가 양쪽에서 미묘하게 달랐고 부호 반전의 후보 원인이다.
    이번 판은 stage2 도 자체 bf16 풀링(pooled/abe1__*.npz)으로 통일한다.
  · **--maxit (수렴 민감도)**: L-BFGS 라 epoch 는 없지만 반복 상한이 그 역할이다.
    β 적합만 기본 1000 -> 5000 으로 올린다(무앵커는 맨바닥 학습이라 최적화가 더 어렵다 —
    저수렴이 N1 을 깎아내렸는지 확인). 팔(crossfit)은 기존 상한 유지 — 대조 조건 불변.

v3 (2026-08-06) — 사용자 지시 5건 반영. v2 실측 요약: B1 재현(경로·maxit 원인 배제),
  부호 미스터리 해소(억제자 — AUC<0.5 이면서 corr(잔차)>0 5/5), M1 종합 0.9631 최선,
  무앵커는 y 는 잘 맞추고 잔차는 못 맞춤(중복의 직접 증명). B1/N1 팔은 두 로그에 기록돼
  제외 — v3 의 참조점은 M1 이다.

  ① 라운드(=epoch): 강소스(tie tb_u / 승부 a1_v)에서 함수공간 부스팅 —
     라운드 k 는 누적 train 점수의 잔차에 βₖ 를 적합, rₖ 열 추가. D2(+r₂)·D3(+r₂+r₃).
  ② "로짓 베끼기 손해" 가설 검증: 라운드별 한계 Δ(D2 vs M1, D3 vs D2)가 음수로 꺾이면
     사용자 가설(앵커가 자원을 로짓 복제에 몰아 epoch 증가가 손해) 지지, 계속 양수면 기각적.
     설계는 그대로 두고 측정으로만 가른다.
  ③ FWLS 게이트 팔(F): Arith 짝피쳐를 **가족별 상위 2개**(train s0-잔차와의 |corr| 기준 —
     선택이 train 에서 일어나 stage2 판정이 오염되지 않는다) 뽑아
     g ⊙ [로짓 기저 + r 4열] 곱항 (+tie 축은 g 주효과). fwls 의 build_inter 재사용.
  ④ 비선형 후보 각 1팔 — 전부 "r-생산자 안에, 잔차 타깃으로" 규율을 따른다:
     A(RFF-β): 패리티 보존 무작위 특징(전 소스 concat, D=2048, c={0.3,1,3,10})에 앵커-β.
     BL(저계수 쌍선형): 라벨-프리 train SVD 좌표(k=16)의 패리티 합법 곱 — 승부 u⊙v(홀),
       tie u⊙u·v⊙v(짝) — 에 앵커-β. 교차-소스 상호작용을 학습 없이 좌표화한 것.
     MP(소형 MLP): SVD-128 좌표 위 등변 g (win 반대칭 / tie 대칭, 출력 0-초기화,
       s0 스케일 자유) — train ES 15%, stage2 는 열로만 소비.
  ⑤ 스케일링·앙상블 전무 — 프로브 계약(probe-contract-single-ckpt) 그대로.

v4 (2026-08-06) — GAT 팔 추가 (사용자 지시: **concat 풀링 벡터 뒤에** 게이티드 어텐션).
  v3 실측: 후보 6개 전부 vs M1 ≈ 0 또는 음수 (F 게이트 tie −0.0016★/−0.0020★ — 게이트
  계열 3번째 사망; 라운드는 r2 의 λ->격자 꼭대기, in-sample Δ +0.0000 으로 **1라운드 만에
  잔차 소진** — 부스팅 자체 종료).
  GT: concat SVD 좌표 x 에 f(x) = w·(tanh(W1 x) ⊙ σ(W2 x)) — 게이트가 행별로 은닉 유닛을
  곱셈 선별한다. MP(순수 가산 MLP)와 같은 입력·같은 프로토콜이라 GT vs MP 가
  "게이팅의 순수 한계 가치"를 잰다. 게이트 포화도(평균 |σ−0.5|)를 진단으로 찍는다.
  GT0: 같은 GAT 를 **풀링만으로**(s0 없이 y 직접) 학습 — w6/w7 대조 (선형 B1/N1 의 신경판).
  FN (v4.3, 사용자 지시): BERT-style FFN d→4d→GELU→d→GELU→readout (~527k 파라미터) —
  newhead 팔 11/12 구조의 잔차-앵커 이식판. FN vs MP 가 용량(527k vs 8k)의 가치를 가른다.
  TE/TD (v4.4, 사용자 지시 — 풀링만·무앵커 2팔): 전문가 4토큰(각 [u|v] SVD-64x2) 위
  트랜스포머 2층. TE = 양방향(인코더) — 전문가끼리 서로 보는 cross-형 헤드의 유비.
  TD = 인과 마스크만 추가(디코더-only, 파라미터 동일) — 순서 없는 집합에 인과 제약이
  지불하는 비용을 TD vs TE 통제쌍이 잰다. 토큰 순서 고정 abe->tb3->a1->tb1.
  SV/MPF (v4.5, 사용자 지시 — SVD 압축손실의 직접 측정 2팔):
    SV  = M1 과 동일 구조(소스별 앵커-β)·입력만 전문가별 SVD-64 -> SV vs M1 이 **선형**이
          보는 압축손실. 동률이면 신경 팔들의 0 에서 "압축 핑계"가 제거된다.
    MPF = MP 와 동일 구조(hidden 32)·입력만 원본 8,192 전차원 -> MPF vs MP 가 **비선형**이
          보는 압축손실이자 "본 헤드 학습 시 전차원/고용량 확대" 방향의 추세 측정.

판정 (holdout paired CI + 문턱 0.0005, vs-M1 비교 14개 -> Bonferroni x14, 판정은 사용자)
  M1 vs L, 그리고 D2/D3/F/A/BL/MP/GT 각각 vs M1 (후보의 한계 가치)

선행 — v2 는 stage2·train 풀링 8블록 전부를 요구한다 (2026-08-05 bf16 체인으로 추출 완료:
  abe-3000 / tb3-4000 / abe1-3109 / tb1-5000, 로짓·풀링 모두 amp=bfloat16, fp32 폴백 0행)
  재추출 필요 시: python Boost/extract_logits.py [--split train] --pooled --only abe1,tb1
                  --steps abe1:3109,tb1:5000 [--overwrite]

사용 (약 50~80분 — train β 적합 축당 ~8회 + SVD 2회 + MLP + stage2 팔 9개)
  python Boost/resfit.py 2>&1 | tee Boost/log_resfit3.txt
  python Boost/resfit.py --axes win                # 부분 실행
"""
import argparse
import sys
import time

import numpy as np
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

import fwls as FW
import reprhead as RH
import repr5 as R5

sys.stdout.reconfigure(line_buffering=True)

SEED, K = RH.SEED, RH.K
LLM, BOOST, ARITH = FW.LLM, FW.BOOST, FW.ARITH
sd3 = lambda B: (B[:, 0] - B[:, 1], B[:, 2] - np.logaddexp(B[:, 0], B[:, 1]))
LAMS_TR = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]     # n 4.3배 — stage2 최적(1e4) 위쪽을 넉넉히
RFF_SCALES = [0.3, 1.0, 3.0, 10.0]                # nlfeat 와이드 스캔과 같은 대역폭 격자
K_BL = 16                                          # 쌍선형 SVD 좌표 수 (곱 256~272열)
K_MLP = 128                                        # MLP 입력 SVD 좌표 수 (채널당)
K_TOK = 64                                         # TF 토큰(전문가)당 SVD 좌표 수 (패리티당)


def tta(z, part):
    return ((z[f'{part}_fwd'] + z[f'{part}_swp']) / 2).astype(np.float64)


def _auc(v, yb):
    """순위 AUC — 적합·보정 없이 방향의 전이 신호만 본다 (mlphead._auc 와 동일)."""
    import pandas as pd
    r = pd.Series(v).rank().to_numpy()
    n1 = int(yb.sum())
    return float((r[yb].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yb) - n1)))


# ══════════════════════════════════════════════════════ 적재
def pick_arith(ids, y):
    """앵커용 Arith 부스터 1개 — valid raw ll 선택. load_all(single_best=True) 와 같은
    규칙이라 평가 기저의 선택과 자동으로 일치한다 (둘 다 온도 피팅 없음)."""
    za = np.load(ARITH / 'outputs/arith_experts_raw.npz')
    assert (np.concatenate([za['ids_valid'], za['ids_hold']]) == ids).all(), 'arith id 불일치'
    lls = {}
    for m in FW.ARITH_MODELS:
        L = np.vstack([za[f'raw_valid__{m}'], za[f'raw_hold__{m}']]).astype(np.float64)
        lls[m] = FW._raw_ll(L[:FW.HALF], y[:FW.HALF])
    best = min(lls, key=lls.get)
    print('  [anchor·arith] valid raw ll: '
          + '  '.join(f'{m}:{lls[m]:.4f}' for m in FW.ARITH_MODELS) + f'  -> 선택 {best}')
    return best


def load_train(ar):
    """train 46,729 로짓·풀링·라벨. 모든 파일을 abe train npz 의 id 순서로 전량 대조한다.
    ar: 앵커에 쓸 Arith 부스터 (pick_arith 선택)."""
    za = np.load(BOOST / 'logits/abe__checkpoint-3000__train.npz')
    ids, y = za['train_ids'], za['train_labels']
    assert len(ids) == 46729 and len(np.unique(ids)) == len(ids)

    def get(rel, n_class, need=True):
        p = BOOST / rel
        if not p.exists():
            assert not need, f'{p} 없음 — 선행 추출 필요 (헤더 참고)'
            return None
        z = np.load(p)
        assert (z['train_ids'] == ids).all() and (z['train_labels'] == y).all(), f'{p.name} 정렬'
        L = tta(z, 'train')
        assert L.shape == (len(ids), n_class) and np.isfinite(L).all(), f'{p.name} 형상/비유한'
        return L

    L_abe = tta(za, 'train')
    L_tb3 = get('logits/tb3__checkpoint-4000__train.npz', 2)
    L_a1 = get(f'logits/abe1__checkpoint-{FW.ABE1_STEP}__train.npz', 3)
    L_tb1 = get(f'logits/tb1__checkpoint-{FW.TB1_STEP}__train.npz', 2, need=False)

    zar = np.load(ARITH / 'outputs/arith_experts_raw.npz')
    assert (zar['ids_train'] == ids).all() and (zar['y_train'] == y).all(), 'arith train 정렬'
    L_ar = zar[f'raw_oof_train__{ar}'].astype(np.float64)

    m3, t3 = sd3(L_abe)
    m_ar, t_ar = sd3(L_ar)
    m_a1, t_a1 = sd3(L_a1)
    D = dict(ids=ids, y=y, m3=m3, t3=t3, m_ar=m_ar, t_ar=t_ar, m_a1=m_a1, t_a1=t_a1,
             s_tie=L_tb3[:, 1] - L_tb3[:, 0],
             s_tb1=(L_tb1[:, 1] - L_tb1[:, 0]) if L_tb1 is not None else None)

    # 풀링 — 4소스 8블록. 짝 (x+y)/2, 홀 (x−y)/2 (swap = 두 뷰/브랜치 교환 항등식)
    def pool(rel, ka, kb):
        z = np.load(BOOST / rel)
        assert (z['train_ids'] == ids).all(), f'{rel} 정렬'
        return z[f'train_{ka}'].astype(np.float64), z[f'train_{kb}'].astype(np.float64)

    zp = np.load(BOOST / 'pooled/tb3__checkpoint-4000__train.npz')
    assert (zp['train_ids'] == ids).all(), 'tb3 train 풀링 정렬'
    g = lambda k: zp[f'train_{k}'].astype(np.float64)
    A_rep, B_rep = (g('a_f') + g('b_s')) / 2, (g('b_f') + g('a_s')) / 2
    D['tb_u'], D['tb_v'] = (A_rep + B_rep) / 2, (A_rep - B_rep) / 2

    ea, eb = pool('pooled/abe__checkpoint-3000__train.npz', 'ea', 'eb')
    D['abe_u'], D['abe_v'] = (ea + eb) / 2, (ea - eb) / 2
    qa, qb = pool(f'pooled/abe1__checkpoint-{FW.ABE1_STEP}__train.npz', 'ea', 'eb')
    D['a1_u'], D['a1_v'] = (qa + qb) / 2, (qa - qb) / 2
    xf, xs = pool(f'pooled/tb1__checkpoint-{FW.TB1_STEP}__train.npz', 'x_f', 'x_s')
    D['tb1_u'], D['tb1_v'] = (xf + xs) / 2, (xf - xs) / 2
    for k in ('tb_u', 'tb_v', 'abe_u', 'abe_v', 'a1_u', 'a1_v', 'tb1_u', 'tb1_v'):
        assert np.isfinite(D[k]).all(), f'train {k} 비유한값'
    return D


def load_pool_s2(ids):
    """stage2 풀링 8블록 — 전부 **자체 bf16 추출본**. train 과 같은 추출 경로라 β 좌표계가
    양쪽에서 동일하다 (1차 런의 a1 은 V3 fp16 임베딩 매핑 — 부호 반전 후보 원인이라 교체)."""
    def get(rel):
        z = np.load(BOOST / rel)
        assert (np.concatenate([z['valid_ids'], z['hold_ids']]) == ids).all(), f'{rel} id'
        return z
    cat = lambda z, k: np.vstack([z[f'valid_{k}'], z[f'hold_{k}']]).astype(np.float64)
    za = get('pooled/abe__checkpoint-3000.npz')
    ea, eb = cat(za, 'ea'), cat(za, 'eb')
    zt = get('pooled/tb3__checkpoint-4000.npz')
    A_rep = (cat(zt, 'a_f') + cat(zt, 'b_s')) / 2
    B_rep = (cat(zt, 'b_f') + cat(zt, 'a_s')) / 2
    zq = get(f'pooled/abe1__checkpoint-{FW.ABE1_STEP}.npz')
    qa, qb = cat(zq, 'ea'), cat(zq, 'eb')
    z1 = get(f'pooled/tb1__checkpoint-{FW.TB1_STEP}.npz')
    xf, xs = cat(z1, 'x_f'), cat(z1, 'x_s')
    P = dict(abe_u=(ea + eb) / 2, abe_v=(ea - eb) / 2,
             tb_u=(A_rep + B_rep) / 2, tb_v=(A_rep - B_rep) / 2,
             a1_u=(qa + qb) / 2, a1_v=(qa - qb) / 2,
             tb1_u=(xf + xs) / 2, tb1_v=(xf - xs) / 2)
    for k, v in P.items():
        assert np.isfinite(v).all(), f'stage2 풀링 {k} 비유한값'
    return P


def stage2_single(ids, y, has_tb1, ar):
    """s0 앵커 적합용 — train 계약과 같은 단일-ckpt·단일-부스터 열을 stage2 에서 구성한다."""
    def get(rel, n_class):
        z = np.load(BOOST / rel)
        assert (np.concatenate([z['valid_ids'], z['hold_ids']]) == ids).all(), f'{rel} id'
        L = np.vstack([(z['valid_fwd'] + z['valid_swp']) / 2,
                       (z['hold_fwd'] + z['hold_swp']) / 2]).astype(np.float64)
        assert L.shape == (len(ids), n_class)
        return L

    L_abe = get('logits/abe__checkpoint-3000.npz', 3)
    L_tb3 = get('logits/tb3__checkpoint-4000.npz', 2)
    L_a1 = get(f'logits/abe1__checkpoint-{FW.ABE1_STEP}.npz', 3)
    za = np.load(ARITH / 'outputs/arith_experts_raw.npz')
    L_ar = np.vstack([za[f'raw_valid__{ar}'], za[f'raw_hold__{ar}']]).astype(np.float64)
    m3, t3 = sd3(L_abe)
    m_ar, t_ar = sd3(L_ar)
    m_a1, t_a1 = sd3(L_a1)
    D = dict(m3=m3, t3=t3, m_ar=m_ar, t_ar=t_ar, m_a1=m_a1, t_a1=t_a1,
             s_tie=L_tb3[:, 1] - L_tb3[:, 0], s_tb1=None)
    if has_tb1:
        L_tb1 = get(f'logits/tb1__checkpoint-{FW.TB1_STEP}.npz', 2)
        D['s_tb1'] = L_tb1[:, 1] - L_tb1[:, 0]
    return D


# ══════════════════════════════════════════════════════ 앵커와 β
def s0_resid(s0, y01, bias):
    """s0 의 1-스칼라 재적합 잔차 — 게이트 선택의 기준 (train 안에서만 라벨을 쓴다)."""
    s0z = s0 / np.clip(np.sqrt((s0 ** 2).mean()), 1e-9, None)
    v, _ = RH._fit(s0z[:, None], y01, np.zeros(1), bias)
    z = s0z * v[0] + (v[1] if bias else 0.0)
    return y01 - expit(z)


def fit_anchor(cols, y01, rows, bias, names):
    """stage2 clean 행 단일 적합 — 로짓 열만, 무벌점, 표준화 없음(열 3~5개, raw 좌표 유지).
    반환 (w, b). 이 좌표계 그대로 train 열에 내적하면 s0 가 된다."""
    X = np.column_stack(cols)
    v, ok = RH._fit(X[rows], y01[rows], np.zeros(X.shape[1]), bias)
    w = v[:X.shape[1]]
    b = float(v[X.shape[1]]) if bias else 0.0
    print('    앵커 가중  ' + '  '.join(f'{nm} {wi:+.3f}' for nm, wi in zip(names, w))
          + (f'  절편 {b:+.3f}' if bias else '') + ('' if ok else '  ** 미수렴'))
    return w, b


def fit_beta(s0, P, y01, even, bias, tag, free=0):
    """train 에서 [s0(무벌점, 선택)] + P(릿지 λ) 를 적합한다. λ 는 층화 3-fold CV, 웜스타트.

    **s0=None 이면 무앵커** — 로짓 열 전무, 풀링 단독으로 판별 전체를 배운다 ("베껴 쓰기"
    가설 팔). 이때 in-sample 이 배우는 것에는 암기 방향이 포함될 수 있다 — 그 전이 여부가
    측정 대상이다.

    정규화는 전 행 통계(라벨 비접촉)로 한 번만 한다 — RH.crossfit 처럼 fold 별 재표준화까지는
    하지 않는다. 통계가 라벨을 안 보므로 λ 선택에 주는 낙관은 무시 가능하고, 이 통계를
    stage2 적용 시 그대로 재사용하는 것이 계약이다(호출부).
    free>0 이면 P 의 **앞 free 열은 무벌점**이다 — 스크래치 모드(headgrid)에서 로짓 열을
    자유 입력으로 줄 때 쓴다 (s0 와 조합 가능; 무벌점 구간은 [s0] + P[:, :free]).
    반환 (β, 진단 dict). β 는 정규화 좌표계 기준 — 호출부는 같은 정규화를 적용해야 한다."""
    n, p = P.shape
    if even:
        mu = P.mean(0)
        sc = np.clip(P.std(0), 1e-9, None)
    else:
        mu = np.zeros(p)                          # 홀 — 중심화 금지 (반대칭)
        sc = np.clip(np.sqrt((P ** 2).mean(0)), 1e-9, None)
    Pz = (P - mu) / sc
    off = 0 if s0 is None else 1
    if off:
        s0z = s0 / np.clip(np.sqrt((s0 ** 2).mean()), 1e-9, None)
        X = np.hstack([s0z[:, None], Pz])
    else:
        X = Pz

    inner = np.zeros(n, int)
    for j, (_, iv) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED)
                                .split(np.zeros((n, 1)), y01)):
        inner[iv] = j

    def penvec(lam):
        pen = np.full(X.shape[1], float(lam))
        pen[:off + int(free)] = 0.0
        return pen

    t0 = time.time()
    score, warm = {}, {}
    for lam in sorted(LAMS_TR, reverse=True):
        tot, nb = 0.0, 0
        for j in range(3):
            tr, te = inner != j, inner == j
            v, ok = RH._fit(X[tr], y01[tr], penvec(lam), bias, warm.get(j))
            warm[j] = v
            nb += (not ok)
            z = X[te] @ v[:X.shape[1]] + (v[X.shape[1]] if bias else 0.0)
            tot += float((np.logaddexp(0, z) - y01[te] * z).sum())
        score[lam] = tot
        print(f'    [{tag}] λ {lam:g}: 내부CV {tot / n:.5f}' + (f'  미수렴 {nb}' if nb else ''))
    lam = min(score, key=score.get)
    edge = ' ** 격자 끝 — 격자를 넓혀 재실행 권장' if lam in (min(LAMS_TR), max(LAMS_TR)) else ''
    v, ok = RH._fit(X, y01, penvec(lam), bias, warm.get(0))

    z1 = X @ v[:X.shape[1]] + (v[X.shape[1]] if bias else 0.0)
    ll1 = float(np.mean(np.logaddexp(0, z1) - y01 * z1))
    if off:
        # 진단 — s0 단독(스케일 1개짜리 재적합) 대비 β 가 in-sample 에서 먹은 양
        v0, _ = RH._fit(s0z[:, None], y01, np.zeros(1), bias)
        z0 = s0z * v0[0] + (v0[1] if bias else 0.0)
        ll0 = float(np.mean(np.logaddexp(0, z0) - y01 * z0))
        base_txt = f's0단독 {ll0:.4f} -> +β {ll1:.4f} (Δ {ll0 - ll1:+.4f}) | w_s0 {v[0]:+.3f}'
    else:
        pbar = float(y01.mean())
        ll0 = float(-(pbar * np.log(pbar) + (1 - pbar) * np.log(1 - pbar))) if bias \
            else float(np.log(2.0))
        base_txt = f'무정보 {ll0:.4f} -> β단독 {ll1:.4f} (Δ {ll0 - ll1:+.4f})'
    print(f'    [{tag}] 선택 λ {lam:g}{edge} | train in-sample ll: {base_txt} '
          f'(in-sample — 전이 보장 아님) | {time.time() - t0:.0f}s'
          + ('' if ok else ' ** 최종 미수렴'))
    # z_tr: 적합된 train 점수 — 라운드 k+1 이 이걸 새 앵커로 받는다 (함수공간 부스팅)
    return v[off:off + p], dict(mu=mu, sc=sc, lam=lam, z_tr=z1)


# ══════════════════════════════════════════════════════ v3 — 게이트·비선형 재료
def load_arith_train(ids_tr):
    """train 행의 Arith 짝피쳐(rankgauss). 게이트 **선택**에만 쓴다 — 팔의 게이트 값은
    stage2 쪽(load_repr 의 arith_Uz)이고, 선택이 train 에서 일어나므로 판정이 오염되지 않는다."""
    import pandas as pd
    cat = pd.read_csv(ARITH / 'feature_catalog.csv')
    feat = pd.read_parquet(ARITH / 'features_57k.parquet')
    pos = pd.Index(feat['id']).get_indexer(ids_tr)
    assert (pos >= 0).all(), 'train 피쳐 id 불일치'
    E, evn, _, _ = RH.parity_split_arith(feat, pos, cat)
    Uz = np.column_stack([FW.rankgauss(E[:, j]) for j in range(E.shape[1])])
    return Uz, evn


def pick_gates(Uz_tr, evn, fams, resid_tr, per_fam=2):
    """가족별 상위 per_fam 개 — train s0-잔차와의 |corr| 기준 (라벨 사용은 train 안에서만)."""
    r = resid_tr - resid_tr.mean()
    c = np.abs((Uz_tr - Uz_tr.mean(0)).T @ r) / np.clip(
        np.sqrt(((Uz_tr - Uz_tr.mean(0)) ** 2).sum(0) * (r ** 2).sum()), 1e-12, None)
    idx = []
    for f in sorted(set(fams)):
        js = [j for j in range(len(evn)) if fams[j] == f]
        idx += sorted(js, key=lambda j: -c[j])[:per_fam]
    return sorted(idx), c


def norm_pools(TR, PS2, keys, even, rows_tr):
    """소스 concat + train 통계 정규화(라벨 비접촉) — train/stage2 가 같은 좌표계를 받는다.
    짝은 중심화+표준화, 홀은 RMS-only (반대칭)."""
    Xt = np.hstack([TR[k] for k in keys])
    Xs = np.hstack([PS2[k] for k in keys])
    if even:
        mu = Xt[rows_tr].mean(0)
        sc = np.clip(Xt[rows_tr].std(0), 1e-9, None)
    else:
        mu = np.zeros(Xt.shape[1])
        sc = np.clip(np.sqrt((Xt[rows_tr] ** 2).mean(0)), 1e-9, None)
    return (Xt - mu) / sc, (Xs - mu) / sc


def rff_pair(Ut, Vt, Us, Vs, D, scales, seed, rows_tr):
    """패리티 보존 무작위 특징을 train/stage2 에 **같은 사영·같은 정규화**로 적용.
    nlfeat.rff 와 같은 항등식: 짝 cos(a)cos(s), 홀 −형 sin(a)sin(s) — 위상은 짝 사영에만."""
    rng = np.random.default_rng(seed)
    per = max(int(D) // len(scales), 1)
    Ev_t, Od_t, Ev_s, Od_s = [], [], [], []
    for c in scales:
        Wu = rng.standard_normal((Ut.shape[1], per)) / np.sqrt(Ut.shape[1])
        Wv = rng.standard_normal((Vt.shape[1], per)) / np.sqrt(Vt.shape[1])
        ph = rng.uniform(0, 2 * np.pi, per)
        at, st_ = Ut @ Wu, Vt @ Wv
        sa = np.clip(at[rows_tr].std(0), 1e-9, None)
        ss = np.clip(np.sqrt((st_[rows_tr] ** 2).mean(0)), 1e-9, None)
        at = c * at / sa + ph
        st_ = c * st_ / ss
        as_, ss2 = c * (Us @ Wu) / sa + ph, c * (Vs @ Wv) / ss
        Ev_t.append(np.cos(at) * np.cos(st_))
        Od_t.append(np.sin(at) * np.sin(st_))
        Ev_s.append(np.cos(as_) * np.cos(ss2))
        Od_s.append(np.sin(as_) * np.sin(ss2))
    return (np.hstack(Ev_t), np.hstack(Od_t)), (np.hstack(Ev_s), np.hstack(Od_s))


def svd_coords(Xt, Xs, k, rows_tr, center):
    """라벨-프리 train SVD 사영 — 쌍선형·MLP 의 공용 좌표. 홀 채널은 무중심(반대칭 보존)."""
    from sklearn.utils.extmath import randomized_svd
    mu = Xt[rows_tr].mean(0) if center else np.zeros(Xt.shape[1])
    _, _, Vt_ = randomized_svd((Xt[rows_tr] - mu).astype(np.float64), n_components=k,
                               random_state=SEED)
    W = Vt_.T
    Ct, Cs = (Xt - mu) @ W, (Xs - mu) @ W
    sc = np.clip(np.sqrt((Ct[rows_tr] ** 2).mean(0)), 1e-9, None)
    return Ct / sc, Cs / sc


def bilinear_feats(cu, cv, even, k):
    """SVD 좌표의 패리티 합법 곱 — 승부(홀): u_i·v_j / tie(짝): u_i·u_j, v_i·v_j (i<=j)."""
    if not even:
        return np.hstack([cu[:, i:i + 1] * cv[:, :k] for i in range(k)])
    cols = [cu[:, i:i + 1] * cu[:, j:j + 1] for i in range(k) for j in range(i, k)]
    cols += [cv[:, i:i + 1] * cv[:, j:j + 1] for i in range(k) for j in range(i, k)]
    return np.hstack(cols)


def fit_gat(s0, CUt, CVt, y01, even, seed, CUs, CVs, epochs=300, hidden=32,
            wd=1e-2, lr=3e-3, patience=30, es_frac=0.15, tag='GAT'):
    """게이티드 어텐션 헤드 — **concat 풀링 벡터 뒤에** 얹는다 (사용자 지시 v4.1).

        f(x) = w · ( tanh(W1 x) ⊙ σ(W2 x) )      x = [u, v] concat SVD 좌표
    σ(W2 x) 게이트가 행별로 은닉 유닛을 켜고 끈다 — MLP(순수 가산 은닉)와의 차이가
    정확히 이 곱셈 선별이고, MP 팔과의 paired 비교가 그 가치를 잰다.
    등변은 구조로: win 은 f([u,v]) − f([u,−v]) (홀), tie 는 합 (짝).
    readout w 0-초기화 -> epoch 0 = s0 단독 (앵커드-잔차 규율). ES 는 train 15%.
    **s0=None 이면 무앵커** — 로짓 전무, 풀링만 보고 y 를 처음부터 배운다 (w6/w7 대조).
    반환: (stage2 r, train r). 게이트 포화도(평균 |σ−0.5|)를 찍는다 —
    0 근처면 게이트가 아무것도 선별하지 않는다는 뜻이다."""
    import torch
    import torch.nn as nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    anchored = s0 is not None
    d = CUt.shape[1] + CVt.shape[1]
    W1 = nn.Linear(d, hidden).to(dev)
    W2 = nn.Linear(d, hidden).to(dev)
    w = nn.Linear(hidden, 1).to(dev)
    nn.init.zeros_(w.weight)
    nn.init.zeros_(w.bias)
    a = nn.Parameter(torch.tensor(1.0, device=dev))
    opt = torch.optim.AdamW(
        [{'params': list(W1.parameters()) + list(W2.parameters()) + list(w.parameters()),
          'weight_decay': wd},
         {'params': [a], 'weight_decay': 0.0}], lr=lr)
    T = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)
    s0z = (s0 / np.clip(np.sqrt((s0 ** 2).mean()), 1e-9, None)) if anchored \
        else np.zeros(len(y01))
    U_, V_, S_, Y_ = T(CUt), T(CVt), T(s0z), T(y01)
    lossf = nn.BCEWithLogitsLoss()

    def f(U, V):
        x = torch.cat([U, V], 1)
        return w(torch.tanh(W1(x)) * torch.sigmoid(W2(x))).squeeze(-1)

    def out(U, V):
        return f(U, V) + (1 if even else -1) * f(U, -V)

    rng = np.random.default_rng(seed)
    pos = rng.permutation(len(y01))
    n_es = max(int(len(y01) * es_frac), 200)
    es, tr = pos[:n_es], pos[n_es:]
    best, best_sd, bad, used = np.inf, None, 0, 0
    mods = (W1, W2, w)
    for ep in range(epochs):
        opt.zero_grad()
        lossf(a * S_[tr] + out(U_[tr], V_[tr]), Y_[tr]).backward()
        opt.step()
        with torch.no_grad():
            v = float(lossf(a * S_[es] + out(U_[es], V_[es]), Y_[es]))
        if v < best - 1e-6:
            best, bad = v, 0
            best_sd = ([{kk: t.detach().clone() for kk, t in m.state_dict().items()}
                        for m in mods], float(a.detach()))
            used = ep + 1
        else:
            bad += 1
            if bad >= patience:
                break
    if best_sd is not None:
        for m, sd in zip(mods, best_sd[0]):
            m.load_state_dict(sd)
    with torch.no_grad():
        r_tr = out(U_, V_).cpu().numpy().astype(np.float64)
        r_s2 = out(T(CUs), T(CVs)).cpu().numpy().astype(np.float64)
        gate = torch.sigmoid(W2(torch.cat([U_, V_], 1)))
        sat = float((gate - 0.5).abs().mean())
    a_txt = f'a(s0) {best_sd[1]:+.3f}' if anchored else 'a(s0) 없음(무앵커)'
    print(f'    [{tag}] ES 최적 epoch {used} | ES ll {best:.4f} | {a_txt} | '
          f'게이트 포화도 {sat:.3f} (0=무선별, 0.5=이진 선별)')
    return r_s2, r_tr


def fit_tf(TU_tr, TV_tr, y01, even, seed, TU_s2, TV_s2, causal=False, epochs=300,
           d_model=64, nhead=4, layers=2, ff=256, wd=1e-2, lr=1e-3, patience=30,
           es_frac=0.15, tag='TF'):
    """트랜스포머 r-생산자 (v4.4, 사용자 지시 — **풀링만·무앵커**, 인코더/디코더 통제쌍).
    토큰 i = 전문가 i 의 [u_i | v_i] SVD-64x2 좌표 -> Linear -> +전문가 식별 임베딩.

    causal=False (TE): 양방향 self-attention — 4토큰이 서로를 본다 (cross-형 헤드의 유비).
    causal=True  (TD): 인과 마스크만 추가 — 그 외 완전 동일(파라미터 수까지). 순서 없는
    전문가 집합(고정 순서 abe->tb3->a1->tb1)에 인과 제약이 지불하는 비용을 TD vs TE
    paired 가 직접 잰다. 디코더-only 는 인코더와 파라미터가 같다 — 차이는 마스크뿐.

    로짓 전무(무앵커) — 풀링만 보고 y 를 처음부터 배운다 (w6 형). readout 0-초기화.
    등변은 ±대칭화. 3-class 는 안 낸다 — 축 스칼라를 결합기가 소비, 확률은 축 합성 몫."""
    import torch
    import torch.nn as nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    S = len(TU_tr)
    pack = lambda U, V, sgn: torch.tensor(
        np.concatenate([np.stack(U, 1), sgn * np.stack(V, 1)], 2), dtype=torch.float32,
        device=dev)                                              # (n, S, ku+kv)
    Xp, Xn = pack(TU_tr, TV_tr, 1.0), pack(TU_tr, TV_tr, -1.0)
    emb = nn.Linear(Xp.shape[2], d_model).to(dev)
    pos = nn.Parameter(torch.randn(S, d_model, device=dev) * 0.02)
    enc = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model, nhead, ff, dropout=0.1, activation='gelu',
                                   batch_first=True), layers).to(dev)
    ro = nn.Linear(d_model, 1).to(dev)
    nn.init.zeros_(ro.weight)
    nn.init.zeros_(ro.bias)
    mask = torch.triu(torch.ones(S, S, device=dev, dtype=torch.bool), 1) if causal else None
    opt = torch.optim.AdamW(list(emb.parameters()) + [pos] + list(enc.parameters())
                            + list(ro.parameters()), lr=lr, weight_decay=wd)
    Y_ = torch.tensor(y01, dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss()

    def f(X):
        h = enc(emb(X) + pos, mask=mask)
        # 인과 모드는 마지막 토큰만 전 토큰을 봤으므로 마지막을 읽고, 양방향은 평균을 읽는다
        return ro(h[:, -1] if causal else h.mean(1)).squeeze(-1)

    def out(A, B, idx=None):
        i = slice(None) if idx is None else idx
        return f(A[i]) + (1 if even else -1) * f(B[i])

    perm = np.random.default_rng(seed).permutation(len(y01))
    n_es = max(int(len(y01) * es_frac), 200)
    es, tr = perm[:n_es], perm[n_es:]
    best, best_sd, bad, used = np.inf, None, 0, 0
    mods = (emb, enc, ro)
    for ep in range(epochs):
        for m in mods:
            m.train()
        opt.zero_grad()
        lossf(out(Xp, Xn, tr), Y_[tr]).backward()
        opt.step()
        for m in mods:
            m.eval()
        with torch.no_grad():
            v = float(lossf(out(Xp, Xn, es), Y_[es]))
        if v < best - 1e-6:
            best, bad = v, 0
            best_sd = ([{kk: t.detach().clone() for kk, t in m.state_dict().items()}
                        for m in mods], pos.detach().clone())
            used = ep + 1
        else:
            bad += 1
            if bad >= patience:
                break
    if best_sd is not None:
        for m, sd in zip(mods, best_sd[0]):
            m.load_state_dict(sd)
        with torch.no_grad():
            pos.copy_(best_sd[1])
    with torch.no_grad():
        r_tr = out(Xp, Xn).cpu().numpy().astype(np.float64)
        Sp, Sn = pack(TU_s2, TV_s2, 1.0), pack(TU_s2, TV_s2, -1.0)
        r_s2 = out(Sp, Sn).cpu().numpy().astype(np.float64)
    n_par = sum(p.numel() for m in mods for p in m.parameters()) + pos.numel()
    print(f'    [{tag}] ES 최적 epoch {used} | ES ll {best:.4f} | 무앵커(풀링만) | '
          f'{layers}층 d{d_model} h{nhead}{" 인과마스크" if causal else " 양방향"} | '
          f'파라미터 {n_par:,}')
    return r_s2, r_tr


def fit_mlp(s0, CUt, CVt, y01, even, seed, CUs, CVs, epochs=300, hidden=32,
            wd=1e-2, lr=3e-3, patience=30, es_frac=0.15, arch='mlp', tag='MLP'):
    """소형 등변 신경 r-생산자 — 출력 0-초기화라 epoch 0 = s0 단독 (앵커드-잔차 규율).
        win(홀): out = f([u,v]) − f([u,−v])   /   tie(짝): out = f([u,v]) + f([u,−v])
    arch='mlp': d -> hidden -> 1 (최소형)
    arch='ffn': **BERT-style FFN** (사용자 지시 v4.3) — d -> 4d -> GELU -> d -> GELU -> 1.
                newhead 팔 11/12 의 구조를 잔차 프로토콜로 이식한 것. 확률로 가는 길은
                이 스칼라를 결합기 로지스틱이 소비하는 것이다 (헤드가 직접 확률을 안 낸다).
    s0 스케일은 자유 스칼라. ES 는 train 안 15% (stage2 비접촉). 반환: stage2 의 r 열."""
    import torch
    import torch.nn as nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    d = CUt.shape[1] + CVt.shape[1]
    if arch == 'ffn':
        f = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.GELU(),
                          nn.Linear(d, 1))
    else:
        f = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1))
    nn.init.zeros_(f[-1].weight)
    nn.init.zeros_(f[-1].bias)
    f = f.to(dev)
    a = torch.nn.Parameter(torch.tensor(1.0, device=dev))
    opt = torch.optim.AdamW([{'params': f.parameters(), 'weight_decay': wd},
                             {'params': [a], 'weight_decay': 0.0}], lr=lr)
    s0z = s0 / np.clip(np.sqrt((s0 ** 2).mean()), 1e-9, None)
    T = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)
    rng = np.random.default_rng(seed)
    pos = rng.permutation(len(y01))
    n_es = max(int(len(y01) * es_frac), 200)
    es, tr = pos[:n_es], pos[n_es:]
    U_, V_, S_, Y_ = T(CUt), T(CVt), T(s0z), T(y01)
    lossf = nn.BCEWithLogitsLoss()

    def out(U, V):
        return (f(torch.cat([U, V], 1)) + (1 if even else -1) * f(torch.cat([U, -V], 1))).squeeze(-1)

    best, best_state, bad, used = np.inf, None, 0, 0
    for ep in range(epochs):
        f.train()
        opt.zero_grad()
        lossf(a * S_[tr] + out(U_[tr], V_[tr]), Y_[tr]).backward()
        opt.step()
        f.eval()
        with torch.no_grad():
            v = float(lossf(a * S_[es] + out(U_[es], V_[es]), Y_[es]))
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = ({k: t.detach().clone() for k, t in f.state_dict().items()},
                          float(a.detach()))
            used = ep + 1
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        f.load_state_dict(best_state[0])
    with torch.no_grad():
        r_tr = out(U_, V_).cpu().numpy().astype(np.float64)
        r_s2 = out(T(CUs), T(CVs)).cpu().numpy().astype(np.float64)
    n_par = sum(p.numel() for p in f.parameters())
    print(f'    [{tag}] ES 최적 epoch {used} | ES ll {best:.4f} | a(s0) {best_state[1]:+.3f} '
          f'| 파라미터 {n_par:,} wd {wd}')
    return r_s2, r_tr


# ══════════════════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--axes', default='tie,win', help='tie,win 중 실행할 축')
    ap.add_argument('--maxit', type=int, default=5000,
                    help='train β 적합의 L-BFGS 반복 상한 (기본 1000 -> 5000, 수렴 민감도 확인). '
                         'stage2 팔(crossfit)에는 적용하지 않는다 — 대조 조건 불변')
    ap.add_argument('--rff-d', type=int, default=2048, help='무작위 특징 수 (대역폭 4개에 분배)')
    args = ap.parse_args()
    axes = [s.strip() for s in args.axes.split(',') if s.strip()]
    t0 = time.time()

    A = R5.load_repr(single_best=True)     # 평가 기저 = 단일 최적 ckpt (앙상블 아님)
    y, nz, hold, cu, fold = A['y'], A['nz'], A['hold'], A['cu'], A['fold']
    n = len(y)

    ar = pick_arith(A['ids'], y)
    TR = load_train(ar)
    y_tr = TR['y']
    has_tb1 = TR['s_tb1'] is not None
    if not has_tb1:
        print('  [주의] tb1 train 로짓 없음 — tie 앵커에서 s_tb1 을 빼고 진행한다')
    S2 = stage2_single(A['ids'], y, has_tb1, ar)
    PS2 = load_pool_s2(A['ids'])
    print(f'train {len(y_tr):,}행 적재 완료 | tie 앵커 {"5" if has_tb1 else "4"}열 / 승부 앵커 3열 | '
          f'λ격자 {[f"{l:g}" for l in LAMS_TR]} | β maxit {args.maxit} (팔은 {RH.MAXIT} 유지)')
    print('  계약: 온도·스케일링·앙상블 전무 — 단일 ckpt·raw (probe-contract-single-ckpt)\n')

    # ── v3 공용 재료 (두 축 공유 — 전부 라벨 비접촉 통계) ──
    t_pre = time.time()
    UzT, evn = load_arith_train(TR['ids'])
    fams = list(A['arith_fam'])
    assert list(evn) == list(A['arith_evn']), '짝피쳐 열 순서 불일치 (train vs stage2)'
    all_tr = np.ones(len(y_tr), bool)
    Ut, Us = norm_pools(TR, PS2, ['abe_u', 'tb_u', 'a1_u', 'tb1_u'], True, all_tr)
    Vt, Vs = norm_pools(TR, PS2, ['abe_v', 'tb_v', 'a1_v', 'tb1_v'], False, all_tr)
    (EvT, OdT), (EvS, OdS) = rff_pair(Ut, Vt, Us, Vs, args.rff_d, RFF_SCALES, SEED, all_tr)
    CUt, CUs = svd_coords(Ut, Us, K_MLP, all_tr, True)
    CVt, CVs = svd_coords(Vt, Vs, K_MLP, all_tr, False)
    # Ut/Vt(전차원 concat)는 MPF(압축손실 측정 팔, v4.5)가 쓴다 — 삭제하지 않는다
    # 전문가별 SVD-64 좌표 — TF 토큰 + SV 팔(선형 압축손실 측정) 공용.
    # COORD[블록키] = (train, stage2). TF 토큰 순서 고정: abe -> tb3 -> a1 -> tb1
    # (TD 의 인과 마스크는 이 순서에 걸린다 — 순서가 임의라는 점 자체가 TD 의 측정 대상)
    COORD = {}
    for ku, kv in (('abe_u', 'abe_v'), ('tb_u', 'tb_v'), ('a1_u', 'a1_v'), ('tb1_u', 'tb1_v')):
        COORD[ku] = svd_coords(TR[ku], PS2[ku], K_TOK, all_tr, True)
        COORD[kv] = svd_coords(TR[kv], PS2[kv], K_TOK, all_tr, False)
    TU_t = [COORD[k][0] for k in ('abe_u', 'tb_u', 'a1_u', 'tb1_u')]
    TU_s = [COORD[k][1] for k in ('abe_u', 'tb_u', 'a1_u', 'tb1_u')]
    TV_t = [COORD[k][0] for k in ('abe_v', 'tb_v', 'a1_v', 'tb1_v')]
    TV_s = [COORD[k][1] for k in ('abe_v', 'tb_v', 'a1_v', 'tb1_v')]
    print(f'  공용 재료: 게이트 짝피쳐 {UzT.shape[1]} | RFF {EvT.shape[1]}열 (c={RFF_SCALES}) | '
          f'SVD 좌표 짝/홀 각 {K_MLP} (쌍선형 상위 {K_BL}, GAT·MLP·FFN 공용) | '
          f'전문가별 SVD-{K_TOK} (TF 토큰·SV 팔 공용) | MPF 는 전차원 {4096 * 2:,} '
          f' [{time.time() - t_pre:.0f}s]\n')

    print('  ── 팔 구성표 — 무엇을 무엇과 비교하는가 ' + '─' * 47)
    print('   R0  | 로짓만                          | 풀링·train 전혀 불사용 — 기준')
    print('   L   | 로짓 + 풀링1024열 stage2릿지    | train-β 불사용 — 대조군')
    print('   M1  | 로짓 + r×4                      | r = train-β(풀링 + s0로짓앵커), 소스당 1열 — 참조점')
    print('   SV  | 로짓 + r_sv×4                   | M1 동일구조·입력만 전문가별 SVD-64 — 선형 압축손실')
    print('   D2  | M1 + r2                         | 라운드2 train-β (같은 소스)')
    print('   D3  | M1 + r2 + r3                    | 라운드3')
    print('   F   | M1 + Arith짝피쳐⊙(로짓·r)       | FWLS 게이트 (피쳐 선택은 train 잔차 |corr|)')
    print('   A   | M1 + r_rff                      | train-β( RFF(풀링) + s0앵커 )')
    print('   BL  | M1 + r_bl                       | train-β( 쌍선형곱(풀링SVD16) + s0앵커 )')
    print('   MP  | M1 + r_mlp                      | MLP(풀링SVD128) + s0앵커 · 잔차 학습')
    print('   GT0 | M1 + r_gat0                     | 게이티드어텐션(풀링만 — 로짓 무관여, w6)')
    print('   GT  | M1 + r_gat                      | 게이티드어텐션(풀링 + s0로짓앵커, w7)')
    print('   FN  | M1 + r_ffn                      | BERT-FFN d→4d→d→GELU (풀링SVD + s0앵커)')
    print('   TE  | M1 + r_tfe                      | TF 인코더(전문가4토큰 양방향, 풀링만·무앵커)')
    print('   TD  | M1 + r_tfd                      | TF 디코더(인과마스크만 추가 — TE 와 통제쌍)')
    print('   MPF | M1 + r_mlpf                     | MP 동일구조·입력만 원본 8,192 전차원 — 압축손실·용량')
    print('   * 모든 r 계열은 train 46,729행에서 학습 — 로짓은 s0(동결 1방향)로만 관여')
    print('   * 판정: M1 은 vs L, 나머지는 vs M1 — holdout paired CI + 문턱 0.0005')
    print('  ' + '─' * 88 + '\n')

    tie_names = ['s_tie', 's_tb1', 't3', 't_a1', 't_ar'] if has_tb1 else \
                ['s_tie', 't3', 't_a1', 't_ar']
    # src[0] = 단일 소스 팔(L/B1/N1)의 소스 — 1차 런과 동일해 연속 비교가 된다
    AXES = {
        'tie': dict(y01=(y == 2).astype(np.float64), y01_tr=(y_tr == 2).astype(np.float64),
                    rows=cu, bias=True, even=True, base=A['T5'], names=tie_names,
                    src=['tb_u', 'abe_u', 'a1_u', 'tb1_u'], hno='1'),
        'win': dict(y01=(y == 0).astype(np.float64), y01_tr=(y_tr == 0).astype(np.float64),
                    rows=cu & nz, bias=False, even=False, base=A['M3'],
                    names=['m3', 'm_a1', 'm_ar'],
                    src=['a1_v', 'abe_v', 'tb_v', 'tb1_v'], hno='2'),
    }

    results = {}
    for ax in axes:
        C = AXES[ax]
        P = ax[0].upper()
        RH.head(C['hno'], f'{"tie축" if ax == "tie" else "승부축"} · clean {int(C["rows"].sum()):,}행'
                          f' — s0 동결 + train-β(소스 4) vs stage2 fold-릿지')

        cols_s2 = [S2[nm] for nm in C['names']]
        cols_tr = [TR[nm] for nm in C['names']]
        w, b = fit_anchor(cols_s2, C['y01'], C['rows'], C['bias'], C['names'])
        s0_tr = np.column_stack(cols_tr) @ w + b

        # 승부축의 train 행은 비-tie 만 — tie 행은 s0/β 적합 대상이 아니다 (축 정의)
        m_tr = np.ones(len(y_tr), bool) if ax == 'tie' else (y_tr != 2)

        s0m, y01m = s0_tr[m_tr], C['y01_tr'][m_tr]
        old_maxit = RH.MAXIT
        RH.MAXIT = args.maxit                      # β 적합만 상향 — 팔(crossfit)은 기본값

        # ── 라운드 1: 소스별 앵커 잔차 4개 ──
        R, ST = {}, {}
        for s in C['src']:
            beta, st = fit_beta(s0m, TR[s][m_tr], y01m, C['even'], C['bias'], f'{ax}·r1·{s}')
            R[s] = ((PS2[s] - st['mu']) / st['sc']) @ beta      # train 좌표계 그대로 적용
            ST[s] = st
        s1 = C['src'][0]

        # ── SV: M1 과 동일 구조·입력만 전문가별 SVD-64 (선형 압축손실 측정, v4.5) ──
        RSV = {}
        for s in C['src']:
            beta, st = fit_beta(s0m, COORD[s][0][m_tr], y01m, C['even'], C['bias'],
                                f'{ax}·sv{K_TOK}·{s}')
            RSV[s] = ((COORD[s][1] - st['mu']) / st['sc']) @ beta

        # ── 라운드 2·3 (강소스 s1) — 누적 train 점수가 새 앵커 (함수공간 부스팅) ──
        b2_, st2 = fit_beta(ST[s1]['z_tr'], TR[s1][m_tr], y01m, C['even'], C['bias'],
                            f'{ax}·r2·{s1}')
        r2 = ((PS2[s1] - st2['mu']) / st2['sc']) @ b2_
        b3_, st3 = fit_beta(st2['z_tr'], TR[s1][m_tr], y01m, C['even'], C['bias'],
                            f'{ax}·r3·{s1}')
        r3 = ((PS2[s1] - st3['mu']) / st3['sc']) @ b3_

        # ── FWLS 게이트 — 가족별 상위 2개 (train s0-잔차 |corr| 선택, stage2 판정 비오염) ──
        resid0 = s0_resid(s0m, y01m, C['bias'])
        gidx, _ = pick_gates(UzT[m_tr], evn, fams, resid0)
        gnames = [evn[j] for j in gidx]
        ggroups = [fams[j] for j in gidx]
        print(f'    게이트 {len(gidx)}개 (가족별<=2): '
              + ', '.join(gnames[:8]) + (' ...' if len(gidx) > 8 else ''))

        # ── 비선형 후보: RFF-β / 저계수 쌍선형-β / 소형 등변 MLP ──
        Ft, Fs = (EvT, EvS) if C['even'] else (OdT, OdS)
        b_rf, st_rf = fit_beta(s0m, Ft[m_tr], y01m, C['even'], C['bias'], f'{ax}·rff')
        r_rff = ((Fs - st_rf['mu']) / st_rf['sc']) @ b_rf
        BLt = bilinear_feats(CUt[:, :K_BL], CVt[:, :K_BL], C['even'], K_BL)
        BLs = bilinear_feats(CUs[:, :K_BL], CVs[:, :K_BL], C['even'], K_BL)
        b_bl, st_bl = fit_beta(s0m, BLt[m_tr], y01m, C['even'], C['bias'], f'{ax}·쌍선형')
        r_bl = ((BLs - st_bl['mu']) / st_bl['sc']) @ b_bl
        del BLt, BLs
        RH.MAXIT = old_maxit
        r_mlp, _ = fit_mlp(s0m, CUt[m_tr], CVt[m_tr], y01m, C['even'], SEED, CUs, CVs)
        r_gat, _ = fit_gat(s0m, CUt[m_tr], CVt[m_tr], y01m, C['even'], SEED, CUs, CVs,
                           tag='GAT·앵커(w7)')
        r_gat0, _ = fit_gat(None, CUt[m_tr], CVt[m_tr], y01m, C['even'], SEED, CUs, CVs,
                            tag='GAT·무앵커(w6)')
        r_ffn, _ = fit_mlp(s0m, CUt[m_tr], CVt[m_tr], y01m, C['even'], SEED, CUs, CVs,
                           arch='ffn', tag='FFN(BERT)')
        r_tfe, _ = fit_tf([c[m_tr] for c in TU_t], [c[m_tr] for c in TV_t], y01m,
                          C['even'], SEED, TU_s, TV_s, causal=False, tag='TF·인코더(양방향)')
        r_tfd, _ = fit_tf([c[m_tr] for c in TU_t], [c[m_tr] for c in TV_t], y01m,
                          C['even'], SEED, TU_s, TV_s, causal=True, tag='TF·디코더(인과)')
        r_mpf, _ = fit_mlp(s0m, Ut[m_tr], Vt[m_tr], y01m, C['even'], SEED, Us, Vs,
                           tag='MLP·전차원 8,192')

        rm = C['rows']
        hm = hold[rm]
        yb = C['y01'][rm] > 0.5

        # 기준 팔을 먼저 — 부호 진단의 잔차 기준이 된다
        r0 = FW.run_arm((f'{P}-R0', '기저 로짓만 — 기준', C['base'],
                         np.zeros((n, 0)), [], None), C['y01'], fold, rm, hold,
                        None, bias=C['bias'])
        print(f'  [{P}-R0] 기저 로짓만 — 기준  ({C["base"].shape[1]}열)')
        FW.arm_line(r0, t0)
        results[f'{P}-R0'] = r0
        base_ll = r0['ll']
        resid = np.full(n, np.nan)
        resid[rm] = C['y01'][rm] - expit(r0['z'][rm])

        # 부호·전이 진단 — r 이 기준 잔차와 맺는 상관의 fold 별 부호 일관성.
        #   5/5 같은 부호(음수 포함) -> 계통적 레짐 차이 = fold-적합 가중이 부호를 맞춰 쓸 수 있다.
        #   부호가 갈림 -> 그 소스의 전이는 불안정 = 노이즈로 읽는다.
        print('    부호·전이 진단 (stage2 clean · 무적합 · 기준 R0 잔차 기준):')
        for nm_, vec in (list(R.items())
                         + [('r2', r2), ('r3', r3), ('rff', r_rff), ('쌍선형', r_bl),
                            ('mlp', r_mlp), ('gat(w7)', r_gat), ('gat0(w6)', r_gat0),
                            ('ffn', r_ffn), ('tfe(풀링만)', r_tfe), ('tfd(풀링만)', r_tfd),
                            ('mlp전차원', r_mpf)]):
            cs = [float(np.corrcoef(vec[rm & (fold == k)], resid[rm & (fold == k)])[0, 1])
                  for k in range(K)]
            dom = np.sign(np.median(cs))
            same = sum(1 for c in cs if np.sign(c) == dom)
            print(f'      {nm_:<12} AUC {_auc(vec[rm], yb):.4f} | corr(잔차) '
                  f'{float(np.corrcoef(vec[rm], resid[rm])[0, 1]):+.3f} | fold '
                  + ' '.join(f'{c:+.2f}' for c in cs) + f' | 부호일관 {same}/5')

        # ── 팔 조립 ──
        def mk(parts):
            cols, blk, off = [], [], 0
            for nm, B in parts:
                cols.append(B)
                blk.append((nm, off, off + B.shape[1]))
                off += B.shape[1]
            return np.hstack(cols), blk

        R4 = [(f'r_{s}', R[s][:, None]) for s in C['src']]
        G_s2 = A['arith_Uz'][:, gidx]
        Lmat = np.hstack([C['base'], np.column_stack([R[s] for s in C['src']])])
        lnames = list(C['names']) + [f'r_{s}' for s in C['src']]
        Xg, gblk = FW.build_inter(G_s2, gnames, ggroups, Lmat, lnames)
        XgF, blkF = mk(R4)
        XgF = np.hstack([XgF, Xg])
        blkF = blkF + [(nm, s_ + 4, e_ + 4) for nm, s_, e_ in gblk]

        arms = [
            (f'{P}-L', f'로짓 + 풀링({s1}) 1,024열 stage2릿지 — 대조군 (train-β 불사용)',
             *mk([(s1, PS2[s1])]), None),
            (f'{P}-M1', '로짓 + r×4 (train-β: 풀링+s0로짓앵커) — 참조점', *mk(R4), None),
            (f'{P}-SV', f'로짓 + r_sv×4 — M1 동일구조·입력만 SVD-{K_TOK} (선형 압축손실 측정)',
             *mk([(f'sv_{s}', RSV[s][:, None]) for s in C['src']]), None),
            (f'{P}-D2', f'M1 + r2 — 라운드2 train-β (소스 {s1})',
             *mk(R4 + [('r2', r2[:, None])]), None),
            (f'{P}-D3', f'M1 + r2 + r3 — 라운드3 (소스 {s1})',
             *mk(R4 + [('r2', r2[:, None]), ('r3', r3[:, None])]), None),
            (f'{P}-F', f'M1 + Arith짝피쳐 {len(gidx)}개 ⊙ (로짓+r) — FWLS 게이트', XgF, blkF,
             G_s2 if C['bias'] else None),
            (f'{P}-A', 'M1 + r_rff — train-β( RFF(풀링) + s0앵커 )',
             *mk(R4 + [('r_rff', r_rff[:, None])]), None),
            (f'{P}-BL', 'M1 + r_bl — train-β( 쌍선형곱(풀링SVD16) + s0앵커 )',
             *mk(R4 + [('r_bl', r_bl[:, None])]), None),
            (f'{P}-MP', 'M1 + r_mlp — MLP(풀링SVD128)+s0앵커 잔차학습',
             *mk(R4 + [('r_mlp', r_mlp[:, None])]), None),
            (f'{P}-MPF', 'M1 + r_mlpf — MLP(풀링 원본 8,192 전차원)+s0앵커 (압축손실·용량 추세)',
             *mk(R4 + [('r_mlpf', r_mpf[:, None])]), None),
            (f'{P}-GT0', 'M1 + r_gat0 — 게이티드어텐션(풀링만, 로짓 무관여 — w6)',
             *mk(R4 + [('r_gat0', r_gat0[:, None])]), None),
            (f'{P}-GT', 'M1 + r_gat — 게이티드어텐션(풀링+s0로짓앵커 — w7)',
             *mk(R4 + [('r_gat', r_gat[:, None])]), None),
            (f'{P}-FN', 'M1 + r_ffn — BERT-FFN(풀링SVD, d→4d→d→GELU)+s0앵커',
             *mk(R4 + [('r_ffn', r_ffn[:, None])]), None),
            (f'{P}-TE', 'M1 + r_tfe — TF인코더(전문가4토큰 양방향, 풀링만·무앵커)',
             *mk(R4 + [('r_tfe', r_tfe[:, None])]), None),
            (f'{P}-TD', 'M1 + r_tfd — TF디코더(인과마스크, 그 외 TE 와 동일, 풀링만·무앵커)',
             *mk(R4 + [('r_tfd', r_tfd[:, None])]), None),
        ]
        lin_ll, m1_ll = None, None
        for code, label, Xi, blk, main in arms:
            print(f'  [{code}] {label}  ({C["base"].shape[1] + Xi.shape[1]}열) 적합 중 ...')
            res = FW.run_arm((code, label, C['base'], Xi, blk, main), C['y01'], fold,
                             rm, hold, base_ll, bias=C['bias'])
            FW.arm_line(res, t0)
            if code.endswith('-L'):
                lin_ll = res['ll']
            elif code.endswith('-M1'):
                m1_ll = res['ll']
                print(f"        vs 대조 L  전체 {RH.ci_str(*RH.paired_ci(lin_ll, res['ll']))}")
                print(f"        vs 대조 L  holdout "
                      f"{RH.ci_str(*RH.paired_ci(lin_ll[hm], res['ll'][hm]))}")
            else:
                print(f"        vs M1     전체 {RH.ci_str(*RH.paired_ci(m1_ll, res['ll']))}")
                print(f"        vs M1     holdout "
                      f"{RH.ci_str(*RH.paired_ci(m1_ll[hm], res['ll'][hm]))}")
            results[code] = res
        print()

    if all(c in results for c in ('T-R0', 'W-R0')):
        RH.head('3', '종합 — 총 ll = tie축 + (1−tie율)·승부축 (clean 우주)')
        nzf, nzf_h = float(nz[cu].mean()), float(nz[hold].mean())
        hw, ht = hold[cu & nz], hold[cu]
        rows_out = []
        for nm, wc, tc in [('R0  로짓만 — 기준', 'W-R0', 'T-R0'),
                           ('L   +풀링 stage2릿지 (train 무)', 'W-L', 'T-L'),
                           ('M1  +r×4 (풀링+s0앵커) — 참조점', 'W-M1', 'T-M1'),
                           ('SV  r_sv×4 (M1 의 SVD-64 판)', 'W-SV', 'T-SV'),
                           ('D2  +라운드2', 'W-D2', 'T-D2'),
                           ('D3  +라운드3', 'W-D3', 'T-D3'),
                           ('F   +Arith게이트⊙(로짓·r)', 'W-F', 'T-F'),
                           ('A   +RFF(풀링)+s0앵커', 'W-A', 'T-A'),
                           ('BL  +쌍선형(풀링)+s0앵커', 'W-BL', 'T-BL'),
                           ('MP  +MLP(풀링SVD128)+s0앵커', 'W-MP', 'T-MP'),
                           ('MPF +MLP(원본 8,192 전차원)+s0앵커', 'W-MPF', 'T-MPF'),
                           ('GT0 +게이티드어텐션(풀링만, w6)', 'W-GT0', 'T-GT0'),
                           ('GT  +게이티드어텐션(풀링+s0, w7)', 'W-GT', 'T-GT'),
                           ('FN  +BERT-FFN(풀링+s0)', 'W-FN', 'T-FN'),
                           ('TE  +TF인코더(양방향, 풀링만)', 'W-TE', 'T-TE'),
                           ('TD  +TF디코더(인과, 풀링만)', 'W-TD', 'T-TD')]:
            if wc not in results or tc not in results:
                continue
            lw, lt = results[wc]['ll'], results[tc]['ll']
            rows_out.append([nm, f'{float(lt.mean() + nzf * lw.mean()):.4f}',
                             f'{float(lt[ht].mean() + nzf_h * lw[hw].mean()):.4f}'])
        RH.table(['구성', '총 ll (clean)', 'holdout'], rows_out, ['l', 'r', 'r'])
        print('  판정 비교는 축별 SV/D2/D3/F/A/BL/MP/MPF/GT0/GT/FN/TE/TD vs M1 26개 — Bonferroni x26.')
        print('  문턱 0.0005. 핵심 쌍: SV vs M1 = 선형이 보는 SVD-64 압축손실 /')
        print('  MPF vs MP = 비선형이 보는 SVD-128 압축손실 + 전차원 확대의 추세 /')
        print('  GT0 vs GT: 앵커 유무 / GT vs MP: 게이팅 / FN vs MP: 용량 527k vs 8k /')
        print('  TE vs M1: 전문가끼리 서로 보기 / TD vs TE: 인과 마스크 비용. 판정은 사용자.')

    np.savez_compressed(BOOST / 'resfit_oof.npz', ids=A['ids'], y=y,
                        **{f'z__{c}': r['z'] for c, r in results.items()})
    print(f'\nsaved: {BOOST / "resfit_oof.npz"}')


if __name__ == '__main__':
    main()
