"""fit_heads.py — 통합 헤드(기둥·GAT·LIN·결합기) 학습. **자립 모듈**.

노트북을 읽지도 exec 하지도 않는다. 아래 코드는 `temp/nbtools/_gen_fit_heads.py` 가
노트북 §6·§7 셀에서 텍스트 그대로 떠 온 것이고, 옮기면서 바꾼 곳은 그 생성기의
SUBS 에 전부 적혀 있다 (V3 분할 원본 경로 / 재사용 게이트 제거 두 가지뿐).

무엇을 하나
    저장 자산(models/logits·models/pooled·ARITH)에서 stage2 우주 10,748 과 train 46,729 를
    읽어 헤드를 학습하고, holdout 5,748 의 **행별 손실**을 돌려준다.
    사다리 실험이 칸마다 이걸 다시 돌린다.

Arith 갈아끼우기
    Arith 는 GBDT 라 풀링 벡터가 없다. 통로는 로짓 2열뿐이다 — t_ar(짝) / m_ar(홀).
    그래서 SVD·풀링 8블록은 그대로 두고 그 2열만 바꾸면 되지만, 그 2열이 기둥 s0 앵커·
    GAT 자유입력·LIN 승부 3열·결합기 base 에 전부 들어가므로 헤드는 통째로 재학습한다.

메모리
    TR 풀링 8블록(46,729 x 1024 float64) 약 3.1GB + PS2 0.7GB + SVD 중간값. 24GB 권장.

사용
    python temp/fit_heads.py                    # 재현 런 (번들 저장 안 함)
    python temp/fit_heads.py --save             # temp/heads_repro 에 번들
    python temp/fit_heads.py --save --deploy    # final/heads 에 배포 번들

    import fit_heads as FH
    S = FH.HeadSession().prepare()      # 적재 + SVD (무겁고 1회)
    r = S.fit()                         # -> {'tot_h', 'row_hold', ...}
    S.set_arith(oof_train=..., s2=...); r2 = S.fit()
    S.drop_arith(); r0 = S.fit()
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.extmath import randomized_svd

# ══════════════════════════════════════════════ 경로·계약 상수
FINAL = Path(__file__).resolve().parent.parent      # 이 파일은 final/temp/
DATA_DIR = FINAL / 'origin'
MODELS = FINAL / 'models'
ARITH = MODELS / 'ARITH'
ARITH_STEP = 4000
ARITH_CKPT = ARITH / f'checkpoint-{ARITH_STEP}'

SEED = 42
HALF = 5000                     # stage2 우주 경계: [:HALF] valid / [HALF:] holdout
K, INNER = 5, 3                 # 결합기 교차적합 fold / 릿지 내부 CV fold
ARITH_MODELS = ('lgbm', 'xgb', 'cat')
ABE1_STEP, TB1_STEP = 3109, 5000
K_TIE_WANT = 512                # 셀 47 의 K_TIE 와 같아야 한다 (캐시 키용)
EVK = ('abe2_u', 'tb2_u', 'abe1_u', 'tb1_u')        # 짝 블록
ODK = ('abe2_v', 'tb2_v', 'abe1_v', 'tb1_v')        # 홀 블록

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HEADS = FINAL / 'temp/heads_repro'                  # fit() 이 덮어쓴다

# ══════════════════════════════════════════════ 캐시 (사다리 실험용)
# 사다리는 칸마다 헤드를 통째로 다시 학습한다. 그중 **Arith 와 무관한 부분**을 다시 낼
# 이유가 없다. 두 가지를 캐시한다.
#   ① SVD-512  풀링 8블록에서만 나온다 -> Arith 가 바뀌어도 불변. 통째로 저장.
#   ② λ 격자   기둥·LIN 의 릿지 λ 는 17개 x 3fold x (4+12) 회 = 816 적합이 든다.
#              기준선이 고른 λ 주변 ±LAM_SPAN 칸만 훑는다. 칸이 바뀌어도 입력의 1024열 중
#              **s0 한 열**만 달라지므로 λ 가 멀리 갈 이유가 없다.
# 캐시를 꺼도(USE_CACHE=False) 결과 계약은 같다 — 시간만 는다.
CACHE_DIR = FINAL / 'temp/head_cache'
LAM_CACHE = CACHE_DIR / 'lams.json'
# SVD 캐시는 **키를 파일명에 넣는다**. 한 파일에 쓰면 설정이 다른 런끼리 서로 밀어낸다
# (실측: factorial 이 tr0 -> tr+ 로 돌면서 tr0 캐시가 tr+ 것에 덮여 다음 런이 재계산했다).
# 269MB x 설정 수 만큼 temp/ 를 먹는다 — 필요 없으면 temp/head_cache/svd512-*.npz 를 지우면 된다.
USE_CACHE = True
LAM_REUSE = True
LAM_SPAN = 1                 # 기준선 λ 의 ±1 칸 (반데케이드 격자라 x/÷ 3.16)

# λ 힌트 네임스페이스. **실험 계열이 다르면 반드시 갈라야 한다.**
#   제거 실험(피쳐 사다리)은 s0 의 한 열만 조금 바뀌므로 같은 ns 에서 ±1 칸이면 충분하다.
#   대체 실험(Arith 를 다른 모델로)은 t_ar/m_ar 가 **다른 모델 계열**의 값이라 s0 가 크게
#   달라진다. 같은 ns 를 쓰면 캐시가 조용히 틀린 힌트를 준다 -> ns 를 갈라 전 격자부터 시작.
LAM_NS = 'base'

# 오염 3,000행을 헤드 TR 에 붙일지. (id 확인 2026-08-09)
#   그 3,000 = stage1 valid 중 V3(ABE1)이 **학습**한 행.
#   ABE1  : in-sample. 단 TR 46,729 도 이미 전부 ABE1 in-sample 이라 성격은 안 바뀐다.
#   ABE2·TB2·Arith : 그들에겐 valid(ES/ckpt 선택) 행 -> TR 에 넣으면 약한 선택 낙관이 생긴다.
#   TB1   : stage2 report-only, 선택에 미사용.
#   stage2 우주(10,748)와 결합기 적합 행(clean 7,748)은 **그대로**다. cu 가 계속 막는다.
#   즉 이 스위치는 "결합기 적합에 안 쓰이던 3,000 을 헤드 학습에만 재활용" 이다.
TR_PLUS_DIRTY = False

# 기둥 4열을 뺄지. 기둥 = 소스별 [s0 무벌점 고정 | 풀링 1024열 λ벌점] -> 1열.
#   GAT 는 같은 풀링의 SVD-512 를 이미 본다 -> **입력 공간이 겹친다.**
#   기둥만 가진 것은 `s0 앵커링` 하나다 — s0(로짓 기저 적합)를 고정하고 풀링이 잔차만
#   배우게 해서 ABE1 암기 방향을 s0 에 흡수시키는 구조. 그게 결합기에서 값을 하는지가 질문.
#   tie 축에만 들어가므로 승부축은 영향이 없다 (깨끗한 A/B).
#   비용: 기둥은 릿지 λ 적합 4소스 x 17λ x 3fold = 204 회. 빼면 그만큼 준다.
DROP_PILLAR = False

# SVD 캐시 키에 들어갈 풀링 파일. 글롭 + 실험이 추가로 등록한 파일.
# 그래프 임베딩처럼 pooled/ 밖에 있는 소스를 5번째 블록으로 쓰려면 여기 등록해야
# 캐시가 자동으로 무효화된다 (안 하면 옛 좌표계를 조용히 재사용한다).
EXTRA_POOL_FILES = []

# ══════════════════════════════════════════════ 진단 (λ 곡선 · 유효 랭크 · 결합기 가중)
# 실험을 닫지 않으려면 재료를 남겨야 한다. 로그를 파싱하지 않고도 다시 볼 수 있게 모은다.
#   λ 곡선     소스별 CV(λ) 전량 -> 곡선 깊이 = 그 풀링이 CV 에서 벌어준 양
#   유효 랭크  Σ σₖ²/(σₖ²+λ). 1024 차원 중 **실효로 몇 개가 살았나**.
#              블록별 차등 λ 나 소스 간 결합이 값을 할지 재기 전에 이걸 먼저 봐야 한다.
#              (릿지 df 공식은 선형 최소제곱 것이라, 로지스틱에서는 해 근방 IRLS 가중
#               그램 X'WX 의 고유값을 쓴다 — 국소 2차 근사다.)
DIAG = {}
DIAG_SPECTRUM = True             # 그램 고유값 계산. 체인당 수 초.


def _record_diag(tag, score, lam, X, y01, bias, n_free):
    """λ 곡선과 유효 랭크를 DIAG 에 남긴다. 선택 자체에는 영향이 없다."""
    d = {'grid': {float(k): float(v) for k, v in score.items()}, 'lam': float(lam),
         'n_free': int(n_free), 'n_pen': int(X.shape[1] - n_free), 'n_rows': int(X.shape[0])}
    g = d['grid']
    d['cv_min'] = min(g.values())
    d['cv_maxlam'] = g[max(g)]                 # λ 최대 = β≈0 = 풀링 미사용
    d['depth'] = d['cv_maxlam'] - d['cv_min']  # 곡선 깊이 = 풀링이 실제 벌어준 양
    if DIAG_SPECTRUM:
        try:
            P = np.ascontiguousarray(X[:, n_free:])
            w = 0.25                            # p(1-p) 상한. 해 근방 IRLS 가중의 보수적 대용
            ev = np.linalg.eigvalsh(P.T @ P * (w / len(P)))
            ev = np.clip(ev, 0, None)[::-1]
            lam_n = lam / len(P)
            shr = ev / (ev + lam_n)             # 방향별 축소율
            d['eff_rank'] = float(shr.sum())
            d['spec_top'] = [float(x) for x in ev[:8]]
            d['shr_q'] = [float(np.quantile(shr, q)) for q in (0.5, 0.9, 0.99)]
        except Exception as e:                  # noqa: BLE001
            d['eff_rank'] = None
            d['spec_err'] = f'{type(e).__name__}: {e}'
    DIAG[tag] = d


def save_diag(path):
    """DIAG 를 json 으로. 로그 파싱 없이 곡선·유효랭크를 다시 볼 수 있다."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(DIAG, ensure_ascii=False, indent=1), encoding='utf-8')
    return path

_LAMS_ALL = {}
if LAM_CACHE.exists():
    _LAMS_ALL = json.loads(LAM_CACHE.read_text(encoding='utf-8'))


def _svd_key():
    """캐시 유효성 키 — 풀링 파일의 (이름·크기·mtime) + 블록 이름 + 랭크."""
    import hashlib
    h = hashlib.sha256()
    for k in list(EVK) + list(ODK):
        h.update(k.encode())
    files = sorted((MODELS / 'pooled').glob('*.npz')) + [Path(f) for f in EXTRA_POOL_FILES]
    for f in sorted(set(files)):
        st = Path(f).stat()
        h.update(f'{Path(f).name}:{st.st_size}:{int(st.st_mtime)}'.encode())
    h.update(f'K_TIE={K_TIE_WANT}'.encode())
    h.update(f'TR_PLUS_DIRTY={TR_PLUS_DIRTY}'.encode())   # TR 이 바뀌면 SVD 기저도 바뀐다
    return h.hexdigest()[:16]


def _lams_for(tag):
    """이 tag 가 훑을 λ 격자. 같은 네임스페이스의 기준선 값이 있으면 그 주변만."""
    if not (USE_CACHE and LAM_REUSE):
        return LAMS_TR
    hint = _LAMS_ALL.get(LAM_NS, {}).get(tag)
    if hint is None:
        return LAMS_TR                       # 처음 보는 tag/ns -> 전 격자 (안전한 기본값)
    i = min(range(len(LAMS_TR)), key=lambda j: abs(np.log10(LAMS_TR[j]) - np.log10(hint)))
    lo, hi = max(0, i - LAM_SPAN), min(len(LAMS_TR), i + LAM_SPAN + 1)
    return LAMS_TR[lo:hi]


def _record_lam(tag, lam):
    if not USE_CACHE:
        return
    _LAMS_ALL.setdefault(LAM_NS, {})[tag] = float(lam)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LAM_CACHE.write_text(json.dumps(_LAMS_ALL, ensure_ascii=False, indent=1), encoding='utf-8')


def set_seeds(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# ══════════════ 노트북 셀 38 (정의) ══════════════
# ── 원본: Boost/reprhead.py — AST 발췌 (모듈 네임스페이스 표기만 직접 참조로) ──
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

# ══════════════ 노트북 셀 39 (정의) ══════════════
# ── 원본: Boost/fwls.py — AST 발췌 (모듈 네임스페이스 표기만 직접 참조로) ──
# 무보정 valid CE — ckpt 선택의 기준 (온도 피팅 없음)
def _raw_ll(z, y_idx):
    """무보정 valid CE — **온도 피팅 없음** (사용자 지시 2026-08-05: 선택 단계에 피팅이
    끼면 '부스터가 못 줄인 것'과 '보정이 흡수한 것'의 귀속이 섞인다). 로짓도 raw 그대로."""
    zs = z - z.max(1, keepdims=True)
    p = np.exp(zs)
    p /= p.sum(1, keepdims=True)
    return float(-np.log(np.clip(p[np.arange(len(y_idx)), y_idx], 1e-15, 1)).mean())

# ══════════════ 노트북 셀 41 (정의) ══════════════
# ── 원본: Boost/extract_logits.py — AST 발췌 (모듈 네임스페이스 표기·루트 상수명 LLM->ROOT 만 치환) ──
def _v3_split():
    """V3(TT_run) 분할 재현 — seed 42 두 번. 반환 (te 5,748, va2 2,000), label 열 포함.
    호출부는 반드시 export 라벨열과 전량 대조해 재현 정확성을 강제할 것."""
    from sklearn.model_selection import train_test_split
    # V3 는 ROOT/'V3/origin/train.csv' 를 썼는데 slim.py 가 그 트리를 지웠다.
    # origin/train_full.csv 가 같은 파일이다 (md5 대조 후 중복 제거).
    # train_test_split 은 행 수와 stratify 배열만 보고 순열을 만들므로 라벨 3열만
    # 읽어도 분할이 정확히 같다 — 176MB 를 다 읽지 않는다.
    full = pd.read_csv(DATA_DIR / 'train_full.csv',
                       usecols=['id', 'winner_model_a', 'winner_model_b'])
    assert len(full) == 57477, f'train_full.csv {len(full)}행 != 57,477'
    lab = np.select([full['winner_model_a'] == 1, full['winner_model_b'] == 1], [0, 1], 2)
    tr51, te = train_test_split(full.assign(label=lab), test_size=0.10, stratify=lab,
                                random_state=42, shuffle=True)
    _, va2 = train_test_split(tr51, test_size=2000, stratify=tr51['label'],
                              random_state=42, shuffle=True)
    return te, va2

# ══════════════ 노트북 셀 43 (정의) ══════════════
# ── 원본: Boost/resfit.py — AST 발췌 (모듈 네임스페이스 표기만 직접 참조로) ──
sd3 = lambda B: (B[:, 0] - B[:, 1], B[:, 2] - np.logaddexp(B[:, 0], B[:, 1]))


def tta(z, part):
    return ((z[f'{part}_fwd'] + z[f'{part}_swp']) / 2).astype(np.float64)


def pick_arith(ids, y):
    """앵커용 Arith 부스터 1개 — valid raw ll 선택. load_all(single_best=True) 와 같은
    규칙이라 평가 기저의 선택과 자동으로 일치한다 (둘 다 온도 피팅 없음)."""
    za = np.load(ARITH / 'outputs/arith_experts_raw.npz')
    assert (np.concatenate([za['ids_valid'], za['ids_hold']]) == ids).all(), 'arith id 불일치'
    lls = {}
    for m in ARITH_MODELS:
        L = np.vstack([za[f'raw_valid__{m}'], za[f'raw_hold__{m}']]).astype(np.float64)
        lls[m] = _raw_ll(L[:HALF], y[:HALF])
    best = min(lls, key=lls.get)
    print('  [anchor·arith] valid raw ll: '
          + '  '.join(f'{m}:{lls[m]:.4f}' for m in ARITH_MODELS) + f'  -> 선택 {best}')
    return best


def load_train(ar):
    """train 46,729 로짓·풀링·라벨. 모든 파일을 abe2 train npz 의 id 순서로 전량 대조한다.
    ar: 앵커에 쓸 Arith 부스터 (pick_arith 선택)."""
    za = np.load(MODELS / 'logits/abe2__checkpoint-3000__train.npz')
    ids, y = za['train_ids'], za['train_labels']
    assert len(ids) == 46729 and len(np.unique(ids)) == len(ids)

    def get(rel, n_class, need=True):
        p = MODELS / rel
        if not p.exists():
            assert not need, f'{p} 없음 — 선행 추출 필요 (헤더 참고)'
            return None
        z = np.load(p)
        assert (z['train_ids'] == ids).all() and (z['train_labels'] == y).all(), f'{p.name} 정렬'
        L = tta(z, 'train')
        assert L.shape == (len(ids), n_class) and np.isfinite(L).all(), f'{p.name} 형상/비유한'
        return L

    L_abe2 = tta(za, 'train')
    L_tb2 = get('logits/tb2__checkpoint-4000__train.npz', 2)
    L_abe1 = get(f'logits/abe1__checkpoint-{ABE1_STEP}__train.npz', 3)
    L_tb1 = get(f'logits/tb1__checkpoint-{TB1_STEP}__train.npz', 2, need=False)

    zar = np.load(ARITH / 'outputs/arith_experts_raw.npz')
    assert (zar['ids_train'] == ids).all() and (zar['y_train'] == y).all(), 'arith train 정렬'
    L_ar = zar[f'raw_oof_train__{ar}'].astype(np.float64)

    m_abe2, t_abe2 = sd3(L_abe2)
    m_ar, t_ar = sd3(L_ar)
    m_abe1, t_abe1 = sd3(L_abe1)
    D = dict(ids=ids, y=y, m_abe2=m_abe2, t_abe2=t_abe2, m_ar=m_ar, t_ar=t_ar, m_abe1=m_abe1, t_abe1=t_abe1,
             s_tb2=L_tb2[:, 1] - L_tb2[:, 0],
             s_tb1=(L_tb1[:, 1] - L_tb1[:, 0]) if L_tb1 is not None else None)

    # 풀링 — 4소스 8블록. 짝 (x+y)/2, 홀 (x−y)/2 (swap = 두 뷰/브랜치 교환 항등식)
    def pool(rel, ka, kb):
        z = np.load(MODELS / rel)
        assert (z['train_ids'] == ids).all(), f'{rel} 정렬'
        return z[f'train_{ka}'].astype(np.float64), z[f'train_{kb}'].astype(np.float64)

    zp = np.load(MODELS / 'pooled/tb2__checkpoint-4000__train.npz')
    assert (zp['train_ids'] == ids).all(), 'tb2 train 풀링 정렬'
    g = lambda k: zp[f'train_{k}'].astype(np.float64)
    A_rep, B_rep = (g('a_f') + g('b_s')) / 2, (g('b_f') + g('a_s')) / 2
    D['tb2_u'], D['tb2_v'] = (A_rep + B_rep) / 2, (A_rep - B_rep) / 2

    ea, eb = pool('pooled/abe2__checkpoint-3000__train.npz', 'ea', 'eb')
    D['abe2_u'], D['abe2_v'] = (ea + eb) / 2, (ea - eb) / 2
    qa, qb = pool(f'pooled/abe1__checkpoint-{ABE1_STEP}__train.npz', 'ea', 'eb')
    D['abe1_u'], D['abe1_v'] = (qa + qb) / 2, (qa - qb) / 2
    xf, xs = pool(f'pooled/tb1__checkpoint-{TB1_STEP}__train.npz', 'x_f', 'x_s')
    D['tb1_u'], D['tb1_v'] = (xf + xs) / 2, (xf - xs) / 2
    for k in ('tb2_u', 'tb2_v', 'abe2_u', 'abe2_v', 'abe1_u', 'abe1_v', 'tb1_u', 'tb1_v'):
        assert np.isfinite(D[k]).all(), f'train {k} 비유한값'
    return D


def load_pool_s2(ids):
    """stage2 풀링 8블록 — 전부 **자체 bf16 추출본**. train 과 같은 추출 경로라 β 좌표계가
    양쪽에서 동일하다 (1차 런의 a1 은 V3 fp16 임베딩 매핑 — 부호 반전 후보 원인이라 교체)."""
    def get(rel):
        z = np.load(MODELS / rel)
        assert (np.concatenate([z['valid_ids'], z['hold_ids']]) == ids).all(), f'{rel} id'
        return z
    cat = lambda z, k: np.vstack([z[f'valid_{k}'], z[f'hold_{k}']]).astype(np.float64)
    za = get('pooled/abe2__checkpoint-3000.npz')
    ea, eb = cat(za, 'ea'), cat(za, 'eb')
    zt = get('pooled/tb2__checkpoint-4000.npz')
    A_rep = (cat(zt, 'a_f') + cat(zt, 'b_s')) / 2
    B_rep = (cat(zt, 'b_f') + cat(zt, 'a_s')) / 2
    zq = get(f'pooled/abe1__checkpoint-{ABE1_STEP}.npz')
    qa, qb = cat(zq, 'ea'), cat(zq, 'eb')
    z1 = get(f'pooled/tb1__checkpoint-{TB1_STEP}.npz')
    xf, xs = cat(z1, 'x_f'), cat(z1, 'x_s')
    P = dict(abe2_u=(ea + eb) / 2, abe2_v=(ea - eb) / 2,
             tb2_u=(A_rep + B_rep) / 2, tb2_v=(A_rep - B_rep) / 2,
             abe1_u=(qa + qb) / 2, abe1_v=(qa - qb) / 2,
             tb1_u=(xf + xs) / 2, tb1_v=(xf - xs) / 2)
    for k, v in P.items():
        assert np.isfinite(v).all(), f'stage2 풀링 {k} 비유한값'
    return P


def stage2_single(ids, y, has_tb1, ar):
    """s0 앵커 적합용 — train 계약과 같은 단일-ckpt·단일-부스터 열을 stage2 에서 구성한다."""
    def get(rel, n_class):
        z = np.load(MODELS / rel)
        assert (np.concatenate([z['valid_ids'], z['hold_ids']]) == ids).all(), f'{rel} id'
        L = np.vstack([(z['valid_fwd'] + z['valid_swp']) / 2,
                       (z['hold_fwd'] + z['hold_swp']) / 2]).astype(np.float64)
        assert L.shape == (len(ids), n_class)
        return L

    L_abe2 = get('logits/abe2__checkpoint-3000.npz', 3)
    L_tb2 = get('logits/tb2__checkpoint-4000.npz', 2)
    L_abe1 = get(f'logits/abe1__checkpoint-{ABE1_STEP}.npz', 3)
    za = np.load(ARITH / 'outputs/arith_experts_raw.npz')
    L_ar = np.vstack([za[f'raw_valid__{ar}'], za[f'raw_hold__{ar}']]).astype(np.float64)
    m_abe2, t_abe2 = sd3(L_abe2)
    m_ar, t_ar = sd3(L_ar)
    m_abe1, t_abe1 = sd3(L_abe1)
    D = dict(m_abe2=m_abe2, t_abe2=t_abe2, m_ar=m_ar, t_ar=t_ar, m_abe1=m_abe1, t_abe1=t_abe1,
             s_tb2=L_tb2[:, 1] - L_tb2[:, 0], s_tb1=None)
    if has_tb1:
        L_tb1 = get(f'logits/tb1__checkpoint-{TB1_STEP}.npz', 2)
        D['s_tb1'] = L_tb1[:, 1] - L_tb1[:, 0]
    return D

# ══════════════ 노트북 셀 51 (정의) ══════════════
LR, EPOCHS, PATIENCE, WD = 3e-4, 3000, 200, 1e-2      # 저속·장기 + ES(patience 200)
H_GAT, RD_CAP, RD_STOP = 128, 8, 0.0003               # 폭 128 / 라운드 상한 8 / 자기종결 문턱

def _es_split(n_, seed, frac=0.15):
    # 헤드 ES 검증 분할 — train 의 15%
    perm = np.random.default_rng(seed).permutation(n_)
    n_es = max(int(n_ * frac), 200)
    return perm[:n_es], perm[n_es:]

class GATNet(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.W1, self.W2, self.w = nn.Linear(d, h), nn.Linear(d, h), nn.Linear(h, 1)
    def forward(self, x):
        return self.w(torch.tanh(self.W1(x)) * torch.sigmoid(self.W2(x)))

def fit_gat(CUt_, CVt_, y01, s0, even, seed, CUs_, CVs_, hidden=H_GAT):
    torch.manual_seed(seed)
    d = CUt_.shape[1] + CVt_.shape[1]
    f = GATNet(d, hidden); nn.init.zeros_(f.w.weight); nn.init.zeros_(f.w.bias); f = f.to(DEVICE)
    anchored = s0 is not None
    a = nn.Parameter(torch.tensor(1.0, device=DEVICE))
    opt = torch.optim.AdamW([{'params': f.parameters(), 'weight_decay': WD},
                             {'params': [a], 'weight_decay': 0.0}], lr=LR)
    s0z = (s0 / np.clip(np.sqrt((s0 ** 2).mean()), 1e-9, None)) if anchored else np.zeros(len(y01))
    T = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device=DEVICE)
    U_, V_, S_, Y_ = T(CUt_), T(CVt_), T(s0z), T(y01)
    lossf = nn.BCEWithLogitsLoss(); sgn = 1 if even else -1
    def out(idx):
        xp = torch.cat([U_[idx], V_[idx]], 1); xn = torch.cat([U_[idx], -V_[idx]], 1)
        return (f(xp) + sgn * f(xn)).squeeze(-1)
    es, tr = _es_split(len(y01), seed)
    best, best_sd, bad, used = np.inf, None, 0, 0
    for ep in range(EPOCHS):
        f.train(); opt.zero_grad(); lossf(a * S_[tr] + out(tr), Y_[tr]).backward(); opt.step()
        f.eval()
        with torch.no_grad():
            v = float(lossf(a * S_[es] + out(es), Y_[es]))
        if v < best - 1e-6:
            best, bad, used = v, 0, ep + 1
            best_sd = ({k2: t.detach().clone() for k2, t in f.state_dict().items()}, float(a.detach()))
        elif (bad := bad + 1) >= PATIENCE:
            break
    f.load_state_dict(best_sd[0])
    with torch.no_grad():
        r_tr = out(np.arange(len(y01))).cpu().numpy().astype(np.float64)
        Usx, Vsx = T(CUs_), T(CVs_)
        r_s2 = (f(torch.cat([Usx, Vsx], 1)) + sgn * f(torch.cat([Usx, -Vsx], 1))).squeeze(-1).cpu().numpy().astype(np.float64)
    z_tr = best_sd[1] * s0z + r_tr if anchored else r_tr
    return r_s2, dict(z_tr=z_tr, es_ll=best, es_epoch=used,
                      sd={k2: t.cpu() for k2, t in best_sd[0].items()})

def apply_gat(sd, CU_, CV_, even):
    # 배포 추론 — 저장된 state_dict 로 test 행에 같은 함수를 태운다
    f = GATNet(CU_.shape[1] + CV_.shape[1], H_GAT); f.load_state_dict(sd); f = f.to(DEVICE).eval()
    sgn = 1 if even else -1
    with torch.no_grad():
        U_ = torch.tensor(CU_, dtype=torch.float32, device=DEVICE)
        V_ = torch.tensor(CV_, dtype=torch.float32, device=DEVICE)
        r = (f(torch.cat([U_, V_], 1)) + sgn * f(torch.cat([U_, -V_], 1))).squeeze(-1)
    return r.cpu().numpy().astype(np.float64)
print('GAT 정의 완료')

# ══════════════ 노트북 셀 53 (정의) ══════════════
# ── 원본: Boost/resfit.py — AST 발췌 (모듈 네임스페이스 표기만 직접 참조로) ──
# 하한을 1 -> 10 으로 올렸다. λ<10 은 어느 체인에서도 선택된 적이 없고(곡선이 그쪽에서
# 단조 상승), 46,729행 x 1,025파라미터를 거의 무벌점으로 풀다 보니 L-BFGS 미수렴만
# 냈다 (기준선 런 미수렴 18줄 전부 λ 1·3.16). 결과 불변, 적합 2/17 절감.
LAMS_TR = [float(10 ** (e / 2)) for e in range(2, 17)]   # 1e1…1e8 반데케이드 15칸

def fit_anchor(cols, y01, rows, bias, names):
    """stage2 clean 행 단일 적합 — 로짓 열만, 무벌점, 표준화 없음(열 3~5개, raw 좌표 유지).
    반환 (w, b). 이 좌표계 그대로 train 열에 내적하면 s0 가 된다."""
    X = np.column_stack(cols)
    v, ok = _fit(X[rows], y01[rows], np.zeros(X.shape[1]), bias)
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

    정규화는 전 행 통계(라벨 비접촉)로 한 번만 한다 — crossfit 처럼 fold 별 재표준화까지는
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
    _grid = _lams_for(tag)                    # 캐시가 있으면 기준선 λ 주변만
    for lam in sorted(_grid, reverse=True):
        tot, nb = 0.0, 0
        for j in range(3):
            tr, te = inner != j, inner == j
            v, ok = _fit(X[tr], y01[tr], penvec(lam), bias, warm.get(j))
            warm[j] = v
            nb += (not ok)
            z = X[te] @ v[:X.shape[1]] + (v[X.shape[1]] if bias else 0.0)
            tot += float((np.logaddexp(0, z) - y01[te] * z).sum())
        score[lam] = tot
        print(f'    [{tag}] λ {lam:g}: 내부CV {tot / n:.5f}' + (f'  미수렴 {nb}' if nb else ''))
    lam = min(score, key=score.get)
    _record_lam(tag, lam)
    _record_diag(tag, score, lam, X, y01, bias, off + int(free))
    edge = ' ** 격자 끝 — 격자를 넓혀 재실행 권장' if lam in (min(_grid), max(_grid)) else ''
    v, ok = _fit(X, y01, penvec(lam), bias, warm.get(0))

    z1 = X @ v[:X.shape[1]] + (v[X.shape[1]] if bias else 0.0)
    ll1 = float(np.mean(np.logaddexp(0, z1) - y01 * z1))
    if off:
        # 진단 — s0 단독(스케일 1개짜리 재적합) 대비 β 가 in-sample 에서 먹은 양
        v0, _ = _fit(s0z[:, None], y01, np.zeros(1), bias)
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

# ══════════════ 노트북 셀 55 (정의) ══════════════
# ── 원본: Boost/reprhead.py — AST 발췌 (모듈 네임스페이스 표기만 직접 참조로) ──
LAMS = [1.0, 10.0, 100.0, 1000.0, 10000.0]   # 결합기 λ 격자 (아래 crossfit 이 위로 자동확장)

MAXIT = 1000              # 람다 경로 웜스타트가 있어 실제로는 훨씬 적게 쓴다


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

# ══════════════ 노트북 셀 45 -> _build_universe() ══════════════
def _build_universe():
    """stage2 우주 — 로짓·풀링 적재, clean 마스크, fold"""
    global PS2, S2, TR, _, ar, cu
    global fold, hold, ids, k_, n, nz
    global te_, te_v3, va2, vidx, y, y_tr
    global za_
    za_ = np.load(MODELS / 'logits/abe2__checkpoint-3000.npz')
    ids = np.concatenate([za_['valid_ids'], za_['hold_ids']])
    y = np.concatenate([za_['valid_labels'], za_['hold_labels']]).astype(np.int64)
    n = len(y)
    assert n == 10748 and len(np.unique(ids)) == n
    hold = np.arange(n) >= HALF
    nz = y != 2

    # clean 마스크 — V3 분할 재현: valid 행 중 V3 의 valid(va2)에 있던 2,000행만 clean
    te_v3, va2 = _v3_split()
    vidx = pd.Index(va2['id']).get_indexer(ids[:HALF])
    cu = np.ones(n, bool)
    cu[:HALF] = vidx >= 0
    assert int(cu.sum()) == 7748, f'clean {int(cu.sum())} != 7748'
    assert (pd.Index(te_v3['id']).get_indexer(ids[HALF:]) >= 0).all(), 'holdout 이 V3 test 밖'

    fold = np.zeros(n, int)
    for k_, (_, te_) in enumerate(StratifiedKFold(K, shuffle=True, random_state=SEED)
                                  .split(np.zeros((n, 1)), y)):
        fold[te_] = k_

    ar = pick_arith(ids, y)                      # Arith 단일 최적 부스터 (valid raw ll)
    TR = load_train(ar); y_tr = TR['y']          # train 46,729 — 로짓·풀링·라벨
    S2 = stage2_single(ids, y, TR['s_tb1'] is not None, ar)   # stage2 로짓 열 (3000/4000 계약)
    PS2 = load_pool_s2(ids)                      # stage2 풀링 8블록
    print(f'stage2 {n:,}행 (clean {int(cu.sum()):,} / holdout {int(hold.sum()):,}) '
          f'| train {len(y_tr):,}행')


# ── 셀 47 내부 정의를 최상위로 끌어올림 (래퍼를 건너뛰어도 살아 있어야 한다) ──
def zstats(X, even):
    # 짝: 평균/표준편차, 홀: 무중심 RMS (반대칭 보존)
    mu = X.mean(0) if even else np.zeros(X.shape[1])
    sc = np.clip(X.std(0) if even else np.sqrt((X ** 2).mean(0)), 1e-9, None)
    return mu, sc

def norm_keep(keys, even):
    Xt = np.hstack([TR[k] for k in keys]); Xs = np.hstack([PS2[k] for k in keys])
    mu, sc = zstats(Xt, even)
    return (Xt - mu) / sc, (Xs - mu) / sc, dict(mu=mu, sc=sc)

def svd_keep(Xt, Xs, k, center):
    mu = Xt.mean(0) if center else np.zeros(Xt.shape[1])
    _, _, Vt_ = randomized_svd((Xt - mu).astype(np.float64), n_components=k, random_state=SEED)
    W = Vt_.T
    Ct, Cs = (Xt - mu) @ W, (Xs - mu) @ W
    sc = np.clip(np.sqrt((Ct ** 2).mean(0)), 1e-9, None)     # 화이트닝 (좌표별 RMS 1)
    return (Ct / sc).astype(np.float32), (Cs / sc).astype(np.float32), dict(mu=mu, W=W, sc=sc)


# ══════════════ 노트북 셀 47 -> _build_svd() ══════════════
def _build_svd():
    """패리티 정규화 + 통합 SVD-512 (Arith 무관, 1회)"""
    global CUs, CUt, CVs, CVt, K_TIE, NU
    global NV, SVD_U, SVD_V, Us, Ut, Vs
    global Vt, norm_keep, svd_keep, zstats
    K_TIE = 512                                  # tie 헤드 입력 랭크 (최종 스윕 승자)

    def zstats(X, even):
        # 짝: 평균/표준편차, 홀: 무중심 RMS (반대칭 보존)
        mu = X.mean(0) if even else np.zeros(X.shape[1])
        sc = np.clip(X.std(0) if even else np.sqrt((X ** 2).mean(0)), 1e-9, None)
        return mu, sc

    def norm_keep(keys, even):
        Xt = np.hstack([TR[k] for k in keys]); Xs = np.hstack([PS2[k] for k in keys])
        mu, sc = zstats(Xt, even)
        return (Xt - mu) / sc, (Xs - mu) / sc, dict(mu=mu, sc=sc)

    def svd_keep(Xt, Xs, k, center):
        mu = Xt.mean(0) if center else np.zeros(Xt.shape[1])
        _, _, Vt_ = randomized_svd((Xt - mu).astype(np.float64), n_components=k, random_state=SEED)
        W = Vt_.T
        Ct, Cs = (Xt - mu) @ W, (Xs - mu) @ W
        sc = np.clip(np.sqrt((Ct ** 2).mean(0)), 1e-9, None)     # 화이트닝 (좌표별 RMS 1)
        return (Ct / sc).astype(np.float32), (Cs / sc).astype(np.float32), dict(mu=mu, W=W, sc=sc)

    Ut, Us, NU = norm_keep(list(EVK), True)
    Vt, Vs, NV = norm_keep(list(ODK), False)
    CUt, CUs, SVD_U = svd_keep(Ut, Us, K_TIE, True)
    CVt, CVs, SVD_V = svd_keep(Vt, Vs, K_TIE, False)
    del Ut, Us, Vt, Vs
    gc.collect(); print('SVD-512 완료', CUt.shape, CVt.shape)


# ══════════════ 노트북 셀 49 -> _build_axes() ══════════════
def _build_axes(drop_arith=False):
    """로짓 기저·축 재료 (Arith 갈아끼우면 여기부터 다시)"""
    global AXCFG, Cg, Ms, _, ax, base_s2
    global m_tr, mu_u, sc_u, sc_v, tie_names, tie_s2
    global tie_tr, win_names, win_s2, win_tr
    _drop = ['t_ar'] if drop_arith else []
    _dropw = ['m_ar'] if drop_arith else []
    tie_names = [c for c in ['s_tb2', 's_tb1', 't_abe2', 't_abe1', 't_ar'] if c not in _drop]
    win_names = [c for c in ['m_abe2', 'm_abe1', 'm_ar'] if c not in _dropw]
    AXCFG = {
        'tie': dict(y01=(y == 2).astype(np.float64), y01_tr=(y_tr == 2).astype(np.float64),
                    rows=cu, bias=True, even=True, names=tie_names,
                    src=['tb2_u', 'abe2_u', 'abe1_u', 'tb1_u']),
        'win': dict(y01=(y == 0).astype(np.float64), y01_tr=(y_tr == 0).astype(np.float64),
                    rows=cu & nz, bias=False, even=False, names=win_names,
                    src=['abe1_v', 'abe2_v', 'tb2_v', 'tb1_v']),
    }
    tie_tr = np.column_stack([TR[nm] for nm in tie_names]); tie_s2 = np.column_stack([S2[nm] for nm in tie_names])
    win_tr = np.column_stack([TR[nm] for nm in win_names]); win_s2 = np.column_stack([S2[nm] for nm in win_names])
    mu_u, sc_u = zstats(tie_tr, True); _, sc_v = zstats(win_tr, False)
    Ms = {}
    for ax in ('tie', 'win'):
        Cg = AXCFG[ax]
        m_tr = np.ones(len(y_tr), bool) if ax == 'tie' else (y_tr != 2)
        base_s2 = np.column_stack([S2[nm] for nm in Cg['names']])    # 통일 base = S2 열
        Ms[ax] = dict(ax=ax, even=Cg['even'], bias=Cg['bias'], y01=Cg['y01'], rows=Cg['rows'],
                      fold=fold, hold=hold, base=base_s2, src=Cg['src'],
                      y01m=Cg['y01_tr'][m_tr], m_tr=m_tr, nl=len(Cg['names']),
                      Lb_tr=np.column_stack([TR[nm] for nm in Cg['names']]), Lb_s2=base_s2,
                      Lu_tr=(tie_tr - mu_u) / sc_u, Lu_s2=(tie_s2 - mu_u) / sc_u,
                      Lv_tr=win_tr / sc_v, Lv_s2=win_s2 / sc_v)
    print('축 재료 완료 — tie 적합', int(Ms['tie']['rows'].sum()), '/ win 적합', int(Ms['win']['rows'].sum()))


# ══════════════ 노트북 셀 60 -> _fit_pillar() ══════════════
def _fit_pillar():
    """[tie] 기둥 — 소스별 앵커드 릿지 4열"""
    global Mt, PILLAR, PIL_DEPLOY, _sk, b_anc, beta
    global i, s, s0_tr, st, t0, w_anc
    t0 = time.time()
    Mt = Ms['tie']
    if True:
        w_anc, b_anc = fit_anchor([S2[nm] for nm in tie_names], Mt['y01'], Mt['rows'], Mt['bias'], tie_names)
        s0_tr = np.column_stack([TR[nm] for nm in tie_names]) @ w_anc + b_anc
        PILLAR, PIL_DEPLOY = [], {}
        for i, s in enumerate(() if DROP_PILLAR else Mt['src']):
            beta, st = fit_beta(s0_tr, TR[s], Mt['y01m'], True, True, f'기둥·{s}')
            PILLAR.append((f'p{i}', ((PS2[s] - st['mu']) / st['sc']) @ beta))
            PIL_DEPLOY[s] = dict(beta=beta, mu=st['mu'], sc=st['sc'])
        if DROP_PILLAR:
            PILLAR = None            # combiner 가 None 을 '기둥 없음' 으로 읽는다
            _sk = len(Mt['src']) * len(LAMS_TR) * 3
            print(f'기둥 생략 (DROP_PILLAR) — 릿지 λ 적합 {_sk}회 절감  '
                  f'[{(time.time() - t0) / 60:.1f}분]')
        else:
            print(f'기둥 4열 완료  [{(time.time() - t0) / 60:.1f}분]')


# ══════════════ 노트북 셀 62 -> _fit_gat() ══════════════
def _fit_gat():
    """[tie] GAT-512-h128-tanh·sig — 라운드 자기 종결"""
    global CU_in, CU_s2, CV_in, CV_s2, GAT_COLS, GAT_SDS
    global aux, gain, prev, r_s2, rd, z
    if True:
        CU_in = np.hstack([CUt, Mt['Lu_tr']]).astype(np.float32)      # tie 는 전 train 행
        CV_in = np.hstack([CVt, Mt['Lv_tr']]).astype(np.float32)
        CU_s2 = np.hstack([CUs, Mt['Lu_s2']]).astype(np.float32)
        CV_s2 = np.hstack([CVs, Mt['Lv_s2']]).astype(np.float32)
        z, GAT_COLS, GAT_SDS, prev = None, [], [], None
        for rd in range(1, RD_CAP + 1):
            r_s2, aux = fit_gat(CU_in, CV_in, Mt['y01m'], z, True, SEED, CU_s2, CV_s2)
            GAT_COLS.append((f'g{rd}', r_s2)); GAT_SDS.append(aux['sd'])
            gain = None if prev is None else prev - aux['es_ll']
            print(f'  r{rd}: es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]}' + (f' gain {gain:+.5f}' if gain else ''))
            prev, z = aux['es_ll'], aux['z_tr']
            if gain is not None and gain < RD_STOP: break
        print(f'tie GAT {len(GAT_COLS)}라운드')


# ══════════════ 노트북 셀 64 -> _fit_lin() ══════════════
def _fit_lin():
    """[win] LIN-FULL-src — 소스별 4체인 x 3라운드"""
    global LIN_COLS, LIN_DEPLOY, Mw, Ps2c, Ptr, beta
    global chain, mw, rd, s, st, z
    Mw = Ms['win']
    if True:
        mw = Mw['m_tr']
        LIN_COLS, LIN_DEPLOY = [], {}
        for s in Mw['src']:
            Ptr = np.hstack([Mw['Lb_tr'][mw], TR[s][mw]]); Ps2c = np.hstack([Mw['Lb_s2'], PS2[s]])
            z, chain = None, []
            for rd in range(1, 4):
                beta, st = fit_beta(z, Ptr, Mw['y01m'], False, False, f'win·{s}·r{rd}', free=Mw['nl'])
                LIN_COLS.append((f'{s}·r{rd}', ((Ps2c - st['mu']) / st['sc']) @ beta))
                chain.append(dict(beta=beta, mu=st['mu'], sc=st['sc'])); z = st['z_tr']
            LIN_DEPLOY[s] = chain
        print(f'win LIN {len(LIN_COLS)}열  [총 {(time.time() - t0) / 60:.1f}분]')


# ── 셀 68 내부 정의를 최상위로 끌어올림 (래퍼를 건너뛰어도 살아 있어야 한다) ──
def combiner(Mx, cols, pillar, rows):
    parts = ([] if pillar is None else pillar) + cols
    X = np.hstack([Mx['base'], np.column_stack([c for _, c in parts])])
    nl = Mx['base'].shape[1]; bias = Mx['bias']; n_even = X.shape[1] if bias else 0
    z, _, picked, n_bad, _, _, _ = crossfit(X, Mx['y01'], fold, bias, LAMS,
                                            rows=rows, n_even=n_even, n_logit=nl)
    lam = sorted(picked)[len(picked) // 2]            # 배포 단일 적합의 λ = fold 중앙값
    Az, Bz = _z(X[rows], X[rows], n_even)
    v, ok = _fit(Az, Mx['y01'][rows], _penvec(X.shape[1], nl, lam), bias)
    return z, dict(v=v, lam=lam, nl=nl, n_even=n_even, X_cols=X.shape[1],
                   mu=X[rows][:, :n_even].mean(0) if n_even else np.zeros(0),
                   sd=np.clip(X[rows][:, :n_even].std(0), 1e-9, None) if n_even else np.zeros(0),
                   rms=np.clip(np.sqrt((X[rows][:, n_even:] ** 2).mean(0)), 1e-9, None))


# ══════════════ 노트북 셀 68 -> _fit_combiner() ══════════════
def _fit_combiner():
    """결합기 — stage2 5-fold 교차적합 로지스틱"""
    global DEP, combiner, dep_t, dep_w, ht, hw
    global llt, llw, res, rt, rw, tag
    global tot_h, zt, zw
    def combiner(Mx, cols, pillar, rows):
        parts = ([] if pillar is None else pillar) + cols
        X = np.hstack([Mx['base'], np.column_stack([c for _, c in parts])])
        nl = Mx['base'].shape[1]; bias = Mx['bias']; n_even = X.shape[1] if bias else 0
        z, _, picked, n_bad, _, _, _ = crossfit(X, Mx['y01'], fold, bias, LAMS,
                                                rows=rows, n_even=n_even, n_logit=nl)
        lam = sorted(picked)[len(picked) // 2]            # 배포 단일 적합의 λ = fold 중앙값
        Az, Bz = _z(X[rows], X[rows], n_even)
        v, ok = _fit(Az, Mx['y01'][rows], _penvec(X.shape[1], nl, lam), bias)
        return z, dict(v=v, lam=lam, nl=nl, n_even=n_even, X_cols=X.shape[1],
                       mu=X[rows][:, :n_even].mean(0) if n_even else np.zeros(0),
                       sd=np.clip(X[rows][:, :n_even].std(0), 1e-9, None) if n_even else np.zeros(0),
                       rms=np.clip(np.sqrt((X[rows][:, n_even:] ** 2).mean(0)), 1e-9, None))

    res = {}
    for tag, rt, rw in (('7k', cu, cu & nz), ('10k', np.ones(n, bool), nz)):
        zt, dep_t = combiner(Ms['tie'], GAT_COLS, PILLAR, rt)
        zw, dep_w = combiner(Ms['win'], LIN_COLS, None, rw)
        llt = rowloss(zt[rt], Ms['tie']['y01'][rt]); llw = rowloss(zw[rw], Ms['win']['y01'][rw])
        ht, hw = hold[rt], hold[rw]
        tot_h = llt[ht].mean() + hw.sum() / ht.sum() * llw[hw].mean()
        res[tag] = dict(zt=zt, zw=zw, dep_t=dep_t, dep_w=dep_w, tot_h=tot_h)
        print(f'[{tag}] tie(hold) {llt[ht].mean():.4f} | win(hold) {llw[hw].mean():.4f} | 총 holdout ll {tot_h:.4f}')
    print(f'Δ(7k - 10k, holdout, 음수면 7k 이득) = {res["7k"]["tot_h"] - res["10k"]["tot_h"]:+.4f}')
    DEP = res['7k']                                       # 배포 = clean 7,748 적합
    np.savez_compressed(HEADS / 'combiner.npz',
        **{f't_{k2}': v for k2, v in DEP['dep_t'].items()},
        **{f'w_{k2}': v for k2, v in DEP['dep_w'].items()})



# ══════════════════════════════════════════════ 세션 API
class HeadSession:
    """적재·SVD 는 1회, §7 만 반복해서 돌린다."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.ready = False
        self._drop = False

    def prepare(self, use_cache=None):
        set_seeds(SEED)
        _build_universe()
        if TR_PLUS_DIRTY:
            _augment_train_with_dirty()
        _build_svd_cached(USE_CACHE if use_cache is None else use_cache)
        assert len(y) == 10748, f'stage2 우주 {len(y)} != 10,748'
        self._orig = {k: (TR[k].copy(), S2[k].copy()) for k in ('t_ar', 'm_ar')}
        self.ready = True
        if self.verbose:
            print(f'세션 준비 완료 — stage2 {len(y):,}행 / train {len(y_tr):,}행 '
                  f'| SVD {CUt.shape} (Arith 무관, 재사용)')
        return self

    def set_arith(self, oof_train, s2, ids_train=None, ids_s2=None):
        """Arith raw 마진(train OOF, stage2)을 갈아끼운다. sd3 로 t_ar/m_ar 를 다시 낸다."""
        assert self.ready, 'prepare() 먼저'
        if ids_train is not None:
            it = np.asarray(ids_train)
            assert (it[:len(it)] == TR['ids'][:len(it)]).all(), 'Arith train id 불일치'
        if ids_s2 is not None:
            assert (np.asarray(ids_s2) == ids).all(), 'Arith stage2 id 불일치'
        o = np.asarray(oof_train, np.float64)
        s = np.asarray(s2, np.float64)
        assert s.shape == (len(y), 3), f'stage2 형상 {s.shape}'
        if len(o) != len(y_tr):
            # TR 증강 모드 — 덧붙인 3,000행은 train OOF 가 없다 (그 행은 Arith 의 train 이
            # 아니라 ES 였다). `_augment_train_with_dirty` 와 같은 순서로 stage2 마진을 잇는다.
            d = np.where(~cu[:HALF])[0]
            assert len(o) + len(d) == len(y_tr),                 f'train OOF {len(o)} + 오염 {len(d)} != TR {len(y_tr)}'
            o = np.concatenate([o, s[d]])
        assert o.shape == (len(y_tr), 3), f'train OOF 형상 {o.shape}'
        TR['m_ar'], TR['t_ar'] = sd3(o)
        S2['m_ar'], S2['t_ar'] = sd3(s)
        self._drop = False
        return self

    def restore_arith(self):
        for k, (a, b) in self._orig.items():
            TR[k], S2[k] = a.copy(), b.copy()
        self._drop = False
        return self

    def drop_arith(self):
        """t_ar/m_ar 를 기저 열 목록에서 통째로 뺀다 — 'Arith 제외' 칸."""
        self._drop = True
        return self

    def fit(self, heads_dir=None, save=False):
        global HEADS
        assert self.ready, 'prepare() 먼저'
        HEADS = Path(heads_dir) if heads_dir else FINAL / 'temp/heads_repro'
        HEADS.mkdir(parents=True, exist_ok=True)
        set_seeds(SEED)
        _build_axes(drop_arith=self._drop)
        _fit_pillar()
        _fit_gat()
        _fit_lin()
        if save:
            _save_bundle()
        _fit_combiner()
        return _score()

    __call__ = fit


def _augment_train_with_dirty():
    """오염 3,000행(stage1 valid 중 V3 학습분)을 헤드 TR 에 덧붙인다.

    그 행들의 로짓·풀링은 이미 stage2 배열(S2 / PS2)에 있다 — 다시 추론할 필요가 없다.
    stage2 우주·cu·hold·fold 는 손대지 않는다. 결합기는 계속 clean 7,748 에서만 적합하고,
    이 행들은 거기서 여전히 제외된다. **헤드(기둥·GAT·LIN) 학습 행만 늘어난다.**

    주의 — SVD 기저 W 는 TR 로 적합하므로 이 증강은 좌표계를 바꾼다. `_svd_key()` 가
    TR_PLUS_DIRTY 를 해시에 넣는 이유다.
    """
    global y_tr                       # TR 은 dict 제자리 변경이라 재바인딩이 없다
    d = np.where(~cu[:HALF])[0]                       # stage2 인덱스 = 오염 행
    assert len(d) == 3000, f'오염 행 {len(d)} != 3,000'
    assert not hold[d].any(), 'holdout 이 섞였다 — 중단'
    n0 = len(TR['ids'])
    for k in list(TR):
        if TR[k] is None:
            continue
        if k == 'ids':
            TR[k] = np.concatenate([TR[k], ids[d]])
        elif k == 'y':
            TR[k] = np.concatenate([TR[k], y[d]])
        elif k in S2:
            TR[k] = np.concatenate([TR[k], np.asarray(S2[k], np.float64)[d]])
        elif k in PS2:
            TR[k] = np.concatenate([TR[k], np.asarray(PS2[k], np.float64)[d]], axis=0)
        else:
            raise AssertionError(f'TR 키 {k} 를 stage2 에서 못 찾았다')
    y_tr = TR['y']
    assert len(np.unique(TR['ids'])) == len(TR['ids']), 'TR id 중복 — 이미 붙였다'
    for k in list(EVK) + list(ODK):
        assert len(TR[k]) == n0 + 3000, f'{k} 길이 불일치'
    print(f'TR 증강: {n0:,} -> {len(TR["ids"]):,}행 (오염 3,000 추가). '
          f'stage2 {len(y):,} / 결합기 적합 {int(cu.sum()):,} 는 불변')


def _svd_cache_path():
    return CACHE_DIR / f'svd512-{_svd_key()}.npz'


def _build_svd_cached(use_cache=True):
    """SVD-512 는 풀링 8블록에서만 나온다 — Arith 와 무관하므로 캐시한다.

    캐시 키는 pooled/*.npz 의 (이름·크기·mtime) + 블록 이름 + 랭크다. 풀링을 다시 뽑으면
    자동으로 무효화된다. 캐시가 어긋난 채로 재사용되면 좌표계가 조용히 달라지므로
    키를 저장분과 대조하고 다르면 다시 낸다.
    """
    global CUt, CUs, CVt, CVs, NU, NV, SVD_U, SVD_V, K_TIE
    cpath = _svd_cache_path()
    if use_cache and cpath.exists():
        try:
            z = np.load(cpath, allow_pickle=False)
            _ = z['key'], z['CUt'].shape          # 중단으로 잘린 파일이면 여기서 터진다
        except Exception as e:                    # noqa: BLE001
            print(f'SVD 캐시 손상 ({type(e).__name__}) -> 다시 낸다')
            z = None
        if z is not None and str(z['key']) == _svd_key():
            CUt, CUs, CVt, CVs = z['CUt'], z['CUs'], z['CVt'], z['CVs']
            NU = dict(mu=z['NU_mu'], sc=z['NU_sc'])
            NV = dict(mu=z['NV_mu'], sc=z['NV_sc'])
            SVD_U = dict(mu=z['U_mu'], W=z['U_W'], sc=z['U_sc'])
            SVD_V = dict(mu=z['V_mu'], W=z['V_W'], sc=z['V_sc'])
            K_TIE = int(z['K_TIE'])
            print(f'SVD 캐시 사용 ({cpath.name}) {CUt.shape} — 재계산 생략')
            return
        print(f'SVD 캐시 키 불일치 ({cpath.name}) -> 다시 낸다')
    t0 = time.time()
    _build_svd()
    print(f'  SVD 계산 {time.time() - t0:.0f}s')
    if not use_cache:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cpath, key=_svd_key(), K_TIE=K_TIE,
             CUt=CUt, CUs=CUs, CVt=CVt, CVs=CVs,
             NU_mu=NU['mu'], NU_sc=NU['sc'], NV_mu=NV['mu'], NV_sc=NV['sc'],
             U_mu=SVD_U['mu'], U_W=SVD_U['W'], U_sc=SVD_U['sc'],
             V_mu=SVD_V['mu'], V_W=SVD_V['W'], V_sc=SVD_V['sc'])
    print(f'  SVD 캐시 저장 -> {cpath.name} ({cpath.stat().st_size / 1e6:.0f}MB)')


def _save_bundle():
    torch.save({'gat_sds': GAT_SDS, 'h': H_GAT, 'k': K_TIE}, HEADS / 'tie_gat.pt')
    np.savez_compressed(
        HEADS / 'deploy.npz',
        w_anc=w_anc, b_anc=b_anc, mu_u=mu_u, sc_u=sc_u, sc_v=sc_v,
        nu_mu=NU['mu'], nu_sc=NU['sc'], nv_mu=NV['mu'], nv_sc=NV['sc'],
        svdu_mu=SVD_U['mu'], svdu_W=SVD_U['W'], svdu_sc=SVD_U['sc'],
        svdv_mu=SVD_V['mu'], svdv_W=SVD_V['W'], svdv_sc=SVD_V['sc'],
        n_gat=len(GAT_COLS), n_lin=len(LIN_COLS), n_pillar=len(PILLAR or []),
        **{f'col_pillar_{i}': c for i, (_, c) in enumerate(PILLAR or [])},
        **{f'col_gat_{i}': c for i, (_, c) in enumerate(GAT_COLS)},
        **{f'col_lin_{i}': c for i, (_, c) in enumerate(LIN_COLS)},
        **{f'pil_{s}_{k2}': v for s, d in (PIL_DEPLOY or {}).items() for k2, v in d.items()},
        **{f'lin_{s}_{r}_{k2}': v for s, ch in LIN_DEPLOY.items()
           for r, d in enumerate(ch) for k2, v in d.items()})
    print('저장:', HEADS / 'tie_gat.pt', '/', HEADS / 'deploy.npz')


def combiner_weights():
    """배포 결합기가 각 열을 실제로 얼마나 쓰는지. 열 이름을 붙여 돌려준다.

    tie:  [로짓 기저 5 | 기둥 4 | GAT n]   (짝 — 중심화+표준화, 절편 있음)
    win:  [로짓 기저 3 | LIN m]            (홀 — RMS only, 절편 없음)
    `v` 는 정규화 좌표계 기준이라 열끼리 크기 비교가 된다 (원 스케일이 아니다).
    """
    out = {}
    for ax, dep, names, extra in (
            ('tie', res['7k']['dep_t'], tie_names,
             [n for n, _ in (PILLAR or [])] + [n for n, _ in GAT_COLS]),
            ('win', res['7k']['dep_w'], win_names, [n for n, _ in LIN_COLS])):
        cols = list(names) + list(extra)
        v = np.asarray(dep['v'], np.float64)
        assert len(cols) == dep['X_cols'], f'{ax} 열 이름 {len(cols)} != {dep["X_cols"]}'
        w = v[:len(cols)]
        out[ax] = {'lam': float(dep['lam']), 'n_logit': int(dep['nl']),
                   'bias': float(v[len(cols)]) if len(v) > len(cols) else None,
                   'cols': [{'name': c, 'w': float(x), 'logit': i < dep['nl']}
                            for i, (c, x) in enumerate(zip(cols, w))]}
    return out


def print_combiner_weights():
    W = combiner_weights()
    for ax in ('tie', 'win'):
        d = W[ax]
        tot = sum(abs(c['w']) for c in d['cols']) or 1.0
        print()
        print(f"[{ax} 결합기] λ {d['lam']:g} | 무벌점 로짓 {d['n_logit']}열"
              + (f" | 절편 {d['bias']:+.3f}" if d['bias'] is not None else ''))
        for c in sorted(d['cols'], key=lambda x: -abs(x['w'])):
            bar = '█' * int(round(abs(c['w']) / tot * 40))
            print(f"    {c['name']:16s} {c['w']:+8.4f} {abs(c['w']) / tot * 100:5.1f}% "
                  f"{'무벌점' if c['logit'] else '   λ  '} {bar}")
    return W


def _score():
    """holdout 행별 3-class 손실. 체인 룰이라 tie축 + (비-tie면) 승부축."""
    D = res['7k']
    rt, rw = cu, cu & nz
    llt = rowloss(D['zt'][rt], Ms['tie']['y01'][rt])
    llw = rowloss(D['zw'][rw], Ms['win']['y01'][rw])
    ht, hw = hold[rt], hold[rw]
    lt_h, lw_h = llt[ht], llw[hw]
    nz_h = nz[hold]
    row = lt_h.astype(np.float64).copy()
    assert int(nz_h.sum()) == len(lw_h), f'승부축 정렬 {int(nz_h.sum())} != {len(lw_h)}'
    row[nz_h] += lw_h
    tot = float(row.mean())
    assert abs(tot - D['tot_h']) < 1e-9, f'총 ll 재계산 {tot} != 셀값 {D["tot_h"]}'
    return {'tot_h': tot, 'tie_h': float(lt_h.mean()), 'win_h': float(lw_h.mean()),
            'row_hold': row, 'n_hold': len(row), 'n_win': int(nz_h.sum()),
            'names': (list(tie_names), list(win_names)),
            'n_pillar': len(PILLAR or []), 'n_gat': len(GAT_COLS), 'n_lin': len(LIN_COLS),
            'tr_rows': len(y_tr), 'tot_h_10k': float(res['10k']['tot_h'])}


def arith_from_ckpt(ck, ids_train, ids_s2, model='xgb'):
    """Arith 체크포인트 -> (train OOF, stage2) raw 마진. 학습 스크래치 캐시를 쓴다."""
    import sys
    sys.path.insert(0, str(MODELS))
    import train_arith as TA
    ck = Path(ck)
    oof, oid, _ = TA.load_oof(ck, model)
    it = np.asarray(ids_train)
    assert len(oid) <= len(it) and (oid == it[:len(oid)]).all(),         'OOF train id 가 TR 앞부분과 다르다 (TR 증강은 뒤에 덧붙는다)'
    return oof, TA.margins_for(ids_s2, ck, model=model)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--save', action='store_true', help='번들 저장 (기본 위치 temp/)')
    ap.add_argument('--deploy', action='store_true', help='배포 번들을 final/heads 에')
    ap.add_argument('--heads-dir', default=None)
    ap.add_argument('--drop-arith', action='store_true')
    ap.add_argument('--arith-ckpt', default=None)
    ap.add_argument('--old-arith', action='store_true',
                    help='구 arith_experts_raw.npz 를 그대로 쓴다')
    ap.add_argument('--no-cache', action='store_true',
                    help='SVD·λ 캐시를 쓰지도 만들지도 않는다 (기준선·배포 런은 이걸로)')
    ap.add_argument('--lam-ns', default='base',
                    help='λ 힌트 네임스페이스. 실험 계열이 바뀌면(예: Arith->그래프) 갈라라')
    ap.add_argument('--tr-plus-dirty', action='store_true',
                    help='오염 3,000행을 헤드 TR 에 추가 (stage2·결합기 적합 행은 불변)')
    ap.add_argument('--no-pillar', action='store_true',
                    help='기둥 4열을 빼고 학습 — GAT 가 같은 풀링을 이미 본다는 가설 검증')
    a = ap.parse_args()

    if a.no_pillar:
        globals()['DROP_PILLAR'] = True
        print('기둥 생략 모드 — tie 결합기 = [base | GAT] (기둥 4열 없음)')

    globals()['LAM_NS'] = a.lam_ns
    if a.tr_plus_dirty:
        globals()['TR_PLUS_DIRTY'] = True
        if a.lam_ns == 'base':
            globals()['LAM_NS'] = 'trplus'      # TR 이 다르면 λ 힌트도 갈라야 한다
        print('TR 증강 모드 — 오염 3,000행을 헤드 학습에 사용 | λ ns =', globals()['LAM_NS'])
    if a.no_cache:
        globals()['USE_CACHE'] = False
        globals()['LAM_REUSE'] = False
        print('캐시 비활성 — SVD 재계산, λ 전 격자 탐색')
    S = HeadSession().prepare()
    ck = None
    if not a.old_arith:
        ck = Path(a.arith_ckpt) if a.arith_ckpt else ARITH_CKPT
        if not (ck / 'meta.json').exists():
            ck = None
    if ck is not None:
        m = json.loads((ck / 'meta.json').read_text(encoding='utf-8'))
        print(f'\nArith <- {ck.name}  피쳐 {m["n_cols"]}종 | ES {m["es"]["rows"]:,}행 '
              f'| holdout raw ll {m["metrics_raw"]["xgb"]["holdout_raw_ll"]:.5f}')
        oof, s2 = arith_from_ckpt(ck, TR['ids'], ids)
        S.set_arith(oof_train=oof, s2=s2, ids_train=TR['ids'], ids_s2=ids)
    else:
        print('\nArith <- 구 outputs/arith_experts_raw.npz')
    if a.drop_arith:
        S.drop_arith()

    hd = Path(a.heads_dir) if a.heads_dir else (
        FINAL / ('heads' if a.deploy else 'temp/heads_repro'))
    r = S.fit(heads_dir=hd, save=a.save)
    print('\n' + '=' * 72)
    print(f'기저 tie {r["names"][0]}')
    print(f'     win {r["names"][1]}')
    print(f'구성  TR {r["tr_rows"]:,}행 | 기둥 {r["n_pillar"]}열 | GAT {r["n_gat"]}열 '
          f'| LIN {r["n_lin"]}열')
    print(f'holdout 총 ll   {r["tot_h"]:.5f}   (tie {r["tie_h"]:.5f} / win {r["win_h"]:.5f}, '
          f'비-tie {r["n_win"]:,}/{r["n_hold"]:,})')
    print(f'  참고 10k 적합 {r["tot_h_10k"]:.5f}')
    print(f'번들 -> {hd}' + ('' if a.deploy else '   (실험. 배포는 --deploy)'))


if __name__ == '__main__':
    main()
