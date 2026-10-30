"""repr5.py — 5-전문가 기저 위의 표현 열: 구 기저 최대 레버(T2 +0.0024★)가 새 기저에서도
버는가 + ABE1 표현(신규 블록)은 얼마를 내는가.

배경
  로짓 레벨 결합은 0.9703~0.9705 천장이 반복 측정됐고, 그걸 뚫은 유일한 채널이 표현
  열이었다(reprhead T2 +0.0024★, W1/T2 조합 0.9675 — 구 3-전문가 기저·전 행 우주).
  이제 기저가 5-전문가([m3,m_a1,m_ar] / [s_tie,s_tb1,t3,t_a1,t_ar])로 확장됐으므로,
  같은 질문을 새 기저에서 다시 잰다. ABE1 표현은 어떤 기저에서도 잰 적이 없는 신규 블록이다.

재료 계약
  로짓: **앙상블 계약** (ABE2 7ckpt 평균 · TB3 4000 · Arith 3종 평균 — 단일 대비 +0.0035
        실측 우세) + TB1 5000 + ABE1 3109.
  표현: abe_u/v = (ea±eb)/2 (pooled abe-3000), tb_u/v = TB3 구간 대수 (pooled tb3-4000),
        a1_u/v = (ea±eb)/2 (V3 embeddings_export — 재추론 없이 분할 재현으로 매핑),
        tb1_u/v = (x_f±x_s)/2 (pooled tb1-5000, ModernBERT 전구간 1벡터/뷰 — 파일이 있을 때만).
  주의: 로짓은 7ckpt 평균인데 표현은 단일 ckpt — reprhead 의 T2 도 같은 계약이었다.

TB1 표현 팔(R5)이 새로 묻는 것
  TB1 헤드는 2-class(tie/no-tie)라 **승부 성분을 로짓으로 낸 적이 단 한 번도 없다**.
  그런데 풀링 벡터의 홀 성분 (x_f−x_s)/2 에는 A/B 비대칭이 남아 있다 — 헤드가 버린 채널이다.
  ABE1 에서 같은 일이 관측됐다(W-R3: m_a1 이 기저에 있는데도 a1_v 가 +0.0018★).
  선행: python Boost/extract_logits.py --pooled --only tb1 --steps tb1:5000

행 우주 — ABE1-clean 7,748 (holdout 5,748 + V3-valid 2,000)
  기저에 ABE1 파생 열(m_a1, t_a1)이 있으므로 V3 학습 행 3,000개를 적합에 넣으면 가중치가
  오염된다(재료 팔 실측: holdout −0.0031). 표현도 같은 채널이다 — 로짓은 표현의 선형
  함수(R² 0.96+)라 오염 논리가 그대로 통한다. 전 팔을 같은 우주에서 재적합해 paired 비교.
  **구 reprhead 수치(0.5716/0.5739, 전 행·3-전문가)와 직접 비교하지 말 것.**

검산
  ABE1 임베딩 매핑은 V3 분할 재현(라벨열 전량 대조) + **헤드 재계산 앵커**로 강제한다:
  체크포인트의 head(LN→Linear→GELU→Linear)를 CPU 로 임베딩에 다시 태워 저장된 fwd 로짓이
  재현되는지 본다. 풀링을 엉뚱한 데서 낚아챘으면 여기서 잡힌다 (pooled_anchor 와 동일 정신).

사용 (약 30~50분 — 표현 팔은 1,029~3,077열 릿지 × 5-fold × 내부 λ)
  python Boost/repr5.py 2>&1 | tee Boost/log_repr5.txt
  python Boost/repr5.py --only R0,a1   # 부분 실행 (기준은 자동 포함)
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import erf
from sklearn.model_selection import StratifiedKFold

import fwls as FW
import reprhead as RH

sys.stdout.reconfigure(line_buffering=True)

SEED, K, HALF = RH.SEED, RH.K, FW.HALF
LLM, BOOST = FW.LLM, FW.BOOST


def _v3_split():
    """V3 분할 재현 (extract_logits._v3_split 과 동일 — 라벨열 대조는 호출부가 강제)."""
    from sklearn.model_selection import train_test_split
    full = pd.read_csv(LLM / 'V3/origin/train.csv')
    lab = np.select([full['winner_model_a'] == 1, full['winner_model_b'] == 1], [0, 1], 2)
    tr51, te = train_test_split(full.assign(label=lab), test_size=0.10, stratify=lab,
                                random_state=42, shuffle=True)
    _, va2 = train_test_split(tr51, test_size=2000, stratify=tr51['label'],
                              random_state=42, shuffle=True)
    return te, va2


def a1_head_logits(a, b):
    """V3 체크포인트의 헤드를 CPU 로 재계산 — LN(3072) → Linear(1024) → GELU(erf) → Linear(3).
    임베딩 매핑 검산용 앵커다. fp16 원본 대비 오차는 corr 로 판정한다."""
    from safetensors.numpy import load_file
    sd = load_file(str(LLM / 'V3/outputs/run/checkpoint-3109/model.safetensors'))
    g, be = sd['head.0.weight'].astype(np.float64), sd['head.0.bias'].astype(np.float64)
    W1, b1 = sd['head.2.weight'].astype(np.float64), sd['head.2.bias'].astype(np.float64)
    W2, b2 = sd['head.5.weight'].astype(np.float64), sd['head.5.bias'].astype(np.float64)
    x = np.hstack([a, b, a - b])
    mu = x.mean(1, keepdims=True)
    x = (x - mu) / np.sqrt(x.var(1, keepdims=True) + 1e-5) * g + be
    h = x @ W1.T + b1
    h = h * 0.5 * (1.0 + erf(h / np.sqrt(2.0)))
    return h @ W2.T + b2


def load_repr(single_best=False):
    """자산 적재 + 검산 + 기저 구성. nlfeat.py·resfit.py 가 같은 계약으로 재사용한다.

    single_best=True 면 로짓 기저(M3/T5)를 valid raw ll 로 고른 **단일 최적 ckpt** 로
    구성한다 (resfit — 앙상블 7-ckpt 는 추론 예산 대비 지나침, 사용자 지시 2026-08-05).
    기본 False = 앙상블 계약 (repr5/nlfeat 본표가 이 계약으로 측정됐다 — 바꾸지 말 것).
    두 계약의 절대값은 서로 직접 비교 금지 — 각 파일 안 paired 만 유효하다.

    반환 dict — ids/y/nz/hold/cu/fold, 표현 블록(짝 *_u · 홀 *_v; 없는 블록은 None),
    기저 M3(승부 3열)·T5(tie 5열). 여기서 하는 일은 전부 라벨 비접촉이다 —
    라벨을 보는 적합은 fold 안 릿지(RH.crossfit)뿐이다. (단일 계약의 ckpt '선택' 은
    valid 5,000 노출 — §1.6 허용 범위이고 holdout 은 무접촉이다.)"""
    print('자산 적재 — 표현 3원(pooled·V3 임베딩) + 로짓 기저 ...')
    D = FW.load_all(single_best=single_best)
    print(f'  로짓 계약: {D["contract"]}')
    ids, y = D['ids'], D['y']
    n = len(y)
    nz = y != 2
    hold = np.arange(n) >= HALF
    cu = ~D['a1_ins']                                # ABE1-clean 7,748
    assert D['m_a1'] is not None, 'abe1 npz 필요'

    fold = np.zeros(n, int)
    for k, (_, te_) in enumerate(StratifiedKFold(K, shuffle=True, random_state=SEED)
                                 .split(np.zeros((n, 1)), y)):
        fold[te_] = k

    # ── 표현 블록 ──
    za = np.load(BOOST / 'pooled/abe__checkpoint-3000.npz')
    zt = np.load(BOOST / 'pooled/tb3__checkpoint-4000.npz')
    for z, tag in ((za, 'abe'), (zt, 'tb3')):
        assert (np.concatenate([z['valid_ids'], z['hold_ids']]) == ids).all(), f'{tag} 풀링 id'
    cat_ = lambda w, k_: np.vstack([w[f'valid_{k_}'], w[f'hold_{k_}']]).astype(np.float64)
    ea, eb = cat_(za, 'ea'), cat_(za, 'eb')
    abe_u, abe_v = (ea + eb) / 2, (ea - eb) / 2
    a_f, b_f = cat_(zt, 'a_f'), cat_(zt, 'b_f')          # 프롬프트 구간(q)은 구 기저에서 노이즈 — 제외
    a_s, b_s = cat_(zt, 'a_s'), cat_(zt, 'b_s')
    A_rep, B_rep = (a_f + b_s) / 2, (b_f + a_s) / 2   # swap = fwd<->swp 뷰 교환 (reprhead 와 동일)
    tb_u, tb_v = (A_rep + B_rep) / 2, (A_rep - B_rep) / 2

    # TB1(ModernBERT) — 전구간 풀링이라 뷰당 벡터가 하나다. swap 은 x_f <-> x_s 교환이므로
    # (x_f+x_s)/2 가 짝, (x_f−x_s)/2 가 홀이다 (tb_seg 의 구간 대수와 같은 항등식).
    p_tb1 = BOOST / 'pooled/tb1__checkpoint-5000.npz'
    tb1_u = tb1_v = None
    if p_tb1.exists():
        z1 = np.load(p_tb1)
        assert (np.concatenate([z1['valid_ids'], z1['hold_ids']]) == ids).all(), 'tb1 풀링 id'
        x_f, x_s = cat_(z1, 'x_f'), cat_(z1, 'x_s')
        assert np.isfinite(x_f).all() and np.isfinite(x_s).all(), 'tb1 풀링 비유한값'
        tb1_u, tb1_v = (x_f + x_s) / 2, (x_f - x_s) / 2
    else:
        print(f'  [주의] {p_tb1.name} 없음 — TB1 표현 팔(R5)을 건너뛴다. 먼저:\n'
              f'         python Boost/extract_logits.py --pooled --only tb1 --steps tb1:5000')

    # ABE1 임베딩 — V3 분할 재현으로 현행 순서 매핑 (clean 행만 존재하면 된다)
    ze = np.load(LLM / 'V3/outputs/embeddings_export.npz')
    zl = np.load(LLM / 'V3/outputs/logits_export.npz')
    te, va2 = _v3_split()
    assert (zl['hold_labels'] == te['label'].to_numpy()).all(), 'V3 hold 순서 재현 실패'
    assert (zl['valid_labels'] == va2['label'].to_numpy()).all(), 'V3 valid 순서 재현 실패'
    h1 = ze['hold_a'].shape[1]
    a1a, a1b = np.full((n, h1), np.nan), np.full((n, h1), np.nan)
    pos_h = pd.Index(te['id']).get_indexer(ids[HALF:])
    assert (pos_h >= 0).all()
    a1a[HALF:], a1b[HALF:] = ze['hold_a'][pos_h], ze['hold_b'][pos_h]
    vidx = pd.Index(va2['id']).get_indexer(ids[:HALF])
    vc = vidx >= 0
    assert (vc == cu[:HALF]).all(), 'clean 경계가 a1_ins 재현과 불일치'
    a1a[:HALF][vc], a1b[:HALF][vc] = ze['valid_a'][vidx[vc]], ze['valid_b'][vidx[vc]]

    # 헤드 재계산 앵커 — 임베딩이 진짜 그 행의 풀링 벡터인지 로짓 재현으로 강제
    ztt = np.load(BOOST / 'logits/tt__checkpoint-3109.npz')
    ref = np.vstack([ztt['valid_fwd'], ztt['hold_fwd']]).astype(np.float64)
    rec = a1_head_logits(a1a[cu], a1b[cu])
    c = np.corrcoef(rec.ravel(), ref[cu].ravel())[0, 1]
    d = np.abs(rec - ref[cu]).max()
    print(f'  [검산] ABE1 헤드 재계산 앵커: corr {c:.6f}  max|Δ| {d:.4f}  '
          + ('OK' if c > 0.999 else '*** 불일치 — 매핑이 틀렸다'))
    assert c > 0.999, 'ABE1 임베딩 매핑 검산 실패'
    a1_u, a1_v = (a1a + a1b) / 2, (a1a - a1b) / 2

    # ── 기저와 팔 ──
    M3 = np.column_stack([D['M'][:, 0], D['m_a1'], D['M'][:, 1]])               # [m3, m_a1, m_ar]
    T5 = np.column_stack([D['T'][:, 0], D['s_tb1'], D['T'][:, 1], D['t_a1'], D['T'][:, 2]])
    print(f'행 우주: ABE1-clean {int(cu.sum()):,} (holdout {int(hold.sum()):,} 포함) | '
          f'표현 블록 각 {h1}차원 | 기저 승부 3열 / tie 5열 | λ격자 {FW.LAMS} + 자동확장')
    print('! 구 reprhead(전 행·3-전문가)·본표(10,748행) 수치와 직접 비교 금지 — 이 파일 안 paired 만 유효\n')
    return dict(ids=ids, y=y, nz=nz, hold=hold, cu=cu, fold=fold, M3=M3, T5=T5, h1=h1,
                abe_u=abe_u, abe_v=abe_v, tb_u=tb_u, tb_v=tb_v,
                a1_u=a1_u, a1_v=a1_v, tb1_u=tb1_u, tb1_v=tb1_v,
                # Arith 짝피쳐(rankgauss)와 가족 — resfit v3 의 FWLS 게이트 팔이 쓴다
                arith_Uz=D['Uz'], arith_evn=D['evn'], arith_fam=D['fam_e'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='', help='팔 코드 부분일치 필터 (기준 R0 은 자동 포함)')
    args = ap.parse_args()
    only = [s.strip() for s in args.only.split(',') if s.strip()]
    t0 = time.time()

    A = load_repr()
    y, nz, hold, cu, fold = A['y'], A['nz'], A['hold'], A['cu'], A['fold']
    n = len(y)
    M3, T5 = A['M3'], A['T5']
    abe_u, abe_v, tb_u, tb_v = A['abe_u'], A['abe_v'], A['tb_u'], A['tb_v']
    a1_u, a1_v, tb1_u, tb1_v = A['a1_u'], A['a1_v'], A['tb1_u'], A['tb1_v']

    def blocks_of(parts):
        cols, blk, off = [], [], 0
        for nm, B in parts:
            cols.append(B)
            blk.append((nm, off, off + B.shape[1]))
            off += B.shape[1]
        return np.hstack(cols), blk

    AXES = [
        ('tie축', (y == 2).astype(np.float64), cu, True, T5,
         [('T-R0', '5로짓만 — 기준', []),
          ('T-R1(tb_u)', '+TB3 표현 짝 (구 기저 T2 = +0.0024★ 재검)', [('tb_u', tb_u)]),
          ('T-R2(abe_u)', '+ABE2 표현 짝 (구 T1 = +0.0015★ 재검)', [('abe_u', abe_u)]),
          ('T-R3(a1_u)', '+ABE1 표현 짝 (신규)', [('a1_u', a1_u)]),
          ('T-R4(전부)', '+표현 짝 셋 다', [('tb_u', tb_u), ('abe_u', abe_u), ('a1_u', a1_u)]),
          ('T-R5(tb1_u)', '+TB1 표현 짝 (신규 · ModernBERT)', [('tb1_u', tb1_u)])]),
        ('승부축', (y == 0).astype(np.float64), cu & nz, False, M3,
         [('W-R0', '3로짓만 — 기준', []),
          ('W-R1(tb_v)', '+TB3 표현 홀 (구 최대가설 W1, m_a1 존재 하 재검)', [('tb_v', tb_v)]),
          ('W-R2(abe_v)', '+ABE2 표현 홀', [('abe_v', abe_v)]),
          ('W-R3(a1_v)', '+ABE1 표현 홀 (신규)', [('a1_v', a1_v)]),
          ('W-R4(전부)', '+표현 홀 셋 다', [('tb_v', tb_v), ('abe_v', abe_v), ('a1_v', a1_v)]),
          ('W-R5(tb1_v)', '+TB1 표현 홀 (신규 — TB1 헤드는 2-class 라 승부를 낸 적이 없다)',
           [('tb1_v', tb1_v)])]),
    ]
    # 풀링 파일이 없는 블록의 팔은 조용히 뺀다 (위에서 이미 안내를 찍었다)
    AXES = [(nm, y01, rm, bi, Lb, [a for a in arms if all(B is not None for _, B in a[2])])
            for nm, y01, rm, bi, Lb, arms in AXES]

    results = {}
    for ax_nm, y01, rows_m, bias, Lb, arms in AXES:
        RH.head('1' if ax_nm == '승부축' else '2',
                f'{ax_nm} · clean {int(rows_m.sum()):,}행 — 로짓 pen=0, 표현 블록만 릿지')
        if only:
            arms = [a for a in arms if a[0].endswith('R0') or any(o in a[0] for o in only)]
        base_ll = None
        for code, label, parts in arms:
            Xi, blk = blocks_of(parts) if parts else (np.zeros((n, 0)), [])
            print(f'  [{code}] {label}  ({Lb.shape[1] + Xi.shape[1]}열) 적합 중 ...')
            r = FW.run_arm((code, label, Lb, Xi, blk, None), y01, fold, rows_m, hold,
                           base_ll, bias=bias)
            if base_ll is None:
                base_ll = r['ll']
            FW.arm_line(r, t0)
            results[code] = r
        print()

    RH.head('3', '종합 — 총 ll = tie축 + (1−tie율)·승부축 (clean 우주, holdout 열은 본표와 비교 가능)')
    nzf = float(nz[cu].mean())
    nzf_h = float(nz[hold].mean())
    hw, ht = hold[cu & nz], hold[cu]
    combos = [('기준 R0/R0', 'W-R0', 'T-R0')]
    bw = min((c for c in results if c.startswith('W-') and c in results),
             key=lambda c: results[c]['full'])
    bt = min((c for c in results if c.startswith('T-') and c in results),
             key=lambda c: results[c]['full'])
    combos.append((f'축별 최선 {bw}/{bt} (사후 선택 — 낙관적)', bw, bt))
    for nm, wc, tc in combos:
        lw, lt = results[wc]['ll'], results[tc]['ll']
        tot = float(lt.mean() + nzf * lw.mean())
        tot_h = float(lt[ht].mean() + nzf_h * lw[hw].mean())
        print(f'  {nm:<40} 총 {tot:.4f} (holdout {tot_h:.4f})')
    print('\n  참조(비교 금지, 맥락용): 구 기저·전 행의 repr 최선 0.9675 / 로짓 천장 0.9705.')
    print('  판정은 각 팔의 Δ CI + 문턱 0.0005 — 팔 8개 다중비교는 Bonferroni 로. 판정은 사용자.')


if __name__ == '__main__':
    main()
