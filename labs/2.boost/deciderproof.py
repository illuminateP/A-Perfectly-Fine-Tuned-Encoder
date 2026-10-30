"""deciderproof.py — 결정자 존재 증명: 신경망 전문가들의 행별 상대 신뢰도가 산술 피쳐로
예측 가능한가. 가능하다면 그 예측(학습된 결정자)의 게이트 가치는 얼마인가.

배경 (2026-08-05 논의)
  FWLS 원논문의 교훈은 "메타피쳐 = 베이스가 못 본 부가 정보"인데, 우리 후보(Arith 240)는
  이미 전문가 하나의 입력이라 부가성이 없었다. 사용자 제안: **결정자의 대상을 신경망
  풀(TB1·TB3·ABE1·ABE2)로 정의**하면 신경망들은 산술 피쳐를 본 적이 없으므로 그 타깃에
  대해 피쳐는 순수 side information 이 된다 — Arith 를 증거 전문가로 유지한 채로.

1단계 — 존재 증명 (이 파일의 본론)
  타깃: 신경망 쌍의 **불일치 행에서 누가 맞나** (스케일 무관 — 온도·보정 무관).
      승부축  ABE2 vs ABE1               sign(m) 의 정오가 갈리는 행
      tie축   {TB3,TB1,ABE2,ABE1} 6쌍    "tie 라고 본다"(자기 점수 상위 tie율 분위)가 갈리는 행
  모델: LightGBM 을 같은 5-fold 로 교차적합. 입력은 짝피쳐 186 raw + |홀피쳐| 67
      (행별 정오는 swap 불변 = 짝 타깃이라, 홀 입력은 절댓값으로만 — 결정자도 swap 일관).
  지표: OOF 판별 AUC + 부트스트랩 CI, 이득 중요도 상위 피쳐 (= "결정자로 적합한 피쳐"의
      비선형·순수부가정보 버전). AUC ~ 0.5 면 이 피쳐 공간에 결정자가 배울 것이 없다.
  주의: 쌍 7개 다중비교 — 개별 CI 는 탐색용, 채택급 주장엔 Bonferroni 로 볼 것.

  ABE1 오염 처리: 현행 valid 의 3,000행은 ABE1 의 학습 행이라 그 행의 '정오'는 암기
  왜곡이다. ABE1 이 낀 타깃은 그 행을 적합·평가에서 뺀다 (clean 7,748행).

2단계 — 게이트 가치 (예측 신뢰도를 선형 블렌더의 게이트로)
  1단계 OOF 예측 g(x) 를 rankgauss 후 로짓과 곱해 게이트 열로 투입. 사용자 지시대로
  **Arith 가 증거 기저에 있을 때/없을 때 둘 다** 잰다 (해석: '결정자 유무'가 아니라
  '증거 풀 소속' — 다르면 정정 바람):
      승부축  [m3,m_a1] ± m_ar   각각 ± g_win 게이트
      tie축   [s_tie,s_tb1,t3,t_a1] ± t_ar   각각 ± g_tie 게이트 (tie 쌍 중 1단계 AUC 최고)
  행 우주: ABE1 오염 3,000행 제외 7,748행 — 재료 팔에서 실측된 가중치 왜곡(-0.0031 holdout)을
  이 측정에 섞지 않기 위함. 전 행 버전이 필요하면 말할 것.
  OOF 예측을 같은 fold 로 쓰므로 스택 규율이 성립한다 (행의 게이트값은 그 행 라벨을 모른다).

사용 (약 3~5분)
  python Boost/deciderproof.py 2>&1 | tee Boost/log_decider.txt
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import fwls as FW
import reprhead as RH

sys.stdout.reconfigure(line_buffering=True)

SEED, K, HALF = RH.SEED, RH.K, FW.HALF
N_BOOT = 2000


def crossfit_gbdt(X, yb, fit_rows, fold):
    """fold k 적합은 (다른 fold ∩ fit_rows) 로만, 예측은 fold k 전 행에 낸다 —
    게이트로 쓰려면 불일치 행 밖에서도 예측이 필요하다."""
    p = np.full(len(yb), np.nan)
    imp = np.zeros(X.shape[1])
    for k in range(K):
        tr = (fold != k) & fit_rows
        te = fold == k
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                               min_child_samples=30, subsample=0.9, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=1.0,
                               random_state=SEED, n_jobs=-1, verbosity=-1)
        m.fit(X[tr], yb[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
        imp += m.booster_.feature_importance('gain')
    return p, imp / K


def auc_ci(yb, p, seed=SEED):
    a = roc_auc_score(yb, p)
    rng = np.random.default_rng(seed)
    n = len(yb)
    bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if yb[i].min() == yb[i].max():
            continue
        bs.append(roc_auc_score(yb[i], p[i]))
    return float(a), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def main():
    t0 = time.time()
    print('자산 적재 (fwls 단일 ckpt 계약 재사용) ...')
    D = FW.load_all(single_best=True)
    y = D['y']
    n = len(y)
    nz = y != 2
    is_tie = (y == 2)
    hold = np.arange(n) >= HALF
    dirty = D['a1_ins']                              # ABE1 의 V3 학습 행 3,000
    assert D['m_a1'] is not None, 'abe1 npz 필요 (extract_logits.py --only abe1)'

    fold = np.zeros(n, int)
    for k, (_, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=SEED)
                                .split(np.zeros((n, 1)), y)):
        fold[te] = k

    # 결정자 입력 — 짝 raw + |홀|. GBDT 라 변환 불필요, 패리티(짝 타깃)만 지킨다.
    cat = pd.read_csv(FW.ARITH / 'feature_catalog.csv')
    feat = pd.read_parquet(FW.ARITH / 'features_57k.parquet')
    pos = pd.Index(feat['id']).get_indexer(D['ids'])
    assert (pos >= 0).all()
    E, evn, O, odn = RH.parity_split_arith(feat, pos, cat)
    X = np.hstack([E, np.abs(O)]).astype(np.float64)
    names = evn + [f'|{c}|' for c in odn]
    print(f'stage2 {n:,}행 | 결정자 입력 {X.shape[1]}열 (짝 {len(evn)} + |홀| {len(odn)}) | '
          f'ABE1 오염 {int(dirty.sum()):,}행은 ABE1 타깃에서 제외\n')

    # ── [1] 존재 증명 — 불일치 판별 ──
    RH.head('1', '존재 증명 — 신경망 쌍의 불일치 행에서 "누가 맞나"가 피쳐로 예측되는가')
    tie_rate = float(is_tie.mean())
    tie_score = {'TB3': D['T'][:, 0], 'TB1': D['s_tb1'], 'ABE2': D['T'][:, 1], 'ABE1': D['t_a1']}
    says_tie = {k: v >= np.nanquantile(v, 1 - tie_rate) for k, v in tie_score.items()}
    tie_right = {k: says_tie[k] == is_tie for k in tie_score}
    win_right = {'ABE2': (D['M'][:, 0] > 0) == (y == 0), 'ABE1': (D['m_a1'] > 0) == (y == 0)}

    preds, results = {}, []
    targets = [('승부', 'ABE2', 'ABE1', win_right, nz)] + \
              [('tie', a, b, tie_right, np.ones(n, bool))
               for a, b in [('TB3', 'TB1'), ('TB3', 'ABE2'), ('TB3', 'ABE1'),
                            ('TB1', 'ABE2'), ('TB1', 'ABE1'), ('ABE2', 'ABE1')]]
    for axis, a, b, right, base_rows in targets:
        rows = base_rows.copy()
        if 'ABE1' in (a, b):
            rows &= ~dirty
        disag = rows & (right[a] != right[b])
        yb = right[a].astype(int)                    # 1 = a 가 맞다
        p, imp = crossfit_gbdt(X, yb, disag, fold)
        m = disag & np.isfinite(p)
        auc, lo, hi = auc_ci(yb[m], p[m])
        top = np.argsort(-imp)[:8]
        tag = f'{axis}:{a}vs{b}'
        preds[tag] = p
        results.append((tag, int(disag.sum()), auc, lo, hi))
        sig = '유의' if lo > 0.5 or hi < 0.5 else '노이즈'
        print(f'  [{tag:<14}] 불일치 {int(disag.sum()):>5,}행 | OOF AUC {auc:.4f} '
              f'[{lo:.4f},{hi:.4f}] {sig}  [{(time.time() - t0) / 60:.0f}분]')
        print('      중요 피쳐: ' + '  '.join(names[j] for j in top if imp[j] > 0))
    print('\n  쌍 7개 다중비교 주의 — 채택급 주장은 Bonferroni(개별 99.3% CI 급)로 볼 것.')

    # ── [2] 게이트 가치 — 예측 신뢰도를 선형 블렌더 게이트로 ──
    RH.head('2', '게이트 가치 — g(x) 를 로짓과 곱해 투입. 행 우주: ABE1-clean 7,748')
    cu = ~dirty
    best_tie = max((r for r in results if r[0].startswith('tie')), key=lambda r: r[2])
    print(f'  tie 게이트로 1단계 AUC 최고 쌍 사용: {best_tie[0]} (AUC {best_tie[2]:.4f})\n')
    g_win = FW.rankgauss(np.nan_to_num(preds['승부:ABE2vsABE1'], nan=0.5))
    g_tie = FW.rankgauss(np.nan_to_num(preds[best_tie[0]], nan=0.5))

    Z0 = np.zeros((n, 0))
    M2 = np.column_stack([D['M'][:, 0], D['m_a1']])                      # [m3, m_a1]
    M3 = np.column_stack([D['M'][:, 0], D['m_a1'], D['M'][:, 1]])        # + m_ar
    T4 = np.column_stack([D['T'][:, 0], D['s_tb1'], D['T'][:, 1], D['t_a1']])
    T5 = np.column_stack([T4, D['T'][:, 2]])                             # + t_ar

    def gate(Lb, g):
        cols = [g * Lb[:, i] for i in range(Lb.shape[1])]
        return np.column_stack(cols), [('g게이트', 0, len(cols))]

    for axis, y01, rows_m, bias, arms in (
            ('승부축', (y == 0).astype(np.float64), cu & nz, False,
             [('D0', '신경망만 [m3,m_a1] — 기준', M2, None),
              ('D0g', '신경망만 + g_win 게이트', M2, g_win),
              ('D1', '+Arith 증거 [m3,m_a1,m_ar] — 기준', M3, None),
              ('D1g', '+Arith 증거 + g_win 게이트', M3, g_win)]),
            ('tie축', is_tie.astype(np.float64), cu, True,
             [('E0', '신경망만 [s_tie,s_tb1,t3,t_a1] — 기준', T4, None),
              ('E0g', '신경망만 + g_tie 게이트', T4, g_tie),
              ('E1', '+Arith 증거 (+t_ar) — 기준', T5, None),
              ('E1g', '+Arith 증거 + g_tie 게이트', T5, g_tie)])):
        print(f'  ── {axis} · {int(rows_m.sum()):,}행 ──')
        base_ll = {}
        for code, label, Lb, g in arms:
            Xi, blk = gate(Lb, g) if g is not None else (Z0, [])
            # 기준 연결: D0 절대기준 / D1 은 D0 대비(=Arith 증거의 한계 가치) /
            #            D0g·D1g 는 각자의 기저 대비(=게이트의 한계 가치)
            ref = base_ll.get('first') if g is None else base_ll[code[:2]]
            r = FW.run_arm((code, label, Lb, Xi, blk, None), y01, fold, rows_m, hold,
                           ref, bias=bias)
            if g is None:
                base_ll[code[:2]] = r['ll']
                base_ll.setdefault('first', r['ll'])
            print(f'  [{code}] {label}  ({Lb.shape[1] + Xi.shape[1]}열)')
            FW.arm_line(r, t0)
        print()
    print('  판독: 게이트 Δ(양수=이득)가 두 기저(신경망만/Arith 포함)에서 함께 서는지,')
    print('  Arith 증거 추가분(D1-D0, E1-E0)과 게이트 이득이 겹치는지가 본론이다. 판정은 사용자.')


if __name__ == '__main__':
    main()
