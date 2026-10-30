"""tie_ensemble.py — tie 축을 여러 tie 모델의 앙상블로 더 깎을 수 있나.

배경
  손실 해부에서 병목이 tie 축으로 좁혀졌다(총 손실의 59%, 무정보 대비 회수율 7%).
  마침 서로 다른 tie 모델이 셋 있다:

    tb3  DeBERTa-v3-large cross, 구간별(q,a,b) AttnPool + 대칭 블록 헤드   현역
    tb2  DeBERTa-v3-large cross, 전구간 AttnPool                          중단본(step 3500)
    tb1  ModernBERT-large  cross, 전구간 AttnPool                          언더핏

  다양성 축은 tb1↔tb2 가 **백본**, tb2↔tb3 가 **풀링·헤드**, tb1↔tb3 가 둘 다다.
  (bi-encoder tie 신호는 별도 모델이 아니라 ABE 의 s_abe 가 이미 담당한다.)

무엇을 재나 — 세 모델의 전 조합 7가지(1개 3 + 2개 3 + 전부 1)를 두 블록에서:
  A. TB 만            순수 다양성. 다른 신호에 가려지지 않은 앙상블 여지를 본다
  B. + ABE + Arith    배포 형태. 현행 tie 헤드(tb3+abe+arith) 위에 tb1/tb2 가 값을 더하나

  tie축은 `총 ll = tie축 + 승부축` 항등식에서 승부축과 독립이라, **여기서 깎은 만큼 총점에 1:1 전달**된다.

프로토콜 (중요 — 비대칭이 있다)
  tb1·tb2 : 학습·ES·best 선택 전부 train_enc(stage2 제외) 안에서만 -> stage2 10,748 전량 무접촉
  tb3·abe : stage2 valid 로 체크포인트 선택 -> valid 에 선택 노출 있음 (§1.6 허용)
  그래서 전체 수치와 **holdout 전용 수치를 나란히** 찍는다. 둘이 갈리면 선택 노출을 의심할 것.

입력   Boost/logits/{tag}__checkpoint-{step}.npz   (extract_logits.py 산물, 통일 스키마)
사용   python Boost/tie_ensemble.py
       python Boost/tie_ensemble.py --booster lgbm --no-lin      # 트리만
       python Boost/tie_ensemble.py --abe-ens best               # ABE 앙상블 대신 단일 최선
"""
import argparse
import glob
import inspect
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

LLM = Path('.')
LOGITS = LLM / 'Boost/logits'
ORIGIN = LLM / 'Arith/origin'
ARITH_NPZ = LLM / 'Arith/outputs/arith_experts.npz'
OUT_NPZ = LLM / 'Boost/tie_oof.npz'

SEED, EPS = 42, 1e-15
N_SPLITS, INNER_FRAC = 5, 0.10
TB_TAGS = ['tb3', 'tb2', 'tb1']                 # 강한 순 — 표에서 tb3 단독이 기준선이 된다
N_CLASS = {'abe': 3, 'tb3': 2, 'tb2': 2, 'tb1': 2}
RECORDED_AUC = {'abe': 0.6421, 'tb3': 0.6563, 'tb2': 0.6370, 'tb1': 0.5983}


# ══════════════════════════════════════════════════════ stage2
def load_stage2():
    va = pd.read_csv(ORIGIN / 'valid.csv')
    ho = pd.read_csv(ORIGIN / 'holdout.csv')
    assert len(va) == 5000 and len(ho) == 5748, f'분할 크기 {len(va)}/{len(ho)}'
    lab = lambda d: np.select([d['winner_model_a'] == 1, d['winner_model_b'] == 1,
                               d['winner_tie'] == 1], [0, 1, 2], -1).astype(np.int64)
    IDS = np.concatenate([va['id'].to_numpy(), ho['id'].to_numpy()])
    Y = np.concatenate([lab(va), lab(ho)])
    assert (Y != -1).all() and len(np.unique(IDS)) == len(IDS)
    return IDS, Y, len(va)


def fit_temperature(logits, labels):
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    lg = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([lg], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(z / lg.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(lg.exp().detach())


def load_tag(tag, IDS, Y, HALF):
    """{step: TTA 평균 로짓}. 로드 즉시 id·라벨·형상·유한성 계약을 강제한다."""
    out = {}
    for p in sorted(glob.glob(str(LOGITS / f'{tag}__checkpoint-*.npz')),
                    key=lambda s: int(re.search(r'checkpoint-(\d+)', s).group(1))):
        z = np.load(p)
        L = np.vstack([(z['valid_fwd'] + z['valid_swp']) / 2,
                       (z['hold_fwd'] + z['hold_swp']) / 2]).astype(np.float64)
        assert (np.concatenate([z['valid_ids'], z['hold_ids']]) == IDS).all(), f'{p}: id 불일치'
        assert (np.concatenate([z['valid_labels'], z['hold_labels']]) == Y).all(), f'{p}: 라벨 불일치'
        assert L.shape == (len(IDS), N_CLASS[tag]), f'{p}: 형상 {L.shape}'
        assert np.isfinite(L).all(), f'{p}: 비유한값'
        out[int(re.search(r'checkpoint-(\d+)', p).group(1))] = L
    return out


def tie_signal(tag, L):
    """logit(p_tie). 3-class 는 logsumexp 로 비-tie 를 묶고, 2-class 는 차이 그대로."""
    return (L[:, 2] - np.logaddexp(L[:, 0], L[:, 1])) if N_CLASS[tag] == 3 else (L[:, 1] - L[:, 0])


def pick_by_valid(tag, mats, Y, HALF):
    """§1.6 규율 — valid 로 T 를 맞추고 valid 이진 tie ll 이 최소인 체크포인트."""
    rows = []
    for step, L in mats.items():
        v, vl = L[:HALF], (Y[:HALF] == 2).astype(np.int64)
        if N_CLASS[tag] == 3:
            T = fit_temperature(v, Y[:HALF])
        else:
            T = fit_temperature(v, vl)
        s = tie_signal(tag, L / T)
        p = 1 / (1 + np.exp(-s))
        rows.append((step,
                     log_loss(vl, np.clip(p[:HALF], EPS, 1 - EPS), labels=[0, 1]),
                     log_loss((Y[HALF:] == 2).astype(int), np.clip(p[HALF:], EPS, 1 - EPS),
                              labels=[0, 1])))
    best = min(rows, key=lambda r: r[1])
    return best[0], rows


# ══════════════════════════════════════════════════════ 지표
def tie_ll(p, y, m=None):
    m = slice(None) if m is None else m
    t = np.clip(p[m], EPS, 1 - EPS)
    it = (y[m] == 2)
    return float(-np.where(it, np.log(t), np.log(1 - t)).mean())


def row_ll(p, y):
    t = np.clip(p, EPS, 1 - EPS)
    return -np.where(y == 2, np.log(t), np.log(1 - t))


def paired_ci(a, b, y, n_boot=2000, seed=SEED):
    d = row_ll(a, y) - row_ll(b, y)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def fmt_ci(m, lo, hi, thresh=0.0005):
    sig = '유의' if (lo > 0 or hi < 0) else '노이즈'
    return f'{m:+.4f} [{lo:+.4f},{hi:+.4f}] {sig}' + (' ★' if sig == '유의' and abs(m) >= thresh else '')


# ══════════════════════════════════════════════════════ OOF
def oof_tie(X, T, FOLD, kind, booster):
    """tie 이진 5-fold OOF. 입력이 전부 swap 불변이라 증강·TTA 가 필요 없다."""
    p = np.zeros(len(T))
    for k in range(N_SPLITS):
        te = FOLD == k
        rest = np.where(~te)[0]
        if kind == 'lin':
            m = LogisticRegression(C=1.0, max_iter=1000).fit(X[rest], T[rest])
            p[te] = m.predict_proba(X[te])[:, 1]
        else:
            tr, es = train_test_split(rest, test_size=int(round(INNER_FRAC * len(T))),
                                      stratify=T[rest], random_state=SEED + k, shuffle=True)
            import lightgbm as lgb
            m = booster()
            # lightgbm 4.7+ 는 eval_set 대신 eval_X/eval_y 를 받는다 — 버전 따라 갈라준다
            ekw = ({'eval_X': X[es], 'eval_y': T[es]}
                   if 'eval_X' in inspect.signature(lgb.LGBMClassifier.fit).parameters
                   else {'eval_set': [(X[es], T[es])]})
            m.fit(X[tr], T[tr], **ekw, callbacks=[lgb.early_stopping(100, verbose=False)])
            p[te] = m.predict_proba(X[te])[:, 1]
    return p


def make_lgbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective='binary', learning_rate=0.03, n_estimators=3000,
                              num_leaves=15, min_child_samples=60, colsample_bytree=0.9,
                              subsample=0.9, subsample_freq=1, reg_lambda=1.0,
                              random_state=SEED, n_jobs=-1, verbosity=-1)


# ══════════════════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--abe-ens', default='all', choices=['all', 'best'],
                    help="ABE tie 신호: 'all'=전 체크포인트 로짓평균(현행 채택) | 'best'=valid 최선 단일")
    ap.add_argument('--no-lin', action='store_true', help='선형 결합 생략')
    ap.add_argument('--no-tree', action='store_true', help='LGBM 결합 생략')
    args = ap.parse_args()

    IDS, Y, HALF = load_stage2()
    N = len(Y)
    T = (Y == 2).astype(np.int64)
    FOLD = np.zeros(N, int)
    for k, (_, te) in enumerate(StratifiedKFold(N_SPLITS, shuffle=True,
                                                random_state=SEED).split(np.zeros((N, 1)), Y)):
        FOLD[te] = k
    HOLD = np.arange(N) >= HALF
    print(f'stage2 {N:,}행 (valid {HALF:,} + hold {N - HALF:,}) | tie율 {T.mean():.4f} | '
          f'{N_SPLITS}-fold (Boost 계열과 동일 분할)')

    # ── 신호 구성 ──────────────────────────────────────────────────────────
    SIG, missing = {}, []
    print(f'\n{"=" * 96}\n[신호 구성]')
    for tag in TB_TAGS:
        mats = load_tag(tag, IDS, Y, HALF)
        if not mats:
            missing.append(tag)
            print(f'  {tag:<5} npz 없음 — extract_logits.py 로 먼저 뽑을 것')
            continue
        if len(mats) == 1:
            step = next(iter(mats))
            note = 'val_state(train_enc 기준) — stage2 무접촉'
        else:
            step, rows = pick_by_valid(tag, mats, Y, HALF)
            note = 'stage2 valid 기준 선택'
            for s, lv, lh in rows:
                print(f'        step {s:>5}  valid {lv:.4f}  hold {lh:.4f}'
                      + ('  <-- 선택' if s == step else ''))
        SIG[tag] = tie_signal(tag, mats[step])
        print(f'  {tag:<5} checkpoint-{step:<5} [{note}]')

    abe_mats = load_tag('abe', IDS, Y, HALF)
    assert abe_mats, 'abe npz 없음 — extract_logits.py --import --only abe'
    if args.abe_ens == 'all':
        L_abe = np.mean(list(abe_mats.values()), axis=0)     # 로짓 공간 평균(현행 채택)
        print(f'  abe   {len(abe_mats)}벌 로짓평균 [선택 아님 — 편향 없음]')
    else:
        st, _ = pick_by_valid('abe', abe_mats, Y, HALF)
        L_abe = abe_mats[st]
        print(f'  abe   checkpoint-{st} [stage2 valid 기준 선택]')
    SIG['abe'] = tie_signal('abe', L_abe)

    za = np.load(ARITH_NPZ)
    assert (np.concatenate([za['ids_valid'], za['ids_hold']]) == IDS).all(), 'Arith id 불일치'
    assert (np.concatenate([za['y_valid'], za['y_hold']]) == Y).all(), 'Arith 라벨 불일치'
    B = np.mean([np.vstack([za[f'logp_valid__{m}'], za[f'logp_hold__{m}']])
                 for m in ('lgbm', 'xgb', 'cat')], axis=0).astype(np.float64)
    SIG['ari'] = B[:, 2] - np.logaddexp(B[:, 0], B[:, 1])
    print(f'  ari   부스터 3종 logp 평균')

    tbs = [t for t in TB_TAGS if t in SIG]
    assert tbs, '사용 가능한 tie 모델이 없다'
    if missing:
        print(f'\n  ** {missing} 가 빠진 채로 진행한다 — 조합이 그만큼 줄어든다 **')

    # ── 앵커 ──────────────────────────────────────────────────────────────
    print(f'\n{"=" * 96}\n[앵커] 단독 tie 판별력 — 기록과 맞아야 자산을 신뢰한다')
    print(f'  {"신호":<6}{"AUC":>9}{"tie ll":>9}{"hold AUC":>10}{"hold ll":>9}   기록')
    for k, v in SIG.items():
        p = 1 / (1 + np.exp(-v))
        rec = f'{RECORDED_AUC[k]:.4f}' if k in RECORDED_AUC else '-'
        print(f'  {k:<6}{roc_auc_score(T, v):9.4f}{tie_ll(p, Y):9.4f}'
              f'{roc_auc_score(T[HOLD], v[HOLD]):10.4f}{tie_ll(p, Y, HOLD):9.4f}   {rec}')

    print(f'\n[상관] 낮을수록 앙상블 여지가 크다')
    ks = list(SIG)
    print('        ' + ''.join(f'{k:>8}' for k in ks))
    for a in ks:
        print(f'  {a:<6}' + ''.join(f'{np.corrcoef(SIG[a], SIG[b])[0, 1]:>8.3f}' for b in ks))

    # ── 블록 A/B ──────────────────────────────────────────────────────────
    COMBOS = [c for r in range(1, len(tbs) + 1) for c in combinations(tbs, r)]
    kinds = ([] if args.no_lin else ['lin']) + ([] if args.no_tree else ['lgb'])
    assert kinds, '결합기를 하나는 켜야 한다'
    STORE = {}

    for extra, title in (((), 'A. TB 만 — 순수 다양성'),
                         (('abe', 'ari'), 'B. + ABE tie + Arith tie — 배포 형태(현행 tie 헤드)')):
        base_key = (tbs[0],) + extra
        print(f'\n{"=" * 96}\n{title}')
        print(f'  기준선 = {"+".join(base_key)}. tie축은 승부축과 독립이라 여기서 깎은 만큼 총점에 1:1 전달된다.')
        hdr = f'  {"조합":<22}'
        for kd in kinds:
            hdr += f'{"tie축(" + kd + ")":>13}{"±std":>8}{"AUC":>8}{"hold":>8}'
        print(hdr + '   vs 기준선')
        for c in COMBOS:
            names = list(c) + list(extra)
            X = np.column_stack([SIG[n] for n in names])
            line = f'  {"+".join(names):<22}'
            for kd in kinds:
                p = oof_tie(X, T, FOLD, kd, make_lgbm)
                STORE[(tuple(names), kd)] = p
                folds = np.array([tie_ll(p, Y, FOLD == k) for k in range(N_SPLITS)])
                line += (f'{folds.mean():13.4f}{folds.std(ddof=1):8.4f}'
                         f'{roc_auc_score(T, p):8.4f}{tie_ll(p, Y, HOLD):8.4f}')
            ref = STORE.get((base_key, kinds[0]))
            cur = STORE[(tuple(names), kinds[0])]
            line += '   ' + ('' if names == list(base_key) else fmt_ci(*paired_ci(ref, cur, Y)))
            print(line)

        # 최선 조합과 기준선의 축 대비 — holdout 만으로도 같은 방향인지
        best = min([c for c in COMBOS], key=lambda c: tie_ll(STORE[(tuple(list(c) + list(extra)),
                                                                    kinds[0])], Y))
        bn = tuple(list(best) + list(extra))
        if bn != base_key:
            pb, pr = STORE[(bn, kinds[0])], STORE[(base_key, kinds[0])]
            d_all = tie_ll(pr, Y) - tie_ll(pb, Y)
            d_ho = tie_ll(pr, Y, HOLD) - tie_ll(pb, Y, HOLD)
            print(f'  최선 {"+".join(bn)}: 기준선 대비 전체 {d_all:+.4f} / holdout {d_ho:+.4f}'
                  f'   (부호가 갈리면 tb3/abe 의 valid 선택 노출을 의심)')

    np.savez_compressed(OUT_NPZ, ids=IDS, y=Y,
                        **{f'oof__{"_".join(k[0])}__{k[1]}': v.astype(np.float32)
                           for k, v in STORE.items()})
    print(f'\nsaved: {OUT_NPZ}')
    print('\n읽는 법')
    print('  · A 블록에서 조합이 유의하게 이기면 tie 모델 간 다양성이 실재한다는 뜻이다.')
    print('  · 그래도 B 블록에서 노이즈면 ABE/Arith 의 tie 신호가 이미 그 다양성을 덮고 있다는 뜻이라')
    print('    배포에 추가할 이유가 없다 — 판정은 B 블록이 한다.')
    print('  · B 에서 이기면 그 Δ 가 곧 총 ll 개선폭이다(tie축 1:1 전달). 문턱 0.0005.')
    print('  · 전체와 holdout 의 부호가 갈리면 tb3/abe 의 valid 선택 노출이 샌 신호다.')


if __name__ == '__main__':
    main()
