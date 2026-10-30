"""ablate_arith.py — Arith 피쳐를 255 에서 줄이며 **헤드 holdout ll 로** 손해를 잰다.

지표를 왜 헤드로 잡나
    Arith 의 값은 자기 혼자 얼마나 맞히느냐가 아니라, **다른 전문가가 못 보는 것을 얼마나
    보느냐**에 있다. Arith 는 GBDT 라 풀링 벡터가 없어서 파이프라인에 들어가는 통로가
    로짓 2열(`t_ar` 짝축 / `m_ar` 홀축)뿐이고, 표현층 8블록은 전부 인코더 것이다.
    즉 Arith 가 기여할 수 있는 건 인코더 표현과 **직교하는 성분**뿐이다.
    단독 ll 이 나빠도 그 성분이 남아 있으면 결합에서 값이 있고, 반대도 성립한다.
    그래서 Arith 단독 ll 은 참고 열로만 두고, 판정은 **헤드 holdout 총 ll** 로 한다.

칸마다 하는 일
    1) rung 피쳐로 xgb 5-fold 재학습        (피쳐는 캐시 — 학습 시간만 든다)
    2) train OOF + stage2 10,748 raw 마진 산출
    3) t_ar/m_ar 갈아끼우고 **헤드 통째로 재학습** (기둥 → GAT → LIN → 결합기)
    4) holdout 5,748 행별 손실 -> paired bootstrap 95% CI (vs 255칸)
    SVD·풀링 8블록은 Arith 와 무관하므로 세션에서 1회만 계산해 재사용한다.

0번째 칸 — `none`
    t_ar/m_ar 를 기저에서 통째로 뺀 칸. "Arith 가 애초에 얼마짜리인가" 가 사다리의
    기준점이고, 그게 없으면 다른 칸의 Δ 를 해석할 자가 없다. 항상 포함한다.

드롭 순서 (--order)
    gain    기준 모델의 추론 구간 fold 합산 total_gain 이 낮은 것부터 (기본)
    dep     비용 계층부터 — 토크나이저(10) → TF-IDF(12) → MiniLM(19) → 무상태(214).
            214 칸에서 임베더 계층이 통째로 빠져 실시간 추출이 21.5분 → 2.1분이 된다.
    random  무작위(seed 42) — gain 순서가 만든 이득인지 보는 대조군

swap 쌍 제약
    카탈로그의 `교환:파트너` 는 A/B 대칭 쌍이라 같이 빼야 involution 이 유지된다.
    각 칸을 쌍 닫힘으로 강제하므로 실제 피쳐 수가 요청과 1~2 다를 수 있다 (표에 찍는다).

사용
    python models/ablate_arith.py --dry-run                      # 계획만
    python models/ablate_arith.py --order dep  --rungs 255,245,236,225,214
    python models/ablate_arith.py --order gain --rungs 255,224,192,160,128,96,64,32
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_F = Path(__file__).resolve().parent.parent          # final/  (이 파일은 final/temp/)
sys.path.insert(0, str(_F / 'models'))               # train_arith
sys.path.insert(0, str(_F / 'temp'))                 # fit_heads (자립 모듈)
import train_arith as TA                                        # noqa: E402
import fit_heads as FH                                          # noqa: E402

FINAL = TA.FINAL
TEMP = FINAL / 'temp'          # 실험 산출물은 전부 여기 (컨테이너 밖)
SEED = 42
THRESH = 0.0005


def verdict(lo, hi):
    if lo > THRESH:
        return '유의 손해 ★'
    if hi < -THRESH:
        return '유의 이득 ★'
    if lo > 0:
        return '손해(문턱 미달)'
    if hi < 0:
        return '이득(문턱 미달)'
    return '노이즈'


def raw_ll(L, y):
    z = L - L.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean())


# ══════════════════════════════════════════════ 드롭 순서
def gain_order(ck, cols, nfold=TA.NFOLD):
    """추론 구간(0..best_iter) 기준 fold 합산 total_gain. 낮은 것부터 뺀다."""
    import xgboost as xgb
    tot = {c: 0.0 for c in cols}
    for k in range(1, nfold + 1):
        m = xgb.XGBClassifier()
        m.load_model(str(Path(ck) / f'xgb_fold{k}.json'))
        bi = getattr(m, 'best_iteration', None)
        bst = m.get_booster()
        names = bst.feature_names or cols
        sub = bst[0:(bi + 1)] if bi is not None else bst
        g = sub.get_score(importance_type='total_gain')
        if g and list(g)[0][0] == 'f' and list(g)[0][1:].isdigit():
            g = {names[int(x[1:])]: v for x, v in g.items()}
        for c, v in g.items():
            tot[c] = tot.get(c, 0.0) + v
    s = sum(tot.values()) or 1.0
    return sorted(cols, key=lambda c: tot[c]), {c: tot[c] / s * 100 for c in cols}


def dep_order(cols, dep):
    rank = {'토크나이저': 0, 'TF-IDF': 1, 'MiniLM': 2, '무상태': 3}
    return sorted(cols, key=lambda c: (rank.get(str(dep.get(c)), 9), c))


def close_pairs(keep, swap_series, all_cols):
    k = set(keep)
    for c in list(k):
        v = swap_series.get(c)
        if isinstance(v, str) and v.startswith('교환:'):
            p = v.split(':', 1)[1]
            if p in set(all_cols):
                k.add(p)
    return [c for c in all_cols if c in k]


# ══════════════════════════════════════════════ 본체
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rungs', default='255,224,192,160,128,96,64,32')
    ap.add_argument('--order', default='gain', choices=['gain', 'dep', 'random'])
    ap.add_argument('--step', type=int, default=TA.DEF_STEP)
    ap.add_argument('--base-ckpt', default=None,
                    help='gain 순서를 뽑을 기준 모델 (기본: ARITH/checkpoint-{step}, 없으면 ARITH/model)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--keep-ckpts', action='store_true')
    ap.add_argument('--no-none', action='store_true', help="'Arith 제외' 칸을 빼고 돈다")
    ap.add_argument('--out', default=str(TEMP / 'ablate_arith.csv'))
    a = ap.parse_args()

    rungs = sorted({int(s) for s in a.rungs.split(',') if s.strip()}, reverse=True)
    all_cols, swap_series = TA.catalog_cols()
    cat = pd.read_csv(TA.ORIGIN / 'feature_catalog.csv').set_index('feature')
    dep = cat['의존성']
    N = len(all_cols)
    assert rungs and rungs[0] <= N, f'사다리 최대값 {rungs[0]} > 피쳐 {N}'
    assert rungs[0] == N, f'첫 칸은 기준선이어야 한다 — {N} 을 포함시켜라'

    # ── 드롭 순서 ──
    if a.order == 'gain':
        base = Path(a.base_ckpt) if a.base_ckpt else TA.ckpt_dir(a.step)
        if not (base / 'xgb_fold1.json').exists():
            base = TA.ARITH / 'model'
        assert (base / 'xgb_fold1.json').exists(), \
            f'{base} 에 xgb_fold1.json 이 없다 — --base-ckpt 로 지정하라'
        order, pct = gain_order(base, all_cols)
        print(f'드롭 순서: gain (기준 {base.name}) — 하위 5: '
              + ', '.join(f'{c}({pct[c]:.4f}%)' for c in order[:5]))
    elif a.order == 'dep':
        order = dep_order(all_cols, dep)
        print('드롭 순서: dep — 토크나이저 → TF-IDF → MiniLM → 무상태')
    else:
        order = list(all_cols)
        np.random.default_rng(SEED).shuffle(order)
        print('드롭 순서: random (seed 42)')

    # ── 칸 구성 ──
    plan = []
    for n in rungs:
        keep = close_pairs(order[len(order) - n:] if n < N else list(all_cols),
                           swap_series, all_cols)
        t = dep.reindex(keep).value_counts()
        plan.append({'요청': n, 'keep': keep, '실제': len(keep),
                     '무상태': int(t.get('무상태', 0)), 'MiniLM': int(t.get('MiniLM', 0)),
                     'TF-IDF': int(t.get('TF-IDF', 0)), '토크나이저': int(t.get('토크나이저', 0))})
    if not a.no_none:
        plan.append({'요청': 0, 'keep': None, '실제': 0, '무상태': 0, 'MiniLM': 0,
                     'TF-IDF': 0, '토크나이저': 0})

    print(f'\n{"요청":>5s} {"실제":>5s} {"무상태":>6s} {"MiniLM":>7s} {"TF-IDF":>7s} '
          f'{"토크나이저":>8s}  비고')
    for p in plan:
        if p['keep'] is None:
            note = 'Arith 제외 — 기저에서 t_ar/m_ar 삭제'
        else:
            note = ('임베더 필요' if p['MiniLM'] else
                    '**임베더 불필요 — 추출 21.5분 → 2.1분/50k**')
        print(f'{p["요청"]:5d} {p["실제"]:5d} {p["무상태"]:6d} {p["MiniLM"]:7d} '
              f'{p["TF-IDF"]:7d} {p["토크나이저"]:8d}  {note}')
    print('\n* 실제 != 요청인 칸은 swap 교환 쌍 닫힘 때문이다.')
    print(f'* 칸당 = xgb 5-fold 재학습 + 헤드 통째 재학습. 총 {len(plan)}칸.')

    if a.dry_run:
        print('\n건조 실행 — 학습하지 않는다.')
        return

    # ── 헤드 세션 1회 ──
    print('\n' + '=' * 78)
    print('[헤드 세션] 노트북 §1~§6 + SVD-512 — 무겁고 1회만 (Arith 와 무관)')
    S = FH.HeadSession().prepare()
    ids_tr, ids_s2 = FH.TR['ids'], FH.ids
    y_tr_nb, y_s2 = FH.y_tr, FH.y
    scratch = TEMP / 'heads_scratch'

    rows, ref = [], None
    for p in plan:
        t0 = time.time()
        print('\n' + '=' * 78)
        if p['keep'] is None:
            print('[칸 none] Arith 제외')
            S.restore_arith().drop_arith()
            solo = float('nan')
            iters = '-'
        else:
            n = p['실제']
            print(f'[칸 {p["요청"]}] 피쳐 {n}/{N} — xgb 5-fold 재학습')
            ck = TA.train(step=a.step, families=('xgb',), overwrite=True,
                          cols=p['keep'], tag=f'ab{n}', use_cache=True)
            meta = json.loads((ck / 'meta.json').read_text(encoding='utf-8'))
            iters = str(meta['best_iters']['xgb'])

            oof, oid, oy = TA.load_oof(ck, 'xgb')
            assert (oid == ids_tr).all(), 'OOF train id 가 헤드의 TR 과 다르다'
            assert (oy == y_tr_nb).all(), 'OOF train 라벨 불일치'
            L_s2 = TA.margins_for(ids_s2, ck, model='xgb')
            solo = raw_ll(L_s2[FH.hold], y_s2[FH.hold])
            S.set_arith(oof_train=oof, s2=L_s2, ids_train=oid, ids_s2=ids_s2)
            print(f'  Arith 단독 holdout raw ll {solo:.5f} (참고)')
            if not a.keep_ckpts:
                shutil.rmtree(ck)

        r = S.fit(heads_dir=scratch)
        if ref is None:
            ref = r['row_hold']
            d, lo, hi, vd = 0.0, 0.0, 0.0, '기준선'
        else:
            d, lo, hi = FH.paired_ci(r['row_hold'], ref)
            vd = verdict(lo, hi)
        rows.append({'order': a.order, '요청': p['요청'], '피쳐': p['실제'],
                     '무상태': p['무상태'], 'MiniLM': p['MiniLM'], 'TF-IDF': p['TF-IDF'],
                     '토크나이저': p['토크나이저'], 'arith단독': solo,
                     '헤드holdout': r['tot_h'], 'tie': r['tie_h'], 'win': r['win_h'],
                     'Δ': d, 'lo': lo, 'hi': hi, '판정': vd,
                     'best_iters': iters, '분': round((time.time() - t0) / 60, 1)})
        print(f'  헤드 holdout {r["tot_h"]:.5f} (tie {r["tie_h"]:.5f} / win {r["win_h"]:.5f})'
              f' | Δ {d:+.4f} [{lo:+.4f},{hi:+.4f}] {vd} | {rows[-1]["분"]}분')

    if scratch.exists():
        shutil.rmtree(scratch)
    R = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(a.out, index=False)
    print('\n' + '=' * 78)
    print(f'[사다리] 기준선 {plan[0]["실제"]}종 | 문턱 ±{THRESH} | holdout 5,748행 paired')
    with pd.option_context('display.width', 220, 'display.max_columns', None):
        print(R[['요청', '피쳐', '무상태', 'MiniLM', 'TF-IDF', '토크나이저', 'arith단독',
                 '헤드holdout', 'tie', 'win', 'Δ', 'lo', 'hi', '판정', '분']]
              .to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    print(f'\n저장: {a.out}')
    print('* Δ 는 헤드 holdout 총 ll 차이(양수 = 손해). 판정은 사용자.')


if __name__ == '__main__':
    main()
