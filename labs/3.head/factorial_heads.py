"""factorial_heads.py — 세 질문을 한 런에서 2x2x3 요인설계로 잡는다.

질문
    A. TR +3,000   버려진 3,000행(stage1 valid 중 V3/ABE1 학습분)을 헤드 TR 에 쓰면 유리한가
                   TR 46,729 -> 49,729. stage2 우주·결합기 적합 행(clean 7,748)은 불변이라
                   "결합기에 안 쓰이던 행을 헤드 학습에만 재활용" 이 정확한 표현이다.
    B. 기둥 4열    값을 하는가. 기둥 = 소스별 [s0 무벌점 | 풀링 1024열 λ벌점] -> 1열.
                   GAT 가 같은 풀링의 SVD-512 를 이미 보므로 입력이 겹친다. 기둥만 가진 것은
                   **소스별 분리**와 **s0 잔차 구조**(ABE1 암기 방향 흡수) 둘.
    C. LIN 라운드  3라운드가 필요한가. 기준선 로그에서 12개 λ 중 6개가 격자 끝(1e+08)이었고
                   그 자리가 전부 r2/r3 다. r3 는 in-sample Δ 가 **정확히 +0.0000** —
                   릿지가 β 를 0 으로 눌러버리려는데 격자가 안 따라간다는 뜻이다.
                   죽은 열이면 결합기 crossfit 의 자유도만 먹는다.

왜 12칸인데 학습은 2번인가
    기둥은 tie 결합기에만, LIN 라운드는 win 결합기에만 들어간다. 둘 다 GAT·LIN **학습**을
    바꾸지 않는다 (LIN 은 3라운드까지 한 번 학습해 두고 앞에서부터 잘라 쓴다 — 체인이
    순차라 r1..R 만 쓰는 것은 잘라내기와 정확히 같다).
    -> TR 설정마다 [기둥+GAT+LIN 3라운드] 를 한 번 학습하고, **결합기만 6번** 돌린다.
       결합기 1회는 7,748행 5-fold 로지스틱 x (tie·win) x (7k·10k) 라 수십 초다.

        TR=46,729 : 적재 -> SVD(캐시) -> 축재료 -> 기둥 -> GAT -> LIN(3R)  ─> 결합기 x6
        TR=49,729 : 〃 (SVD 는 TR 이 바뀌어 다시 낸다)                      ─> 결합기 x6

판정
    holdout 5,748 은 전 칸 무접촉. 행별 손실로 paired bootstrap 95% CI, 문턱 ±0.0005.
    기준 칸 = (TR 46,729, 기둥 4열, LIN 3R) = 현행 배포 구성. 기준선 런의 0.95970 을
    그대로 내야 정상이고, 그게 이 스크립트의 회귀 테스트다.

자동 검산 — 배선이 새면 여기서 멈춘다
    같은 TR·같은 라운드에서 기둥 on/off 의 **win_h 가 비트 동일** (기둥은 tie 축 전용)
    같은 TR·같은 기둥에서 라운드가 달라도 **tie_h 가 비트 동일** (LIN 은 win 축 전용)

사용
    python -u temp/factorial_heads.py
    python -u temp/factorial_heads.py --lin-rounds 1,3 --lam-span 2
    python -u temp/factorial_heads.py --only tr0_p4_r3,tr0_p0_r3
"""
import argparse
import gc
import re
import sys
import time
from itertools import product
from pathlib import Path

import json

import numpy as np
import pandas as pd

_F = Path(__file__).resolve().parent.parent          # final/
sys.path.insert(0, str(_F / 'temp'))
sys.path.insert(0, str(_F / 'models'))
import fit_heads as FH                                          # noqa: E402

TEMP = _F / 'temp'
THRESH = 0.0005
BASE = 'tr0_p4_r3'                                   # 현행 배포 구성 = Δ 의 기준


def cell_name(trp, drop, rounds):
    return f'{"tr+" if trp else "tr0"}_{"p0" if drop else "p4"}_r{rounds}'


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


_RD = re.compile(r'·r(\d+)$')


def lin_upto(cols, rounds):
    """LIN 체인을 r1..rounds 만 남긴다. 체인이 순차라 잘라내기 = 그 라운드까지 쓰기."""
    out = [(nm, c) for nm, c in cols if int(_RD.search(nm).group(1)) <= rounds]
    assert len(out) == len({nm for nm, _ in cols if _RD.search(nm)}) // 3 * rounds or out, \
        'LIN 열 이름이 예상 형식이 아니다'
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--only', default='', help='쉼표 구분 칸 이름')
    ap.add_argument('--lin-rounds', default='1,2,3', help='LIN 체인 라운드 수 후보')
    ap.add_argument('--tr', default='0,+', help="TR 설정: '0'=46,729 '+'=49,729")
    ap.add_argument('--lam-span', type=int, default=2, help='λ 힌트 반경. 2 -> 후보 5개')
    ap.add_argument('--arith-ckpt', default=None)
    ap.add_argument('--old-arith', action='store_true')
    ap.add_argument('--out', default=str(TEMP / 'factorial_heads.csv'))
    a = ap.parse_args()

    rounds = sorted({int(x) for x in a.lin_rounds.split(',') if x.strip()})
    trs = [x.strip() == '+' for x in a.tr.split(',') if x.strip()]
    assert rounds and max(rounds) <= 3 and min(rounds) >= 1, 'LIN 라운드는 1~3'
    cells = [(cell_name(t, d, r), t, d, r) for t, d, r in product(trs, (False, True), rounds)]
    if a.only:
        keep = set(a.only.split(','))
        bad = keep - {c[0] for c in cells}
        assert not bad, f'모르는 칸: {sorted(bad)}'
        cells = [c for c in cells if c[0] in keep]
    FH.LAM_SPAN = a.lam_span

    print(f'요인설계 {len(cells)}칸 — TR {len(trs)}설정 x 기둥 2 x LIN 라운드 {len(rounds)}')
    print(f'  기준 칸 {BASE} (TR 46,729 / 기둥 4열 / LIN 3R) — 기준선 런 0.95970 을 내야 정상')
    print(f'  λ 후보 {2 * a.lam_span + 1}개 (전 격자 {len(FH.LAMS_TR)}개) | 문턱 ±{THRESH}')
    print('  학습은 TR 설정당 1회 — 기둥·GAT·LIN(3R). 결합기만 칸마다.\n')

    ck = None
    if not a.old_arith:
        ck = Path(a.arith_ckpt) if a.arith_ckpt else FH.ARITH_CKPT
        if not (ck / 'meta.json').exists():
            ck = None

    rows, rowloss, weights, chk_win, chk_tie = [], {}, {}, {}, {}
    for trp in sorted({c[1] for c in cells}):
        mine = [c for c in cells if c[1] == trp]
        tag_tr = '49,729' if trp else '46,729'
        print('=' * 88)
        print(f'[TR {tag_tr}] 적재 -> SVD -> 축재료 -> 기둥 -> GAT -> LIN(3R)   (이 넷은 1회)')
        t0 = time.time()

        FH.TR_PLUS_DIRTY = trp
        FH.DROP_PILLAR = False                       # 기둥은 항상 계산해 둔다 (칸에서 골라 씀)
        FH.LAM_NS = 'trplus' if trp else 'base'
        gc.collect()
        S = FH.HeadSession().prepare()
        if ck is not None:
            oof, s2 = FH.arith_from_ckpt(ck, FH.TR['ids'], FH.ids)
            S.set_arith(oof_train=oof, s2=s2, ids_train=FH.TR['ids'], ids_s2=FH.ids)
            print(f'Arith <- {ck.name}')

        FH.HEADS = TEMP / f'heads_fact_{tag_tr.replace(",", "")}'
        FH.HEADS.mkdir(parents=True, exist_ok=True)
        FH.set_seeds(FH.SEED)
        FH._build_axes()
        FH._fit_pillar()
        FH._fit_gat()
        FH._fit_lin()
        pillar_full, lin_full = FH.PILLAR, FH.LIN_COLS
        print(f'[TR {tag_tr}] 공용 학습 {(time.time() - t0) / 60:.1f}분 | 기둥 {len(pillar_full)}열 '
              f'/ GAT {len(FH.GAT_COLS)}열 / LIN {len(lin_full)}열')

        for nm, _, drop, rd in mine:
            t1 = time.time()
            FH.PILLAR = None if drop else pillar_full
            FH.LIN_COLS = lin_upto(lin_full, rd)
            print(f'\n── [{nm}] 기둥 {"없음" if drop else "4열"} / LIN r1..r{rd} '
                  f'({len(FH.LIN_COLS)}열) — 결합기만 ──')
            FH._fit_combiner()
            r = FH._score()
            rowloss[nm] = r['row_hold']       # Δ 는 전 칸이 끝난 뒤 한 번에 (순서 무관)
            weights[nm] = FH.combiner_weights()
            # 검산 ① 기둥은 tie 축 전용 -> 같은 (TR, 라운드) 에서 win 이 같아야
            kw = (trp, rd)
            if kw in chk_win:
                assert abs(r['win_h'] - chk_win[kw]) < 1e-12, \
                    f'기둥이 승부축으로 샜다: {r["win_h"]} != {chk_win[kw]}'
            else:
                chk_win[kw] = r['win_h']
            # 검산 ② LIN 은 win 축 전용 -> 같은 (TR, 기둥) 에서 tie 가 같아야
            kt = (trp, drop)
            if kt in chk_tie:
                assert abs(r['tie_h'] - chk_tie[kt]) < 1e-12, \
                    f'LIN 라운드가 tie 축으로 샜다: {r["tie_h"]} != {chk_tie[kt]}'
            else:
                chk_tie[kt] = r['tie_h']
            rows.append(dict(칸=nm, TR=r['tr_rows'], 기둥=r['n_pillar'], GAT=r['n_gat'],
                             LIN=r['n_lin'], holdout=r['tot_h'], tie=r['tie_h'], win=r['win_h'],
                             ll_10k=r['tot_h_10k'], 초=round(time.time() - t1)))
            print(f'  holdout {r["tot_h"]:.5f} (tie {r["tie_h"]:.5f} / win {r["win_h"]:.5f}) '
                  f'| {rows[-1]["초"]}s')
        FH.PILLAR, FH.LIN_COLS = pillar_full, lin_full
        del pillar_full, lin_full
        gc.collect()

    # ── Δ 는 여기서. 순회 순서에 의존하지 않으므로 기준 선점 버그가 구조적으로 불가능 ──
    assert BASE in rowloss, f'기준 칸 {BASE} 가 없다 — Δ 를 낼 수 없다'
    ref = rowloss[BASE]
    for r in rows:
        if r['칸'] == BASE:
            r['Δ'], r['lo'], r['hi'], r['판정'] = 0.0, 0.0, 0.0, '기준'
        else:
            d, lo, hi = FH.paired_ci(rowloss[r['칸']], ref)
            r['Δ'], r['lo'], r['hi'], r['판정'] = d, lo, hi, verdict(lo, hi)

    # ── 재료 저장 — 재실행 없이 임의 쌍 대비·가중·λ곡선을 다시 볼 수 있게 ──
    base = Path(a.out).with_suffix('')
    np.savez_compressed(f'{base}.rowloss.npz', **rowloss)
    Path(f'{base}.weights.json').write_text(
        json.dumps(weights, ensure_ascii=False, indent=1), encoding='utf-8')
    FH.save_diag(f'{base}.diag.json')
    print('\n재료 저장')
    print(f'  {base}.rowloss.npz   {len(rowloss)}칸 x 5,748행 — 임의 쌍 paired CI')
    print(f'  {base}.weights.json  결합기 열별 가중')
    print(f'  {base}.diag.json     λ 곡선 + 유효 랭크 ({len(FH.DIAG)}체인)')

    R = pd.DataFrame(rows)[['칸', 'TR', '기둥', 'GAT', 'LIN', 'holdout', 'tie', 'win',
                            'Δ', 'lo', 'hi', '판정', 'll_10k', '초']]
    R = R.sort_values('holdout').reset_index(drop=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(a.out, index=False)
    print('\n' + '=' * 88)
    print(f'[결과] 기준 {BASE} | holdout 5,748 paired | 문턱 ±{THRESH} | 좋은 순')
    with pd.option_context('display.width', 240, 'display.max_columns', None):
        print(R.to_string(index=False, float_format=lambda x: f'{x:.5f}'))

    if BASE in weights:
        print('\n' + '=' * 88)
        print(f'[결합기 가중 — 기준 칸 {BASE}] 정규화 좌표계라 열끼리 크기 비교가 된다')
        for ax in ('tie', 'win'):
            d = weights[BASE][ax]
            tot = sum(abs(c['w']) for c in d['cols']) or 1.0
            print(f"\n  [{ax}] λ {d['lam']:g} | 무벌점 로짓 {d['n_logit']}열"
                  + (f" | 절편 {d['bias']:+.3f}" if d['bias'] is not None else ''))
            for c in sorted(d['cols'], key=lambda x: -abs(x['w'])):
                bar = '█' * int(round(abs(c['w']) / tot * 40))
                print(f"      {c['name']:16s} {c['w']:+8.4f} {abs(c['w']) / tot * 100:5.1f}% "
                      f"{'무벌점' if c['logit'] else '   λ  '} {bar}")

    if FH.DIAG:
        print('\n' + '=' * 88)
        print('[릿지 진단 — 곡선 깊이 = 그 풀링이 CV 에서 벌어준 양 / 유효랭크 = 살아남은 방향 수]')
        print(f"  {'체인':16s} {'λ':>10s} {'곡선깊이':>9s} {'유효랭크':>9s} {'벌점열':>7s}")
        for t, d in FH.DIAG.items():
            er = f"{d['eff_rank']:9.1f}" if d.get('eff_rank') is not None else '        —'
            print(f"  {t:16s} {d['lam']:>10g} {d['depth']:9.5f} {er} {d['n_pen']:>7d}")

    g = {r['칸']: r['holdout'] for r in rows}
    print('\n[주효과] 양수 = 손해')
    for rd in rounds:
        p0, p1 = g.get(cell_name(False, False, rd)), g.get(cell_name(False, True, rd))
        if p0 and p1:
            print(f'  기둥 제거 (TR 46,729, r{rd})  {p1 - p0:+.5f}')
    for drop in (False, True):
        a3, a1 = g.get(cell_name(False, drop, 3)), g.get(cell_name(False, drop, 1))
        if a3 and a1:
            print(f'  LIN 3R->1R (TR 46,729, 기둥 {"없음" if drop else "4열"})  {a1 - a3:+.5f}')
    for drop, rd in product((False, True), rounds):
        b0, b1 = g.get(cell_name(False, drop, rd)), g.get(cell_name(True, drop, rd))
        if b0 and b1:
            print(f'  TR +3,000 (기둥 {"없음" if drop else "4열"}, r{rd})  {b1 - b0:+.5f}')
    print(f'\n저장: {a.out}')
    print('* Δ 의 CI 는 기준 칸 대비 paired 다. 판정은 사용자.')


if __name__ == '__main__':
    main()
