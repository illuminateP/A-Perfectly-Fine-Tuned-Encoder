"""Arith 풀링 스내칭 프로브 — 산술 피쳐 255를 다섯 번째 소스로 양축에 붙인다.

지금 Arith 는 로짓(m_ar·t_ar)으로만 참여한다. 이 프로브는 로짓 뒤에 있던 피쳐 벡터
자체를 헤드에 공급한다 — 다른 4소스의 풀링 스내칭과 같은 문법.

파리티 조립 — 카탈로그 `swap` 열로 **대수적으로** 만든다 (스왑 재추론 없음):
    불변 178                     → u
    부호반전 58                  → v
    교환쌍 9쌍(18)               → 합/2 → u 9 · 차/2 → v 9
    역수 1 (ratio_token_len)     → log 후 부호반전 → v 1
    합계  u 187 | v 68 · 정규화 TR 통계 (u z-score · v 무중심 RMS — 홀 계약)

승부축 (시드 5,6,7,8,9 — winconf 저장 열 재사용 · **ar 헤드 5망만 신규 학습**):
    W4-BASE   [m3 | W4]          winconf SK-RAW 1R 재현 — 하네스 검사 (비트 일치 기대)
    W5-AR     [m3 | W4 | ar]     ar 헤드 = GATE·h128·SK 판독·1R·무절편·seed+4 (si=4 관례)
    AR-ONLY   [m3 | ar]          ar 단독 정보량 (참고)

tie축 (시드 42,1,2 — 두 팔 모두 in-run 학습 · §3.6 확정 TB 재료):
    TIE-BASE  몸통 [CUt 512 | CVt 512]             r1 TANHTANH → r2 BILIN 잔차앵커
    TIE-AR    몸통 [CUt|ar_u 187 | CVt|ar_v 68]    동일 구조 · 몸통 입력만 +255
    판독 LIN_SK(로짓 8 직결·절편 有) · 결합기 [tie5|g1 g2]+절편 (base 벌점 — Δ 전용,
    blk2 를 안 켰으므로 절대값은 §3.7 대비 +0.0004 높게 나온다)
    참조점: TIE-BASE g2 3시드 평균 ≈ 0.56883 (§3.7 표의 base · 같은 시드·같은 하네스)

사전등록:
    승부 전진: (W5-AR − W4-BASE) ≤ −0.0003 & 시드 4/5     (층2 열 1개 분산 비용 ~0.0001 내재)
    tie  전진: (TIE-AR − TIE-BASE) ≤ −0.0005 & 시드 3/3   (전진 시 새 시드 확인이 다음 단계)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import tie_ladder as TL
import readout as RO
import winlab as WL

FINAL = TL.FINAL


# ══════════════════════════════════════════════ Arith 소스 조립
def parity_assemble(M, use):
    """swap 사상 → (u, v). 성질: M 을 스왑하면 u 불변 · v 부호반전 (테스트가 이를 검증)."""
    col = {f: j for j, f in enumerate(use.index)}
    inv = [f for f, s in use['swap'].items() if s == '불변']
    neg = [f for f, s in use['swap'].items() if s == '부호반전']
    rec = [f for f, s in use['swap'].items() if s == '역수']
    pairs = sorted({tuple(sorted((f, s.split(':', 1)[1])))
                    for f, s in use['swap'].items() if s.startswith('교환:')})
    for a, b in pairs:                                    # 사상 왕복 검증
        assert use['swap'][a] == f'교환:{b}' and use['swap'][b] == f'교환:{a}', (a, b)
    assert (len(inv), len(neg), len(pairs), len(rec)) == (178, 58, 9, 1), \
        f'카탈로그 구성 변경? {len(inv)}/{len(neg)}/{len(pairs)}/{len(rec)}'

    u_parts = [M[:, [col[f] for f in inv]]] + \
              [((M[:, col[a]] + M[:, col[b]]) / 2)[:, None] for a, b in pairs]
    v_parts = [M[:, [col[f] for f in neg]]] + \
              [((M[:, col[a]] - M[:, col[b]]) / 2)[:, None] for a, b in pairs]
    r = M[:, col[rec[0]]]
    assert (r > 0).all(), f'역수 피쳐 {rec[0]} 에 비양수 — log 불가'
    v_parts.append(np.log(r)[:, None])                    # swap: x→1/x ⇒ log x 는 부호반전
    return np.hstack(u_parts), np.hstack(v_parts), dict(
        inv=inv, neg=neg, pairs=pairs, rec=rec)


def load_ar(U):
    """카탈로그 swap 사상으로 (u, v) 를 조립한다. TR·stage2 정렬은 id 로 검증."""
    cat = pd.read_csv(FINAL / 'origin/feature_catalog.csv')
    use = cat[cat['사용'] == 1].set_index('feature')
    F = pd.read_parquet(FINAL / 'models/ARITH/features_57k.parquet')
    fid = pd.Index(F['id'])

    tr_ids = np.load(FINAL / 'models/logits/abe2__checkpoint-3000__train.npz')['train_ids']
    i_tr, i_s2 = fid.get_indexer(tr_ids), fid.get_indexer(U['ids'])
    assert (i_tr >= 0).all() and (i_s2 >= 0).all(), '파케이 id 커버리지'

    M = F[use.index.tolist()].to_numpy(np.float64)
    assert np.isfinite(M).all(), '비유한 피쳐값'
    u_all, v_all, meta = parity_assemble(M, use)
    inv, neg, pairs = meta['inv'], meta['neg'], meta['pairs']
    mu_u, sc_u = TL.zstats(u_all[i_tr], True)
    _, sc_v = TL.zstats(v_all[i_tr], False)
    AR = dict(u_tr=((u_all - mu_u) / sc_u)[i_tr].astype(np.float32),
              v_tr=(v_all / sc_v)[i_tr].astype(np.float32),
              u_s2=((u_all - mu_u) / sc_u)[i_s2].astype(np.float32),
              v_s2=(v_all / sc_v)[i_s2].astype(np.float32))
    q = np.quantile(np.abs(AR['v_tr']), 0.999)
    print(f'  arith 소스 — u {AR["u_tr"].shape[1]} | v {AR["v_tr"].shape[1]} '
          f'(불변 {len(inv)} · 부호반전 {len(neg)} · 교환쌍 {len(pairs)} · 역수 1)\n'
          f'    정규화 후 |v| 99.9분위 {q:.1f}  (꼬리 포화 진단 — 풀링은 보통 3~5)')
    return AR


# ══════════════════════════════════════════════ 승부축
def win_probe(U, A, AR, seeds, res, roww):
    Z = np.load(FINAL / 'temp/winconf.rowloss.npz')
    conf = json.load(open(FINAL / 'temp/winconf.json'))['cells']
    m = A['y_tr'] != 2
    y01w = (A['y_tr'] == 0).astype(np.float64)[m]

    du, dv = AR['u_tr'].shape[1], AR['v_tr'].shape[1]
    n_lu, n_lv = A['Lu_tr'].shape[1], A['Lv_tr'].shape[1]
    d = du + n_lu + dv + n_lv
    sk = list(range(du, du + n_lu)) + list(range(du + n_lu + dv, d))
    pool = list(range(0, du)) + list(range(du + n_lu, du + n_lu + dv))
    cu_t = np.ascontiguousarray(np.hstack([AR['u_tr'][m], A['Lu_tr'][m]]), np.float32)
    cv_t = np.ascontiguousarray(np.hstack([AR['v_tr'][m], A['Lv_tr'][m]]), np.float32)
    cu_s = np.ascontiguousarray(np.hstack([AR['u_s2'], A['Lu_s2']]), np.float32)
    cv_s = np.ascontiguousarray(np.hstack([AR['v_s2'], A['Lv_s2']]), np.float32)

    rows = U['cu'] & U['nz']
    y01 = (U['y'] == 0).astype(np.float64)

    def ev(tag, W, note=''):
        X = U['win_base'] if W is None else np.hstack([U['win_base'], W])
        z, dep = TL.combiner(X, y01, U['fold'], rows, bias=False,
                             n_logit=U['win_base'].shape[1])
        ll = TL.rowloss(z[rows], y01[rows])[U['hold'][rows]]
        res[tag], roww[tag] = dict(win=float(ll.mean())), ll
        print(f'    [{tag:14s}] 승부(hold) {ll.mean():.5f}  '
              f'(열 {X.shape[1]} · λ {dep["lam"]:g}){note}')

    print(f'\n{"="*78}\n[승부축] ar 헤드 GATE·h128·SK·1R — 시드당 1망\n{"="*78}')
    ev('logit', None, '  ← 로짓만 [m3]')
    for sd in seeds:
        t0 = time.time()
        net = WL.make_net('GATE', sk, pool, du)
        r_s2, aux = TL.fit_gat(cu_t, cv_t, y01w, None, False, sd + 4, cu_s, cv_s,
                               hidden=128, net=net, sym_fn=RO.sym_post)
        print(f'  [시드 {sd}] ar 헤드 — es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]} '
              f'[{time.time() - t0:.0f}s]')
        W4 = Z[f'cols__SK-RAW-s{sd}'][:, :4].astype(np.float64)
        ar = r_s2[:, None]
        ev(f'W4-BASE-s{sd}', W4)
        ref = conf[f'SK-RAW-s{sd}-g1']['win']
        d0 = abs(res[f'W4-BASE-s{sd}']['win'] - ref)
        assert d0 < 5e-6, f'winconf 재현 실패 s{sd}: Δ={d0:.2e}'
        ev(f'W5-AR-s{sd}', np.hstack([W4, ar]))
        ev(f'AR-ONLY-s{sd}', ar)
    print('  하네스 검사 — W4-BASE 5칸 전부 winconf g1 과 일치')


# ══════════════════════════════════════════════ tie축
def tie_chain(A, seed, with_ar, AR, h=128):
    """TB 체인 (r1 TANHTANH → r2 BILIN 잔차앵커) · LIN_SK 판독 · g1 g2 반환."""
    if with_ar:
        cu_t = np.ascontiguousarray(
            np.hstack([A['CUt'], AR['u_tr'], A['Lu_tr']]), np.float32)
        cv_t = np.ascontiguousarray(
            np.hstack([A['CVt'], AR['v_tr'], A['Lv_tr']]), np.float32)
        cu_s = np.ascontiguousarray(
            np.hstack([A['CUs'], AR['u_s2'], A['Lu_s2']]), np.float32)
        cv_s = np.ascontiguousarray(
            np.hstack([A['CVs'], AR['v_s2'], A['Lv_s2']]), np.float32)
    else:
        cu_t, cv_t, cu_s, cv_s = A['CU_in'], A['CV_in'], A['CU_s2'], A['CV_s2']
    n_cu, n_cv = cu_t.shape[1], cv_t.shape[1]
    n_lu, n_lv = A['Lu_tr'].shape[1], A['Lv_tr'].shape[1]
    pool = list(range(0, n_cu - n_lu)) + list(range(n_cu, n_cu + n_cv - n_lv))
    sk = list(range(n_cu - n_lu, n_cu)) + list(range(n_cu + n_cv - n_lv, n_cu + n_cv))

    cols, zs = [], None
    for unit in ('TANHTANH', 'BILIN'):
        def net(d, hh, _u=unit, _s=sk, _p=pool):
            return RO.SplitNet(d, hh, _u, 'LIN', _s, _p)
        r_s2, aux = TL.fit_gat(cu_t, cv_t, A['y01m'], zs, True, seed, cu_s, cv_s,
                               hidden=h, net=net, sym_fn=RO.sym_post)
        cols.append(r_s2)
        print(f'      r{len(cols)} {unit:9s} es_ll {aux["es_ll"]:.5f} ep{aux["es_epoch"]}')
        zs = aux['z_tr']
    return np.column_stack(cols)


def tie_probe(U, A, AR, seeds, res, roww):
    rows = U['cu']
    y01 = (U['y'] == 2).astype(np.float64)

    def ev(tag, G):
        X = np.hstack([U['tie_base'], G])
        z, dep = TL.combiner(X, y01, U['fold'], rows, bias=True,
                             n_logit=U['tie_base'].shape[1])
        ll = TL.rowloss(z[rows], y01[rows])[U['hold'][rows]]
        res[tag], roww[tag] = dict(tie=float(ll.mean())), ll
        print(f'    [{tag:14s}] tie(hold) {ll.mean():.5f}  (λ {dep["lam"]:g})')

    print(f'\n{"="*78}\n[tie축] TB·h128·2R — 팔당 시드당 2망\n{"="*78}')
    for sd in seeds:
        for arm, with_ar in (('TIE-BASE', False), ('TIE-AR', True)):
            print(f'  [{arm}-s{sd}]  몸통 {"[CUt|ar_u|CVt|ar_v]" if with_ar else "[CUt|CVt]"}')
            t0 = time.time()
            G = tie_chain(A, sd, with_ar, AR)
            print(f'      [{time.time() - t0:.0f}s]')
            ev(f'{arm}-s{sd}', G)


# ══════════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bundle', default='temp/heads_repro')
    ap.add_argument('--win_seeds', default='5,6,7,8,9')
    ap.add_argument('--tie_seeds', default='42,1,2')
    ap.add_argument('--axes', default='win,tie')
    ap.add_argument('--out', default='temp/arithlab')
    args = ap.parse_args()

    axes = [a for a in args.axes.split(',') if a]
    ws = [int(s) for s in args.win_seeds.split(',') if s]
    ts = [int(s) for s in args.tie_seeds.split(',') if s]

    print('[적재]')
    U = TL.build(FINAL / args.bundle)
    A = TL.gat_assets(U)
    AR = load_ar(U)

    res, roww = {}, {}
    if 'win' in axes:
        win_probe(U, A, AR, ws, res, roww)
    if 'tie' in axes:
        tie_probe(U, A, AR, ts, res, roww)

    op = FINAL / args.out
    with open(op.with_suffix('.json'), 'w') as f:
        json.dump({'cells': res, 'win_seeds': ws, 'tie_seeds': ts}, f,
                  ensure_ascii=False, indent=1)
    np.savez_compressed(op.with_suffix('.rowloss.npz'), **roww)
    print(f'\n저장: {op}.json / {op}.rowloss.npz')

    print(f'\n{"="*78}\n[요약]\n{"="*78}')
    if 'win' in axes:
        mw = lambda t: np.mean([res[f'{t}-s{s}']['win'] for s in ws])
        rw = lambda t: np.mean([roww[f'{t}-s{s}'] for s in ws], axis=0)
        vw = lambda t: np.array([res[f'{t}-s{s}']['win'] for s in ws])
        print(f'  승부축 ({len(ws)}시드) — W4-BASE {mw("W4-BASE"):.5f} · '
              f'로짓만 {res["logit"]["win"]:.5f}')
        for t in ('W5-AR', 'AR-ONLY'):
            m, lo, hi = TL.paired_ci(rw(t), rw('W4-BASE'))
            d = vw(t) - vw('W4-BASE')
            print(f'    {t:8s} − W4-BASE  {TL.ci_str(m, lo, hi)}  '
                  f'시드 {d.mean():+.5f} {int((d < 0).sum())}/{len(d)}')
        print('    사전등록: W5-AR ≤ −0.0003 & 4/5 → 새 시드 확인으로 전진')
    if 'tie' in axes:
        mt = lambda t: np.mean([res[f'{t}-s{s}']['tie'] for s in ts])
        rt = lambda t: np.mean([roww[f'{t}-s{s}'] for s in ts], axis=0)
        vt = lambda t: np.array([res[f'{t}-s{s}']['tie'] for s in ts])
        print(f'  tie축 ({len(ts)}시드) — TIE-BASE {mt("TIE-BASE"):.5f} '
              f'(§3.7 base 참조 0.56883 · 같은 시드)')
        m, lo, hi = TL.paired_ci(rt('TIE-AR'), rt('TIE-BASE'))
        d = vt('TIE-AR') - vt('TIE-BASE')
        print(f'    TIE-AR   − TIE-BASE  {TL.ci_str(m, lo, hi)}  '
              f'시드 {d.mean():+.5f} {int((d < 0).sum())}/{len(d)}')
        print('    사전등록: ≤ −0.0005 & 3/3 → 새 시드 확인으로 전진')
    print('\n  ! 총점 환산 — 승부 Δ×0.691 · tie Δ×1.0 (체인 룰 분해)')
    print('  ! tie 결합기는 base 벌점 (blk2 미적용) — Δ 로만 읽는다')


if __name__ == '__main__':
    main()
