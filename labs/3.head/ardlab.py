"""ardlab.py — 단계 2: 층2 벌점 **모양을 train 46,729 에서 학습**해 stage2 로 전이한다.

가설(사용자): stage2 의 ard 가 죽은 건 열별 판정에 ~34,000행이 필요한데 clean 이
7,748행뿐이라서다. train 은 46,729행 — 단, 열별 오염이 균등해야 한다. 그래서
arith_insample.py 가 만든 in-sample t_ar 로 5로짓 전부를 "자기 학습 행 예측"으로
통일한 판과, 통일 안 한 판(OOF t_ar)을 **둘 다** 잰다 — 차이가 균등화의 값어치다.

팔 5개 (전부 같은 stage2 결합기 계약 · 같은 교차적합 · 같은 holdout):
    base        [0×로짓5 | λ×g2]           현행 — 짝 기준
    ard         stage2 clean 에서 열별 적응   penlab 재현 대조 (기측정 +0.00007 노이즈)
    ard_tr_oof  train 에서 모양 학습, t_ar=OOF (오염 비대칭 그대로) → 전이
    ard_tr_ins  train 에서 모양 학습, t_ar=in-sample (동일 조건) → 전이
    base_hins   **층1 대조** — §3.6 헤드의 판독 직결 로짓에서 arith 두 열(t_ar·m_ar)의
                train 측만 in-sample 로 교체해 재학습, 결합기는 base 그대로.
                왜: 판독의 로짓 항은 "로짓 몫 흡수" 가 임무인데(ro_logit), 그 몫이
                deberta ×1.14~1.31 과대 / t_ar ×0.93 과소의 오염 혼합으로 측정된
                조건에서 §3.6 전체가 선택·학습됐다. 몸통의 잔차 목표가 바뀌는
                학습 동역학이라 층2가 못 고친다. 단, t_ar 는 현재 train↔배포가
                일치하는 유일한 열이라 균등화가 "전 열 배포 불일치" 와 맞바꿈이
                된다 — 방향은 실측만이 안다. 기대 스케일 참조: SK−PURE +0.00030.

전이 규약: train 층2 설계행렬 [로짓5(train) | g1_tr g2_tr] 에 penlab 'ard' 규약
(5회 재가중·전 열 적응, 전역 λ 는 train 5-fold CV)으로 적합한 **배포 벌점 벡터**를
평균 1 로 정규화해 모양으로 고정하고, stage2 crossfit 은 전역 세기 s 만 고른다.
g1_tr/g2_tr 은 배포 헤드(§3.6 TB)가 자기 학습 행을 예측한 값 — 로짓과 동일 조건.

주의: train 에서 모양이 잘 학습됐는지의 1차 판독은 t_ar 벌점이다. clean 관련도는
t_ar 가 5로짓 중 2위이므로, 동일 조건화 후에도 train-ARD 가 t_ar 를 세게 벌하면
행 수 문제가 아니라 모양 자체가 전이 불가라는 뜻이 된다.

예상: 시드당 헤드 2×17s + base 2×18s + ard 55s + 모양학습 2×~20s + 전이 2×~18s ≈ 3.5분
     → 3시드 ≈ 11분 + 적재 1분.
"""
import argparse
import json
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

import tie_ladder as TL
import penlab as PL
import readout as RO

FINAL = TL.FINAL
NAMES = TL.TIE_NAMES + ['g1', 'g2']
ARMS = ('base', 'ard', 'ard_tr_oof', 'ard_tr_ins', 'base_hins')


def assets_ins(A, U, tie_ins, win_ins):
    """gat_assets 사본 — arith 열의 **train 측만** in-sample 로 바꾼 헤드 학습 자산.

    정규화 통계는 계약대로(gat_assets 와 같은 식) 수정된 train 로짓에서 다시 계산해
    train·stage2 양쪽에 적용한다. stage2 로짓 값 자체(U['tie_base']·U['win_base'])와
    SVD 풀링은 불변 — 바뀌는 건 헤드가 학습 중에 보는 판독 직결 입력뿐이다."""
    mu_u, sc_u = TL.zstats(tie_ins, True)
    _, sc_v = TL.zstats(win_ins, False)
    A2 = dict(A)
    A2['Lu_tr'] = (tie_ins - mu_u) / sc_u
    A2['Lv_tr'] = win_ins / sc_v
    A2['Lu_s2'] = (U['tie_base'] - mu_u) / sc_u
    A2['Lv_s2'] = U['win_base'] / sc_v
    return A2


def tb_cols2(A, seed):
    """penlab.tb_cols 와 같은 §3.6 헤드 — stage2 2열에 더해 **train 행의 라운드 증분
    2열**을 복원한다. fit_gat 은 z_tr = a·(s0/rms) + r_tr 로 누적만 돌려주므로
    앵커 몫을 되빼서 r_tr 을 얻는다 (r1 은 무앵커라 z_tr 그대로)."""
    K = A['CUt'].shape[1]
    nu, nv = A['Lu_tr'].shape[1], A['Lv_tr'].shape[1]
    sk = list(range(K, K + nu)) + list(range(2 * K + nu, 2 * K + nu + nv))
    pool = list(range(0, K)) + list(range(K + nu, 2 * K + nu))
    ct = np.ascontiguousarray(np.hstack([A['CUt'], A['Lu_tr']]), np.float32)
    cv = np.ascontiguousarray(np.hstack([A['CVt'], A['Lv_tr']]), np.float32)
    cs = np.ascontiguousarray(np.hstack([A['CUs'], A['Lu_s2']]), np.float32)
    vs = np.ascontiguousarray(np.hstack([A['CVs'], A['Lv_s2']]), np.float32)
    z, cols_s2, cols_tr = None, [], []
    for unit in ('TANHTANH', 'BILIN'):
        net = RO.make_factory(unit, 'LIN', sk, pool)
        r, aux = TL.fit_gat(ct, cv, A['y01m'], z, True, seed, cs, vs,
                            net=net, sym_fn=RO.sym_post)
        cols_s2.append(r)
        cols_tr.append(aux['z_tr'] if z is None
                       else aux['z_tr'] - aux['a'] * (z / aux['rms']))
        z = aux['z_tr']
    return np.column_stack(cols_s2), np.column_stack(cols_tr)


def train_shape(Xd, y01, tag):
    """train 우주에서 ARD 모양 학습 — penlab 'ard' 규약(5회 재가중·전 열 적응).
    전역 λ 는 5-fold CV(웜스타트·큰 λ 부터), 표준화는 층2 tie 규약(전 열
    중심화+표준화·절편 O). 전 행 단일적합의 벌점 벡터를 평균 1 로 돌려준다."""
    n, p = Xd.shape
    fold = np.zeros(n, int)
    for j, (_, iv) in enumerate(StratifiedKFold(TL.K, shuffle=True, random_state=TL.SEED)
                                .split(np.zeros((n, 1)), y01)):
        fold[iv] = j
    Z = [TL._z(Xd[fold != j], Xd[fold == j], p) for j in range(TL.K)]
    score, warm = {}, [None] * TL.K
    for lam in sorted(PL.LAMS, reverse=True):
        tot = 0.0
        for j in range(TL.K):
            A_, B_ = Z[j]
            v, _ = PL.fit_struct('ard', A_, y01[fold != j], True, 0, (lam,), (), warm[j])
            warm[j] = v
            s = TL._scores(v, B_, p, True)
            yy = y01[fold == j]
            tot += float((np.logaddexp(0, s) - yy * s).sum())
        score[lam] = tot
    lam = min(score, key=score.get)
    A_, _ = TL._z(Xd, Xd, p)
    v, pen = PL.fit_struct('ard', A_, y01, True, 0, (lam,), ())
    shape = pen / pen.mean()
    print(f'    [{tag}] train λ* {lam:g} · 모양 ' +
          ' '.join(f'{nm} {q:.2f}' for nm, q in zip(NAMES, shape)))
    return shape, lam, v


def crossfit_fixed(X, y01, fold, rows, shape):
    """penlab.crossfit_struct 사본 — 구조 대신 **고정 모양** shape 의 전역 세기 s 만
    안쪽 CV 로 고른다. pen = s·shape (shape 평균 1 → s 는 λ 와 같은 눈금)."""
    p = X.shape[1]
    z = np.full(len(y01), np.nan)
    picked = []
    G = [(l,) for l in sorted(PL.LAMS, reverse=True)]
    for k in range(PL.K_OUT):
        tr, te = (fold != k) & rows, (fold == k) & rows
        Xtr, ytr = X[tr], y01[tr]
        inner = np.zeros(len(ytr), int)
        for j, (_, iv) in enumerate(StratifiedKFold(PL.K_IN, shuffle=True,
                                                    random_state=TL.SEED)
                                    .split(np.zeros((len(ytr), 1)), ytr)):
            inner[iv] = j
        Zi = [TL._z(Xtr[inner != j], Xtr[inner == j], p) for j in range(PL.K_IN)]
        score, warm = {}, [None] * PL.K_IN
        for q in G:
            tot = 0.0
            for j in range(PL.K_IN):
                A_, B_ = Zi[j]
                v, _ = TL._fit(A_, ytr[inner != j], q[0] * shape, True, warm[j])
                warm[j] = v
                s = TL._scores(v, B_, p, True)
                yy = ytr[inner == j]
                tot += float((np.logaddexp(0, s) - yy * s).sum())
            score[q] = tot
        q = min(score, key=score.get)
        picked.append(q)
        A_, B_ = TL._z(Xtr, X[te], p)
        v, _ = TL._fit(A_, ytr, q[0] * shape, True)
        z[te] = TL._scores(v, B_, p, True)
    A_, _ = TL._z(X[rows], X[rows], p)
    q = picked[len(picked) // 2]
    v, _ = TL._fit(A_, y01[rows], q[0] * shape, True)
    return z, dict(picked=[float(x[0]) for x in picked], v=[float(x) for x in v],
                   pen=[float(x) for x in (q[0] * shape)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bundle', default='temp/heads_repro')
    ap.add_argument('--ins', default='temp/arith_xgb_insample_train.npz')
    ap.add_argument('--seeds', default='42,1,2')
    ap.add_argument('--lams', default=None,
                    help='λ 격자 (콤마). penlab 격자 오버라이드 — 기본 1,10,100,1000,10000')
    ap.add_argument('--out', default='temp/ardlab')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s]
    if args.lams:
        PL.LAMS = [float(x) for x in args.lams.split(',')]

    print('[적재]')
    U = TL.build(FINAL / args.bundle)
    y_t = (U['y'] == 2).astype(np.float64)
    hold, cu = U['hold'], U['cu']
    A = TL.gat_assets(U)
    ids, y46, tie_tr, win_tr = TL.load_train_logits()

    z = np.load(FINAL / args.ins)
    assert (z['ids_train'] == ids).all() and (z['y_train'] == y46).all(), 'in-sample 정렬'
    m_ins, t_ins = TL.sd3(z['logp_ins_train__xgb'].astype(np.float64))
    tie_ins = tie_tr.copy()
    tie_ins[:, 4] = t_ins
    win_ins = win_tr.copy()
    win_ins[:, 2] = m_ins
    A2 = assets_ins(A, U, tie_ins, win_ins)
    print(f'  in-sample t_ar 적재 (best_iter {int(z["best_iteration"])}) · '
          f'라벨상관 {np.corrcoef(t_ins, A["y01m"])[0, 1]:.4f} '
          f'(OOF {np.corrcoef(tie_tr[:, 4], A["y01m"])[0, 1]:.4f})')

    res, rows_ = {}, {}

    def record(arm, sd, z_, dep, sec):
        ll = TL.rowloss(z_[cu], y_t[cu])
        h = hold[cu]
        tag = f'{arm}-s{sd}'
        res[tag] = dict(arm=arm, seed=sd, ll=float(ll[h].mean()), **dep,
                        sec=round(sec, 1))
        rows_[tag] = ll[h]
        print(f'  [{tag}] ll(hold) {ll[h].mean():.5f}  '
              f'(선택 {dep["picked"]} · {sec:.0f}s)')

    for sd in seeds:
        t0 = time.time()
        G_s2, G_tr = tb_cols2(A, sd)
        print(f'\n[seed {sd}] 헤드 TB [{time.time() - t0:.0f}s]')
        X_s2 = np.hstack([U['tie_base'], G_s2])
        for arm, kind in (('base', 'base'), ('ard', 'ard')):
            t0 = time.time()
            z_, dep = PL.crossfit_struct(X_s2, y_t, U['fold'], cu, True, 5, kind)
            record(arm, sd, z_, dep, time.time() - t0)
        for arm, L46 in (('ard_tr_oof', tie_tr), ('ard_tr_ins', tie_ins)):
            t0 = time.time()
            shape, lam_t, v_t = train_shape(np.hstack([L46, G_tr]), A['y01m'],
                                            f'{arm}-s{sd}')
            z_, dep = crossfit_fixed(X_s2, y_t, U['fold'], cu, shape)
            dep.update(train_lam=float(lam_t), shape=[float(x) for x in shape],
                       train_v=[float(x) for x in v_t])
            record(arm, sd, z_, dep, time.time() - t0)
        # 층1 대조 — 판독 직결의 arith 열만 in-sample 로 바꾼 헤드, 결합기는 base
        t0 = time.time()
        G_s2i, _ = tb_cols2(A2, sd)
        Xi = np.hstack([U['tie_base'], G_s2i])
        z_, dep = PL.crossfit_struct(Xi, y_t, U['fold'], cu, True, 5, 'base')
        record('base_hins', sd, z_, dep, time.time() - t0)

    print(f'\n{"=" * 78}\n[대조] base 기준 · 같은 열 · paired · 시드평균\n{"=" * 78}')
    m0 = np.mean([rows_[f'base-s{s}'] for s in seeds], axis=0)
    print(f'  {"팔":12s}{"ll(hold)":>11s}   base 대비 (시드평균 paired)      시드부호')
    for arm in ARMS:
        v = np.array([res[f'{arm}-s{s}']['ll'] for s in seeds])
        if arm == 'base':
            print(f'  {arm:12s}{v.mean():11.5f}   기준')
            continue
        cur = np.mean([rows_[f'{arm}-s{s}'] for s in seeds], axis=0)
        d = v - np.array([res[f'base-s{s}']['ll'] for s in seeds])
        m, lo, hi = TL.paired_ci(cur, m0)
        print(f'  {arm:12s}{v.mean():11.5f}   {TL.ci_str(m, lo, hi)}  '
              f'{int((d < 0).sum())}/{len(d)}')

    print(f'\n{"=" * 78}\n[진단] 배포 단일적합의 벌점 (열 이름 순: {" ".join(NAMES)})\n{"=" * 78}')
    for tag in sorted(res):
        d = res[tag]
        print(f'  {tag:18s} 벌점 {np.round(np.array(d["pen"]), 1)}')
        print(f'  {"":18s} 계수 {np.round(np.array(d["v"])[:7], 3)}')

    op = FINAL / args.out
    np.savez_compressed(op.with_suffix('.rowloss.npz'), **rows_)
    with open(op.with_suffix('.json'), 'w') as f:
        json.dump({'cells': res}, f, ensure_ascii=False, indent=1)
    print(f'\n저장: {op}.rowloss.npz / {op}.json')


if __name__ == '__main__':
    main()
