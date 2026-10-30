"""monohead.py — **일반적인 BERT CLS 헤드**로 3진 분류했으면 어땠나. 기준선 실험.

우리 설계는 전부 패리티 분해 위에 서 있다 — 짝/홀로 쪼개고, 축마다 헤드를 따로 두고,
결합기를 두 개 돌린다. 그게 값을 하는지 재려면 **분해를 안 한 판**과 붙여야 한다.
이 파일은 그 대조군이다.

    입력    로짓 8 + 풀링을 **한 번에**. 축 분해 없음. 소스별 A_s·B_s 를 그대로 쓴다
    피쳐    raw   [A │ B]                                        2×4,096 =  8,192
            feat  [A │ B │ A−B │ |A−B| │ tanh(A)⊙tanh(B)]        5×4,096 = 20,480
                  ← InferSent/SBERT 쌍 인코딩의 고전 피쳐. |A−B|·⊙ 는 **원 좌표 원소별**
                    이라 SVD 계수 공간에서 만들 수 없다 → raw 로 간다
    헤드    d → h (gelu) → FFN(h → 4h → h, gelu, 잔차) → 3 → softmax
            BERT 분류 헤드의 표준 형태. h=128 이면 약 2.75M 파라미터 (우리 tie 헤드 525K)
    학습    TR 46,729 · **3-class CrossEntropy** · 전체 배치 · ES = TR 15% · AdamW

평가 두 가지
    (a) 단독   이 헤드의 softmax 만으로 holdout 3-class ll. 층 2 없음 —
               "결합기 없이 그냥 BERT 헤드" 의 값
    (b) 통합   출력 3로짓을 sd3 로 (m, t) 분해해 각 축 결합기에 열로 넣는다.
               우리 프로토콜 그대로라 **공정 비교**. 대체(mono 만)와 얹기(현행+mono) 둘 다

패리티 주의 — 이 헤드에는 대칭 제약이 **없다**. A/B 를 바꾸면 예측이 자리만 바뀌는 보장이
없어서, 표준 관행대로 추론 때 swap-TTA(정방향 + 역방향[1,0,2] 평균)를 쓴다.
TTA 유/무를 모두 보고한다 — 그 차이가 곧 **패리티 구조가 공짜로 주던 것**의 크기다.
"""
import argparse
import gc
import json
import time

import numpy as np
import torch
import torch.nn as nn

import tie_ladder as TL

FINAL = TL.FINAL
SRCS = ('abe2', 'tb2', 'abe1', 'tb1')
FEATS = ('raw', 'feat')
EPOCHS, PATIENCE, LR, WD = 3000, 200, 3e-4, 1e-2


class MonoHead(nn.Module):
    """d → h(gelu) → FFN(h→4h→h, gelu, 잔차) → 3.  BERT 분류 헤드의 표준 형태."""

    def __init__(self, d, h):
        super().__init__()
        self.proj = nn.Linear(d, h)
        self.f1, self.f2 = nn.Linear(h, 4 * h), nn.Linear(4 * h, h)
        self.out = nn.Linear(h, 3)

    def forward(self, x):
        g = torch.nn.functional.gelu
        u = g(self.proj(x))
        u = u + self.f2(g(self.f1(u)))
        return self.out(u)


def raw_ab(ids_s2):
    """소스별 raw A·B (패리티 분해 **전**). 뷰평균만 접는다. 반환 (TR, stage2) 각 (n, 4096)."""
    M = TL.MODELS
    spec = [('abe2__checkpoint-3000', ('ea',), ('eb',)),
            ('tb2__checkpoint-4000', ('a_f', 'b_s'), ('b_f', 'a_s')),
            (f'abe1__checkpoint-{TL.ABE1_STEP}', ('ea',), ('eb',)),
            (f'tb1__checkpoint-{TL.TB1_STEP}', ('x_f',), ('x_s',))]
    At, Bt, As, Bs = [], [], [], []
    for tag, ka, kb in spec:
        zt, zs = np.load(M / f'pooled/{tag}__train.npz'), np.load(M / f'pooled/{tag}.npz')
        assert (np.concatenate([zs['valid_ids'], zs['hold_ids']]) == ids_s2).all(), f'{tag} id'
        g = lambda z, pre, ks: sum(z[f'{pre}_{k}'].astype(np.float32) for k in ks) / len(ks)
        c = lambda z, ks: sum(np.vstack([z[f'valid_{k}'], z[f'hold_{k}']]).astype(np.float32)
                              for k in ks) / len(ks)
        At.append(g(zt, 'train', ka)); Bt.append(g(zt, 'train', kb))
        As.append(c(zs, ka)); Bs.append(c(zs, kb))
    out = [np.hstack(x) for x in (At, Bt, As, Bs)]
    del At, Bt, As, Bs
    gc.collect()
    print(f'  raw A·B 적재 — TR {out[0].shape} · stage2 {out[2].shape}')
    return out


def build_feats(A, B, kind):
    """피쳐 조립. 미리 할당하고 블록마다 채워 임시 사본을 줄인다."""
    n, d = A.shape
    nb = 2 if kind == 'raw' else 5
    X = np.empty((n, d * nb), np.float32)
    X[:, :d], X[:, d:2 * d] = A, B
    if kind == 'feat':
        np.subtract(A, B, out=X[:, 2 * d:3 * d])
        np.abs(X[:, 2 * d:3 * d], out=X[:, 3 * d:4 * d])
        tA = np.tanh(A)
        np.multiply(tA, np.tanh(B), out=X[:, 4 * d:5 * d])
        del tA
        gc.collect()
    return X


def fit_mono(Xtr, y_tr, Xs2, Xs2_sw, h, seed, tag):
    """3-class CE · 전체 배치 · ES = TR 15%. 반환 stage2 3로짓 (원본 / swap 입력)."""
    torch.manual_seed(seed)
    dev = TL.DEVICE
    f = MonoHead(Xtr.shape[1], h).to(dev)
    opt = torch.optim.AdamW(f.parameters(), lr=LR, weight_decay=WD)
    eidx, tidx = TL._es_split(len(y_tr), seed)
    X = torch.tensor(Xtr, device=dev)
    Y = torch.tensor(y_tr, dtype=torch.long, device=dev)
    Xe, Ye, Xt, Yt = X[eidx], Y[eidx], X[tidx], Y[tidx]
    lossf = nn.CrossEntropyLoss()
    best, best_sd, bad, used = np.inf, None, 0, 0
    for ep in range(EPOCHS):
        f.train(); opt.zero_grad()
        lossf(f(Xt), Yt).backward()
        opt.step()
        f.eval()
        with torch.no_grad():
            v = float(lossf(f(Xe), Ye))
        if v < best - 1e-6:
            best, bad, used = v, 0, ep + 1
            best_sd = {k: t.detach().clone() for k, t in f.state_dict().items()}
        elif (bad := bad + 1) >= PATIENCE:
            break
    f.load_state_dict(best_sd)
    del X, Xe, Xt
    torch.cuda.empty_cache()
    with torch.no_grad():
        L = f(torch.tensor(Xs2, device=dev)).cpu().numpy().astype(np.float64)
        Lsw = f(torch.tensor(Xs2_sw, device=dev)).cpu().numpy().astype(np.float64)
    del f
    torch.cuda.empty_cache()
    print(f'      [{tag}] es_ce {best:.5f} ep{used}')
    return L, Lsw, best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bundle', default='temp/heads_repro')
    ap.add_argument('--feats', default='raw,feat')
    ap.add_argument('--seeds', default='42,1,2')
    ap.add_argument('--h', type=int, default=128)
    ap.add_argument('--out', default='temp/monohead')
    args = ap.parse_args()
    kinds = [k for k in args.feats.split(',') if k]
    seeds = [int(s) for s in args.seeds.split(',') if s]
    assert set(kinds) <= set(FEATS), f'모르는 피쳐 {set(kinds) - set(FEATS)}'

    print('[적재]')
    U = TL.build(FINAL / args.bundle)
    _, y_tr, tie_tr, win_tr = TL.load_train_logits()
    A_tr, B_tr, A_s2, B_s2 = raw_ab(U['ids'])

    # 로짓 8 — (m, t) 는 3-class 로짓의 무손실 재매개화라 정보 손실이 없다
    mu_u, sc_u = TL.zstats(tie_tr, True)
    _, sc_v = TL.zstats(win_tr, False)
    Lg_tr = np.hstack([(tie_tr - mu_u) / sc_u, win_tr / sc_v]).astype(np.float32)
    Lg_s2 = np.hstack([(U['tie_base'] - mu_u) / sc_u, U['win_base'] / sc_v]).astype(np.float32)
    # swap 입력용 로짓 — 홀 성분만 부호 반전 (A↔B 교환의 로짓 표현)
    Lg_s2_sw = np.hstack([(U['tie_base'] - mu_u) / sc_u,
                          -U['win_base'] / sc_v]).astype(np.float32)

    win = TL.fit_win(U)
    y_t = (U['y'] == 2).astype(np.float64)
    y_w = (U['y'] == 0).astype(np.float64)
    rt, rw = U['cu'], U['cu'] & U['nz']
    res, rowc = {}, {}

    def eval_pair(tag, t_col, m_col, overlay):
        """(t, m) 열을 각 축 결합기에 넣어 합성 총점. overlay 면 현행 열 위에 얹는다."""
        Xt = [U['tie_base']] + ([U['GAT']] if overlay else []) + [t_col[:, None]]
        Xw = [U['win_base']] + ([U['LIN']] if overlay else []) + [m_col[:, None]]
        zt, dt = TL.combiner(np.hstack(Xt), y_t, U['fold'], rt, True, U['tie_base'].shape[1])
        zw, dw = TL.combiner(np.hstack(Xw), y_w, U['fold'], rw, False, U['win_base'].shape[1])
        llt = TL.rowloss(zt[rt], y_t[rt])
        llw = TL.rowloss(zw[rw], y_w[rw])
        row, lt, lw = TL.compose_ll(U, llt, llw, rt, rw)
        res[tag] = dict(tot=float(row.mean()), tie=lt, win=lw,
                        n_tie=len(Xt) and np.hstack(Xt).shape[1],
                        n_win=np.hstack(Xw).shape[1], lam=(dt['lam'], dw['lam']))
        rowc[tag] = row
        print(f'  [{tag}] tot {row.mean():.5f} (tie {lt:.5f} · win {lw:.5f}) '
              f'열 {res[tag]["n_tie"]}/{res[tag]["n_win"]}')

    # 기준선 — 현행 배포 구조
    zt, _ = TL.combiner(np.hstack([U['tie_base'], U['GAT']]), y_t, U['fold'], rt, True,
                        U['tie_base'].shape[1])
    llt0 = TL.rowloss(zt[rt], y_t[rt])
    row0, lt0, lw0 = TL.compose_ll(U, llt0, win['ll'], rt, win['rows'])
    res['cur'] = dict(tot=float(row0.mean()), tie=lt0, win=lw0)
    rowc['cur'] = row0
    print(f'\n  [cur] 현행 배포 구조 tot {row0.mean():.5f} (tie {lt0:.5f} · win {lw0:.5f})')

    for kind in kinds:
        print(f'\n{"="*78}\n[{kind}] 피쳐 조립\n{"="*78}')
        t0 = time.time()
        Xt_ = build_feats(A_tr, B_tr, kind)
        Xs_ = build_feats(A_s2, B_s2, kind)
        Xs_sw = build_feats(B_s2, A_s2, kind)          # A↔B 교환 입력 (TTA용)
        mu = Xt_.mean(0)
        sd = np.clip(Xt_.std(0), 1e-9, None)
        for M_ in (Xt_, Xs_, Xs_sw):
            M_ -= mu
            M_ /= sd
        Xt_ = np.ascontiguousarray(np.hstack([Xt_, Lg_tr]))
        Xs_ = np.ascontiguousarray(np.hstack([Xs_, Lg_s2]))
        Xs_sw = np.ascontiguousarray(np.hstack([Xs_sw, Lg_s2_sw]))
        print(f'  d {Xt_.shape[1]:,} (피쳐 {Xt_.shape[1] - 8:,} + 로짓 8) · '
              f'TR {Xt_.nbytes / 2**30:.2f} GB [{time.time() - t0:.0f}s]')
        gc.collect()

        for sd_ in seeds:
            tag = f'{kind}-s{sd_}'
            t0 = time.time()
            L, Lsw, ce = fit_mono(Xt_, y_tr, Xs_, Xs_sw, args.h, sd_, tag)
            # TTA — swap 입력의 출력은 [A,B,tie] 자리를 되돌려 평균
            Ltta = (L + Lsw[:, [1, 0, 2]]) / 2
            for mode, LL in (('noTTA', L), ('TTA', Ltta)):
                p = LL - LL.max(1, keepdims=True)
                p = p - np.log(np.exp(p).sum(1, keepdims=True))
                solo = float(-p[U['hold'], U['y'][U['hold']]].mean())
                m_, t_ = TL.sd3(LL)
                res[f'{tag}-{mode}-solo'] = dict(tot=solo, ce=ce)
                print(f'  [{tag}·{mode}] 단독 3-class ll(hold) {solo:.5f}')
                eval_pair(f'{tag}-{mode}-alone', t_, m_, overlay=False)
                eval_pair(f'{tag}-{mode}-over', t_, m_, overlay=True)
            print(f'    [{tag}] {time.time() - t0:.0f}s')
        del Xt_, Xs_, Xs_sw
        gc.collect()

    op = FINAL / args.out
    np.savez_compressed(op.with_suffix('.rowloss.npz'), **rowc)
    with open(op.with_suffix('.json'), 'w') as f:
        json.dump({'cells': res}, f, ensure_ascii=False, indent=1)
    print(f'\n저장: {op}.rowloss.npz / {op}.json')

    print(f'\n{"="*78}\n[요약] 현행 {res["cur"]["tot"]:.5f} 기준 · {len(seeds)}시드 평균'
          f'\n{"="*78}')
    print(f'  {"구성":22s}{"단독 3진":>10s}{"대체":>10s}{"얹기":>10s}   얹기 Δ vs 현행')
    for kind in kinds:
        for mode in ('noTTA', 'TTA'):
            solo = np.mean([res[f'{kind}-s{s}-{mode}-solo']['tot'] for s in seeds])
            al = np.mean([res[f'{kind}-s{s}-{mode}-alone']['tot'] for s in seeds])
            ov = np.mean([res[f'{kind}-s{s}-{mode}-over']['tot'] for s in seeds])
            cur = np.mean([rowc[f'{kind}-s{s}-{mode}-over'] for s in seeds], axis=0)
            m, lo, hi = TL.paired_ci(cur, rowc['cur'])
            print(f'  {kind + "·" + mode:22s}{solo:10.5f}{al:10.5f}{ov:10.5f}   '
                  f'{TL.ci_str(m, lo, hi)}')
    print('\n  단독 = 결합기 없이 이 헤드의 softmax 만 · 대체 = [로짓|mono] · '
          '얹기 = [로짓|현행열|mono]')
    print('  ! TTA − noTTA 가 곧 **패리티 구조가 공짜로 주던 것**의 크기다')
    print('  ! 파라미터: mono(feat, h128) ≈ 2.75M vs 현행 tie헤드 525K')


if __name__ == '__main__':
    main()
