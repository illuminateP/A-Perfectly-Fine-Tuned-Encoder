"""headgrid_solo.py — 신경 셀의 **기둥 없는(단독) 평가** 전수 (v5 보완, 2026-08-06).

무엇을 메우나
  headgrid.py 는 신경 셀을 항상 M1 기둥(앵커드 선형 r×4) **위에** 얹어 평가한다
  (headgrid.py: `pil = None if h == 'LIN' else pillar`). 그래서 신경 셀의 Δ 는
  "더 얹을 수 있나"(한계 가치)만 재고, **"기둥을 대신할 수 있나"(대체 가능성)는
  LIN 셀에서만 측정된다.** 이 파일이 그 공백을 신경 5종에 대해 전수로 닫는다.

    본 그리드 팔 :  로짓 기저 + 기둥 r×4 + 셀 열 3개      <- 얹기
    이 파일 팔   :  로짓 기저 +           셀 열 3개      <- 대체 (기둥 없음)

  판정 의미
    · 단독으로 M1 수준에 닿으면 -> 본학습 헤드가 신경 하나로 단순해진다
      (소스별 릿지 4회 적합 = β 4,096개 생산 과정이 통째로 빠진다).
    · 닿지 않으면 -> "기둥은 앵커드 선형이 맡고 신경은 얹기만" 이 전면 확정된다.
      (참고: LIN-*-plain 은 이미 R0 로짓만보다도 나쁘게 나오는 중 — 순수 선형 풀링
       체계는 받침 없이 못 선다는 증거가 tie축에서 먼저 나왔다.)

전수 범위 (기본 = 3모드 전부)
  헤드 5 (MLP/GAT/FFN/TE/TD) x 입력 4 (FULL/256/128/64, FFN 은 FULL 불가) x 모드 3
    = 57셀/축, 양축 114셀.  --modes plain 이면 19셀/축(양축 38).
  LIN 은 본 그리드에서 이미 기둥 없이 평가되므로 제외한다.
  모드 의미 재확인 — mode 는 **r-생산자의 학습 입력**, 기둥 제거는 **평가 팔**의 문제다:
    anchor  s0(동결 로짓)를 앵커로 잔차 학습  <- 대체 후보로는 이쪽이 가장 유력
    plain   풀링만 보고 y 직접 학습            <- 사용자가 지목한 순수 풀링 체계
    scratch 풀링 + 로짓을 자유 입력으로 y 학습

JOINT 계열 (v5.1, 사용자 의도의 원형 — "Pooler -> MLP -> 3-class logit -> softmax")
  위 셀들은 축마다 **별도 모델**(이진 BCE)이고 로짓 기저 결합기를 거친다. 사용자가 처음
  의도한 구조는 **한 모델이 3-class 확률을 직접 내는 단독 헤드**다. mlphead(stage2 적합)·
  newhead(train 적합)가 이 구조를 측정했지만 당시 재료가 3-전문가뿐(ABE1·TB1 풀링 부재)
  이었으므로, 4-전문가 재료·현행 위생으로 재측정해 그 문을 정확히 닫는다.
    구조: net([U,V]) -> (p0,p1), net([U,−V]) -> (q0,q1)
          d = p0−q0 (홀·승부), nu = p1+q1 (짝·tie)          <- 등변은 구조로
          3-class 로짓 = [d/2, −d/2, nu+logaddexp(d/2,−d/2)] -> softmax -> CE
          (softmax 는 자유도 1을 죽이므로 이 (d,nu) 매개화는 Linear(h,3)+softmax 와 동치)
    셀: JMLP/JGAT/JFFN/JTE/JTD x 입력 4(JFFN 은 FULL 불가) x 방식 2(**plain/scratch**)
        = 38셀 (양축 공동 학습이라 셀당 1회 학습)
        plain = 풀링만 / scratch = 로짓을 자유 입력으로 추가 (anchor 는 단독 합성의
        오프셋 재구성이 필요해 제외 — 사용자 결정. 앵커 계열은 축별 솔로 셀이 덮는다)
        라운드는 1 고정 (라운드 실측 2회 사망) — ES 상한 플래그로 학습 여지만 감시한다.
    평가 2중: ① 단독 — softmax 확률의 3-class 행별 ll 그대로 (stage2 적합 0개, 결합기 없음)
              ② 열 소비 — d 를 승부축, nu 를 tie축의 r 열로 (위 솔로 팔과 같은 잣대)
    저장: ① j3__{셀}__s42  ② ll__{축}__{셀}__s42 (리포트 [5]절이 자동으로 집는다)

규율·계약
  학습·재료·시드·라운드·ES 전부 headgrid.py 와 동일 함수를 그대로 호출한다
  (fit_flat/fit_tok/run_cell/eval_cols) — 팔 조립에서 pillar=None 만 다르다.
  저장은 **별도 npz**(headgrid_solo_oof.npz) — 본 그리드가 아직 돌고 있어도 파일 경합이 없다.
  계약: 단일 ckpt·raw·무온도·무앙상블 (probe-contract-single-ckpt).

사용 — **인자 없음**. 본 그리드 완주 후 이 한 줄이면 끝난다 (양축 · 3모드 전수 ~3시간).
  python Boost/headgrid_solo.py
  이어하기는 자동이다 — headgrid_solo_oof.npz 에 있는 셀은 건너뛴다(중단·재실행 안전).
  결과 정리는 headgrid_report.py 가 [5]절에 자동으로 합쳐 출력한다.
"""
import sys
import time

import numpy as np
from tqdm import tqdm

import fwls as FW
import reprhead as RH
import repr5 as R5
import resfit as RF
import headgrid as HG

sys.stdout.reconfigure(line_buffering=True)

SOLO_NPZ = FW.BOOST / 'headgrid_solo_oof.npz'
SOLO_HEADS = ('MLP', 'GAT', 'FFN', 'TE', 'TD')       # LIN 은 본 그리드가 이미 기둥 없이 평가
JOINT_HEADS = ('JMLP', 'JGAT', 'JFFN', 'JTE', 'JTD')
JOINT_MODES = ('plain', 'scratch')                    # 사용자 결정 — 둘 다 수행, anchor 제외
AXES = ('tie', 'win')


# ══════════════════════════════════════════════════════ JOINT — 단독 3-class 헤드
def fit_joint(head, y3, seed, flatX=None, tokX=None, hidden=32, wd=1e-2,
              d_model=64, nhead=4, layers=2, ff=256):
    """단독 3-class 헤드 1개 학습 (양축 공동, CrossEntropy). 앵커 없음 — plain/scratch 전용
    (scratch 는 호출부가 로짓을 입력에 이어붙여 준다).

        net([U,V]) -> (p0,p1),  net([U,−V]) -> (q0,q1)
        d = p0−q0 (홀), nu = p1+q1 (짝)
        3-class 로짓 = [d/2, −d/2, nu+logaddexp(d/2,−d/2)]  ->  softmax  ->  CE

    출력층 0-초기화 -> epoch 0 = 균등확률. 반환 (d_s2, nu_s2, aux{es_ll, es_epoch, es_cap})."""
    import torch
    import torch.nn as nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    T = lambda x, dt=torch.float32: torch.tensor(np.ascontiguousarray(x), dtype=dt, device=dev)
    logits3 = lambda d, nu: torch.stack(
        [d / 2, -d / 2, nu + torch.logaddexp(d / 2, -d / 2)], 1)

    tok = head in ('JTE', 'JTD')
    if not tok:
        CUt, CVt, CUs, CVs = flatX
        dim = CUt.shape[1] + CVt.shape[1]
        if head == 'JMLP':
            f = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 2))
            last = f[-1]
        elif head == 'JFFN':
            f = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim),
                              nn.GELU(), nn.Linear(dim, 2))
            last = f[-1]
        else:                                        # JGAT — 게이티드 유닛
            class G(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.W1 = nn.Linear(dim, hidden)
                    self.W2 = nn.Linear(dim, hidden)
                    self.w = nn.Linear(hidden, 2)

                def forward(self, x):
                    return self.w(torch.tanh(self.W1(x)) * torch.sigmoid(self.W2(x)))
            f = G()
            last = f.w
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        f = f.to(dev)
        params = list(f.parameters())
        U_, V_, Us_, Vs_ = T(CUt), T(CVt), T(CUs), T(CVs)

        def dnu(U, V, idx):
            p = f(torch.cat([U[idx], V[idx]], 1))
            q = f(torch.cat([U[idx], -V[idx]], 1))
            return p[:, 0] - q[:, 0], p[:, 1] + q[:, 1]

        fwd_tr = lambda idx: dnu(U_, V_, idx)
        fwd_s2 = lambda: dnu(Us_, Vs_, slice(None))
        sw = f                                        # train()/eval() 스위치
    else:
        TU_tr, TV_tr, TU_s2, TV_s2 = tokX
        S = len(TU_tr)
        pack = lambda U, V, sg: T(np.concatenate([np.stack(U, 1), sg * np.stack(V, 1)], 2))
        Xp, Xn = pack(TU_tr, TV_tr, 1.0), pack(TU_tr, TV_tr, -1.0)
        emb = nn.Linear(Xp.shape[2], d_model).to(dev)
        pos = nn.Parameter(torch.randn(S, d_model, device=dev) * 0.02)
        enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, ff, dropout=0.1, activation='gelu',
                                       batch_first=True), layers).to(dev)
        ro = nn.Linear(d_model, 2).to(dev)
        nn.init.zeros_(ro.weight)
        nn.init.zeros_(ro.bias)
        causal = head == 'JTD'
        msk = torch.triu(torch.ones(S, S, device=dev, dtype=torch.bool), 1) if causal else None
        params = (list(emb.parameters()) + [pos] + list(enc.parameters())
                  + list(ro.parameters()))

        def f2(X, idx):
            h = enc(emb(X[idx]) + pos, mask=msk)
            return ro(h[:, -1] if causal else h.mean(1))

        def dnu_tok(A, B, idx):
            p, q = f2(A, idx), f2(B, idx)
            return p[:, 0] - q[:, 0], p[:, 1] + q[:, 1]

        fwd_tr = lambda idx: dnu_tok(Xp, Xn, idx)
        Sp, Sn = pack(TU_s2, TV_s2, 1.0), pack(TU_s2, TV_s2, -1.0)
        fwd_s2 = lambda: dnu_tok(Sp, Sn, slice(None))
        sw = enc

    opt = torch.optim.AdamW(params, lr=(1e-3 if tok else 3e-3), weight_decay=wd)
    Y_ = T(y3, torch.long)
    lossf = nn.CrossEntropyLoss()
    perm = np.random.default_rng(seed).permutation(len(y3))
    n_es = max(int(len(y3) * 0.15), 200)
    es, tr = perm[:n_es], perm[n_es:]

    def loss_at(idx):
        d, nu = fwd_tr(idx)
        return lossf(logits3(d, nu), Y_[idx])

    best, best_w, bad, used, cap = np.inf, None, 0, 0, True
    for ep in range(HG.EPOCHS):
        sw.train()
        opt.zero_grad()
        loss_at(tr).backward()
        opt.step()
        sw.eval()
        with torch.no_grad():
            v = float(loss_at(es))
        if v < best - 1e-6:
            best, bad, used = v, 0, ep + 1
            best_w = [p.detach().clone() for p in params]
        else:
            bad += 1
            if bad >= HG.PATIENCE:
                cap = False
                break
    if best_w is not None:
        with torch.no_grad():
            for p, w_ in zip(params, best_w):
                p.copy_(w_)
    with torch.no_grad():
        d_s2, nu_s2 = fwd_s2()
    return (d_s2.cpu().numpy().astype(np.float64),
            nu_s2.cpu().numpy().astype(np.float64),
            dict(es_ll=best, es_epoch=used, es_cap=cap))


def joint_rowll(d_s2, nu_s2, y, rows):
    """단독 평가 — (d, nu) 를 3-class 확률로 합성해 행별 ll (우주 밖 NaN, 결합기 없음)."""
    z = np.stack([d_s2 / 2, -d_s2 / 2, nu_s2 + np.logaddexp(d_s2 / 2, -d_s2 / 2)], 1)
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))
    out = np.full(len(y), np.nan)
    out[rows] = ll[rows]
    return out


def main():
    axes, modes, seed = list(AXES), list(HG.MODES), HG.SEED_GRID     # 인자 없음 — 전수 고정
    t0 = time.time()

    A = R5.load_repr(single_best=True)
    y, nz, hold, cu, fold = A['y'], A['nz'], A['hold'], A['cu'], A['fold']
    ar = RF.pick_arith(A['ids'], y)
    TR = RF.load_train(ar)
    y_tr = TR['y']
    S2 = RF.stage2_single(A['ids'], y, TR['s_tb1'] is not None, ar)
    PS2 = RF.load_pool_s2(A['ids'])
    print('  계약: 단일 ckpt·raw·무온도·무앙상블 | 기둥 없는 단독 평가 | '
          f'seed {seed} | 라운드 {HG.ROUNDS} + ES\n')

    done = {}
    if SOLO_NPZ.exists():                       # 이어하기는 자동 — 완료 셀은 건너뛴다
        old = np.load(SOLO_NPZ, allow_pickle=True)
        done = {k: old[k] for k in old.files}
        print(f'  이어하기: 기존 {len(done)}키 적재\n')

    def save(key, vec):
        done[key] = vec
        np.savez_compressed(SOLO_NPZ, **done)

    # ── 공용 재료 (headgrid.main 과 동일 절차 — 라벨 비접촉, train 통계) ──
    t_pre = time.time()
    all_tr = np.ones(len(y_tr), bool)
    LINMAT = {'FULL': {k: (TR[k], PS2[k]) for k in HG.EVK + HG.ODK}}
    for K in (256, 128, 64):
        LINMAT[K] = {k: RF.svd_coords(TR[k], PS2[k], K, all_tr, k.endswith('_u'))
                     for k in HG.EVK + HG.ODK}
    Ut, Us = RF.norm_pools(TR, PS2, list(HG.EVK), True, all_tr)
    Vt, Vs = RF.norm_pools(TR, PS2, list(HG.ODK), False, all_tr)
    FLAT = {'FULL': (Ut.astype(np.float32), Vt.astype(np.float32),
                     Us.astype(np.float32), Vs.astype(np.float32))}
    for K in (256, 128, 64):
        cu_t, cu_s = RF.svd_coords(Ut, Us, K, all_tr, True)
        cv_t, cv_s = RF.svd_coords(Vt, Vs, K, all_tr, False)
        FLAT[K] = (cu_t, cv_t, cu_s, cv_s)
    del Ut, Us, Vt, Vs
    TOK = {}
    for K in HG.KS:
        gu_t, gu_s, gv_t, gv_s = [], [], [], []
        for ku, kv in zip(HG.EVK, HG.ODK):
            if K == 'FULL':
                mu_u, sc_u = HG.zstats(TR[ku], True)
                mu_v, sc_v = HG.zstats(TR[kv], False)
                gu_t.append(((TR[ku] - mu_u) / sc_u).astype(np.float32))
                gu_s.append(((PS2[ku] - mu_u) / sc_u).astype(np.float32))
                gv_t.append(((TR[kv] - mu_v) / sc_v).astype(np.float32))
                gv_s.append(((PS2[kv] - mu_v) / sc_v).astype(np.float32))
            else:
                gu_t.append(LINMAT[K][ku][0]); gu_s.append(LINMAT[K][ku][1])
                gv_t.append(LINMAT[K][kv][0]); gv_s.append(LINMAT[K][kv][1])
        TOK[K] = (gu_t, gv_t, gu_s, gv_s)
    print(f'  공용 재료 준비  [{(time.time() - t_pre) / 60:.1f}분]\n')

    tie_names = ['s_tie', 's_tb1', 't3', 't_a1', 't_ar']
    AXCFG = {
        'tie': dict(y01=(y == 2).astype(np.float64), y01_tr=(y_tr == 2).astype(np.float64),
                    rows=cu, bias=True, even=True, base=A['T5'], names=tie_names,
                    src=['tb_u', 'abe_u', 'a1_u', 'tb1_u']),
        'win': dict(y01=(y == 0).astype(np.float64), y01_tr=(y_tr == 0).astype(np.float64),
                    rows=cu & nz, bias=False, even=False, base=A['M3'],
                    names=['m3', 'm_a1', 'm_ar'], src=['a1_v', 'abe_v', 'tb_v', 'tb1_v']),
    }

    cells = [(h, K, m) for h in SOLO_HEADS for K in HG.KS for m in modes
             if not (h == 'FFN' and K == 'FULL')]
    cells_j = [(h, K, m) for h in JOINT_HEADS for K in HG.KS for m in JOINT_MODES
               if not (h == 'JFFN' and K == 'FULL')]
    bar = tqdm(total=len(axes) * len(cells) + len(cells_j), desc='단독평가', unit='셀',
               smoothing=0.05)

    Ms = {}
    for ax in axes:
        C = AXCFG[ax]
        m_tr = np.ones(len(y_tr), bool) if ax == 'tie' else (y_tr != 2)
        cols_s2 = [S2[nm] for nm in C['names']]
        cols_tr = [TR[nm] for nm in C['names']]
        w, b = RF.fit_anchor(cols_s2, C['y01'], C['rows'], C['bias'], C['names'])
        s0_tr = np.column_stack(cols_tr) @ w + b
        tie_tr = np.column_stack([TR[nm] for nm in tie_names])
        tie_s2 = np.column_stack([S2[nm] for nm in tie_names])
        win_tr = np.column_stack([TR[nm] for nm in ('m3', 'm_a1', 'm_ar')])
        win_s2 = np.column_stack([S2[nm] for nm in ('m3', 'm_a1', 'm_ar')])
        mu_u, sc_u = HG.zstats(tie_tr, True)
        mu_v, sc_v = HG.zstats(win_tr, False)
        M = dict(ax=ax, even=C['even'], bias=C['bias'], y01=C['y01'], rows=C['rows'],
                 fold=fold, hold=hold, base=C['base'], src=C['src'],
                 y01m=C['y01_tr'][m_tr], m_tr=m_tr, s0m=s0_tr[m_tr], nl=len(C['names']),
                 Lb_tr=np.column_stack(cols_tr), Lb_s2=np.column_stack(cols_s2),
                 Lu_tr=(tie_tr - mu_u) / sc_u, Lu_s2=(tie_s2 - mu_u) / sc_u,
                 Lv_tr=win_tr / sc_v, Lv_s2=win_s2 / sc_v,
                 lin={K: {s: (LINMAT[K][s][0][m_tr], LINMAT[K][s][1]) for s in C['src']}
                      for K in HG.KS},
                 flat={K: (FLAT[K][0][m_tr], FLAT[K][1][m_tr], FLAT[K][2], FLAT[K][3])
                       for K in HG.KS},
                 tok=TOK)
        Ms[ax] = M

        # 기준 팔 — 이 파일 안에서도 R0(로짓만)를 다시 계산해 자립적으로 읽히게 한다
        k0 = f'll__{ax}__R0'
        if k0 not in done:
            save(k0, HG.eval_cols([], M, None))

        for h, K, mode in cells:
            name = f'{h}-{K}-{mode}'
            key = f'll__{ax}__{name}__s{seed}'
            bar.set_postfix_str(f'{ax}·{name}')
            if key in done:
                bar.update(1)
                continue
            cols, flgs = HG.run_cell(h, K, mode, M, seed)
            for fl in flgs:
                HG.flag(fl.replace('학습여지', '학습여지·단독'))
            save(key, HG.eval_cols(cols, M, None))         # ← 기둥 없음
            bar.update(1)

        # 축별 요약 (본 그리드 npz 가 있으면 얹기 대비도 같이)
        llR = done[f'll__{ax}__R0']
        gz = np.load(HG.OOF_NPZ) if HG.OOF_NPZ.exists() else None
        llM = gz[f'll__{ax}__M1'] if gz is not None and f'll__{ax}__M1' in gz.files else None
        rows = []
        for h, K, mode in cells:
            name = f'{h}-{K}-{mode}'
            v = done.get(f'll__{ax}__{name}__s{seed}')
            if v is None:
                continue
            m = ~np.isnan(v)
            d0 = RH.paired_ci(llR[m], v[m])
            dm = RH.paired_ci(llM[m], v[m]) if llM is not None else None
            gk = f'll__{ax}__{name}__s{seed}'
            g = gz[gk] if gz is not None and gk in gz.files else None
            rows.append([name, f'{float(v[m].mean()):.4f}', RH.ci_str(*d0),
                         RH.ci_str(*dm) if dm else '—',
                         f'{float(g[m].mean()):.4f}' if g is not None else '—'])
        rows.sort(key=lambda r: float(r[1]))
        RH.head(f'단독·{ax}', f'{ax}축 기둥 없는 단독 평가 — 로짓 + 셀 열 3개 (기둥 4열 제외)')
        RH.table(['셀', '단독 ll', 'Δ vs R0(로짓만)', 'Δ vs M1(기둥)', '참고: 얹기 ll'],
                 rows, ['l', 'r', 'l', 'l', 'r'])
        if llM is not None:
            mM = ~np.isnan(llM)
            print(f'  기준: R0 {float(llR[~np.isnan(llR)].mean()):.4f} | '
                  f'M1(기둥) {float(llM[mM].mean()):.4f}')
        print('  읽기: "Δ vs M1" 이 0 근처면 그 신경 헤드가 기둥을 **대신할 수 있다**는 뜻이다.')
        print('        크게 음수면 대체 불가 — 기둥은 앵커드 선형이 맡아야 한다.')

    # ── JOINT: 단독 3-class 헤드 (plain/scratch) — 양축 공동 1학습, 결합기·기둥 없음 ──
    if all(a in Ms for a in AXES):
        Mt, Mw = Ms['tie'], Ms['win']
        y3_tr = y_tr.astype(np.int64)
        for h, K, mode in cells_j:
            name = f'{h}-{K}-{mode}'
            kj = f'j3__{name}__s{seed}'
            bar.set_postfix_str(f'joint·{name}')
            if kj in done and f'll__tie__{name}__s{seed}' in done:
                bar.update(1)
                continue
            if h in ('JTE', 'JTD'):
                TU_tr, TV_tr = list(TOK[K][0]), list(TOK[K][1])
                TU_s2, TV_s2 = list(TOK[K][2]), list(TOK[K][3])
                if mode == 'scratch':                # 로짓 토큰 — u=tie기저(짝), v=승부기저(홀)
                    kd, kv = TU_tr[0].shape[1], TV_tr[0].shape[1]
                    pad = lambda X, n_: (np.hstack([X, np.zeros((len(X), n_ - X.shape[1]))])
                                         if X.shape[1] < n_ else X[:, :n_])
                    TU_tr = TU_tr + [pad(Mt['Lu_tr'], kd)]
                    TU_s2 = TU_s2 + [pad(Mt['Lu_s2'], kd)]
                    TV_tr = TV_tr + [pad(Mt['Lv_tr'], kv)]
                    TV_s2 = TV_s2 + [pad(Mt['Lv_s2'], kv)]
                d_s2, nu_s2, aux = fit_joint(h, y3_tr, seed,
                                             tokX=(TU_tr, TV_tr, TU_s2, TV_s2))
            else:
                CUt, CVt, CUs, CVs = FLAT[K]
                if mode == 'scratch':
                    CUt = np.hstack([CUt, Mt['Lu_tr']])
                    CUs = np.hstack([CUs, Mt['Lu_s2']])
                    CVt = np.hstack([CVt, Mt['Lv_tr']])
                    CVs = np.hstack([CVs, Mt['Lv_s2']])
                d_s2, nu_s2, aux = fit_joint(h, y3_tr, seed, flatX=(CUt, CVt, CUs, CVs))
            if aux['es_cap']:
                HG.flag(f'⚠ 학습여지(epoch) 단독3c·{name}·s{seed}')
            done[kj] = joint_rowll(d_s2, nu_s2, y, cu)
            done[f'll__win__{name}__s{seed}'] = HG.eval_cols([('d', d_s2)], Mw, None)
            save(f'll__tie__{name}__s{seed}', HG.eval_cols([('nu', nu_s2)], Mt, None))
            bar.update(1)

        rows_j = []
        for h, K, mode in cells_j:
            v = done.get(f'j3__{h}-{K}-{mode}__s{seed}')
            if v is None:
                continue
            m = ~np.isnan(v)
            rows_j.append([f'{h}-{K}-{mode}', f'{float(v[m].mean()):.4f}',
                           f'{float(v[m & hold].mean()):.4f}'])
        rows_j.sort(key=lambda r: float(r[1]))
        RH.head('단독3c', '단독 3-class 헤드 (Pooler→net→3class→softmax) — 결합기·기둥 없음')
        RH.table(['셀', 'stage2 총ll', 'holdout 총ll'], rows_j, ['l', 'r', 'r'])
        print('  기준·CI 는 report [6]절이 정식으로 낸다 (M1·R0 의 행별 3-class 합성과 paired).')
        print('  이 셀들의 (d,nu) 열 소비 성적은 report [5]절에 J* 이름으로 병합된다.')

    bar.close()
    print(f'\n총 {(time.time() - t0) / 3600:.2f}시간 | 저장: {SOLO_NPZ}')
    print('정리: python Boost/headgrid_report.py  ([5]·[6]절에 단독 평가가 합쳐 나온다)')


if __name__ == '__main__':
    main()
