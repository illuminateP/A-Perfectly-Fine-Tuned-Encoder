"""headgrid.py — 본학습 전 마지막 전수 그리드 (v5, 2026-08-06).

목적
  본학습(Kaggle 노트북) 헤드를 확정하기 위해, 지금까지 팔 단위로 재던 조합 공간을
  **3축 전수**로 닫는다. 이 파일의 결과표가 곧 본학습 헤드의 선정 근거 문서다.

3축 (축당 69셀, 전체 138셀)
  헤드 6   : LIN(소스별 선형 β) / MLP-32 / GAT / FFN(BERT형) / TE(TF인코더) / TD(TF디코더)
  입력 4   : FULL / SVD-256 / SVD-128 / SVD-64   (FFN 은 FULL 불가 — 5.4억 파라미터)
  방식 3   : anchor(s0 잔차) / plain(풀링만, y 직접) / scratch(풀링+로짓 자유입력, y 직접)
  제외: RFF·쌍선형 — "고정특징+LIN" 변형, 실측 0 (log_resfit3/4).

라운드(epoch) = 3 고정 + ES (사용자 지시)
  모든 셀은 함수공간 부스팅 3라운드: 라운드 k 는 누적 train 점수를 앵커로 잔차에 재학습.
  ES(patience)가 죽은 라운드를 조기 종료시키므로 비용은 3배가 아니라 ~1.7배다.
  **과소적합 플래그** — 예산 끝에서도 ll 이 내려가던 셀은 별도 출력:
    ⚠ 학습여지(epoch)  : 신경 ES 최적 epoch 가 상한(300)에 닿음
    ⚠ 학습여지(라운드) : 라운드 3 의 train ES ll 이 라운드 2 대비 0.001 이상 개선
    (λ 격자 끝 경고는 fit_beta 가 기존대로 찍는다)

시드 (사용자 결정: 그리드 1시드 + 선발만 시드평균)
  그리드 스윕 = seed 42 단일. 선발 상위 --topk(기본 8) 신경 셀만 seed 43·44 추가 학습,
  **행별 ll 의 3시드 평균**으로 최종 순위 = "임의 시드 1개짜리 모형의 기대 성능"
  (r 열 자체의 시드 평균은 앙상블이라 계약 위반 — 하지 않는다. 배포는 단일 시드.)

판정 위생
  선발 통계 = clean 7,748 전체 OOF 의 paired Δ(vs M1). **holdout 열은 선발전 표에서만
  개봉**한다 — 그리드 표에서 holdout 을 반복 참조하면 무접촉성이 소선된다.
  모든 학습·통계·선택은 train 46,729 안에서만. stage2 는 fold-결합기와 판정 전용.
  계약: 단일 ckpt·raw·무온도·무앙상블 (probe-contract-single-ckpt).

평가 구조
  신경 셀 팔 = 로짓 + M1기둥(r×4) + 셀의 라운드 열 3개.
  LIN 셀 팔 = 로짓 + 셀 자신의 (소스4 × 라운드3) 12열 — **기둥 대체 후보**로 판정.
  스크래치 입력의 로짓: tie 기저(짝)는 U 쪽, 승부 기저(홀)는 V 쪽 — 패리티 유지.
  TE/TD 스크래치는 로짓 토큰 1개 추가(u=tie기저, v=승부기저, K 차원 0-패딩).

증분 저장·이어하기
  셀 완료마다 headgrid_oof.npz 에 행별 ll 저장(merge) — 튕겨도 --resume 으로 이어 돈다.

사용 (총 9~12시간; 축 분할 권장)
  python Boost/headgrid.py --axes tie 2>&1 | tee Boost/log_headgrid_tie.txt
  python Boost/headgrid.py --axes win --resume 2>&1 | tee Boost/log_headgrid_win.txt
  python Boost/headgrid.py --resume        # 중단 지점부터 (완료 셀 자동 건너뜀)
"""
import argparse
import sys
import time

import numpy as np
from tqdm import tqdm

import fwls as FW
import reprhead as RH
import repr5 as R5
import resfit as RF

sys.stdout.reconfigure(line_buffering=True)

SEED_GRID = 42
SEEDS_FIN = (42, 43, 44)
ROUNDS = 3
EPOCHS, PATIENCE = 300, 30
HEADS = ('LIN', 'MLP', 'GAT', 'FFN', 'TE', 'TD')
KS = ('FULL', 256, 128, 64)
MODES = ('anchor', 'plain', 'scratch')
EVK = ('abe_u', 'tb_u', 'a1_u', 'tb1_u')
ODK = ('abe_v', 'tb_v', 'a1_v', 'tb1_v')
OOF_NPZ = FW.BOOST / 'headgrid_oof.npz'
FLAG_TXT = FW.BOOST / 'headgrid_flags.txt'
RD_GAIN = 0.001          # 학습여지(라운드) 문턱 — 라운드3 ES ll 개선량


def flag(msg):
    print(f'    {msg}')
    with open(FLAG_TXT, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


# ══════════════════════════════════════════════════════ torch 학습기 (통합)
def _es_split(n, seed, frac=0.15):
    perm = np.random.default_rng(seed).permutation(n)
    n_es = max(int(n * frac), 200)
    return perm[:n_es], perm[n_es:]


def fit_flat(head, CUt, CVt, y01, s0, even, seed, CUs, CVs,
             hidden=32, wd=1e-2, lr=3e-3):
    """MLP/GAT/FFN 공용 — 출력 0-초기화, ±대칭화, s0=None 이면 y 직접.
    반환 (r_s2, aux{z_tr, es_ll, es_epoch, es_cap})."""
    import torch
    import torch.nn as nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    d = CUt.shape[1] + CVt.shape[1]
    if head == 'MLP':
        f = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1))
        last = f[-1]
    elif head == 'FFN':
        f = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.GELU(),
                          nn.Linear(d, 1))
        last = f[-1]
    else:                                            # GAT — 게이티드 유닛
        class G(nn.Module):
            def __init__(self):
                super().__init__()
                self.W1 = nn.Linear(d, hidden)
                self.W2 = nn.Linear(d, hidden)
                self.w = nn.Linear(hidden, 1)

            def forward(self, x):
                return self.w(torch.tanh(self.W1(x)) * torch.sigmoid(self.W2(x)))
        f = G()
        last = f.w
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    f = f.to(dev)
    anchored = s0 is not None
    a = nn.Parameter(torch.tensor(1.0, device=dev))
    opt = torch.optim.AdamW([{'params': f.parameters(), 'weight_decay': wd},
                             {'params': [a], 'weight_decay': 0.0}], lr=lr)
    s0z = (s0 / np.clip(np.sqrt((s0 ** 2).mean()), 1e-9, None)) if anchored \
        else np.zeros(len(y01))
    T = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)
    U_, V_, S_, Y_ = T(CUt), T(CVt), T(s0z), T(y01)
    lossf = torch.nn.BCEWithLogitsLoss()
    sgn = 1 if even else -1

    def out(idx):
        xp = torch.cat([U_[idx], V_[idx]], 1)
        xn = torch.cat([U_[idx], -V_[idx]], 1)
        return (f(xp) + sgn * f(xn)).squeeze(-1)

    es, tr = _es_split(len(y01), seed)
    best, best_sd, bad, used, cap = np.inf, None, 0, 0, True
    for ep in range(EPOCHS):
        f.train()
        opt.zero_grad()
        lossf(a * S_[tr] + out(tr), Y_[tr]).backward()
        opt.step()
        f.eval()
        with torch.no_grad():
            v = float(lossf(a * S_[es] + out(es), Y_[es]))
        if v < best - 1e-6:
            best, bad, used = v, 0, ep + 1
            best_sd = ({k: t.detach().clone() for k, t in f.state_dict().items()},
                       float(a.detach()))
        else:
            bad += 1
            if bad >= PATIENCE:
                cap = False
                break
    if best_sd is not None:
        f.load_state_dict(best_sd[0])
    with torch.no_grad():
        idx_all = np.arange(len(y01))
        r_tr = out(idx_all).cpu().numpy().astype(np.float64)
        Us_, Vs_ = T(CUs), T(CVs)
        xp = torch.cat([Us_, Vs_], 1)
        xn = torch.cat([Us_, -Vs_], 1)
        r_s2 = (f(xp) + sgn * f(xn)).squeeze(-1).cpu().numpy().astype(np.float64)
    z_tr = best_sd[1] * s0z + r_tr if anchored else r_tr
    return r_s2, dict(z_tr=z_tr, es_ll=best, es_epoch=used, es_cap=cap)


def fit_tok(causal, TU_tr, TV_tr, y01, s0, even, seed, TU_s2, TV_s2,
            d_model=64, nhead=4, layers=2, ff=256, wd=1e-2, lr=1e-3):
    """TE/TD 공용 — 전문가 토큰(+선택적 로짓 토큰은 호출부가 리스트에 포함).
    s0 지원(anchor 모드) + aux 반환. 그 외 resfit.fit_tf 와 동일 설계."""
    import torch
    import torch.nn as nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    S = len(TU_tr)
    pack = lambda U, V, sg: torch.tensor(
        np.concatenate([np.stack(U, 1), sg * np.stack(V, 1)], 2), dtype=torch.float32,
        device=dev)
    Xp, Xn = pack(TU_tr, TV_tr, 1.0), pack(TU_tr, TV_tr, -1.0)
    emb = nn.Linear(Xp.shape[2], d_model).to(dev)
    pos = nn.Parameter(torch.randn(S, d_model, device=dev) * 0.02)
    enc = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model, nhead, ff, dropout=0.1, activation='gelu',
                                   batch_first=True), layers).to(dev)
    ro = nn.Linear(d_model, 1).to(dev)
    nn.init.zeros_(ro.weight)
    nn.init.zeros_(ro.bias)
    mask = torch.triu(torch.ones(S, S, device=dev, dtype=torch.bool), 1) if causal else None
    anchored = s0 is not None
    a = nn.Parameter(torch.tensor(1.0, device=dev))
    opt = torch.optim.AdamW(
        [{'params': list(emb.parameters()) + [pos] + list(enc.parameters())
          + list(ro.parameters()), 'weight_decay': wd},
         {'params': [a], 'weight_decay': 0.0}], lr=lr)
    s0z = (s0 / np.clip(np.sqrt((s0 ** 2).mean()), 1e-9, None)) if anchored \
        else np.zeros(len(y01))
    S_ = torch.tensor(s0z, dtype=torch.float32, device=dev)
    Y_ = torch.tensor(y01, dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss()
    sgn = 1 if even else -1

    def f(X):
        h = enc(emb(X) + pos, mask=mask)
        return ro(h[:, -1] if causal else h.mean(1)).squeeze(-1)

    def out(A, B, idx):
        return f(A[idx]) + sgn * f(B[idx])

    es, tr = _es_split(len(y01), seed)
    mods = (emb, enc, ro)
    best, best_sd, bad, used, cap = np.inf, None, 0, 0, True
    for ep in range(EPOCHS):
        for m in mods:
            m.train()
        opt.zero_grad()
        lossf(a * S_[tr] + out(Xp, Xn, tr), Y_[tr]).backward()
        opt.step()
        for m in mods:
            m.eval()
        with torch.no_grad():
            v = float(lossf(a * S_[es] + out(Xp, Xn, es), Y_[es]))
        if v < best - 1e-6:
            best, bad, used = v, 0, ep + 1
            best_sd = ([{k: t.detach().clone() for k, t in m.state_dict().items()}
                        for m in mods], pos.detach().clone(), float(a.detach()))
        else:
            bad += 1
            if bad >= PATIENCE:
                cap = False
                break
    if best_sd is not None:
        for m, sd in zip(mods, best_sd[0]):
            m.load_state_dict(sd)
        with torch.no_grad():
            pos.copy_(best_sd[1])
    idx_all = np.arange(len(y01))
    with torch.no_grad():
        r_tr = out(Xp, Xn, idx_all).cpu().numpy().astype(np.float64)
        Sp, Sn = pack(TU_s2, TV_s2, 1.0), pack(TU_s2, TV_s2, -1.0)
        r_s2 = (f(Sp) + sgn * f(Sn)).cpu().numpy().astype(np.float64)
    z_tr = best_sd[2] * s0z + r_tr if anchored else r_tr
    return r_s2, dict(z_tr=z_tr, es_ll=best, es_epoch=used, es_cap=cap)


# ══════════════════════════════════════════════════════ 셀 실행
def zstats(X, even):
    mu = X.mean(0) if even else np.zeros(X.shape[1])
    sc = np.clip(X.std(0) if even else np.sqrt((X ** 2).mean(0)), 1e-9, None)
    return mu, sc


def run_cell(head, K, mode, M, seed, do_flags=True):
    """한 셀 = 라운드 3 체인. 반환 (열 목록 [(이름, stage2 벡터)], 플래그 목록).
    M: 축별 재료 dict (main 에서 조립). LIN 은 소스별 4체인(12열), 신경은 3열."""
    even, bias = M['even'], M['bias']
    y01m, m_tr = M['y01m'], M['m_tr']
    name = f'{head}-{K}-{mode}'
    flags = []

    def note(kind, rd):
        flags.append(f'⚠ 학습여지({kind}) {M["ax"]}·{name}·r{rd}·s{seed}')

    if head == 'LIN':
        cols = []
        for s in M['src']:
            Ptr, Ps2 = M['lin'][K][s]                       # (train, stage2) — 해당 소스 입력
            z = M['s0m'] if mode == 'anchor' else None
            if mode == 'scratch':
                Ptr = np.hstack([M['Lb_tr'][m_tr], Ptr])
                Ps2 = np.hstack([M['Lb_s2'], Ps2])
            prev_ll = None
            for rd in range(1, ROUNDS + 1):
                beta, st = RF.fit_beta(z, Ptr, y01m, even, bias,
                                       f'{M["ax"]}·{name}·{s}·r{rd}',
                                       free=(M['nl'] if mode == 'scratch' else 0))
                cols.append((f'{name}·{s}·r{rd}', ((Ps2 - st['mu']) / st['sc']) @ beta))
                cur_ll = float(np.mean(np.logaddexp(0, st['z_tr']) - y01m * st['z_tr']))
                if do_flags and rd == ROUNDS and prev_ll is not None \
                        and prev_ll - cur_ll > RD_GAIN:
                    note('라운드', rd)
                prev_ll = cur_ll
                z = st['z_tr']
        return cols, flags

    if head in ('MLP', 'GAT', 'FFN'):
        CUt, CVt, CUs, CVs = M['flat'][K]
        if mode == 'scratch':
            CUt = np.hstack([CUt, M['Lu_tr'][m_tr]])
            CUs = np.hstack([CUs, M['Lu_s2']])
            CVt = np.hstack([CVt, M['Lv_tr'][m_tr]])
            CVs = np.hstack([CVs, M['Lv_s2']])
        z = M['s0m'] if mode == 'anchor' else None
        cols, prev = [], None
        for rd in range(1, ROUNDS + 1):
            r_s2, aux = fit_flat(head, CUt, CVt, y01m, z, even, seed, CUs, CVs)
            cols.append((f'{name}·r{rd}', r_s2))
            if do_flags and aux['es_cap']:
                note('epoch', rd)
            if do_flags and rd == ROUNDS and prev is not None and prev - aux['es_ll'] > RD_GAIN:
                note('라운드', rd)
            prev = aux['es_ll']
            z = aux['z_tr']
        return cols, flags

    # TE / TD — 전문가 토큰 (+스크래치는 로짓 토큰 추가)
    TU_tr = [c[m_tr] for c in M['tok'][K][0]]
    TV_tr = [c[m_tr] for c in M['tok'][K][1]]
    TU_s2, TV_s2 = list(M['tok'][K][2]), list(M['tok'][K][3])
    if mode == 'scratch':
        kd = TU_tr[0].shape[1]
        pad = lambda X, n_: np.hstack([X, np.zeros((len(X), n_ - X.shape[1]))]) \
            if X.shape[1] < n_ else X[:, :n_]
        TU_tr = TU_tr + [pad(M['Lu_tr'][m_tr], kd)]
        TU_s2 = TU_s2 + [pad(M['Lu_s2'], kd)]
        TV_tr = TV_tr + [pad(M['Lv_tr'][m_tr], TV_tr[0].shape[1])]
        TV_s2 = TV_s2 + [pad(M['Lv_s2'], TV_s2[0].shape[1])]
    z = M['s0m'] if mode == 'anchor' else None
    cols, prev = [], None
    for rd in range(1, ROUNDS + 1):
        r_s2, aux = fit_tok(head == 'TD', TU_tr, TV_tr, y01m, z, even, seed, TU_s2, TV_s2)
        cols.append((f'{name}·r{rd}', r_s2))
        if do_flags and aux['es_cap']:
            note('epoch', rd)
        if do_flags and rd == ROUNDS and prev is not None and prev - aux['es_ll'] > RD_GAIN:
            note('라운드', rd)
        prev = aux['es_ll']
        z = aux['z_tr']
    return cols, flags


def eval_cols(cols, M, pillar):
    """열 목록을 팔로 평가 — 반환: 전 행 길이 ll 벡터(비우주 행 NaN). 빈 목록 = R0."""
    parts = (pillar or []) + cols
    if parts:
        Xi = np.column_stack([c for _, c in parts])
        blk = [(nm, i, i + 1) for i, (nm, _) in enumerate(parts)]
    else:
        Xi, blk = np.zeros((len(M['y01']), 0)), []
    res = FW.run_arm(('cell', 'cell', M['base'], Xi, blk, None), M['y01'], M['fold'],
                     M['rows'], M['hold'], None, bias=M['bias'])
    out = np.full(len(M['y01']), np.nan)
    out[M['rows']] = res['ll']
    return out


# ══════════════════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--axes', default='tie,win')
    ap.add_argument('--resume', action='store_true', help='headgrid_oof.npz 의 완료 셀 건너뜀')
    ap.add_argument('--topk', type=int, default=8, help='시드평균 선발전에 올릴 신경 셀 수')
    args = ap.parse_args()
    axes = [s.strip() for s in args.axes.split(',') if s.strip()]
    t0 = time.time()

    A = R5.load_repr(single_best=True)
    y, nz, hold, cu, fold = A['y'], A['nz'], A['hold'], A['cu'], A['fold']
    ar = RF.pick_arith(A['ids'], y)
    TR = RF.load_train(ar)
    y_tr = TR['y']
    S2 = RF.stage2_single(A['ids'], y, TR['s_tb1'] is not None, ar)
    PS2 = RF.load_pool_s2(A['ids'])
    print('  계약: 단일 ckpt·raw·무온도·무앙상블 | 그리드 seed 42 | 선발 시드평균 '
          f'{SEEDS_FIN} | 라운드 {ROUNDS} + ES\n')

    done = {}
    if args.resume and OOF_NPZ.exists():
        old = np.load(OOF_NPZ, allow_pickle=True)
        done = {k: old[k] for k in old.files}
        print(f'  --resume: 기존 {len(done)}키 적재\n')

    def save(key, vec):
        done[key] = vec
        np.savez_compressed(OOF_NPZ, **done)

    # ── 공용 재료: SVD 사다리 (전부 라벨 비접촉, train 통계) ──
    t_pre = time.time()
    all_tr = np.ones(len(y_tr), bool)
    LINMAT = {'FULL': {k: (TR[k], PS2[k]) for k in EVK + ODK}}
    for K in (256, 128, 64):
        LINMAT[K] = {k: RF.svd_coords(TR[k], PS2[k], K, all_tr,
                                      k.endswith('_u')) for k in EVK + ODK}
    Ut, Us = RF.norm_pools(TR, PS2, list(EVK), True, all_tr)
    Vt, Vs = RF.norm_pools(TR, PS2, list(ODK), False, all_tr)
    FLAT = {'FULL': (Ut.astype(np.float32), Vt.astype(np.float32),
                     Us.astype(np.float32), Vs.astype(np.float32))}
    for K in (256, 128, 64):
        cu_t, cu_s = RF.svd_coords(Ut, Us, K, all_tr, True)
        cv_t, cv_s = RF.svd_coords(Vt, Vs, K, all_tr, False)
        FLAT[K] = (cu_t, cv_t, cu_s, cv_s)
    del Ut, Us, Vt, Vs
    TOK = {}
    for K in KS:
        gu_t, gu_s, gv_t, gv_s = [], [], [], []
        for ku, kv in zip(EVK, ODK):
            if K == 'FULL':
                mu_u, sc_u = zstats(TR[ku], True)
                mu_v, sc_v = zstats(TR[kv], False)
                gu_t.append(((TR[ku] - mu_u) / sc_u).astype(np.float32))
                gu_s.append(((PS2[ku] - mu_u) / sc_u).astype(np.float32))
                gv_t.append(((TR[kv] - mu_v) / sc_v).astype(np.float32))
                gv_s.append(((PS2[kv] - mu_v) / sc_v).astype(np.float32))
            else:
                cu_t, cu_s = LINMAT[K][ku]
                cv_t, cv_s = LINMAT[K][kv]
                gu_t.append(cu_t); gu_s.append(cu_s); gv_t.append(cv_t); gv_s.append(cv_s)
        TOK[K] = (gu_t, gv_t, gu_s, gv_s)
    print(f'  공용 재료(SVD 사다리 완료)  [{(time.time() - t_pre) / 60:.1f}분]\n')

    tie_names = ['s_tie', 's_tb1', 't3', 't_a1', 't_ar']
    AXCFG = {
        'tie': dict(y01=(y == 2).astype(np.float64), y01_tr=(y_tr == 2).astype(np.float64),
                    rows=cu, bias=True, even=True, base=A['T5'], names=tie_names,
                    src=['tb_u', 'abe_u', 'a1_u', 'tb1_u']),
        'win': dict(y01=(y == 0).astype(np.float64), y01_tr=(y_tr == 0).astype(np.float64),
                    rows=cu & nz, bias=False, even=False, base=A['M3'],
                    names=['m3', 'm_a1', 'm_ar'], src=['a1_v', 'abe_v', 'tb_v', 'tb1_v']),
    }

    cells = [(h, K, m) for h in HEADS for K in KS for m in MODES
             if not (h == 'FFN' and K == 'FULL')]
    total = len(axes) * len(cells) + len(axes) * args.topk * (len(SEEDS_FIN) - 1)
    bar = tqdm(total=total, desc='그리드', unit='셀', smoothing=0.05)

    report = {}
    for ax in axes:
        C = AXCFG[ax]
        m_tr = np.ones(len(y_tr), bool) if ax == 'tie' else (y_tr != 2)
        cols_s2 = [S2[nm] for nm in C['names']]
        cols_tr = [TR[nm] for nm in C['names']]
        w, b = RF.fit_anchor(cols_s2, C['y01'], C['rows'], C['bias'], C['names'])
        s0_tr = np.column_stack(cols_tr) @ w + b
        Lb_tr = np.column_stack(cols_tr)
        Lb_s2 = np.column_stack(cols_s2)
        # 스크래치용 로짓 입력 — 패리티 자리: 짝(tie 기저)은 U 쪽, 홀(승부 기저)은 V 쪽
        tie_tr = np.column_stack([TR[nm] for nm in tie_names])
        tie_s2 = np.column_stack([S2[nm] for nm in tie_names])
        win_tr = np.column_stack([TR[nm] for nm in ('m3', 'm_a1', 'm_ar')])
        win_s2 = np.column_stack([S2[nm] for nm in ('m3', 'm_a1', 'm_ar')])
        mu_u, sc_u = zstats(tie_tr, True)
        mu_v, sc_v = zstats(win_tr, False)
        M = dict(ax=ax, even=C['even'], bias=C['bias'], y01=C['y01'], rows=C['rows'],
                 fold=fold, hold=hold, base=C['base'], src=C['src'],
                 y01m=C['y01_tr'][m_tr], m_tr=m_tr, s0m=s0_tr[m_tr], nl=len(C['names']),
                 Lb_tr=Lb_tr, Lb_s2=Lb_s2,
                 Lu_tr=(tie_tr - mu_u) / sc_u, Lu_s2=(tie_s2 - mu_u) / sc_u,
                 Lv_tr=win_tr / sc_v, Lv_s2=win_s2 / sc_v,
                 lin={K: {s: (LINMAT[K][s][0][m_tr], LINMAT[K][s][1])
                          for s in C['src']} for K in KS},
                 flat={K: (FLAT[K][0][m_tr], FLAT[K][1][m_tr], FLAT[K][2], FLAT[K][3])
                       for K in KS},
                 tok=TOK)

        # 기둥(M1) = LIN-FULL-anchor 라운드1 — 신경 셀들의 공통 받침
        pil_key = f'pil__{ax}'
        if pil_key + '__0' in done:
            pillar = [(f'p{i}', done[pil_key + f'__{i}']) for i in range(4)]
        else:
            pillar = []
            for i, s in enumerate(C['src']):
                beta, st = RF.fit_beta(M['s0m'], TR[s][m_tr], M['y01m'], C['even'],
                                       C['bias'], f'{ax}·기둥·{s}')
                pillar.append((f'p{i}', ((PS2[s] - st['mu']) / st['sc']) @ beta))
                save(pil_key + f'__{i}', pillar[-1][1])
        # 기준 팔들
        for key, cols in (('R0', []), ('M1', pillar)):
            k = f'll__{ax}__{key}'
            if k not in done:
                save(k, eval_cols(cols, M, None) if key == 'R0'
                     else eval_cols([], M, pillar))

        for h, K, mode in cells:
            name = f'{h}-{K}-{mode}'
            key = f'll__{ax}__{name}__s{SEED_GRID}'
            bar.set_postfix_str(f'{ax}·{name}')
            if key in done:
                bar.update(1)
                continue
            cols, flgs = run_cell(h, K, mode, M, SEED_GRID)
            for fl in flgs:
                flag(fl)
            pil = None if h == 'LIN' else pillar
            save(key, eval_cols(cols, M, pil))
            # 신경 셀은 r 열도 저장 — 선발전·본학습 재현용
            bar.update(1)

        # ── 그리드 순위 (전체 OOF paired, holdout 미개봉) ──
        llM1 = done[f'll__{ax}__M1']
        rows_rank = []
        for h, K, mode in cells:
            name = f'{h}-{K}-{mode}'
            v = done[f'll__{ax}__{name}__s{SEED_GRID}']
            m = ~np.isnan(v)
            d_, lo, hi = RH.paired_ci(llM1[m], v[m])
            rows_rank.append((d_, lo, hi, name, h != 'LIN'))
        rows_rank.sort(key=lambda r: -r[0])
        RH.head(ax, f'{ax} 그리드 순위 (vs M1, 전체 OOF — holdout 은 선발전에서만)')
        RH.table(['셀', 'Δ vs M1', 'CI'],
                 [[nm, f'{d_:+.4f}', RH.ci_str(d_, lo, hi)]
                  for d_, lo, hi, nm, _ in rows_rank[:15]], ['l', 'r', 'l'])

        # ── 선발전: 상위 신경 셀 + 2시드 → 시드평균 ──
        fins = [r for r in rows_rank if r[4]][:args.topk]
        fin_rows = []
        for d_, lo, hi, name, _ in fins:
            h, K, mode = name.split('-')
            K = K if K == 'FULL' else int(K)
            lls = [done[f'll__{ax}__{name}__s{SEED_GRID}']]
            for sd in SEEDS_FIN[1:]:
                k2 = f'll__{ax}__{name}__s{sd}'
                bar.set_postfix_str(f'{ax}·{name}·s{sd}')
                if k2 not in done:
                    cols, flgs = run_cell(h, K, mode, M, sd)
                    for fl in flgs:
                        flag(fl)
                    save(k2, eval_cols(cols, M, pillar))
                lls.append(done[k2])
                bar.update(1)
            avg = np.nanmean(np.stack(lls), 0)
            m = ~np.isnan(avg)
            d2, lo2, hi2 = RH.paired_ci(llM1[m], avg[m])
            hm = hold & m
            d2h, lo2h, hi2h = RH.paired_ci(llM1[hm], avg[hm])
            sds = [float(np.nanmean(v)) for v in lls]
            fin_rows.append([name, RH.ci_str(d2, lo2, hi2), RH.ci_str(d2h, lo2h, hi2h),
                             f'{np.std(sds):.4f}'])
        # LIN 참조(결정론 — 시드 불필요)도 최종표에
        for _, _, _, name, isn in rows_rank:
            if not isn and any(name.startswith(p) for p in ('LIN-FULL', 'LIN-256')):
                v = done[f'll__{ax}__{name}__s{SEED_GRID}']
                m = ~np.isnan(v)
                d2, lo2, hi2 = RH.paired_ci(llM1[m], v[m])
                hm = hold & m
                d2h, lo2h, hi2h = RH.paired_ci(llM1[hm], v[hm])
                fin_rows.append([name + ' (결정론)', RH.ci_str(d2, lo2, hi2),
                                 RH.ci_str(d2h, lo2h, hi2h), '—'])
        RH.head(ax + '·선발', f'{ax} 선발전 — 시드평균 (vs M1) · holdout 개봉')
        RH.table(['셀', 'Δ 전체(시드평균)', 'Δ holdout', '시드 표준편차'],
                 fin_rows, ['l', 'l', 'l', 'r'])
        report[ax] = (rows_rank, fin_rows)

    bar.close()
    print(f'\n총 {(time.time() - t0) / 3600:.1f}시간 | 저장: {OOF_NPZ}')
    print('판정(본학습 헤드 확정)은 사용자 — 동률권은 단순한 쪽 우선을 권고.')
    print(f'과소적합 플래그 전체 목록: {FLAG_TXT}')


if __name__ == '__main__':
    main()
