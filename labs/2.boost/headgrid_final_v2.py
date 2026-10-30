"""headgrid_final_v2.py — 최종 후보 3헤드(GAT·LIN·FFN) 아키텍처 확정 스윕 (v2, 2026-08-07).

사용자 확정 명세
  방식      = scratch 고정 (풀링 + 로짓 자유입력, 앵커 없음), dropout 없음, seed 42 단일
  입력      = SVD 사다리 확장. GAT/FFN 6단(64·128·256·512·1024·FULL) / LIN 5단(1024 제외)
  GAT       = 유닛 2종(tanh·gauss · tanh·sig) x 폭 h 2점(64·128)
  FFN       = 초기 확장 비율 3종(2d · 4d · 8d) — 파라미터 2,000만 초과 조합은 자동 제외
  신규 3팔  = 화이트닝 해제 / 미니배치 / 전문가별 예산 — **2단계**에서 1단계 승자 K에만
  예산      = lr 3e-4 · epoch 상한 3000 · patience 200 · 라운드 상한 8(자기 종결 0.0003)

왜 LIN 은 1024 가 없고 신규 3팔도 안 걸리나 (측정·증명된 사실)
  1) LIN 재료는 **소스별**(각 1024차원) SVD 다. K=1024 는 전랭크 회전이라 LIN-FULL 과
     열 공간이 같다 — 같은 셀이므로 뺀다.
  2) 화이트닝 해제가 LIN 에는 **무효**다. resfit.fit_beta 가 입력 P 를 열별로 다시
     표준화(짝 std / 홀 RMS)하므로, 열 j 를 상수 c 로 스케일하면 mu·sc 도 c 배 되어
     정확히 상쇄된다. 적합 결과가 비트 단위로 같다.
  3) 미니배치는 릿지(lbfgs)에 해당 개념이 없고, 전문가별 예산은 LIN 이 이미 소스별
     4체인이라 기본값이다.
  => 신규 3팔은 GAT/FFN 전용.

화이트닝이란 (이 런의 핵심 신규 축)
  resfit.svd_coords 는 사영 후 좌표를 **열별 RMS 로 나눈다** — K 개 방향에 동일 진폭을
  준다. 즉 250번째(특이값이 작아 대부분 잡음) 방향이 1번째와 같은 크기로 헤드에 들어간다.
  해제 팔(-nw)은 전역 스칼라 하나로만 나눠 특이값 가중(sigma 비례)을 살린다 — 하드 절단
  대신 부드러운 스펙트럼 사전이 되고, K 사다리의 의미 자체가 달라진다.

평가 (headgrid 와 동일 계약: 단일 ckpt·raw·무온도·무앙상블)
  얹기 ll__  = 로짓 + 기둥 r x 4 + 셀 라운드 열   (GAT/FFN)
  단독 llS__ = 로짓 + 셀 라운드 열 (기둥 없음)    (GAT/FFN)
  LIN 은 기둥 대체 후보이므로 단독 평가만 한다 (원그리드와 동일 취급).
  기둥·M1 기준 벡터는 headgrid_oof.npz 의 pil__ / ll__{ax}__M1 을 그대로 재사용한다.
  선택 통계 = 얹기 stage2 clean 7,748 전체 OOF Δ vs M1. holdout 도 같이 찍는다.

증분 저장 headgrid_final_v2_oof.npz — 셀 완료마다 저장, 재실행 시 자동 이어하기.
플래그는 headgrid_final_v2_flags.txt (기존 flags 파일은 ext 의 입력이므로 건드리지 않는다).

단계 순서 (중요 — 후보 선정 전에 LIN 을 먼저 수렴시킨다)
  1     기본 사다리 42셀/축
  lin   LIN λ 자동확장(-lx) — 1단계 실측상 격자 끝 74건이 전부 라운드 2·3 이었다.
        상한에 묶인 채로 LIN 을 후보 순위에 올리면 불공정하므로 **2단계 전에** 돌린다.
        여기서 LIN 이 올라오면 후보 목록 자체가 바뀐다.
  2     확정된 후보에 신규 3팔 (--cells 로 명시 지정)

사용 (1단계 84셀 ~8시간 / lin 10셀 ~3.5시간 / 2단계 후보당 3셀)
  python Boost/headgrid_final_v2.py --stage 1 --axes tie 2>&1 | tee Boost/log_hgv2_tie.txt
  python Boost/headgrid_final_v2.py --stage lin 2>&1 | tee Boost/log_hgv2_lin.txt
  python Boost/headgrid_final_v2.py --stage lin --lam-ks FULL      # LIN-FULL 만 (~1.7시간)
  python Boost/headgrid_final_v2.py --stage 2 --cells \\
      'tie:GAT-512-h128-tanh·sig,tie:FFN-1024-r2,win:GAT-1024-h64-tanh·sig'
  python Boost/headgrid_final_v2.py            # 1->lin->2 연속 + 결합표 (이어하기 자동)
"""
import argparse
import gc
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

SEED = 42
LR, EPOCHS, PATIENCE = 3e-4, 3000, 200
MB_BS, MB_EPOCHS, MB_PATIENCE = 2048, 1000, 50
WD = 1e-2
RD_CAP, RD_STOP, RD_GAIN = 8, 0.0003, 0.001
ROUNDS_LIN = 3
PARAM_CAP = 20_000_000

KS_NEURAL = (64, 128, 256, 512, 1024, 'FULL')
KS_LIN = (64, 128, 256, 512, 'FULL')
GAT_HS = (64, 128)
GAT_UNITS = ('tanh·gauss', 'tanh·sig')
FFN_RATIOS = (2, 4, 8)
ARMS = ('nw', 'mb', 'pe')                    # 화이트닝해제 / 미니배치 / 전문가별예산
ARM_NAME = {'nw': '화이트닝 해제', 'mb': '미니배치', 'pe': '전문가별 예산'}
LAMS_FINAL = [float(10 ** (e / 2)) for e in range(17)]     # 1e0 … 1e8 반데케이드
LAMS_LX = [float(10 ** (e / 2)) for e in range(25)]        # 1e0 … 1e12 — λ 자동확장 팔(-lx)
LAM_CEIL = 1e16                                            # 무한 루프 방지용 수치 안전판

OOF_NPZ = FW.BOOST / 'headgrid_final_v2_oof.npz'
FLAG_TXT = FW.BOOST / 'headgrid_final_v2_flags.txt'


def flag(msg):
    print(f'    {msg}')
    with open(FLAG_TXT, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


# ══════════════════════════════════════════════════════ 재료
def svd_arm(Xt, Xs, k, rows_tr, center, whiten):
    """라벨-프리 train SVD 사영. whiten=True 면 resfit.svd_coords 와 동일(열별 RMS),
    False 면 **전역 스칼라 하나**로만 나눠 특이값 가중을 보존한다."""
    from sklearn.utils.extmath import randomized_svd
    mu = Xt[rows_tr].mean(0) if center else np.zeros(Xt.shape[1])
    _, _, Vt_ = randomized_svd((Xt[rows_tr] - mu).astype(np.float64), n_components=k,
                               random_state=RF.SEED)
    W = Vt_.T
    Ct, Cs = (Xt - mu) @ W, (Xs - mu) @ W
    sc = (np.clip(np.sqrt((Ct[rows_tr] ** 2).mean(0)), 1e-9, None) if whiten
          else np.clip(np.sqrt((Ct[rows_tr] ** 2).mean()), 1e-9, None))
    return (Ct / sc).astype(np.float32), (Cs / sc).astype(np.float32)


def flat_mats(TR, PS2, FULLMAT, K, rows_tr, whiten, per_expert):
    """GAT/FFN 입력 (CUt, CVt, CUs, CVs). K='FULL' 이면 비압축 정규화 풀링 그대로.
    per_expert=True 면 전문가마다 K/4 씩 따로 압축해 concat (약한 전문가 예산 보장)."""
    if K == 'FULL':
        return FULLMAT
    if per_expert:
        assert K % 4 == 0, f'전문가별 예산은 K 가 4 의 배수여야 한다 (K={K})'
        ke = K // 4
        out = []
        for keys, center in ((HG.EVK, True), (HG.ODK, False)):
            ct, cs = [], []
            for kk in keys:
                a, b = svd_arm(TR[kk], PS2[kk], ke, rows_tr, center, whiten)
                ct.append(a)
                cs.append(b)
            out.append((np.hstack(ct), np.hstack(cs)))
        (cu_t, cu_s), (cv_t, cv_s) = out
        return cu_t, cv_t, cu_s, cv_s
    Ut, Vt, Us, Vs = FULLMAT
    cu_t, cu_s = svd_arm(Ut, Us, K, rows_tr, True, whiten)
    cv_t, cv_s = svd_arm(Vt, Vs, K, rows_tr, False, whiten)
    return cu_t, cv_t, cu_s, cv_s


def lin_mats(TR, PS2, srcs, K, rows_tr):
    """LIN 입력 — 소스별 (train, stage2). K='FULL' 이면 원좌표 그대로."""
    if K == 'FULL':
        return {s: (TR[s], PS2[s]) for s in srcs}
    return {s: svd_arm(TR[s], PS2[s], K, rows_tr, s.endswith('_u'), True) for s in srcs}


def fit_beta_auto(s0, P, y01, even, bias, tag, free=0):
    """λ **자동 확장** 래퍼 — reprhead.crossfit 이 이미 갖고 있는 규율을 fit_beta 에도 준다.

    왜 래퍼인가: resfit.fit_beta 자체를 고치면 headgrid.py·headgrid_final.py 의 과거
    결과가 재현되지 않는다. RF.LAMS_TR 은 모듈 전역이므로 여기서만 갈아끼우고
    try/finally 로 반드시 원복한다.

    왜 필요한가(1단계 실측): 격자 끝(1e8) 경고 74건이 **전부 라운드 2·3** 이었다
    (라운드 1 은 λ 1~3,162 에서 정상 선택). 즉 라운드 1 이 신호를 다 먹고 2·3 이
    "더 배울 게 없다"며 λ→∞ 로 도망친 것인데, 상한에 묶여 β 가 정확히 0 이 되지 못하고
    미세 잡음 열이 결합기로 넘어간다. 확장은 그 잡음을 없앤다.

    절차: (1) LAMS_LX(1e0…1e12, 25점) 전체 격자로 적합. (2) 그래도 상단이 뽑히면
    [상단, 상단x10] 2점 창으로 옮겨가며 확장 — CV 손실이 λ 에 단봉이라 창 비교로
    충분하고 전체 격자를 다시 훑는 것보다 훨씬 싸다. (3) LAM_CEIL 에서 멈춘다.
    """
    saved = RF.LAMS_TR
    try:
        RF.LAMS_TR = list(LAMS_LX)
        beta, st = RF.fit_beta(s0, P, y01, even, bias, tag, free=free)
        n_ext = 0
        while st['lam'] >= max(RF.LAMS_TR) and st['lam'] * 10 <= LAM_CEIL:
            RF.LAMS_TR = [st['lam'], st['lam'] * 10]
            beta, st = RF.fit_beta(s0, P, y01, even, bias, f'{tag}·확장{n_ext + 1}',
                                   free=free)
            n_ext += 1
        if n_ext:
            print(f'    [{tag}] λ 자동확장 {n_ext}회 -> 최종 λ {st["lam"]:g}')
        return beta, st
    finally:
        RF.LAMS_TR = saved


# ══════════════════════════════════════════════════════ 신경 헤드
def _uf(nm, torch):
    if nm == 'tanh':
        return torch.tanh
    if nm == 'gelu':
        return torch.nn.functional.gelu
    if nm == 'sig':
        return torch.sigmoid
    if nm == 'gauss':
        return lambda z: torch.exp(-0.5 * z * z)
    raise ValueError(f'유닛 {nm}')


def n_params(kind, d, cfg):
    if kind == 'GAT':
        h = cfg['h']
        return 2 * (d * h + h) + h + 1
    r = cfg['ratio']
    return d * (r * d) + r * d + (r * d) * d + d + d + 1


def fit_net(kind, cfg, CUt, CVt, y01, s0, even, seed, CUs, CVs, batch=None):
    """GAT(게이티드 유닛) / FFN(BERT형, 확장비율 r) 공용.
    출력 0-초기화, ±대칭화, s0 는 라운드 2+ 의 누적 앵커. batch 를 주면 미니배치 SGD.
    반환 (r_s2, aux{z_tr, es_ll, es_epoch, es_cap})."""
    import torch
    import torch.nn as nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    d = CUt.shape[1] + CVt.shape[1]
    if kind == 'GAT':
        a_nm, g_nm = cfg['unit'].split('·')
        act, gate = _uf(a_nm, torch), _uf(g_nm, torch)
        h = cfg['h']

        class G(nn.Module):
            def __init__(self):
                super().__init__()
                self.W1 = nn.Linear(d, h)
                self.W2 = nn.Linear(d, h)
                self.w = nn.Linear(h, 1)

            def forward(self, x):
                return self.w(act(self.W1(x)) * gate(self.W2(x)))
        f = G()
        last = f.w
    else:
        r = cfg['ratio']
        f = nn.Sequential(nn.Linear(d, r * d), nn.GELU(), nn.Linear(r * d, d), nn.GELU(),
                          nn.Linear(d, 1))
        last = f[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    f = f.to(dev)
    anchored = s0 is not None
    a = nn.Parameter(torch.tensor(1.0, device=dev))
    opt = torch.optim.AdamW([{'params': f.parameters(), 'weight_decay': WD},
                             {'params': [a], 'weight_decay': 0.0}], lr=LR)
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

    es, tr = HG._es_split(len(y01), seed)
    n_ep = MB_EPOCHS if batch else EPOCHS
    pat = MB_PATIENCE if batch else PATIENCE
    rng = np.random.default_rng(seed + 1)
    best, best_sd, bad, used, cap = np.inf, None, 0, 0, True
    for ep in range(n_ep):
        f.train()
        if batch is None:
            opt.zero_grad()
            lossf(a * S_[tr] + out(tr), Y_[tr]).backward()
            opt.step()
        else:
            order = tr[rng.permutation(len(tr))]
            for i in range(0, len(order), batch):
                bi = order[i:i + batch]
                opt.zero_grad()
                lossf(a * S_[bi] + out(bi), Y_[bi]).backward()
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
            if bad >= pat:
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


def gpu_flush():
    """라운드 사이 GPU 캐시 반환 — FULL·K1024 셀이 연속으로 오면 필요하다."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ══════════════════════════════════════════════════════ 셀 실행
def run_net_cell(name, kind, cfg, M, mats, batch=None):
    """스크래치 신경 셀 — 라운드 상한 RD_CAP + 자기 종결.
    반환 (열, 플래그, es_epoch 목록)."""
    m_tr = M['m_tr']
    CU, CV, CUs, CVs = mats
    CUt = np.hstack([CU[m_tr], M['Lu_tr'][m_tr]]).astype(np.float32)
    CVt = np.hstack([CV[m_tr], M['Lv_tr'][m_tr]]).astype(np.float32)
    CUs2 = np.hstack([CUs, M['Lu_s2']]).astype(np.float32)
    CVs2 = np.hstack([CVs, M['Lv_s2']]).astype(np.float32)
    z, cols, flags, prev, es_hist, gain = None, [], [], None, [], None
    for rd in range(1, RD_CAP + 1):
        r_s2, aux = fit_net(kind, cfg, CUt, CVt, M['y01m'], z, M['even'], SEED,
                            CUs2, CVs2, batch=batch)
        gpu_flush()
        cols.append((f'{name}·r{rd}', r_s2))
        es_hist.append(aux['es_epoch'])
        if aux['es_cap']:
            flags.append(f'⚠ 학습여지(epoch) {M["ax"]}·{name}·r{rd}·s{SEED}')
        gain = None if prev is None else prev - aux['es_ll']
        prev = aux['es_ll']
        z = aux['z_tr']
        if gain is not None and gain < RD_STOP:
            break                                       # 죽은 라운드 — 사슬 자기 종결
    else:
        if gain is None or gain > RD_GAIN:
            flags.append(f'⚠ 학습여지(라운드) {M["ax"]}·{name}·r{RD_CAP}·s{SEED}')
    del CUt, CVt, CUs2, CVs2
    gc.collect()
    return cols, flags, es_hist


def run_lin_cell(name, M, srcmats, auto=False, cat=False):
    """LIN 스크래치 (로짓 열은 무벌점 선두, 라운드 3).

    variant='src' (기본) — 소스별 4체인 x 3라운드 = 12열. 소스마다 자기 λ 를 고르고
      소스 간 가중은 stage2 결합기가 정한다. **이것이 LIN 판 '전문가별 예산 보장'이다.**
    variant='cat' — 전소스 concat 1체인 x 3라운드 = 3열. λ 하나로 4,096열을 묶고 소스
      혼합을 train 안에서 확정한다. 위 팔의 **대조군**.
    auto=True 면 λ 자동확장(-lx 팔).
    """
    fitter = fit_beta_auto if auto else RF.fit_beta
    cols, flags = [], []
    if cat:
        groups = [('cat', np.hstack([srcmats[s][0] for s in M['src']]),
                   np.hstack([srcmats[s][1] for s in M['src']]))]
    else:
        groups = [(s, srcmats[s][0], srcmats[s][1]) for s in M['src']]
    for s, Praw, Ps2raw in groups:
        Ptr = np.hstack([M['Lb_tr'][M['m_tr']], Praw[M['m_tr']]])
        Ps2 = np.hstack([M['Lb_s2'], Ps2raw])
        z, prev = None, None
        for rd in range(1, ROUNDS_LIN + 1):
            beta, st = fitter(z, Ptr, M['y01m'], M['even'], M['bias'],
                              f'{M["ax"]}·{name}·{s}·r{rd}', free=M['nl'])
            cols.append((f'{name}·{s}·r{rd}', ((Ps2 - st['mu']) / st['sc']) @ beta))
            cur = float(np.mean(np.logaddexp(0, st['z_tr']) - M['y01m'] * st['z_tr']))
            if rd == ROUNDS_LIN and prev is not None and prev - cur > RD_GAIN:
                flags.append(f'⚠ 학습여지(라운드) {M["ax"]}·{name}·r{rd}·s{SEED}')
            prev = cur
            z = st['z_tr']
        del Ptr, Ps2
        gc.collect()
    return cols, flags


def cell_name(c):
    if c['kind'] == 'GAT':
        base = f"GAT-{c['K']}-h{c['h']}-{c['unit']}"
    elif c['kind'] == 'FFN':
        base = f"FFN-{c['K']}-r{c['ratio']}"
    else:
        base = f"LIN-{c['K']}-{c.get('variant', 'src')}"
    return base + ''.join(f'-{a}' for a in c.get('arms', ()))


def stage1_cells():
    cells = []
    for K in KS_NEURAL:
        for h in GAT_HS:
            for u in GAT_UNITS:
                cells.append(dict(kind='GAT', K=K, h=h, unit=u))
    for K in KS_NEURAL:
        if K == 'FULL':
            continue                                    # 5.4억 파라미터 — 원그리드에서 제외
        for r in FFN_RATIOS:
            cells.append(dict(kind='FFN', K=K, ratio=r))
    for K in KS_LIN:
        cells.append(dict(kind='LIN', K=K))
    return cells


def dim_of(K):
    return (8192 if K == 'FULL' else 2 * K) + 8          # 로짓 자유열 5(짝)+3(홀)


# ══════════════════════════════════════════════════════ main
def main():
    global LR, EPOCHS, PATIENCE, RD_CAP, MB_BS
    ap = argparse.ArgumentParser()
    ap.add_argument('--axes', default='tie,win')
    ap.add_argument('--stage', default='all', choices=['1', 'lin', '2', 'all'],
                    help="1=기본 사다리 / lin=LIN λ자동확장(-lx) / 2=신규3팔 / all=1->lin->2")
    ap.add_argument('--lam-ks', default='all',
                    help="-lx 를 걸 LIN K 목록. 'all'(기본) 또는 'FULL' 또는 '256,512,FULL'")
    ap.add_argument('--cells', default='',
                    help="2단계 대상 명시 지정. 쉼표 구분, 축 접두 가능: "
                         "'tie:GAT-512-h128-tanh·sig,win:LIN-FULL-src-lx'. "
                         "생략하면 헤드(GAT/FFN)별 1위를 자동 선정한다.")
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--patience', type=int, default=PATIENCE)
    ap.add_argument('--rounds', type=int, default=RD_CAP)
    ap.add_argument('--batch', type=int, default=MB_BS, help='미니배치 팔의 배치 크기')
    args = ap.parse_args()
    LR, EPOCHS, PATIENCE, RD_CAP = args.lr, args.epochs, args.patience, args.rounds
    MB_BS = args.batch
    axes = [s.strip() for s in args.axes.split(',') if s.strip()]
    t0 = time.time()

    A = R5.load_repr(single_best=True)
    y, nz, hold, cu, fold = A['y'], A['nz'], A['hold'], A['cu'], A['fold']
    ar = RF.pick_arith(A['ids'], y)
    TR = RF.load_train(ar)
    y_tr = TR['y']
    S2 = RF.stage2_single(A['ids'], y, TR['s_tb1'] is not None, ar)
    PS2 = RF.load_pool_s2(A['ids'])
    RF.LAMS_TR = LAMS_FINAL
    print(f'  계약: 단일 ckpt·raw·무온도·무앙상블 | seed {SEED} | scratch 고정 | '
          f'lr {LR:g} · epoch {EPOCHS} · patience {PATIENCE} · 라운드 상한 {RD_CAP}\n'
          f'  미니배치 팔: bs {MB_BS} · epoch {MB_EPOCHS} · patience {MB_PATIENCE} | '
          f'파라미터 캡 {PARAM_CAP:,}\n')

    assert HG.OOF_NPZ.exists(), 'headgrid_oof.npz 필요 — 기둥(pil__)·M1 기준 재사용'
    gz = np.load(HG.OOF_NPZ)

    done = {}
    if OOF_NPZ.exists():
        old = np.load(OOF_NPZ, allow_pickle=True)
        done = {k: old[k] for k in old.files}
        print(f'  이어하기: 기존 {len(done)}키 적재\n')

    def save(key, vec):
        done[key] = vec
        np.savez_compressed(OOF_NPZ, **done)

    # ── 축 무관 원본 (라벨 비접촉, train 통계) ──
    t_pre = time.time()
    all_tr = np.ones(len(y_tr), bool)
    Ut, Us = RF.norm_pools(TR, PS2, list(HG.EVK), True, all_tr)
    Vt, Vs = RF.norm_pools(TR, PS2, list(HG.ODK), False, all_tr)
    FULLMAT = (Ut.astype(np.float32), Vt.astype(np.float32),
               Us.astype(np.float32), Vs.astype(np.float32))
    del Ut, Us, Vt, Vs
    gc.collect()
    print(f'  공용 재료(비압축 풀링)  [{(time.time() - t_pre) / 60:.1f}분]\n')

    tie_names = ['s_tie', 's_tb1', 't3', 't_a1', 't_ar']
    AXCFG = {
        'tie': dict(y01=(y == 2).astype(np.float64), y01_tr=(y_tr == 2).astype(np.float64),
                    rows=cu, bias=True, even=True, base=A['T5'], names=tie_names,
                    src=['tb_u', 'abe_u', 'a1_u', 'tb1_u']),
        'win': dict(y01=(y == 0).astype(np.float64), y01_tr=(y_tr == 0).astype(np.float64),
                    rows=cu & nz, bias=False, even=False, base=A['M3'],
                    names=['m3', 'm_a1', 'm_ar'], src=['a1_v', 'abe_v', 'tb_v', 'tb1_v']),
    }
    Ms, PIL, LLM1 = {}, {}, {}
    for ax in ('tie', 'win'):
        C = AXCFG[ax]
        m_tr = np.ones(len(y_tr), bool) if ax == 'tie' else (y_tr != 2)
        cols_tr = [TR[nm] for nm in C['names']]
        cols_s2 = [S2[nm] for nm in C['names']]
        tie_tr = np.column_stack([TR[nm] for nm in tie_names])
        tie_s2 = np.column_stack([S2[nm] for nm in tie_names])
        win_tr = np.column_stack([TR[nm] for nm in ('m3', 'm_a1', 'm_ar')])
        win_s2 = np.column_stack([S2[nm] for nm in ('m3', 'm_a1', 'm_ar')])
        mu_u, sc_u = HG.zstats(tie_tr, True)
        _, sc_v = HG.zstats(win_tr, False)
        Ms[ax] = dict(ax=ax, even=C['even'], bias=C['bias'], y01=C['y01'], rows=C['rows'],
                      fold=fold, hold=hold, base=C['base'], src=C['src'],
                      y01m=C['y01_tr'][m_tr], m_tr=m_tr, nl=len(C['names']),
                      Lb_tr=np.column_stack(cols_tr), Lb_s2=np.column_stack(cols_s2),
                      Lu_tr=(tie_tr - mu_u) / sc_u, Lu_s2=(tie_s2 - mu_u) / sc_u,
                      Lv_tr=win_tr / sc_v, Lv_s2=win_s2 / sc_v)
        assert f'pil__{ax}__0' in gz.files, f'기둥(pil__{ax}) 없음 — headgrid_oof.npz 확인'
        PIL[ax] = [(f'p{i}', gz[f'pil__{ax}__{i}']) for i in range(4)]
        LLM1[ax] = gz[f'll__{ax}__M1']

    def dstat(ax, v, extra=None):
        m = ~np.isnan(v)
        if extra is not None:
            m = m & extra
        return RH.paired_ci(LLM1[ax][m], v[m])

    # ── 셀 배치 실행기 (K 로 묶어 재료를 한 번만 만든다) ──
    def run_batch(cells, tag, bar):
        """cells: dict 목록. 같은 (K, whiten, per_expert) 는 재료를 공유한다."""
        order = sorted(cells, key=lambda c: (
            KS_NEURAL.index(c['K']) if c['K'] in KS_NEURAL else 99,
            'nw' in c.get('arms', ()), 'pe' in c.get('arms', ()), c['kind']))
        cur_key, mats, srcmats = None, None, None
        for c in order:
            arms = c.get('arms', ())
            name = cell_name(c)
            for ax in (c.get('axes') or axes):        # 셀이 축을 지정하면 그 축만
                k1 = f'll__{ax}__{name}__s{SEED}'
                k2 = f'llS__{ax}__{name}__s{SEED}'
                need = (k2 not in done) if c['kind'] == 'LIN' else \
                    (k1 not in done or k2 not in done)
                if not need:
                    bar.update(1)
                    continue
                if c['kind'] != 'LIN':
                    d = dim_of(c['K'])
                    npar = n_params(c['kind'], d, c)
                    if npar > PARAM_CAP:
                        flag(f'⏭ 캡 초과로 제외 {ax}·{name} — 파라미터 {npar:,} '
                             f'> {PARAM_CAP:,}')
                        bar.update(1)
                        continue
                key = (c['K'], 'nw' in arms, 'pe' in arms, c['kind'] == 'LIN')
                if key != cur_key:
                    del mats, srcmats
                    mats = srcmats = None
                    gc.collect()
                    t_m = time.time()
                    if c['kind'] == 'LIN':
                        srcmats = {}
                        for a2 in axes:
                            srcmats.update(lin_mats(TR, PS2, Ms[a2]['src'], c['K'], all_tr))
                    else:
                        mats = flat_mats(TR, PS2, FULLMAT, c['K'], all_tr,
                                         whiten=('nw' not in arms),
                                         per_expert=('pe' in arms))
                    cur_key = key
                    print(f'  [재료] K={c["K"]} nw={"nw" in arms} pe={"pe" in arms} '
                          f'{"LIN" if c["kind"] == "LIN" else "FLAT"}  '
                          f'{(time.time() - t_m) / 60:.1f}분')
                bar.set_postfix_str(f'{tag}·{ax}·{name}')
                t_c = time.time()
                if c['kind'] == 'LIN':
                    cols, flgs = run_lin_cell(name, Ms[ax], srcmats,
                                              auto=('lx' in arms),
                                              cat=(c.get('variant') == 'cat'))
                    es_hist = []
                else:
                    cols, flgs, es_hist = run_net_cell(
                        name, c['kind'], c, Ms[ax], mats,
                        batch=(MB_BS if 'mb' in arms else None))
                for fl in flgs:
                    flag(fl)
                if c['kind'] != 'LIN':
                    save(k1, HG.eval_cols(cols, Ms[ax], PIL[ax]))
                save(k2, HG.eval_cols(cols, Ms[ax], None))
                d2 = dstat(ax, done[k2])
                lead = (f'얹기 {RH.ci_str(*dstat(ax, done[k1]))} | 단독 {RH.ci_str(*d2)}'
                        if c['kind'] != 'LIN' else f'기둥대체 {RH.ci_str(*d2)}')
                print(f'  {ax}·{name}  라운드{max(len(es_hist), len(cols))} es{es_hist}  '
                      f'{(time.time() - t_c) / 60:.1f}분  {lead}')
                bar.update(1)
        del mats, srcmats
        gc.collect()

    def n_runs(cells):
        return sum(len(c.get('axes') or axes) for c in cells)

    # ── 1단계 ──
    s1 = stage1_cells()
    do1 = args.stage in ('1', 'all')
    dolx = args.stage in ('lin', 'all')
    do2 = args.stage in ('2', 'all')
    if do1:
        RH.head('1단계', f'1단계 기본 사다리 — {len(s1)}셀/축 x {len(axes)}축 '
                         '(scratch·화이트닝 on·풀배치·통합 SVD)')
        bar = tqdm(total=n_runs(s1), desc='stage1', unit='셀', smoothing=0.05)
        run_batch(s1, 's1', bar)
        bar.close()

    # ── LIN λ 자동확장 (-lx) — 후보 선정 전에 먼저 돌려야 순위가 공정하다 ──
    slx = []
    if dolx:
        lam_ks = (list(KS_LIN) if args.lam_ks == 'all' else
                  [k if k == 'FULL' else int(k) for k in args.lam_ks.split(',')])
        slx = [dict(kind='LIN', K=K, arms=('lx',)) for K in lam_ks]
        RH.head('LIN·λ확장', f'LIN λ 자동확장 -lx — {len(slx)}셀/축 x {len(axes)}축 '
                             f'(격자 1e0…1e12 시작, 상단이면 x10 확장, 상한 {LAM_CEIL:g})')
        bar = tqdm(total=n_runs(slx), desc='lin-lx', unit='셀', smoothing=0.05)
        run_batch(slx, 'lx', bar)
        bar.close()

    # ── 2단계: 후보에 신규 3팔 (--cells 로 명시 지정 가능) ──
    s2 = []
    if do2 and args.cells:
        name2cfg = {cell_name(c): c for c in s1 + slx}
        for tok in args.cells.split(','):
            tok = tok.strip()
            if not tok:
                continue
            ax_pref, _, nm = tok.rpartition(':')
            cfg = name2cfg.get(nm)
            assert cfg is not None, (f'--cells 이름 불일치: {nm}\n'
                                     f'  가능한 이름 예: {sorted(name2cfg)[:3]} …')
            if cfg['kind'] == 'LIN':
                # LIN 은 3팔 중 둘이 **구조적으로 정의되지 않는다**:
                #   화이트닝 해제 -> fit_beta 가 열별로 재표준화(짝 std / 홀 RMS)하므로
                #                    열 스케일 c 배는 mu·sc 도 c 배 되어 정확히 상쇄된다.
                #                    적합 결과가 비트 단위로 같다 = 팔이 성립하지 않는다.
                #   미니배치     -> 릿지는 L-BFGS 로 정확해를 찾는다. 최적화 예산이라는
                #                    개념 자체가 없다.
                # 남는 하나(전문가별 예산)만 대응물이 있다: LIN 의 기본형 src 가 이미
                # '소스별 예산 보장'이므로, 그 **대조군은 전소스 concat 단일 체인(cat)**이다.
                print(f'  [2단계] {nm}(LIN) — 화이트닝/미니배치 팔은 구조적으로 정의되지 '
                      '않는다(사유는 소스 주석). 예산 배분 팔만 대조군 cat(전소스 concat '
                      '단일 체인)으로 실행한다')
                c2 = dict(cfg)
                c2['variant'] = 'cat'
                c2['arms'] = ()
                if ax_pref:
                    c2['axes'] = [a.strip() for a in ax_pref.split('|')]
                if c2 not in s2:
                    s2.append(c2)
                continue
            for arm in ARMS:
                if cfg['K'] == 'FULL' and arm in ('nw', 'pe'):
                    print(f'  [2단계] {nm}·{ARM_NAME[arm]} 불가(FULL 은 SVD 좌표가 '
                          '아니다) — 건너뜀')
                    continue
                c2 = dict(cfg)
                c2['arms'] = (arm,)
                if ax_pref:
                    c2['axes'] = [a.strip() for a in ax_pref.split('|')]
                if c2 not in s2:
                    s2.append(c2)
    elif do2:
        for ax in axes:
            for kind in ('GAT', 'FFN'):
                cand = [c for c in s1 if c['kind'] == kind
                        and f'll__{ax}__{cell_name(c)}__s{SEED}' in done]
                if not cand:
                    print(f'  [2단계] {ax}·{kind} 1단계 결과 없음 — 건너뜀')
                    continue
                best = max(cand, key=lambda c: dstat(
                    ax, done[f'll__{ax}__{cell_name(c)}__s{SEED}'])[0])
                print(f'  [2단계] {ax}·{kind} 승자 = {cell_name(best)} '
                      f'{RH.ci_str(*dstat(ax, done[f"ll__{ax}__{cell_name(best)}__s{SEED}"]))}')
                for arm in ARMS:
                    if best['K'] == 'FULL' and arm in ('nw', 'pe'):
                        print(f'         └ {ARM_NAME[arm]} 팔 불가(FULL 은 SVD 좌표가 '
                              f'아니다) — 건너뜀')
                        continue
                    c2 = dict(best)
                    c2['arms'] = (arm,)
                    if c2 not in s2:
                        s2.append(c2)
    if s2:
        RH.head('2단계', f'2단계 신규 3팔 — {len(s2)}셀 / 실행 {n_runs(s2)}회 '
                         '(화이트닝 해제 / 미니배치 / 전문가별 예산)')
        bar = tqdm(total=n_runs(s2), desc='stage2', unit='셀', smoothing=0.05)
        run_batch(s2, 's2', bar)
        bar.close()

    # ── 축별 최종표 ──
    allc = s1 + slx + s2
    for ax in axes:
        rows = []
        for c in allc:
            name = cell_name(c)
            k1, k2 = f'll__{ax}__{name}__s{SEED}', f'llS__{ax}__{name}__s{SEED}'
            if k2 not in done:
                continue
            if c['kind'] == 'LIN':
                d2 = dstat(ax, done[k2])
                rows.append((d2[0], [name, '— (기둥대체)', RH.ci_str(*d2),
                                     RH.ci_str(*dstat(ax, done[k2], extra=hold))]))
            elif k1 in done:
                d1 = dstat(ax, done[k1])
                rows.append((d1[0], [name, RH.ci_str(*d1), RH.ci_str(*dstat(ax, done[k2])),
                                     RH.ci_str(*dstat(ax, done[k1], extra=hold))]))
        rows.sort(key=lambda r: -r[0])
        RH.head(ax + '·최종', f'{ax}축 전체 순위 — Δ vs M1 (얹기 / 단독 / holdout)')
        RH.table(['셀', '얹기 Δ', '단독 Δ', 'holdout Δ'], [r for _, r in rows],
                 ['l', 'l', 'l', 'l'])

    # ── 두 축 결합 ──
    def best_of(ax, prefix, kinds):
        cand = [(cell_name(c), done[f'{prefix}__{ax}__{cell_name(c)}__s{SEED}'])
                for c in allc if c['kind'] in kinds
                and f'{prefix}__{ax}__{cell_name(c)}__s{SEED}' in done]
        return max(cand, key=lambda t: dstat(ax, t[1])[0]) if cand else None

    if 'tie' in axes and 'win' in axes:
        clean = ~np.isnan(LLM1['tie'])

        def combo(lt, lw, msk):
            mt, mw = msk & ~np.isnan(lt), msk & ~np.isnan(lw)
            return float(lt[mt].mean() + mw.sum() / mt.sum() * lw[mw].mean())

        base_c = combo(LLM1['tie'], LLM1['win'], clean)
        base_h = combo(LLM1['tie'], LLM1['win'], clean & hold)
        ties = [(nm + ' (얹기)', v) for nm, v in
                [x for x in [best_of('tie', 'll', ('GAT', 'FFN'))] if x]]
        ties += [(nm + ' (단독)', v) for nm, v in
                 [x for x in [best_of('tie', 'llS', ('GAT', 'FFN'))] if x]]
        wins = [(nm + ' (기둥대체)', v) for nm, v in
                [x for x in [best_of('win', 'llS', ('LIN',))] if x]]
        wins += [(nm + ' (얹기)', v) for nm, v in
                 [x for x in [best_of('win', 'll', ('GAT', 'FFN'))] if x]]
        if ties and wins:
            rows = [['M1 x M1 (참조)', f'{base_c:.4f}', '+0.0000', f'{base_h:.4f}',
                     '+0.0000']]
            for tnm, tv in ties:
                for wnm, wv in wins:
                    c_ = combo(tv, wv, clean)
                    h_ = combo(tv, wv, clean & hold)
                    rows.append([f'tie {tnm} x win {wnm}', f'{c_:.4f}',
                                 f'{base_c - c_:+.4f}', f'{h_:.4f}', f'{base_h - h_:+.4f}'])
            RH.head('결합', '두 축 결합 총 ll = tie축 + (1-tie율)·승부축 — 배포 후보')
            RH.table(['구성', 'stage2 총ll', 'Δ', 'holdout 총ll', 'Δ'], rows,
                     ['l', 'r', 'r', 'r', 'r'])
    else:
        print('\n  (두 축 결합표는 양축 결과가 모두 모이면 출력 — 이어하기로 재실행)')

    print(f'\n총 {(time.time() - t0) / 3600:.1f}시간 | 저장: {OOF_NPZ}')
    print(f'과소적합·제외 플래그: {FLAG_TXT} | 판정(본학습 헤드 확정)은 사용자.')


if __name__ == '__main__':
    main()
