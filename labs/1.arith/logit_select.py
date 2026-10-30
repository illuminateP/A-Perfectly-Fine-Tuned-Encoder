"""logit_select.py — ABE·TB 최적 체크포인트로 train 46,729 로짓을 뽑고(1회 캐시),
[로짓 + 산술 피처] 섀도잉 선별을 train/stage2 이중으로 돌려 생존 피처를 보고한다.

파이프라인에서의 위치 (합의 2026-08-02)
---------------------------------------
  [1] 선별(이 스크립트) — 로짓은 '심판'으로만 쓰고 버린다. 산출물 = 선별 리포트 CSV
  [2] Arith 본학습(Train.ipynb) — 생존 피처만, 로짓 0열, 표준 학습 (독립 전문가)
  [3] 결합(stage2) — [ABE, TB, Arith] 로짓 3벌만 섞는다

왜 이중 선별인가
----------------
train 위의 본체 로짓은 in-sample(부분 암기)이라 섀도잉이 배포 시점보다 과하게 걸린다
→ 직교 피처의 false rejection 방향 편향. 그래서:
  런 A (train 46,729)  : 선별력 큼, 암기 편향 있음
  런 B (stage2 10,748) : clean 로짓, 표본 작음
일치 그룹은 확정, 불일치는 B(clean) 우선. in-sample 낙관 갭(train ll − stage2 ll)을
함께 출력해 편향의 크기를 계측한다.

생존 지표
---------
[로짓+피처] 트리에서 gain 은 로짓이 독식하므로 절대값은 무의미하다. 대신
  survival = (로짓 동석 시 피처 간 gain 점유율) / (피처 단독 런의 gain 점유율)
— 1 에 가까우면 로짓이 못 가리는 신호(직교), 0 에 가까우면 이미 흡수된 신호.
확정은 유형(그룹) 단위로 읽는다. 개별 피처 순위는 노이즈가 크다.

최적 체크포인트 선정 (holdout 기준, 노트북 검증 셀과 동일)
----------------------------------------------------------
각 checkpoint-*/logits_export.npz 에서 valid TTA 로짓으로 T 를 피팅(LBFGS)하고
holdout TTA+T 로 채점(ABE: 3-class ll / TB: binary tie ll) → 최소 ll 체크포인트 채택.
GPU 불필요·수 초. 추출 npz 에 step·T 를 기록하고 재실행 시 일치 assert.

추출 (bf16, 로컬 9070XT — ROCm 갱신으로 bf16 사용 가능)
-------------------------------------------------------
  ABE  bi-encoder : 브랜치별 1회 인코딩 -> 풀링 벡터 재사용으로 fwd/swp 두 뷰 헤드 계산
                    (인코더 2N 패스, TTA 공짜). swp 는 [1,0,2] 재정렬 저장(export 관례).
  TB   cross      : (Q,A,B)/(Q,B,A) 두 패스. tie 2-class 라 재정렬 없음.
  앵커: valid 부분집합(최장 512 + 랜덤 512)을 같은 파이프라인으로 뽑아 저장본과
        행 단위 corr > 0.98 assert (L40S bf16 -> 로컬 bf16 오차 허용).

사용 (env ML):
  P=python
  $P Arith/logit_select.py                  # 전부: 선정 -> 추출(캐시 스킵) -> 선별
  $P Arith/logit_select.py --select-only    # 캐시로 선별만
  $P Arith/logit_select.py --tok-budget 6144    # OOM 시 하향
예상 시간: ABE 추출 ~1h + TB 추출 ~2.5-3.5h + 선별 ~15분. 재실행은 선별만.

산출물 (Arith/logits/):
  abe_train_logits.npz / tb_train_logits.npz   ids, y, fwd, swp, step, T
  stage2_folds.csv                             stage2 5-fold 동결
  select_report.csv                            피처별 gain 점유율·생존비 (런 4종)
"""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

ARITH = Path('./Arith')
OUT = ARITH / 'logits'
ABE_BEST = Path('./V5/aws/outputs/best_checkpoints')
TB_BEST = Path('./V6/pod/outputs/best_checkpoints')

SEED = 42
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP = torch.bfloat16                              # 학습(L40S bf16)과 동일 정밀도 경로
HEAD_FRAC = 0.6
ABE_MAX_LEN, ABE_Q_FLOOR = 2048, 256              # Train_ABE cell 13
TB_MAX_LEN, TB_Q_FLOOR = 3072, 256                # Train_TB cell 14
LOGIT_COLS = ['abe_z_a', 'abe_z_b', 'abe_z_tie', 'tb_z_notie', 'tb_z_tie']


# ── 전처리 — 각 노트북에서 그대로 (하나라도 다르면 앵커가 깨진다) ──────────────
def parse_text(x):
    if pd.isna(x):
        return ''
    try:
        turns = json.loads(x)
    except (json.JSONDecodeError, TypeError):
        return str(x)
    if isinstance(turns, list):
        return ' '.join(t for t in turns if isinstance(t, str))
    return str(turns)


def to_model_text(t):
    return t.replace('\n', ' [BR] ')


def head_tail(ids, k):
    ids = list(ids)
    if len(ids) <= k:
        return ids
    h = int(k * HEAD_FRAC)
    t = k - h
    return ids[:h] + ids[-t:] if t > 0 else ids[:h]


def abe_build_row(iq, ia, ib, CLS, SEP):          # Train_ABE cell 17 (BUDGET = 2048-3)
    budget = ABE_MAX_LEN - 3
    lq, la, lb = len(iq), len(ia), len(ib)
    kq = min(lq, ABE_Q_FLOOR)
    k_resp = min(max(la, lb), budget - kq)
    kq = min(lq, kq + (budget - kq - k_resp))     # alloc_q: 답변이 안 쓴 예산 회수
    ka, kb = min(la, budget - kq), min(lb, budget - kq)
    q = head_tail(iq, kq)
    sa = [CLS] + q + [SEP] + head_tail(ia, ka) + [SEP]
    sb = [CLS] + q + [SEP] + head_tail(ib, kb) + [SEP]
    return sa, sb


def tb_build_triple(iq, ia, ib, CLS, SEP):        # Train_TB cell 17 (BUDGET = 3072-4)
    budget = TB_MAX_LEN - 4
    lq, la, lb = len(iq), len(ia), len(ib)
    kq = min(lq, TB_Q_FLOOR)
    rem = budget - kq
    half = rem // 2
    if la <= half and lb <= half:
        ka, kb = la, lb
    elif la <= half:
        ka, kb = la, min(lb, rem - la)
    elif lb <= half:
        kb, ka = lb, min(la, rem - lb)
    else:
        ka, kb = half, rem - half
    kq = min(lq, kq + (budget - kq - ka - kb))
    q, a, b = head_tail(iq, kq), head_tail(ia, ka), head_tail(ib, kb)
    seq = [CLS] + q + [SEP] + a + [SEP] + b + [SEP]
    qs = (1, 1 + len(q))
    as_ = (qs[1] + 1, qs[1] + 1 + len(a))
    bs = (as_[1] + 1, as_[1] + 1 + len(b))
    return seq, (qs, as_, bs)


# ── 모델 재현 — 저장 state_dict 와 키가 정확히 일치해야 strict 로드가 된다 ──────
class AttnPool(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.score = nn.Linear(h, 1)

    def forward(self, last_hidden, mask):
        s = self.score(last_hidden).squeeze(-1)
        s = s.float().masked_fill(mask == 0, float('-inf'))
        w = torch.softmax(s, dim=1)
        # empty span (blank response) -> all -inf -> NaN. Deployed TieModel maps it to a zero
        # vector; reproducing that line is mandatory or the export differs on those rows.
        # 빈 구간(공백 응답)은 NaN 이 되는데 배포본이 0 벡터로 누른다 — 이 줄이 없으면 재현이 깨진다
        w = torch.nan_to_num(w).unsqueeze(-1)
        return (last_hidden.float() * w).sum(dim=1)


def build_model(tag, ckpt_dir, root):
    from safetensors.torch import load_file
    from transformers import AutoConfig, DebertaV2Model
    cfg = AutoConfig.from_pretrained(root)
    h = cfg.hidden_size

    if tag == 'abe':                              # PrefModel (Train_ABE cell 25)
        class PrefModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = DebertaV2Model(cfg)
                self.pool = AttnPool(h)
                self.LN = nn.LayerNorm(h)
                self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h * 4, 1024), nn.GELU(),
                                          nn.Dropout(0.1), nn.Linear(1024, 3))
        model = PrefModel()
    else:                                         # TieModel (Train_TB cell 25)
        class TieModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = DebertaV2Model(cfg)
                self.pool_ab = AttnPool(h)
                self.pool_q = AttnPool(h)
                self.LN = nn.LayerNorm(h)
                self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h * 5, 1024), nn.GELU(),
                                          nn.Dropout(0.1), nn.Linear(1024, 2))
        model = TieModel()

    sd = load_file(Path(ckpt_dir) / 'model.safetensors')
    model.load_state_dict(sd, strict=True)
    print(f'[{tag}] {Path(ckpt_dir).name} 로드: 텐서 {len(sd)}개 strict 통과')
    return model.to(DEV).eval()


# ── 최적 체크포인트 선정 — holdout TTA+T (노트북 검증 셀과 동일 기준) ───────────
def fit_temperature(logits, labels):
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(z / logT.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.exp().detach())


def score_ckpt(tag, npz_path):
    z = np.load(npz_path)
    v_avg = (z['valid_fwd'] + z['valid_swp']) / 2
    h_avg = (z['hold_fwd'] + z['hold_swp']) / 2
    vl, hl = z['valid_labels'], z['hold_labels']
    if tag == 'abe':
        T = fit_temperature(v_avg, vl)
        p = torch.softmax(torch.tensor(h_avg / T, dtype=torch.float32), 1).numpy()
        ll = log_loss(hl, p, labels=[0, 1, 2])
    else:
        T = fit_temperature(v_avg, (vl == 2).astype(np.int64))
        p = torch.softmax(torch.tensor(h_avg / T, dtype=torch.float32), 1).numpy()[:, 1]
        ll = log_loss((hl == 2).astype(int), np.c_[1 - p, p], labels=[0, 1])
    return ll, T


def pick_best(tag, root):
    rows = []
    for p in sorted(glob.glob(str(root / 'checkpoint-*')), key=lambda s: int(s.rsplit('-', 1)[1])):
        step = int(p.rsplit('-', 1)[1])
        ll, T = score_ckpt(tag, Path(p) / 'logits_export.npz')
        rows.append((step, ll, T, p))
    rows.sort(key=lambda r: r[1])
    print(f'\n[{tag}] holdout TTA+T 채점 ({"3-class ll" if tag == "abe" else "binary tie ll"}):')
    for step, ll, T, _ in sorted(rows):
        mark = ' <-- best' if step == rows[0][0] else ''
        print(f'  checkpoint-{step:<6} ll {ll:.4f}  T {T:.3f}{mark}')
    step, ll, T, path = rows[0]
    return step, T, path


# ── 추출 ──────────────────────────────────────────────────────────────────────
def load_train():
    df = pd.read_csv(ARITH / 'origin' / 'train.csv')
    assert len(df) == 46729, f'train {len(df)}행'
    y = np.select([df['winner_model_a'] == 1, df['winner_model_b'] == 1, df['winner_tie'] == 1],
                  [0, 1, 2], -1).astype(np.int64)
    assert (y != -1).all()
    return df, y


def tokenize_cols(df, tokenizer):
    tt = lambda s: tokenizer([to_model_text(parse_text(t)) for t in s],
                             add_special_tokens=False)['input_ids']
    return tt(df['prompt']), tt(df['response_a']), tt(df['response_b'])


def batched(order, lengths, tok_budget):
    i = 0
    while i < len(order):
        L0 = int(lengths[order[i]])
        bs = max(1, min(32, tok_budget // max(L0, 1)))
        yield order[i:i + bs]
        i += bs


def pad_batch(seqs, PAD):
    L = max(len(s) for s in seqs)
    ids = np.full((len(seqs), L), PAD, np.int64)
    m = np.zeros((len(seqs), L), np.int64)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = s
        m[r, :len(s)] = 1
    return torch.from_numpy(ids).to(DEV), torch.from_numpy(m).to(DEV)


@torch.no_grad()
def abe_pooled(model, seqs, PAD, tok_budget, desc):
    """브랜치 시퀀스 리스트 -> LN(AttnPool(enc)) 벡터 (N,1024) fp32. 길이 내림차순 배칭."""
    N = len(seqs)
    out = np.zeros((N, 1024), np.float32)
    lengths = np.array([len(s) for s in seqs])
    order = np.argsort(-lengths)
    bar = tqdm(total=N, desc=desc, unit='행')
    for idx in batched(order, lengths, tok_budget):
        ids, m = pad_batch([seqs[j] for j in idx], PAD)
        with torch.autocast('cuda', dtype=AMP, enabled=DEV == 'cuda'):
            H = model.encoder(input_ids=ids, attention_mask=m).last_hidden_state
            e = model.LN(model.pool(H, m))
        out[idx] = e.float().cpu().numpy()
        bar.update(len(idx))
    bar.close()
    return out


@torch.no_grad()
def abe_heads(model, ea, eb):
    """풀링 벡터 재사용으로 fwd/swp 두 뷰 로짓. swp 는 [1,0,2] 재정렬(export 관례)."""
    zf, zs = [], []
    for st in range(0, len(ea), 8192):
        a = torch.tensor(ea[st:st + 8192], device=DEV)
        b = torch.tensor(eb[st:st + 8192], device=DEV)
        with torch.autocast('cuda', dtype=AMP, enabled=DEV == 'cuda'):
            f = model.head(torch.cat([a, b, a - b, torch.tanh(a) * torch.tanh(b)], 1))
            s = model.head(torch.cat([b, a, b - a, torch.tanh(b) * torch.tanh(a)], 1))
        zf.append(f.float().cpu().numpy())
        zs.append(s.float().cpu().numpy()[:, [1, 0, 2]])
    return np.vstack(zf), np.vstack(zs)


@torch.no_grad()
def tb_view(model, Q, A, B, PAD, CLS, SEP, tok_budget, desc):
    """(Q,A,B) 순서 한 뷰의 2-class 로짓 (N,2)."""
    N = len(Q)
    out = np.zeros((N, 2), np.float32)
    lengths = np.array([len(tb_build_triple(Q[i], A[i], B[i], CLS, SEP)[0]) for i in range(N)])
    order = np.argsort(-lengths)
    bar = tqdm(total=N, desc=desc, unit='행')
    for idx in batched(order, lengths, tok_budget):
        seqs, spans = [], []
        for j in idx:
            seq, sp = tb_build_triple(Q[j], A[j], B[j], CLS, SEP)
            seqs.append(seq)
            spans.append(sp)
        ids, m = pad_batch(seqs, PAD)
        masks = {}
        for name, k in (('q', 0), ('a', 1), ('b', 2)):
            sm = torch.zeros_like(m)
            for r, sp in enumerate(spans):
                sm[r, sp[k][0]:sp[k][1]] = 1
            masks[name] = sm
        with torch.autocast('cuda', dtype=AMP, enabled=DEV == 'cuda'):
            H = model.encoder(input_ids=ids, attention_mask=m).last_hidden_state
            q = model.LN(model.pool_q(H, masks['q']))
            a = model.LN(model.pool_ab(H, masks['a']))
            b = model.LN(model.pool_ab(H, masks['b']))
            x = torch.cat([q, a, b, (a - b).abs(), torch.tanh(a) * torch.tanh(b)], 1)
            logits = model.head(x)
        out[idx] = logits.float().cpu().numpy()
        bar.update(len(idx))
    bar.close()
    return out


def anchor_check(tag, root, step, run_fn, anchor_n):
    """valid 부분집합(최장/랜덤 반반)을 같은 파이프라인으로 뽑아 저장 로짓과 대조."""
    z = np.load(root / f'checkpoint-{step}' / 'logits_export.npz')
    vdf = pd.read_csv(ARITH / 'origin' / 'valid.csv')
    assert (vdf['id'].to_numpy() == z['valid_ids']).all(), 'valid 순서 불일치'
    fwd_ref, swp_ref = z['valid_fwd'], z['valid_swp']
    n = len(vdf)
    rng = np.random.default_rng(SEED)
    lens = (vdf['prompt'].str.len().fillna(0) + vdf['response_a'].str.len().fillna(0)
            + vdf['response_b'].str.len().fillna(0)).to_numpy()
    idx = np.unique(np.concatenate([np.argsort(lens)[-anchor_n // 2:],
                                    rng.choice(n, anchor_n // 2, replace=False)]))
    fwd, swp = run_fn(vdf.iloc[idx].reset_index(drop=True))
    for name, mine, ref in (('fwd', fwd, fwd_ref[idx]), ('swp', swp, swp_ref[idx])):
        # 비유한값을 먼저 분리해 보고한다 — corr 이 통째로 nan 이 되면 원인이 안 보인다
        bad_mine = int((~np.isfinite(mine)).any(1).sum())
        bad_ref = int((~np.isfinite(ref)).any(1).sum())
        ok = np.isfinite(mine).all(1) & np.isfinite(ref).all(1)
        assert bad_mine == 0, (f'앵커[{tag}/{name}] 재현본에 비유한값 {bad_mine}행 '
                               f'(참조 {bad_ref}행) — AttnPool 빈 구간 가드 확인')
        corr = np.corrcoef(mine[ok].ravel(), ref[ok].ravel())[0, 1]
        mad = np.abs(mine[ok] - ref[ok]).max()
        print(f'  앵커[{tag}/{name}] n={int(ok.sum())}/{len(idx)}  corr {corr:.5f}  '
              f'max|diff| {mad:.3f}  (참조 비유한 {bad_ref}행)')
        assert corr > 0.98, f'앵커 붕괴({tag}/{name}) — 전처리/가중치 배선 점검'


def extract(tag, args):
    path = OUT / f'{tag}_train_logits.npz'
    root = ABE_BEST if tag == 'abe' else TB_BEST
    step, T, ckpt = pick_best(tag, root)
    if path.exists():
        z = np.load(path)
        assert int(z['step']) == step, (f'{path.name} 캐시 step {int(z["step"])} != 현 최적 {step}'
                                        ' — 캐시 삭제 후 재실행')
        print(f'[{tag}] 캐시 있음 (step {step}) → 추출 스킵')
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root)
    CLS, SEP, PAD = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    assert (CLS, SEP, PAD) == (1, 2, 0) and len(tokenizer) == 128002, '토크나이저 확인'
    model = build_model(tag, ckpt, root)
    df, y = load_train()

    def run_rows(frame):
        Q, A, B = tokenize_cols(frame, tokenizer)
        if tag == 'abe':
            seq_a, seq_b = [], []
            for iq, ia, ib in zip(Q, A, B):
                sa, sb = abe_build_row(iq, ia, ib, CLS, SEP)
                seq_a.append(sa)
                seq_b.append(sb)
            ea = abe_pooled(model, seq_a, PAD, args.tok_budget, 'abe/branch-A')
            eb = abe_pooled(model, seq_b, PAD, args.tok_budget, 'abe/branch-B')
            return abe_heads(model, ea, eb)
        fwd = tb_view(model, Q, A, B, PAD, CLS, SEP, args.tok_budget, 'tb/fwd')
        swp = tb_view(model, Q, B, A, PAD, CLS, SEP, args.tok_budget, 'tb/swp')
        return fwd, swp                            # tie 2-class — 재정렬 없음

    print(f'[{tag}] 앵커 검증 (valid 부분집합 {args.anchor_n}행)')
    anchor_check(tag, root, step, run_rows, args.anchor_n)

    t0 = time.time()
    fwd, swp = run_rows(df)
    tta = (fwd + swp) / 2
    if tag == 'abe':
        ll_tr = log_loss(y, torch.softmax(torch.tensor(tta), 1).numpy(), labels=[0, 1, 2])
    else:
        p = torch.softmax(torch.tensor(tta), 1).numpy()[:, 1]
        ll_tr = log_loss((y == 2).astype(int), np.c_[1 - p, p], labels=[0, 1])
    print(f'[{tag}] train(in-sample) TTA ll {ll_tr:.4f}  ({(time.time()-t0)/60:.0f}분)'
          '  ※ stage2 와의 격차 = 암기 낙관 (선별 리포트에서 대조)')
    tmp = path.with_name(f'{tag}_train_logits.tmp.npz')
    np.savez(tmp, ids=df['id'].to_numpy(), y=y, fwd=fwd, swp=swp,
             step=np.int64(step), T=np.float64(T))
    os.replace(tmp, path)
    del model
    if DEV == 'cuda':
        torch.cuda.empty_cache()
    print(f'저장: {path.name}')


# ── 선별 — [로짓 + 피처] 섀도잉, train/stage2 이중 ─────────────────────────────
def build_swap(cols, swap_series):                # Train.ipynb cell 2 그대로
    neg = [c for c in cols if swap_series[c] == '부호반전']
    recip = [c for c in cols if swap_series[c] == '역수']
    pairs = []
    for c in cols:
        v = swap_series[c]
        if isinstance(v, str) and v.startswith('교환:') and c < v.split(':', 1)[1]:
            partner = v.split(':', 1)[1]
            assert partner in set(cols), f'교환 파트너 {partner} 미포함'
            pairs.append((c, partner))
    inv = [c for c in cols if swap_series[c] == '불변']
    assert len(neg) + len(recip) + len(inv) + 2 * len(pairs) == len(cols), 'swap 분류 누락!'
    idx = {c: i for i, c in enumerate(cols)}
    neg_i = np.array([idx[c] for c in neg], int)
    rec_i = np.array([idx[c] for c in recip], int)
    a_i = np.array([idx[a] for a, _ in pairs], int)
    b_i = np.array([idx[b] for _, b in pairs], int)

    def swap_X(X):
        Xs = X.copy()
        if len(neg_i):
            Xs[:, neg_i] *= -1.0
        if len(rec_i):
            Xs[:, rec_i] = 1.0 / np.clip(Xs[:, rec_i], 1e-12, None)
        if len(a_i):
            Xs[:, a_i], Xs[:, b_i] = Xs[:, b_i].copy(), Xs[:, a_i].copy()
        return Xs
    return swap_X


def swap_label(yy):
    return np.where(yy == 0, 1, np.where(yy == 1, 0, 2)).astype(yy.dtype)


def lgbm_run(X, y, folds, X_es, y_es, swap_X, desc, es_on_fold=False, es_mode=None):
    """Train.ipynb 레시피 LGBM 5-fold (swap 증강 + TTA). 반환 (OOF ll, 평균 gain, OOF 확률).

    es_mode: None      -> X_es/y_es 고정 세트로 ES (런 A: valid)
             'fold'    -> 자기 평가 fold 로 ES (= es_on_fold=True, 낙관 편향 있음)
             'inner'   -> 다음 fold 를 ES 로 떼고 학습에서 제외 (누수 없음, 그룹 델타 기본)
    """
    import inspect
    import lightgbm as lgb
    if es_on_fold and es_mode is None:
        es_mode = 'fold'
    oof = np.zeros((len(X), 3), np.float64)
    gains = np.zeros(X.shape[1])
    ks = np.sort(np.unique(folds))
    new_api = 'eval_X' in inspect.signature(lgb.LGBMClassifier.fit).parameters
    for i, k in enumerate(tqdm(ks, desc=desc, leave=False)):
        va = folds == k
        if es_mode == 'inner':
            k_in = ks[(i + 1) % len(ks)]           # ES 전용 fold — 학습에서도 뺀다
            tr = (folds != k) & (folds != k_in)
            eX, ey = X[folds == k_in], y[folds == k_in]
        else:
            tr = folds != k
            eX, ey = (X[va], y[va]) if es_mode == 'fold' else (X_es, y_es)
        Xf = np.vstack([X[tr], swap_X(X[tr])])
        yf = np.hstack([y[tr], swap_label(y[tr])])
        m = lgb.LGBMClassifier(objective='multiclass', num_class=3, learning_rate=0.03,
                               n_estimators=3000, num_leaves=63, min_child_samples=100,
                               colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                               reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1)
        ekw = {'eval_X': eX, 'eval_y': ey} if new_api else {'eval_set': [(eX, ey)]}
        m.fit(Xf, yf, **ekw, callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = (m.predict_proba(X[va]) + m.predict_proba(swap_X(X[va]))[:, [1, 0, 2]]) / 2
        gains += m.booster_.feature_importance('gain') / len(ks)
    ll = log_loss(y, oof, labels=[0, 1, 2])
    return ll, gains, oof


def per_row_ll(p, y):
    p = np.clip(np.asarray(p, np.float64), 1e-15, 1.0)
    p = p / p.sum(1, keepdims=True)
    return -np.log(p[np.arange(len(y)), y])


def paired_ci(ll_a, ll_b, n_boot=2000):
    """Δ = mean(ll_a − ll_b) 의 95% CI. 양수 = b 가 좋다(= a 가 손해)."""
    d = np.asarray(ll_a) - np.asarray(ll_b)
    rng = np.random.default_rng(SEED)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def load_matrices(verbose=True):
    """추출 캐시 + 피처 + 카탈로그 -> train/stage2 행렬, swap 기계, fold. select/group_delta 공용."""
    for tag in ('abe', 'tb'):
        assert (OUT / f'{tag}_train_logits.npz').exists(), f'{tag} 추출 캐시 없음'
    za = np.load(OUT / 'abe_train_logits.npz')
    zt = np.load(OUT / 'tb_train_logits.npz')

    # 피처·카탈로그 (Train.ipynb cell 1 과 동일 규율)
    df_tr, y_tr = load_train()
    feat = pd.read_parquet(ARITH / 'features_57k.parquet')
    catalog = pd.read_csv(ARITH / 'feature_catalog.csv').fillna({'제외사유': ''})
    ucols = catalog[catalog['사용'] == 1]['feature'].tolist()
    folds_df = pd.read_csv(ARITH / 'arith_folds.csv')
    m_tr = feat['role'].to_numpy() == 'train' if 'role' in feat.columns else None
    assert m_tr is not None, 'features_57k 에 role 컬럼 필요'
    assert (feat.loc[m_tr, 'id'].to_numpy() == df_tr['id'].to_numpy()).all(), '피처 train 정렬 불일치'
    assert (folds_df['id'].to_numpy() == df_tr['id'].to_numpy()).all(), 'folds 불일치'
    assert (za['ids'] == df_tr['id'].to_numpy()).all() and (zt['ids'] == df_tr['id'].to_numpy()).all()
    folds = folds_df['fold'].to_numpy()

    # 로짓 피처 = TTA 평균 (T 는 단조 변환이라 트리에 무영향 -> 미적용)
    abe_tta = (za['fwd'] + za['swp']) / 2
    tb_tta = (zt['fwd'] + zt['swp']) / 2
    L_tr = np.concatenate([abe_tta, tb_tta], 1).astype(np.float32)

    # stage2 로짓 (clean) — 각 최적 ckpt 의 logits_export
    def stage2_logits(root, step):
        z = np.load(root / f'checkpoint-{step}' / 'logits_export.npz')
        v = (z['valid_fwd'] + z['valid_swp']) / 2
        h = (z['hold_fwd'] + z['hold_swp']) / 2
        return np.vstack([v, h]), np.concatenate([z['valid_ids'], z['hold_ids']]), \
            np.concatenate([z['valid_labels'], z['hold_labels']])
    La, ids_s2, lab_s2 = stage2_logits(ABE_BEST, int(za['step']))
    Lt, ids_s2b, _ = stage2_logits(TB_BEST, int(zt['step']))
    assert (ids_s2 == ids_s2b).all()
    m_s2 = feat['role'].isin(['valid', 'holdout']).to_numpy()
    assert (feat.loc[m_s2, 'id'].to_numpy() == ids_s2).all(), 'stage2 피처 정렬 불일치'
    L_s2 = np.concatenate([La, Lt], 1).astype(np.float32)
    y_s2 = lab_s2.astype(np.int64)

    # in-sample 낙관 갭
    if verbose:
        ll3 = lambda z, yy: log_loss(yy, torch.softmax(torch.tensor(z), 1).numpy(), labels=[0, 1, 2])
        print(f'\n=== in-sample 낙관 갭 (TTA raw, T 미적용 동일 기준) ===')
        print(f'  ABE 3-class ll   train {ll3(abe_tta, y_tr):.4f}  vs  stage2 {ll3(La, y_s2):.4f}')
        pt = lambda z: torch.softmax(torch.tensor(z), 1).numpy()[:, 1]
        llb = lambda z, yy: log_loss((yy == 2).astype(int), np.c_[1 - pt(z), pt(z)], labels=[0, 1])
        print(f'  TB  tie ll       train {llb(tb_tta, y_tr):.4f}  vs  stage2 {llb(Lt, y_s2):.4f}')
        print('  격차가 클수록 런 A 의 섀도잉이 과함 -> 판정에서 런 B(clean) 비중을 높일 것')

    # stage2 fold 동결
    f2_path = OUT / 'stage2_folds.csv'
    if f2_path.exists():
        f2 = pd.read_csv(f2_path)
        assert (f2['id'].to_numpy() == ids_s2).all()
        folds2 = f2['fold'].to_numpy()
    else:
        folds2 = np.zeros(len(y_s2), int)
        for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED)
                                    .split(ids_s2, y_s2)):
            folds2[te] = k
        pd.DataFrame({'id': ids_s2, 'fold': folds2}).to_csv(f2_path, index=False)

    # swap 기계: 피처(카탈로그) + 로짓 5열(abe a/b 교환, tie·tb 불변)
    sw = catalog.set_index('feature')['swap']
    swap_feat = build_swap(ucols, sw)
    F_tr = feat.loc[m_tr, ucols].to_numpy(np.float32)
    F_s2 = feat.loc[m_s2, ucols].to_numpy(np.float32)

    def swap_logit(X):                             # 로짓 5열: abe z_a<->z_b 교환, tie·tb 불변
        Xs = X.copy()
        Xs[:, 0], Xs[:, 1] = X[:, 1].copy(), X[:, 0].copy()
        return Xs

    def swap_full(X):                              # [로짓 5 | 피처 d] 결합 행렬용
        Xs = swap_logit(X)
        Xs[:, 5:] = swap_feat(X[:, 5:])
        return Xs

    # ES 세트: 런 A 는 valid(clean 로짓+피처), 런 B 는 fold 기반
    m_va = feat['role'].to_numpy() == 'valid'
    F_va = feat.loc[m_va, ucols].to_numpy(np.float32)
    L_va = L_s2[:m_va.sum()]                       # stage2 앞 5,000 = valid
    y_va = y_s2[:m_va.sum()]
    return dict(catalog=catalog, ucols=ucols, folds=folds, folds2=folds2,
                F_tr=F_tr, L_tr=L_tr, y_tr=y_tr, F_s2=F_s2, L_s2=L_s2, y_s2=y_s2,
                F_va=F_va, L_va=L_va, y_va=y_va,
                swap_feat=swap_feat, swap_logit=swap_logit, swap_full=swap_full)


def select(args):
    M = load_matrices()
    catalog, ucols = M['catalog'], M['ucols']
    folds, folds2 = M['folds'], M['folds2']
    F_tr, L_tr, y_tr = M['F_tr'], M['L_tr'], M['y_tr']
    F_s2, L_s2, y_s2 = M['F_s2'], M['L_s2'], M['y_s2']
    F_va, L_va, y_va = M['F_va'], M['L_va'], M['y_va']
    swap_feat, swap_logit, swap_full = M['swap_feat'], M['swap_logit'], M['swap_full']

    runs = {}
    runs['A0 피처만@train'] = lgbm_run(F_tr, y_tr, folds, F_va, y_va, swap_feat, 'A0')
    runs['A1 로짓+피처@train'] = lgbm_run(np.hstack([L_tr, F_tr]), y_tr, folds,
                                          np.hstack([L_va, F_va]), y_va, swap_full, 'A1')
    runs['LA 로짓만@train'] = lgbm_run(L_tr, y_tr, folds, L_va, y_va, swap_logit, 'LA')
    runs['B0 피처만@stage2'] = lgbm_run(F_s2, y_s2, folds2, None, None, swap_feat, 'B0',
                                        es_on_fold=True)
    runs['B1 로짓+피처@stage2'] = lgbm_run(np.hstack([L_s2, F_s2]), y_s2, folds2, None, None,
                                           swap_full, 'B1', es_on_fold=True)
    runs['LB 로짓만@stage2'] = lgbm_run(L_s2, y_s2, folds2, None, None, swap_logit, 'LB',
                                        es_on_fold=True)

    print('\n=== 전역 OOF ll (3-class) — "피처 전체의 한계 가치" ===')
    for k, r in runs.items():
        print(f'  {k:<22} {r[0]:.4f}')
    print(f'  증분(train)  : 로짓만 {runs["LA 로짓만@train"][0]:.4f} → +피처 '
          f'{runs["A1 로짓+피처@train"][0]:.4f}  (Δ {runs["LA 로짓만@train"][0] - runs["A1 로짓+피처@train"][0]:+.4f})')
    print(f'  증분(stage2) : 로짓만 {runs["LB 로짓만@stage2"][0]:.4f} → +피처 '
          f'{runs["B1 로짓+피처@stage2"][0]:.4f}  (Δ {runs["LB 로짓만@stage2"][0] - runs["B1 로짓+피처@stage2"][0]:+.4f})')

    # 피처별 gain 점유율 (피처끼리만 정규화) + 생존비
    def share(gains, offset):
        g = gains[offset:]
        return g / max(g.sum(), 1e-12) * 100
    rep = pd.DataFrame({'feature': ucols})
    rep['gainA0%'] = share(runs['A0 피처만@train'][1], 0)
    rep['gainA1%'] = share(runs['A1 로짓+피처@train'][1], 5)
    rep['gainB0%'] = share(runs['B0 피처만@stage2'][1], 0)
    rep['gainB1%'] = share(runs['B1 로짓+피처@stage2'][1], 5)
    rep['surv_train'] = rep['gainA1%'] / rep['gainA0%'].clip(lower=1e-6)
    rep['surv_s2'] = rep['gainB1%'] / rep['gainB0%'].clip(lower=1e-6)
    rep = rep.merge(catalog[['feature', '유형']], on='feature')
    for run_key in ('A1 로짓+피처@train', 'B1 로짓+피처@stage2'):
        total = runs[run_key][1].sum()
        print(f'  [{run_key}] 로짓 5열의 gain 점유율: {runs[run_key][1][:5].sum() / max(total, 1e-12) * 100:.1f}%')
    rep.sort_values('gainB1%', ascending=False).to_csv(OUT / 'select_report.csv', index=False)

    print('\n=== 유형별 집계 (판정은 그룹 단위 — 개별 순위는 노이즈) ===')
    g = rep.groupby('유형').agg(n=('feature', 'size'),
                                A0=('gainA0%', 'sum'), A1=('gainA1%', 'sum'),
                                B0=('gainB0%', 'sum'), B1=('gainB1%', 'sum'))
    g['surv_tr'] = g['A1'] / g['A0'].clip(lower=1e-9)
    g['surv_s2'] = g['B1'] / g['B0'].clip(lower=1e-9)
    print(g.sort_values('B1', ascending=False).round(2).to_string())
    print('\n  읽는 법: surv ≈ 1 = 로짓이 못 가리는 직교 신호 / ≈ 0 = 이미 흡수.')
    print('  A·B 불일치 그룹은 B(clean) 우선. 확정 채택/기각은 그룹-드랍 델타로 재확인 후')
    print('  feature_catalog.csv 사용 플래그 갱신 -> Train.ipynb 재학습 -> stage2 증분 사다리.')
    print(f'\nsaved: {OUT / "select_report.csv"}')


def group_delta(args):
    """그룹 단위 판정 — gain 점유율(정찰)이 아니라 '빼고/넣고 손실이 얼마나 움직이나'(판정).

      단독 증분  [로짓] -> [로짓 + 그룹 g]          그 그룹의 총 가치
      제거 델타  [로짓+전체] -> [로짓+전체 − g]      고유 기여 (다른 그룹이 대체 못 하는 몫)

    둘이 갈리는 지점이 정보다: 단독은 큰데 제거가 0 이면 '값은 있으나 중복' -> 빼도 된다.
    ES 는 'inner'(다음 fold 를 떼어 ES 전용) — select() 의 stage2 런(자기 fold ES)에 있던
    낙관을 없앴으므로 base/full 절대값이 B1/LB 와 조금 다르다. 그룹 간 비교는 동일 조건이라 유효.
    """
    M = load_matrices(verbose=False)
    catalog, ucols = M['catalog'], M['ucols']
    scopes = {'stage2': dict(F=M['F_s2'], L=M['L_s2'], y=M['y_s2'], folds=M['folds2'],
                             es=None, ey=None, es_mode='inner'),
              'train': dict(F=M['F_tr'], L=M['L_tr'], y=M['y_tr'], folds=M['folds'],
                            es=(M['L_va'], M['F_va']), ey=M['y_va'], es_mode=None)}
    todo = ['stage2', 'train'] if args.scope == 'both' else [args.scope]
    swap_logit, swap_full = M['swap_logit'], M['swap_full']

    gcol = catalog.set_index('feature')['유형']
    sw_series = catalog.set_index('feature')['swap']
    groups = {}                                     # 유형 -> 피처 열 인덱스(로짓 5열 오프셋 적용)
    for j, c in enumerate(ucols):
        groups.setdefault(gcol[c], []).append(5 + j)
    all_cols = list(range(5, 5 + len(ucols)))

    def subset_swap(cols):
        """열 부분집합 전용 swap — 전체 245열 기준 인덱스를 그대로 쓰면 범위를 벗어난다.
        교환쌍 9개는 전부 같은 유형 안에 닫혀 있어(사전 확인) 그룹 슬라이싱이 안전하다."""
        names = [ucols[c - 5] for c in cols if c >= 5]
        f_swap = build_swap(names, sw_series) if names else None

        def _sw(Xsub):
            Xs = Xsub.copy()
            Xs[:, 0], Xs[:, 1] = Xsub[:, 1].copy(), Xsub[:, 0].copy()   # abe z_a <-> z_b
            if f_swap is not None:
                Xs[:, 5:] = f_swap(Xsub[:, 5:])
            return Xs
        return _sw
    rows = []
    for scope in todo:
        S = scopes[scope]
        X = np.hstack([S['L'], S['F']])
        X_es = np.hstack(S['es']) if S['es'] is not None else None
        y, folds = S['y'], S['folds']

        def run(cols, desc):
            sub = np.ascontiguousarray(X[:, cols])
            es = np.ascontiguousarray(X_es[:, cols]) if X_es is not None else None
            return lgbm_run(sub, y, folds, es, S['ey'], subset_swap(cols), desc,
                            es_mode=S['es_mode'])

        print(f'\n[{scope}] 기준선 2런 + 그룹 {len(groups)}종 x 2방향 = {2 + 2 * len(groups)}런')
        t0 = time.time()
        base_ll, _, base_oof = run(list(range(5)), f'{scope}/base')
        full_ll, _, full_oof = run(list(range(5)) + all_cols, f'{scope}/full')
        e_base, e_full = per_row_ll(base_oof, y), per_row_ll(full_oof, y)
        d, lo, hi = paired_ci(e_base, e_full)
        print(f'  base(로짓만) {base_ll:.4f} | full(로짓+전체) {full_ll:.4f} | '
              f'전체 증분 {d:+.4f} [{lo:+.4f}, {hi:+.4f}]')

        for gname, gcols in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            add_ll, _, add_oof = run(list(range(5)) + gcols, f'{scope}/+{gname}')
            keep = [c for c in all_cols if c not in set(gcols)]
            drop_ll, _, drop_oof = run(list(range(5)) + keep, f'{scope}/-{gname}')
            a_d, a_lo, a_hi = paired_ci(e_base, per_row_ll(add_oof, y))        # 양수 = 그룹이 도움
            r_d, r_lo, r_hi = paired_ci(per_row_ll(drop_oof, y), e_full)       # 양수 = 빼면 손해
            uniq = '유의' if r_lo > 0 else ('역효과' if r_hi < 0 else '노이즈')
            solo = '유의' if a_lo > 0 else ('역효과' if a_hi < 0 else '노이즈')
            rows.append(dict(scope=scope, 유형=gname, n=len(gcols),
                             add_ll=add_ll, add_delta=a_d, add_lo=a_lo, add_hi=a_hi, add=solo,
                             drop_ll=drop_ll, drop_delta=r_d, drop_lo=r_lo, drop_hi=r_hi, drop=uniq))
            print(f'  {gname:<12} n={len(gcols):>3}  단독 {a_d:+.4f} [{a_lo:+.4f},{a_hi:+.4f}] {solo:<4}'
                  f'   고유 {r_d:+.4f} [{r_lo:+.4f},{r_hi:+.4f}] {uniq}', flush=True)
        print(f'  [{scope}] {(time.time() - t0) / 60:.0f}분')

    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / 'group_delta.csv', index=False)
    print('\n=== 판정표 (문턱 %0.4f, 고유 기여 내림차순) ===' % 0.0005)
    for scope in todo:
        sub = rep[rep['scope'] == scope].sort_values('drop_delta', ascending=False)
        print(f'\n[{scope}]')
        print(sub[['유형', 'n', 'add_delta', 'add', 'drop_delta', 'drop']]
              .to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
    print('\n  읽는 법: 고유(제거 델타) 유의 -> 유지 확정 / 단독 유의 + 고유 노이즈 -> 중복(제거 가능)')
    print('           둘 다 노이즈 -> 제거 후보. 동률이면 단순한 쪽(제거).')
    print(f'saved: {OUT / "group_delta.csv"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--select-only', action='store_true', help='추출 캐시로 선별만')
    ap.add_argument('--group-delta', action='store_true',
                    help='그룹 단위 판정(단독 증분 + 제거 델타 + bootstrap CI)')
    ap.add_argument('--scope', choices=['stage2', 'train', 'both'], default='stage2',
                    help='그룹 델타를 잴 데이터 (기본 stage2 = clean 로짓)')
    ap.add_argument('--tok-budget', type=int, default=8192, help='배치당 토큰 예산 (16GB 기본)')
    ap.add_argument('--anchor-n', type=int, default=1024, help='앵커 검증 valid 행 수')
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True)
    for p in (ABE_BEST / 'config.json', TB_BEST / 'config.json',
              ARITH / 'origin' / 'train.csv', ARITH / 'features_57k.parquet',
              ARITH / 'feature_catalog.csv', ARITH / 'arith_folds.csv'):
        assert Path(p).exists(), f'경로 없음: {p}'
    print(f'device {DEV}' + (f' | {torch.cuda.get_device_name(0)} | bf16' if DEV == 'cuda' else ''))
    if args.group_delta:                      # 판정 모드 — 추출 캐시만 쓰고 GPU 미사용
        group_delta(args)
        return
    if not args.select_only:
        for tag in ('abe', 'tb'):
            extract(tag, args)
    select(args)


if __name__ == '__main__':
    main()
