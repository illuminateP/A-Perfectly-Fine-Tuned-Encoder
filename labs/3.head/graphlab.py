"""텍스트 구조 그래프 프로브 — Q·A·B 를 이산 기호 그래프로 만들어 유사도를 재고,
그 유사도가 (1) 기존 헤드가 못 보는 정보를 갖는가(증분), (2) 단독 가치는 얼마인가를 잰다.

층 2 전용 프로브다 — 헤드 학습 없음. 텍스트 → 그래프 → 유사도 스칼라 → 결합기 열.

그래프 (텍스트마다 1개 × 알파벳 2종)
    토큰화       단어 / 수 / 기호 정규식. 수는 정수부 자릿수로 묶는다(NUM1, NUM2, …)
    word 알파벳  소문자 단어 + NUM빈 + 기호 전부      — 내용 겹침 (bertscore 와 경쟁)
    struct 알파벳 NUM빈 + 기호만 (단어 제거)           — 구조 골격 (기존 헤드가 못 볼 후보)
    엣지         인접 공기(窓 2) 무향 가중 · 노드 상위 96개 (빈도, 동률은 사전순 — 결정론)

유사도 4종 (클수록 유사 · spec/curv 는 −거리)
    jac    노드 집합 자카드            겹침 0차
    cos    엣지 가중 코사인            겹침 1차
    spec   정규화 라플라시안 상위 16 고유값 L2 거리    — "모양·거리"
    curv   Forman 곡률(4−dᵢ−dⱼ) 분포의 W1 거리        — "모양·곡률"

파리티 — 스왑(A↔B) 시 sim(Q,A)↔sim(Q,B), sim(A,B) 불변이므로
    V = sim(Q,A) − sim(Q,B)   홀 8열 → 승부축
    U = sim(Q,A) + sim(Q,B)   짝 8열 → tie축
    W = sim(A,B)              짝 8열 → tie축 ("답변끼리 닮으면 무승부" 가설)

판정 (전부 기존 crossfit 프로토콜 · holdout 판정)
    단독   [V8] / [U8|W8] 만으로 결합기 — 상수예측 ll 과 비교
    증분   승부 [m3|W4(신형)|V8] − [m3|W4] 시드 paired 5개 (winconf 저장 열)
          tie  [tie5|GAT|U8|W8] − [tie5|GAT] 행 부트스트랩
    겹침   각 열의 기존 축 점수와의 상관 — "이미 보고 있는 정보인가"

예산: 층 2 열당 분산 비용 ~0.0001–0.0002 실측(combinelab2) · 문턱 0.0005.
특징 추출은 --feats 캐시로 저장해 결합기 변형 재실행 시 재계산하지 않는다.
"""
import argparse
import json
import re
import time

import numpy as np
import pandas as pd

import tie_ladder as TL

FINAL = TL.FINAL
TOK = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|[^\sA-Za-z0-9]")
CAP_NODES = 96
SPEC_K = 16
CURV_Q = 32
MEASURES = ('jac', 'cos', 'spec', 'curv')
ALPHAS = ('word', 'struct')


# ══════════════════════════════════════════════ 텍스트 → 그래프
def parse_turns(s):
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return '\n'.join(x for x in v if isinstance(x, str))
    except (json.JSONDecodeError, TypeError):
        pass
    return s if isinstance(s, str) else ''


def tokens(text):
    out = []
    for t in TOK.findall(text):
        if t[0].isdigit():
            out.append(f'NUM{min(len(t.split(".")[0]), 6)}')
        elif t[0].isalpha():
            out.append(t.lower())
        else:
            out.append(t)
    return out


def build_graph(toks, alpha):
    """(노드가중 dict, 엣지가중 dict) — 무향 · 자기루프 제거 · 상위 CAP_NODES 노드."""
    if alpha == 'struct':
        toks = [t for t in toks if not t[0].isalpha() or t.startswith('NUM')]
    if not toks:
        return {}, {}
    nw = {}
    for t in toks:
        nw[t] = nw.get(t, 0) + 1
    keep = set(sorted(nw, key=lambda k: (-nw[k], k))[:CAP_NODES])
    nw = {k: v for k, v in nw.items() if k in keep}
    ew = {}
    for a, b in zip(toks[:-1], toks[1:]):
        if a == b or a not in keep or b not in keep:
            continue
        e = (a, b) if a < b else (b, a)
        ew[e] = ew.get(e, 0) + 1
    return nw, ew


def spec_vec(nw, ew):
    """정규화 라플라시안 상위 SPEC_K 고유값 (내림차순 · 0 채움)."""
    if len(ew) == 0:
        return np.zeros(SPEC_K)
    nodes = {k: i for i, k in enumerate(sorted(nw))}
    n = len(nodes)
    A = np.zeros((n, n))
    for (a, b), w in ew.items():
        A[nodes[a], nodes[b]] = A[nodes[b], nodes[a]] = w
    d = np.clip(A.sum(1), 1e-9, None) ** -0.5
    L = np.eye(n) - (A * d).T * d
    lam = np.sort(np.linalg.eigvalsh(L))[::-1][:SPEC_K]
    out = np.zeros(SPEC_K)
    out[:len(lam)] = lam
    return out


def curv_vec(ew):
    """Forman 곡률 4−dᵢ−dⱼ 의 CURV_Q 분위 벡터 (엣지 없으면 0)."""
    if len(ew) == 0:
        return np.zeros(CURV_Q)
    deg = {}
    for (a, b) in ew:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    f = np.array([4.0 - deg[a] - deg[b] for (a, b) in ew])
    return np.quantile(f, np.linspace(0, 1, CURV_Q))


def sims(g1, g2):
    """그래프 쌍 → 유사도 4종. 빈 그래프가 끼면 0 (정보 없음)."""
    (n1, e1, s1, c1), (n2, e2, s2, c2) = g1, g2
    out = {}
    k1, k2 = set(n1), set(n2)
    out['jac'] = len(k1 & k2) / max(len(k1 | k2), 1)
    if e1 and e2:
        keys = set(e1) | set(e2)
        v1 = np.array([e1.get(k, 0.0) for k in keys])
        v2 = np.array([e2.get(k, 0.0) for k in keys])
        out['cos'] = float(v1 @ v2 / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-12))
        out['spec'] = -float(np.linalg.norm(s1 - s2) / np.sqrt(SPEC_K))
        out['curv'] = -float(np.abs(c1 - c2).mean())
    else:
        out['cos'] = out['spec'] = out['curv'] = 0.0
    return out


def load_text(ids):
    """stage2 전 행의 텍스트. origin 의 작은 파일 우선, 부족분만 train_full 에서 채운다.

    `origin/valid.csv` 는 clean 2,000행(앵커)뿐이라 stage2 의 valid 5,000행을 못 덮는다.
    """
    cols = ['id', 'prompt', 'response_a', 'response_b']
    need = pd.Index(ids)
    parts = [pd.read_csv(TL.DATA_DIR / f, usecols=cols)
             for f in ('valid.csv', 'holdout.csv') if (TL.DATA_DIR / f).exists()]
    have = (pd.concat(parts).drop_duplicates('id') if parts
            else pd.DataFrame(columns=cols))
    miss = set(need.difference(pd.Index(have['id'])))
    if miss:
        print(f'    origin 소파일이 {len(miss):,}행 부족 → train_full.csv 에서 보충')
        add = [c[c['id'].isin(miss)] for c in
               pd.read_csv(TL.DATA_DIR / 'train_full.csv', usecols=cols,
                           chunksize=20000)]
        have = pd.concat([have] + add)
    df = have.drop_duplicates('id').set_index('id')
    gap = need.difference(df.index)
    assert len(gap) == 0, f'텍스트 없는 id {len(gap)}개 (예: {list(gap[:5])})'
    return df.loc[need]


def extract(ids):
    """stage2 전 행의 V·U·W 열 — id 정렬을 검증하며 뽑는다."""
    df = load_text(ids)
    assert len(df) == len(ids) and (df.index.values == ids).all(), 'id 정렬 실패'
    V = np.zeros((len(ids), len(ALPHAS) * len(MEASURES)))
    U = np.zeros_like(V)
    W = np.zeros_like(V)
    n_empty = np.zeros(len(ALPHAS), int)
    t0 = time.time()
    for r, (_, row) in enumerate(df.iterrows()):
        txt = {k: parse_turns(row[c]) for k, c in
               (('q', 'prompt'), ('a', 'response_a'), ('b', 'response_b'))}
        tk = {k: tokens(v) for k, v in txt.items()}
        for ai, alpha in enumerate(ALPHAS):
            G = {}
            for k in ('q', 'a', 'b'):
                nw, ew = build_graph(tk[k], alpha)
                G[k] = (nw, ew, spec_vec(nw, ew), curv_vec(ew))
            n_empty[ai] += (len(G['q'][0]) == 0)
            qa, qb, ab = sims(G['q'], G['a']), sims(G['q'], G['b']), sims(G['a'], G['b'])
            for mi, m in enumerate(MEASURES):
                j = ai * len(MEASURES) + mi
                V[r, j] = qa[m] - qb[m]
                U[r, j] = qa[m] + qb[m]
                W[r, j] = ab[m]
        if r % 2000 == 1999:
            print(f'    {r + 1:,}/{len(ids):,}행 [{time.time() - t0:.0f}s]')
    names = [f'{a}_{m}' for a in ALPHAS for m in MEASURES]
    print(f'  추출 완료 [{time.time() - t0:.0f}s] · Q 빈 그래프: '
          + ' '.join(f'{a} {n_empty[i]:,}행' for i, a in enumerate(ALPHAS)))
    return V, U, W, names


# ══════════════════════════════════════════════ 평가
def fit_axis(U_, X, y01, rows, bias, n_logit):
    z, dep = TL.combiner(X, y01, U_['fold'], rows, bias=bias, n_logit=n_logit)
    ll = TL.rowloss(z[rows], y01[rows])
    return z, ll[U_['hold'][rows]], dep


def const_ll(y_ev, p):
    z = np.log(p / (1 - p))
    return float(np.mean(np.logaddexp(0, z) - y_ev * z))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bundle', default='temp/heads_repro')
    ap.add_argument('--cols', default='temp/winconf.rowloss.npz')
    ap.add_argument('--tag', default='SK-RAW')
    ap.add_argument('--seeds', default='5,6,7,8,9')
    ap.add_argument('--rounds', type=int, default=1)
    ap.add_argument('--feats', default='temp/graphfeat.npz')
    ap.add_argument('--out', default='temp/graphlab')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s]

    print('[적재]')
    U_ = TL.build(FINAL / args.bundle)
    y = U_['y']

    fp = FINAL / args.feats
    if fp.exists():
        z = np.load(fp, allow_pickle=True)
        assert (z['ids'] == U_['ids']).all(), '캐시 id 불일치 — 캐시를 지우고 재실행'
        V, Ue, We, names = z['V'], z['U'], z['W'], list(z['names'])
        print(f'  특징 캐시 재사용: {fp} (V{V.shape[1]} U{Ue.shape[1]} W{We.shape[1]})')
    else:
        print('  [특징 추출 — 텍스트 → 그래프 → 유사도]')
        V, Ue, We, names = extract(U_['ids'])
        np.savez_compressed(fp, ids=U_['ids'], V=V, U=Ue, W=We, names=names)
        print(f'  저장: {fp}')
    for nm, A in (('V', V), ('U', Ue), ('W', We)):
        assert np.isfinite(A).all(), f'{nm} 비유한'

    rows_w = U_['cu'] & U_['nz']
    rows_t = U_['cu']
    y_w = (y == 0).astype(np.float64)
    y_t = (y == 2).astype(np.float64)
    ev_w = U_['hold'][rows_w]
    fit_w = rows_w & ~U_['hold']
    p_a = float(y_w[fit_w].mean())
    p_t = float(y_t[rows_t & ~U_['hold']].mean())

    Z = np.load(FINAL / args.cols)
    nc = args.rounds * 4
    res, roww = {}, {}

    def rec(tag, ll, note=''):
        res[tag] = float(ll.mean())
        roww[tag] = ll
        print(f'    [{tag:14s}] {ll.mean():.5f}{note}')

    # ── 승부축 ────────────────────────────────────────────
    print(f'\n{"="*78}\n[승부축] 판정 {int(ev_w.sum()):,}행 · 상수예측(비-tie 중 A '
          f'{p_a:.5f}) ll {const_ll(y_w[rows_w][ev_w], p_a):.5f}\n{"="*78}')
    _, ll, _ = fit_axis(U_, U_['win_base'], y_w, rows_w, False, 3)
    rec('logit', ll, '  ← 로짓만 [m3]')
    _, ll, d = fit_axis(U_, np.hstack([U_['win_base'], V]), y_w, rows_w, False, 3)
    rec('m3+V', ll, f'  ← 로짓과 독립인 몫 (λ {d["lam"]:g})')
    _, ll, d = fit_axis(U_, V, y_w, rows_w, False, 0)
    rec('V-alone', ll, f'  ← 단독 (λ {d["lam"]:g})')

    zb = {}
    for sd in seeds:
        key = f'cols__{args.tag}-s{sd}'
        Wc = Z[key][:, :nc].astype(np.float64)
        zbs, ll, _ = fit_axis(U_, np.hstack([U_['win_base'], Wc]), y_w, rows_w, False, 3)
        rec(f'BASE-s{sd}', ll)
        zb[sd] = zbs
        _, ll, d = fit_axis(U_, np.hstack([U_['win_base'], Wc, V]), y_w, rows_w,
                            False, 3)
        rec(f'BASE+V-s{sd}', ll, f'  (λ {d["lam"]:g})')

    # ── tie축 ────────────────────────────────────────────
    print(f'\n{"="*78}\n[tie축] 상수예측(tie율 {p_t:.5f}) ll '
          f'{const_ll(y_t[rows_t][U_["hold"][rows_t]], p_t):.5f}\n{"="*78}')
    UW = np.hstack([Ue, We])
    _, ll, _ = fit_axis(U_, U_['tie_base'], y_t, rows_t, True, 5)
    rec('t-logit', ll, '  ← 로짓만 [tie5]')
    _, ll, d = fit_axis(U_, np.hstack([U_['tie_base'], UW]), y_t, rows_t, True, 5)
    rec('t5+UW', ll, f'  ← 로짓과 독립인 몫 (λ {d["lam"]:g})')
    _, ll, d = fit_axis(U_, UW, y_t, rows_t, True, 0)
    rec('UW-alone', ll, f'  ← 단독 (λ {d["lam"]:g})')
    zt, ll, _ = fit_axis(U_, np.hstack([U_['tie_base'], U_['GAT']]), y_t, rows_t,
                         True, 5)
    rec('TBASE', ll, '  ← [tie5|GAT]')
    _, ll, d = fit_axis(U_, np.hstack([U_['tie_base'], U_['GAT'], UW]), y_t, rows_t,
                        True, 5)
    rec('TBASE+UW', ll, f'  (λ {d["lam"]:g})')

    # ── 겹침 진단 ─────────────────────────────────────────
    print(f'\n{"="*78}\n[겹침] 열별 — 기존 점수와의 상관(clean) · 라벨 상관\n{"="*78}')
    zw = np.mean([zb[s] for s in seeds], axis=0)
    cw, ct = rows_w & ~U_['hold'], rows_t & ~U_['hold']
    print(f'  {"열":13s}{"corr(z_win)":>12}{"corr(y_win)":>12}   '
          f'{"corr(z_tie)":>12}{"corr(y_tie)":>12}')
    for j, nm in enumerate(names):
        vw = np.corrcoef(V[cw, j], zw[cw])[0, 1]
        vy = np.corrcoef(V[cw, j], y_w[cw])[0, 1]
        uw = np.corrcoef(Ue[ct, j], zt[ct])[0, 1]
        uy = np.corrcoef(Ue[ct, j], y_t[ct])[0, 1]
        wy = np.corrcoef(We[ct, j], y_t[ct])[0, 1]
        print(f'  V/U {nm:9s}{vw:+12.3f}{vy:+12.3f}   {uw:+12.3f}{uy:+12.3f}'
              f'   (W corr(y_tie) {wy:+.3f})')

    op = FINAL / args.out
    with open(op.with_suffix('.json'), 'w') as f:
        json.dump({'cells': res, 'names': names, 'seeds': seeds}, f,
                  ensure_ascii=False, indent=1)
    np.savez_compressed(op.with_suffix('.rowloss.npz'), **roww)
    print(f'\n저장: {op}.json / {op}.rowloss.npz')

    # ── 요약 ─────────────────────────────────────────────
    rw = lambda t: np.mean([roww[f'{t}-s{s}'] for s in seeds], axis=0)
    print(f'\n{"="*78}\n[요약]\n{"="*78}')
    print(f'  단독   V-alone {res["V-alone"]:.5f} (상수 '
          f'{const_ll(y_w[rows_w][ev_w], p_a):.5f}) · UW-alone {res["UW-alone"]:.5f} '
          f'(상수 {const_ll(y_t[rows_t][U_["hold"][rows_t]], p_t):.5f})')
    m, lo, hi = TL.paired_ci(roww['m3+V'], roww['logit'])
    print(f'  로짓 증분   m3+V − logit    {TL.ci_str(m, lo, hi)}')
    m, lo, hi = TL.paired_ci(rw('BASE+V'), rw('BASE'))
    d = np.array([res[f'BASE+V-s{s}'] - res[f'BASE-s{s}'] for s in seeds])
    print(f'  승부 증분   BASE+V − BASE   {TL.ci_str(m, lo, hi)}  '
          f'시드 {d.mean():+.5f} {int((d < 0).sum())}/{len(d)}')
    m, lo, hi = TL.paired_ci(roww['t5+UW'], roww['t-logit'])
    print(f'  tie 로짓 증분 t5+UW − t5    {TL.ci_str(m, lo, hi)}')
    m, lo, hi = TL.paired_ci(roww['TBASE+UW'], roww['TBASE'])
    print(f'  tie 증분    TBASE+UW−TBASE  {TL.ci_str(m, lo, hi)}')
    print('\n  ! 판정: 증분 −0.0003 이상 & (승부는 시드 4/5) — 열 8~16개 분의 분산 비용을'
          ' 이겨야 한다')
    print('  ! 단독이 상수예측을 크게 못 이기면 표현 자체가 빈약한 것 — 증분 해석 이전에'
          ' 죽는다')


if __name__ == '__main__':
    main()
