"""lab.py — 통합 헤드 반복 실험 키트. 노트북 코드(nbhead.py)로 층1(stage1 46,729행)→층2(stage2 교차적합)를 시드별로
학습하고, holdout 5,748행을 OOF(교차적합)·배포적합(in-sample) 두 기준으로 채점해 results/<arm>/ 에 남긴다.

  python lab.py check                      nbhead.py 원문 ↔ 노트북 셀 글자 대조 + 두 번들의 npz 이름
  python lab.py sync [--nb PATH]           노트북 헤드 셀이 바뀌면 nbhead.py 원문 부분을 다시 떠 온다
  python lab.py replay --bundle v1         번들의 outputs/heads/deploy.npz 를 복원해 holdout 채점 (노트북 §8 과 같은 값)
  python lab.py run --bundle v2 --arm v2 --seeds 42,1,2,3,4 [--also v1=v2@7748] [--set Layer1.K_TIE=1024]
  python lab.py compare v1 v2 [--seeds 42,1,2]

--bundle  v1 = $APFTE_V1_BUNDLE (ABE1-3109 · stage2 7,748) · v2 = ../bundle (ABE1-4380 · stage2 10,748) · 또는 번들 경로
--also    REF=ARM — 같은 층1 로 층2 를 REF 번들의 stage2 행에만(순서까지 같게) 한 번 더 적합해 ARM 으로 저장한다
--set     전역·클래스 속성 덮어쓰기, 예 Layer1.K_TIE=1024 · Layer1.WD=0.1 · Layer2.K=10 (summary.json 에 기록)
판정 (plan.md §5) Δ = 뒤 팔 − 앞 팔, 음수 = 개선. 채택 = Δ ≤ −0.0005 · 행 부트스트랩 95% CI 가 0 배제 · 시드 부호 전부.
"""
import sys
sys.dont_write_bytecode = True       # 번들 폴더에 __pycache__ 를 남기지 않는다 / keep the bundle free of __pycache__
import os, re, ast, gc, json, time, random, hashlib, argparse, platform
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import numpy as np
import pandas as pd
import torch
import nbhead as N

ROOT = LAB.parents[1]                                                # repository root
# nbhead.py holds the head cells of the v1 notebook (Kaggle: a-perfectly-fine-tuned-encoder) verbatim
NOTEBOOK = Path(os.environ.get('APFTE_V1_NOTEBOOK', ROOT / 'a-perfectly-fine-tuned-encoder.ipynb'))
BUNDLES = {'v1': Path(os.environ.get('APFTE_V1_BUNDLE', ROOT / 'bundle_v1')), 'v2': ROOT / 'bundle'}
RESULTS = LAB / 'results'
SENTINEL = '# ======== 아래는 노트북 셀 원문'
# 노트북 셀은 번호가 아니라 정의 서명으로 찾는다 — 노트북에 셀이 늘어도 그대로 동작한다 / located by signature, not index
CELLS = ('def build(stage):', 'class HeadLin(nn.Module):', 'class Layer1:', 'class Layer2:',
         'def assemble(z_tie, z_win):', 'def wiring_checks(L1, L2):')
MARK = re.compile(r'^# ---- notebook cell (\d+) · (.+?) · sha256 ([0-9a-f]{16}) ----$', re.M)
NBOOT, THR = 10_000, 5e-4
ENV = dict(python=platform.python_version(), torch=torch.__version__,
           device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')


# ────────── 노트북 원문 대조 / verbatim contract ──────────
def sha(text):
    return hashlib.sha256(text.rstrip('\n').encode('utf-8')).hexdigest()[:16]


def nb_cells(nb=NOTEBOOK):
    cells = json.loads(Path(nb).read_text(encoding='utf-8'))['cells']
    out = []
    for sig in CELLS:
        hit = [(i, ''.join(c['source'])) for i, c in enumerate(cells)
               if c['cell_type'] == 'code' and sig in ''.join(c['source'])]
        assert len(hit) == 1, f'{sig!r}: found in {len(hit)} code cells of {Path(nb).name}!!'
        out.append((hit[0][0], sig, hit[0][1]))
    return out


def blocks():
    text = (LAB / 'nbhead.py').read_text(encoding='utf-8')
    body = text[text.index(SENTINEL):]
    ms = list(MARK.finditer(body))
    return [(int(m[1]), m[2], m[3], body[m.end() + 1:(ms[i + 1].start() if i + 1 < len(ms) else len(body))])
            for i, m in enumerate(ms)]


def sync(nb=NOTEBOOK):
    p = LAB / 'nbhead.py'
    text = p.read_text(encoding='utf-8')
    head = text[:text.index('\n', text.index(SENTINEL)) + 1]
    parts = [head] + [f'\n\n# ---- notebook cell {i} · {sig} · sha256 {sha(src)} ----\n{src.rstrip(chr(10))}\n'
                      for i, sig, src in nb_cells(nb)]
    p.write_text(''.join(parts), encoding='utf-8')
    print(f' [sync] nbhead.py <- {Path(nb).name} cells {[i for i, _, _ in nb_cells(nb)]}')


def verbatim(nb=NOTEBOOK, show=False):
    cur = {sig: (i, src) for i, sig, src in nb_cells(nb)}
    bl = blocks()
    ok = [b[1] for b in bl] == list(CELLS)
    for i, sig, h, blk in bl:
        same, intact = sig in cur and sha(cur[sig][1]) == sha(blk), sha(blk) == h
        ok &= same and intact
        if show:
            print(f'  cell {i:>3} {sig:28s} {"= notebook" if same else "!= notebook (python lab.py sync)"}'
                  f'{"" if intact else " · nbhead.py was edited by hand"}')
    return bool(ok)


# ────────── 준비 / setup ──────────
def bundle_path(b):
    p = Path(BUNDLES.get(b, b)).resolve()
    assert (p / 'origin' / 'split_ids.csv').exists(), f'{p}: not a bundle (origin/split_ids.csv missing)!!'
    return p


def set_seed(sd):
    """노트북 셀 22 set_seeds + nbhead 전역 SEED (Layer1·Layer2 가 호출 시점에 읽는다)."""
    N.SEED = sd
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)


# stage1 은 시드마다 같은 파일이라 한 번만 읽는다 (Layer1._assets 가 모듈 전역 build 를 부르므로 여기서 감싼다)
_build, _cache = getattr(N, 'build', None), {}        # sync 전의 빈 nbhead 에는 build 가 없다


def _build_cached(stage):
    if stage != 'stage1':
        return _build(stage)
    if str(N.OUTPUT_DIR) not in _cache:
        _cache.clear()
        _cache[str(N.OUTPUT_DIR)] = _build(stage)
    return _cache[str(N.OUTPUT_DIR)]


if _build is not None:
    N.build = _build_cached


def apply_sets(sets):
    done = {}
    for s in sets:
        k, v = s.split('=', 1)
        obj, attr = (getattr(N, k.split('.')[0]), k.split('.')[1]) if '.' in k else (N, k)
        assert hasattr(obj, attr), f'--set {k}: no such attribute in nbhead!!'
        setattr(obj, attr, ast.literal_eval(v))
        done[k] = ast.literal_eval(v)
    return done


def take(S, idx):
    """stage2 행 부분집합 (idx 순서 그대로) / a row subset of a build('stage2') dict, in idx order."""
    return dict(ids=S['ids'][idx], y=S['y'][idx], nz=S['nz'][idx], hold=S['hold'][idx],
                tie_base=S['tie_base'][idx], win_base=S['win_base'][idx],
                pool={k: v[idx] for k, v in S['pool'].items()})


def rows_of(ref, S2):
    """REF 번들의 stage2(valid → holdout) id 순서대로 현재 S2 의 행 번호. 순서가 같아야 층2 fold 가 같다."""
    sp = pd.read_csv(bundle_path(ref) / 'origin' / 'split_ids.csv')
    hold_ids = sp.loc[sp['split'] == 'holdout', 'id'].to_numpy()
    want = np.concatenate([sp.loc[sp['split'] == 'valid', 'id'].to_numpy(), hold_ids])
    idx = pd.Index(S2['ids']).get_indexer(want)
    assert (idx >= 0).all(), f'{ref}: {int((idx < 0).sum())} stage2 rows are not in this bundle!!'
    assert (S2['ids'][idx][S2['hold'][idx]] == hold_ids).all() and S2['hold'][idx].sum() == len(hold_ids), \
        f'{ref}: holdout block disagrees!!'
    return idx


def scores(y, nz, zt, zw, hold):
    row, t, w = N.score(y, nz, zt, zw, hold)
    return dict(tot=float(row.mean()), tie=t, win=w)


# ────────── 실행 / run ──────────
def run(bundle, arm, seeds, also=(), sets=(), wiring=True):
    bp = bundle_path(bundle)
    N.use(bp)
    over = apply_sets(sets)
    S2 = N.build('stage2')
    views = [(arm, None, None)]
    for spec in also:
        ref, name = spec.split('=', 1)
        views.append((name, rows_of(ref, S2), str(bundle_path(ref))))
    base = dict(bundle=str(bp), npz=[N.NPZ[t]['logits_stage1'] for t in N.SPECS] + [N.ARITH_NPZ['logits_stage1']],
                set=over, K_TIE=N.Layer1.K_TIE, notebook_verbatim=verbatim())
    summ = {}
    for name, idx, ref in views:
        (RESULTS / name).mkdir(parents=True, exist_ok=True)
        cfg = json.loads(json.dumps(dict(base, stage2_rows=len(S2['y']) if idx is None else len(idx), rows_from=ref)))
        f = RESULTS / name / 'summary.json'
        old = json.loads(f.read_text(encoding='utf-8')) if f.exists() else None
        assert old is None or old['config'] == cfg, \
            f'results/{name} holds a run with another config — use a new --arm name or delete that folder!!'
        summ[name] = old or dict(config=cfg, seeds={})
    print(f' [run] arms {[v[0] for v in views]} · seeds {seeds} · set {over or "-"} · '
          f'notebook verbatim {base["notebook_verbatim"]} · {ENV["device"]}', flush=True)

    for sd in seeds:
        if all(str(sd) in summ[n]['seeds'] and (RESULTS / n / f's{sd}.npz').exists() for n, _, _ in views):
            print(f'[seed {sd}] already in results — skip', flush=True)
            continue
        t0 = time.time()
        set_seed(sd)
        L1 = N.Layer1.fit()
        g, w = L1.transform(S2['tie_base'], S2['win_base'], S2['pool'])
        t1 = time.time()
        for name, idx, _ in views:
            S = S2 if idx is None else take(S2, idx)
            gg, ww = (g, w) if idx is None else (g[idx], w[idx])
            L2 = N.Layer2.fit(S, gg, ww)
            if wiring:
                N.wiring_checks(L1, L2)
            zt_d, zw_d = L2.transform(S['tie_base'], gg, S['win_base'], ww)
            o = scores(S['y'], S['nz'], L2.z_tie, L2.z_win, S['hold'])
            d = scores(S['y'], S['nz'], zt_d, zw_d, S['hold'])
            np.savez(RESULTS / name / f's{sd}.npz', ids=S['ids'], y=S['y'], hold=S['hold'],
                     zt_oof=L2.z_tie, zw_oof=L2.z_win, zt_dep=zt_d, zw_dep=zw_d, g=gg, w=ww)
            summ[name]['seeds'][str(sd)] = dict(oof=o, dep=d, lam_tie=list(L2.dep_t['lam']), lam_win=list(L2.dep_w['lam']),
                                                picked_tie=L2.dep_t['picked'], picked_win=L2.dep_w['picked'],
                                                sec_layer1=round(t1 - t0, 1), sec=round(time.time() - t0, 1),
                                                at=time.strftime('%Y-%m-%d %H:%M:%S'), env=ENV)
            (RESULTS / name / 'summary.json').write_text(json.dumps(summ[name], indent=1), encoding='utf-8')
            print(f'[{name} seed {sd}] OOF {o["tot"]:.5f} (tie {o["tie"]:.5f} win {o["win"]:.5f}) | '
                  f'deploy-fit {d["tot"]:.5f} (tie {d["tie"]:.5f} win {d["win"]:.5f}) | '
                  f'λ tie {L2.dep_t["lam"]} win {L2.dep_w["lam"]} | {time.time() - t0:.0f}s', flush=True)
        del L1
        gc.collect()
        torch.cuda.empty_cache()

    for name, _, _ in views:
        sd_ = sorted(summ[name]['seeds'], key=int)
        o = [summ[name]['seeds'][s]['oof']['tot'] for s in sd_]
        d = [summ[name]['seeds'][s]['dep']['tot'] for s in sd_]
        print(f'[{name}] seeds {",".join(sd_)} · OOF mean {np.mean(o):.5f} (sd {np.std(o):.5f}) · '
              f'deploy-fit mean {np.mean(d):.5f} (sd {np.std(d):.5f})', flush=True)


# ────────── 재현 / replay a deploy bundle ──────────
def replay(bundle):
    bp = bundle_path(bundle)
    N.use(bp)
    p = N.OUTPUT_DIR / 'heads' / 'deploy.npz'
    if not p.exists():
        print(f' [replay] {p} does not exist — build it with a SUBMIT=False notebook run first')
        return None
    D0 = np.load(p)
    set_seed(int(D0['seed']))                          # Layer1.restore 가 seed == SEED 를 요구한다
    L1, L2 = N.Layer1.restore(D0), N.Layer2.restore(D0)
    N.wiring_checks(L1, L2)
    S2 = N.build('stage2')
    g, w = L1.transform(S2['tie_base'], S2['win_base'], S2['pool'])
    zt, zw = L2.transform(S2['tie_base'], g, S2['win_base'], w)
    r = scores(S2['y'], S2['nz'], zt, zw, S2['hold'])
    print(f' [replay] {p}\n          holdout {int(S2["hold"].sum()):,} rows — tie {r["tie"]:.5f} · win {r["win"]:.5f} · '
          f'total {r["tot"]:.5f} (deploy coefficients, in-sample) · λ tie {L2.dep_t["lam"]} win {L2.dep_w["lam"]}')
    return r


# ────────── 비교 / paired comparison on the holdout rows ──────────
def holdout_rows(Z, kind):
    """holdout 행별 손실 — N.score 와 같은 식(tot = tie 손실 + 승부 손실(비tie 행))."""
    h = Z['hold']
    y, zt, zw = Z['y'][h], Z[f'zt_{kind}'][h], Z[f'zw_{kind}'][h]
    m = y != 2
    lt = N.rowloss(zt, (y == 2).astype(np.float64))
    lw = np.full(len(y), np.nan)
    lw[m] = N.rowloss(zw[m], (y[m] == 0).astype(np.float64))
    return Z['ids'][h], y, dict(tot=lt + np.where(m, lw, 0.0), tie=lt, win=lw)


def boot(d, seed=42):
    d = d[~np.isnan(d)]
    rng, acc = np.random.default_rng(seed), np.empty(NBOOT)
    for s in range(0, NBOOT, 1000):
        k = min(1000, NBOOT - s)
        acc[s:s + k] = d[rng.integers(0, len(d), size=(k, len(d)))].mean(1)
    return float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))


def compare(a, b, seeds=None):
    assert a != b, 'compare needs two different arms!!'
    S = {x: json.loads((RESULTS / x / 'summary.json').read_text(encoding='utf-8')) for x in (a, b)}
    common = [int(s) for s in S[a]['seeds'] if s in S[b]['seeds']]
    seeds = [s for s in (seeds or common)]
    assert seeds and all(s in common for s in seeds), f'seeds {seeds} are not in both arms (common {common})!!'
    Z = {x: {s: dict(np.load(RESULTS / x / f's{s}.npz')) for s in seeds} for x in (a, b)}
    out = dict(a=a, b=b, seeds=seeds)
    print(f'\n=== {a} → {b} · seeds {",".join(map(str, seeds))} · Δ = {b} − {a} (음수 = {b} 가 나음) ===')
    for kind, label in (('oof', 'OOF (교차적합 — 판정 기준)'), ('dep', '배포적합 (in-sample, 노트북 §8 숫자)')):
        R = {x: {} for x in (a, b)}
        for x in (a, b):
            for s in seeds:
                ids, y, r = holdout_rows(Z[x][s], kind)
                R[x][s] = (ids, y, r)
        ids0, y0 = R[a][seeds[0]][:2]
        for x in (a, b):                                           # 행을 a 의 holdout 순서로 맞춘다
            for s in seeds:
                ids, y, r = R[x][s]
                ix = pd.Index(ids).get_indexer(ids0)
                assert (ix >= 0).all() and len(ids) == len(ids0) and (y[ix] == y0).all(), 'holdout rows disagree!!'
                R[x][s] = {k: v[ix] for k, v in r.items()}
        mean = lambda r, k: float(np.nanmean(r[k]))
        print(f'\n[{label}]  holdout {len(ids0):,} rows\n  {"seed":>6s} {a + " tot":>12s} {b + " tot":>12s}'
              f' {"Δtot":>9s} {"Δtie":>9s} {"Δwin":>9s}')
        per = []
        for s in seeds:
            dd = {k: mean(R[b][s], k) - mean(R[a][s], k) for k in ('tot', 'tie', 'win')}
            per.append(dd)
            print(f'  {s:>6d} {mean(R[a][s], "tot"):12.5f} {mean(R[b][s], "tot"):12.5f}'
                  f' {dd["tot"]:+9.5f} {dd["tie"]:+9.5f} {dd["win"]:+9.5f}')
        res = {}
        for k in ('tot', 'tie', 'win'):
            d = np.mean([R[b][s][k] for s in seeds], 0) - np.mean([R[a][s][k] for s in seeds], 0)
            m, (lo, hi) = float(np.nanmean(d)), boot(d)
            better = sum(p[k] < 0 for p in per)
            res[k] = dict(mean=m, lo=lo, hi=hi, better=better, n=len(seeds),
                          a=float(np.mean([mean(R[a][s], k) for s in seeds])),
                          b=float(np.mean([mean(R[b][s], k) for s in seeds])))
            print(f'  {"mean " + k:>10s}  {res[k]["a"]:.5f} → {res[k]["b"]:.5f}  Δ {m:+.5f} [{lo:+.5f},{hi:+.5f}]'
                  f'  CI {"0 배제" if lo > 0 or hi < 0 else "0 포함"} · 시드 개선 {better}/{len(seeds)}')
        t = res['tot']
        if t['mean'] <= -THR and t['hi'] < 0 and t['better'] == len(seeds):
            v = f'채택 기준 충족 — {b} 가 나음 (Δ ≤ −0.0005 · CI 0 배제 · 시드 {len(seeds)}/{len(seeds)})'
        elif t['mean'] >= THR and t['lo'] > 0 and t['better'] == 0:
            v = f'악화 확정 — {b} 가 나쁨 (Δ ≥ +0.0005 · CI 0 배제 · 시드 0/{len(seeds)})'
        else:
            v = f'판정 문턱 미달 (|Δ| {abs(t["mean"]):.5f} · CI [{t["lo"]:+.5f},{t["hi"]:+.5f}] · 시드 개선 {t["better"]}/{len(seeds)})'
        print(f'  판정: {v}')
        if len(seeds) > 1:                                         # 시드 앙상블 — z 를 시드 평균 (lever.md L1)
            E = {}
            for x in (a, b):
                ZZ = dict(Z[x][seeds[0]])
                ZZ[f'zt_{kind}'] = np.mean([Z[x][s][f'zt_{kind}'] for s in seeds], 0)
                ZZ[f'zw_{kind}'] = np.mean([Z[x][s][f'zw_{kind}'] for s in seeds], 0)
                ids, y, r = holdout_rows(ZZ, kind)
                E[x] = {k: v_[pd.Index(ids).get_indexer(ids0)] for k, v_ in r.items()}
            d = E[b]['tot'] - E[a]['tot']
            lo, hi = boot(d)
            res['ens'] = dict(a=float(E[a]['tot'].mean()), b=float(E[b]['tot'].mean()), mean=float(d.mean()), lo=lo, hi=hi)
            print(f'  시드 {len(seeds)}개 z 평균(앙상블): {res["ens"]["a"]:.5f} → {res["ens"]["b"]:.5f}'
                  f'  Δ {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}]')
        out[kind] = dict(per_seed=dict(zip(map(str, seeds), per)), **res, verdict=v)
    (RESULTS / '_compare').mkdir(parents=True, exist_ok=True)
    (RESULTS / '_compare' / f'{a}__{b}__s{"-".join(map(str, seeds))}.json').write_text(json.dumps(out, indent=1),
                                                                                       encoding='utf-8')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for c in ('check', 'sync'):
        sub.add_parser(c).add_argument('--nb', default=str(NOTEBOOK))
    sub.add_parser('replay').add_argument('--bundle', default='v2')
    p = sub.add_parser('run')
    p.add_argument('--bundle', required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--seeds', default='42,1,2')
    p.add_argument('--also', action='append', default=[])
    p.add_argument('--set', action='append', default=[], dest='sets')
    p.add_argument('--no-wiring', action='store_true', help='구조를 바꾼 실험 — 셀 114 의 파라미터 수 assert 생략')
    p = sub.add_parser('compare')
    p.add_argument('a')
    p.add_argument('b')
    p.add_argument('--seeds', default=None)
    a = ap.parse_args()
    print(f'# lab.py {" ".join(sys.argv[1:])} · {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    if a.cmd == 'sync':
        sync(a.nb)
    elif a.cmd == 'check':
        print(f' [check] nbhead.py vs {Path(a.nb).name}')
        ok = verbatim(a.nb, show=True)
        print(f' [check] notebook verbatim: {ok}')
        for b in ('v1', 'v2'):
            N.use(bundle_path(b))
    elif a.cmd == 'replay':
        replay(a.bundle)
    elif a.cmd == 'run':
        run(a.bundle, a.arm, [int(s) for s in a.seeds.split(',')], a.also, a.sets, wiring=not a.no_wiring)
    elif a.cmd == 'compare':
        compare(a.a, a.b, [int(s) for s in a.seeds.split(',')] if a.seeds else None)


if __name__ == '__main__':
    main()
