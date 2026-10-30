"""nlfeat.py — 비선형 판독이 선형 판독 **위에** 더 얹어지는가 (앵커 유지 · 학습 없음).

왜 이걸 먼저 재는가
  풀링에서 정보를 회수하는 통로 중 지금까지 흑자인 것은 **선형 릿지 열** 하나뿐이다
  (repr5: tie +0.0019★ tb_u / 승부 +0.0018★ a1_v). 반대로 풀링 위에 헤드를 새로 학습한
  시도는 전부 적자였다 (mlphead −0.05~−0.065★, newhead 12팔 −0.031~−0.093★).
  두 결과 사이에는 "비선형 자체가 없나" 와 "비선형은 있는데 학습 분산이 그걸 삼켰나" 가
  구분되지 않은 채 남아 있다. 이 파일은 **학습을 하지 않는 비선형**으로 그 둘을 가른다:
  무작위 특징은 적합할 파라미터가 없고(사영 행렬은 라벨·데이터와 무관한 난수), 결합은
  기존 릿지 그대로다. 즉 신경 헤드의 분산세를 내지 않고 비선형 신호의 유무만 본다.

  · 여기서 이득이 나온다 -> 비선형 신호는 있다. 신경 헤드의 실패는 분산·암기 문제였고,
    앵커드-잔차 헤드로 다시 갈 근거가 된다.
  · 안 나온다          -> 이 대역폭 범위에서 풀링의 비선형은 선형 판독 위에 없다.
    (모든 비선형을 닫는 건 아니다 — 대역폭 격자 밖은 미측정이다. 아래 주석 참고.)

패리티 보존 무작위 푸리에 피쳐 — 근사가 아니라 항등식
  φ(u,v) = cos(a + s),  a = 짝 사영(u만) + 위상,  s = 홀 사영(v만)
      짝 채널  ½[φ(u,v) + φ(u,−v)] =  cos(a)·cos(s)
      홀 채널  ½[φ(u,v) − φ(u,−v)] = −sin(a)·sin(s)
  swap 은 v -> −v 이므로 s -> −s 이고, cos(s) 는 그대로 / sin(s) 는 부호가 뒤집힌다.
  따라서 짝 채널은 tie축에, 홀 채널은 승부축에 **구조적으로** 적법하다 (EquiHead 의
  대칭화/반대칭화를 커널 근사에 그대로 옮긴 것이다).

  홀 채널의 의미: sin(a)·sin(s) 는 "짝(국면)이 홀(승부 방향)을 변조" 하는 항이다.
  FWLS 게이트가 산술피쳐 ⊙ 로짓이었다면 이건 풀링짝 ⊙ 풀링홀 — 게이트 가족이 훨씬 넓다.
  그래서 이 팔은 '비선형 판독' 과 '표현 게이트' 두 질문을 동시에 친다.

대역폭
  사영 후 스케일을 1 로 맞춘 뒤 c 를 곱한다. c 가 작으면 거의 선형, 크면 고주파다.
  c 를 팔로 쪼개면 다중비교가 늘어나므로 **한 팔 안에 여러 c 를 섞어 넣고** 릿지가 고르게 한다
  (RFF 관례). 기본 c = 1, 3.

규율
  · 우주·fold·기저는 repr5 와 동일 (ABE1-clean 7,748, 5-fold, 앙상블 계약 5-전문가).
  · 라벨을 보는 적합은 fold 안 릿지뿐이다. 사영 행렬은 난수, 정규화 스칼라는 라벨 비접촉.
  · 로짓 열 pen=0 -> λ->∞ 면 기준 팔로 정확히 수렴한다. 구조적으로 질 수 없는 베팅이다.
  · 판정 비교는 4개 (축별 N1/N2 vs 선형 L) -> Bonferroni 는 x4 로 읽을 것.

사용 (약 40~60분)
  python Boost/nlfeat.py 2>&1 | tee Boost/log_nlfeat.txt
  python Boost/nlfeat.py --only N1          # 부분 (기준 N0·선형 L 은 자동 포함)
  python Boost/nlfeat.py --scales 0.3,1,3 --D 768
"""
import argparse
import sys
import time

import numpy as np

import fwls as FW
import reprhead as RH
import repr5 as R5

sys.stdout.reconfigure(line_buffering=True)

SEED, K = RH.SEED, RH.K

# (이름, 짝 블록 키, 홀 블록 키) — 풀링 파일이 없으면 자동으로 빠진다
SRC = [('abe', 'abe_u', 'abe_v'), ('tb3', 'tb_u', 'tb_v'),
       ('a1', 'a1_u', 'a1_v'), ('tb1', 'tb1_u', 'tb1_v')]


def prep(U, V, rows):
    """짝은 중심화+표준화, 홀은 RMS 로만 나눈다.

    홀을 중심화하면 상수항이 생겨 반대칭이 깨진다 (reprhead._z 와 같은 규율).
    통계는 **clean 행에서만** 낸다 — ABE1 오염 행은 NaN 이라 전 행 평균이 오염된다."""
    mu = U[rows].mean(0)
    sd = np.clip(U[rows].std(0), 1e-9, None)
    rms = np.clip(np.sqrt((V[rows] ** 2).mean(0)), 1e-9, None)
    return (U - mu) / sd, V / rms


def rff(Uz, Vz, rows, D, scales, seed):
    """패리티 보존 무작위 특징. 반환 (짝 행렬, 홀 행렬) 각 (n, D).

    사영 후 스케일 정규화는 clean 행 통계로 한다 (라벨 비접촉 스칼라).
    짝 사영에는 위상 b 를 더하고, 홀 사영에는 **더하지 않는다** — 위상을 더하면
    sin(s + b) 가 홀함수가 아니게 되어 승부축 반대칭이 깨진다."""
    rng = np.random.default_rng(seed)
    du, dv = Uz.shape[1], Vz.shape[1]
    per = max(int(D) // len(scales), 1)
    Ev, Od = [], []
    for c in scales:
        Wu = rng.standard_normal((du, per)) / np.sqrt(du)
        Wv = rng.standard_normal((dv, per)) / np.sqrt(dv)
        a = Uz @ Wu
        s = Vz @ Wv
        a = c * a / np.clip(a[rows].std(0), 1e-9, None) + rng.uniform(0, 2 * np.pi, per)
        s = c * s / np.clip(np.sqrt((s[rows] ** 2).mean(0)), 1e-9, None)
        Ev.append(np.cos(a) * np.cos(s))
        Od.append(np.sin(a) * np.sin(s))
    return np.hstack(Ev), np.hstack(Od)


def parity_check(Uz, Vz, rows, D, scales, seed, tol=1e-9):
    """v -> −v 를 실제로 넣어 짝/홀이 성립하는지 검산한다. 여기가 깨지면 뒤가 전부 무의미하다."""
    E1, O1 = rff(Uz, Vz, rows, D, scales, seed)
    E2, O2 = rff(Uz, -Vz, rows, D, scales, seed)
    de = float(np.abs(E1[rows] - E2[rows]).max())
    do = float(np.abs(O1[rows] + O2[rows]).max())
    assert de < tol and do < tol, f'패리티 위반 짝 {de:.2e} 홀 {do:.2e}'
    return de, do


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--D', type=int, default=512, help='무작위 특징 수 (대역폭들에 균등 분배)')
    ap.add_argument('--scales', default='1,3', help='대역폭 c 목록 — 한 팔 안에 섞어 넣는다')
    ap.add_argument('--only', default='', help='팔 코드 부분일치 필터 (N0·L 은 자동 포함)')
    ap.add_argument('--seed', type=int, default=SEED)
    # 선형 대조군 블록 — 기본값은 repr5 에서 유의했던 블록(사전등록). R5(TB1)가 더 크게
    # 이기면 --lin-tie tb1_u 처럼 갈아끼워 같은 질문을 그 위에서 다시 물을 수 있다.
    ap.add_argument('--lin-tie', default='tb_u', help='tie축 선형 대조 블록 키')
    ap.add_argument('--lin-win', default='a1_v', help='승부축 선형 대조 블록 키')
    args = ap.parse_args()
    only = [s.strip() for s in args.only.split(',') if s.strip()]
    scales = [float(s) for s in args.scales.split(',') if s.strip()]
    t0 = time.time()

    A = R5.load_repr()
    y, nz, hold, cu, fold = A['y'], A['nz'], A['hold'], A['cu'], A['fold']
    n = len(y)

    # ── 소스별 전처리 (없는 블록은 뺀다) ──
    src = []
    for nm, ku, kv in SRC:
        if A[ku] is None or A[kv] is None:
            print(f'  [주의] {nm} 풀링 없음 — 소스에서 뺀다')
            continue
        Uz, Vz = prep(A[ku], A[kv], cu)
        assert np.isfinite(Uz[cu]).all() and np.isfinite(Vz[cu]).all(), f'{nm} 비유한값'
        src.append((nm, Uz, Vz))
    assert src, '풀링 소스가 하나도 없다'
    names = [s[0] for s in src]

    de, do = parity_check(src[0][1], src[0][2], cu, args.D, scales, args.seed)
    print(f'  [검산] 패리티 (v -> −v): 짝 max|Δ| {de:.2e} | 홀 max|합| {do:.2e}  OK')

    # 단일 소스 RFF — 축별로 repr5 에서 유의했던 블록의 소스를 쓴다
    #   tie축 tb_u(TB3) +0.0019★ / 승부축 a1_v(ABE1) +0.0018★
    ONE = {'tie': 'tb3', 'win': 'a1'}
    assert set(ONE.values()) <= set(names), f'단일 소스 팔에 필요한 풀링이 없다: {ONE} vs {names}'
    feats = {}
    for nm, Uz, Vz in src:
        if nm in ONE.values():                       # 팔에 쓰이는 소스만 만든다 (메모리)
            feats[nm] = rff(Uz, Vz, cu, args.D, scales, args.seed)
    # 전 소스 — 사영 안에서 전문가끼리 섞인다 (교차 압축을 학습 없이 보는 유일한 경로)
    Uall = np.hstack([s[1] for s in src])
    Vall = np.hstack([s[2] for s in src])
    d_all = (Uall.shape[1], Vall.shape[1])
    feats['all'] = rff(Uall, Vall, cu, args.D, scales, args.seed)
    del Uall, Vall

    print(f'소스 {len(src)}개 {names} | 무작위 특징 {args.D}개 x 대역폭 {scales} | '
          f'단일 소스: tie {ONE["tie"]} / 승부 {ONE["win"]}')
    print(f'전 소스 사영 입력 차원 짝 {d_all[0]:,} + 홀 {d_all[1]:,}\n')

    def linblock(key):
        return [(key, A[key])]

    lt, lw = args.lin_tie, args.lin_win
    assert A.get(lt) is not None and A.get(lw) is not None, f'선형 대조 블록 없음: {lt}/{lw}'
    AXES = [
        ('tie축', (y == 2).astype(np.float64), cu, True, A['T5'], 0, [
            ('T-N0', '5로짓만 — 기준', []),
            ('T-L', f'+{lt} (선형 · repr5 재현)', linblock(lt)),
            ('T-N1', f'+{lt} + 비선형({ONE["tie"]} 소스)',
             linblock(lt) + [(f'rff_{ONE["tie"]}', feats[ONE['tie']][0])]),
            ('T-N2', f'+{lt} + 비선형(전 소스 — 교차)',
             linblock(lt) + [('rff_all', feats['all'][0])]),
        ]),
        ('승부축', (y == 0).astype(np.float64), cu & nz, False, A['M3'], 1, [
            ('W-N0', '3로짓만 — 기준', []),
            ('W-L', f'+{lw} (선형 · repr5 재현)', linblock(lw)),
            ('W-N1', f'+{lw} + 비선형({ONE["win"]} 소스)',
             linblock(lw) + [(f'rff_{ONE["win"]}', feats[ONE['win']][1])]),
            ('W-N2', f'+{lw} + 비선형(전 소스 — 교차)',
             linblock(lw) + [('rff_all', feats['all'][1])]),
        ]),
    ]

    def blocks_of(parts):
        cols, blk, off = [], [], 0
        for nm, B in parts:
            cols.append(B)
            blk.append((nm, off, off + B.shape[1]))
            off += B.shape[1]
        return np.hstack(cols), blk

    results, store = {}, {}
    for ax_nm, y01, rows_m, bias, Lb, hno, arms in AXES:
        RH.head(str(hno + 1), f'{ax_nm} · clean {int(rows_m.sum()):,}행 — 로짓 pen=0, 표현·비선형만 릿지')
        if only:
            arms = [a for a in arms if a[0][2:] in ('N0', 'L') or any(o in a[0] for o in only)]
        base_ll, lin_ll = None, None
        hm = hold[rows_m]
        for code, label, parts in arms:
            Xi, blk = blocks_of(parts) if parts else (np.zeros((n, 0)), [])
            print(f'  [{code}] {label}  ({Lb.shape[1] + Xi.shape[1]}열) 적합 중 ...')
            r = FW.run_arm((code, label, Lb, Xi, blk, None), y01, fold, rows_m, hold,
                           base_ll, bias=bias)
            if base_ll is None:
                base_ll = r['ll']
            FW.arm_line(r, t0)
            # 이 파일의 판정 비교는 **선형 대조군 대비**다 — 기준 대비 Δ 는 위 줄에 이미 있다
            if code.endswith('-L'):
                lin_ll = r['ll']
            elif lin_ll is not None:
                print(f"        vs 선형 L  전체 {RH.ci_str(*RH.paired_ci(lin_ll, r['ll']))}")
                print(f"        vs 선형 L  holdout {RH.ci_str(*RH.paired_ci(lin_ll[hm], r['ll'][hm]))}")
            results[code] = r
            store[code] = r['z']
        print()

    RH.head('3', '종합 — 총 ll = tie축 + (1−tie율)·승부축 (clean 우주)')
    nzf = float(nz[cu].mean())
    nzf_h = float(nz[hold].mean())
    hw, ht = hold[cu & nz], hold[cu]
    have = lambda c: c in results
    rows_out = []
    for nm, wc, tc in [('기준 (로짓만)', 'W-N0', 'T-N0'),
                       ('선형 표현 (repr5 재현)', 'W-L', 'T-L'),
                       ('+비선형 단일 소스', 'W-N1', 'T-N1'),
                       ('+비선형 전 소스', 'W-N2', 'T-N2')]:
        if not (have(wc) and have(tc)):
            continue
        llw, llt = results[wc]['ll'], results[tc]['ll']       # lt/lw(대조 블록 키)와 이름 충돌 주의
        rows_out.append([nm, f'{float(llt.mean() + nzf * llw.mean()):.4f}',
                         f'{float(llt[ht].mean() + nzf_h * llw[hw].mean()):.4f}'])
    RH.table(['구성', '총 ll (clean)', 'holdout'], rows_out, ['l', 'r', 'r'])
    print('  · 비선형 팔이 선형 팔보다 위(값이 큼)면 이 대역폭에서 비선형 신호가 없다는 뜻이다.')
    print('  · 아래면 크기와 CI 를 본다 — 판정 비교 4개이므로 Bonferroni x4.')
    print('  · 미측정으로 남는 것: 대역폭 격자 밖, 그리고 학습된(적응적) 사영.')
    print('    이 파일이 닫는 것은 "무작위 사영 + 이 대역폭" 범위뿐이다. 판정은 사용자.')

    np.savez_compressed(FW.BOOST / 'nlfeat_oof.npz', ids=A['ids'], y=y,
                        **{f'z__{c}': v for c, v in store.items()})
    print(f'\nsaved: {FW.BOOST / "nlfeat_oof.npz"}')


if __name__ == '__main__':
    main()
