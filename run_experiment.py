from __future__ import annotations

"""
实验入口：参数化运行单国或两国博弈并导出结果为 CSV。

用法示例：
- 单国：python run_experiment.py --mode single --rounds 50 --k 5 --country H --out single.csv
- 两国同时：python run_experiment.py --mode bi-sim --rounds 50 --k 5 --out bi_sim.csv
- 两国轮流：python run_experiment.py --mode bi-alt --rounds 50 --k 5 --out bi_alt.csv
"""

import argparse
import csv
from typing import Any, Dict
import sys
import time
import numpy as np

from eco_simu import simulate, SimulationConfig, create_symmetric_parameters, bootstrap_simulator
from eco_simu.plotting import plot_history, plot_sector_paths, plot_diagnostics, plot_history_agent_view
from agent_loop import (
    MultiCountryLoopState,
    run_multilateral_with_graph,
    build_policy_from_spec,
    parse_reward_weights,
)


def _export_single(state: LoopState, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "reward", "income_growth", "price_mean", "trade_balance"])
        for item in state.log:
            m = item["obs"]["metrics"]
            w.writerow([
                item.get("t"),
                item.get("reward"),
                m.get("income_growth_last"),
                m.get("price_mean_last"),
                m.get("trade_balance_last"),
            ])


def _export_bi(state: TwoCountryLoopState, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t",
            "reward_H",
            "reward_F",
            "income_growth_H",
            "income_growth_F",
            "price_mean_H",
            "price_mean_F",
            "trade_balance_H",
            "trade_balance_F",
        ])
        for item in state.log:
            mH = item["obs"]["H"]["metrics"]
            mF = item["obs"]["F"]["metrics"]
            r = item["reward"]
            w.writerow([
                item.get("t"),
                r.get("H"), r.get("F"),
                mH.get("income_growth_last"), mF.get("income_growth_last"),
                mH.get("price_mean_last"), mF.get("price_mean_last"),
                mH.get("trade_balance_last"), mF.get("trade_balance_last"),
            ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["multi-sim-graph", "multi-alt-graph"], default="multi-sim-graph")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--k", type=int, default=5, help="steps per round")
    ap.add_argument("--countries", type=str, default="H,F", help="comma-separated country codes for multi")
    ap.add_argument("--out", type=str, default="experiment.csv")
    ap.add_argument("--early", action="store_true", help="enable early stop")
    ap.add_argument("--policy", type=str, required=True, help="policy spec or country:spec list for multi (production: use llm:... only)")
    ap.add_argument("--rw", type=str, default=None, help="reward weights like w_income=1.0,w_price=0.2,w_trade=0.001; multi: country:{..}")
    ap.add_argument("--obs-full", action="store_true", help="include full vectors in observations (default on)")
    ap.add_argument("--obs-topk", type=int, default=5, help="top-K sector deltas to include in prompt")
    ap.add_argument("--max-sectors-per-type", type=int, default=None, help="per action type sector limit (None=no limit)")
    ap.add_argument("--recursion-limit", type=int, default=None)
    # warmup controls
    ap.add_argument("--warmup-window", type=int, default=10, help="window size for stability check during warmup")
    ap.add_argument("--warmup-eps", type=float, default=0.01, help="tolerance for stability during warmup (checks adjacent-period changes in percentage points)")
    ap.add_argument("--warmup-max", type=int, default=1000, help="max steps during warmup before giving up")
    ap.add_argument("--require-stable-warmup", action="store_true", help="abort if warmup does not reach stability within max steps")
    ap.add_argument("--fixed-import-multiplier", type=float, default=None, help="if set, fix import multiplier for all sectors to this value and ignore agent's import_multiplier")
    
    # plotting controls
    ap.add_argument("--plot", action="store_true", help="generate summary and sector plots using eco_simu.plotting")
    ap.add_argument("--plot-dir", type=str, default="results", help="directory to save plot images")
    ap.add_argument("--plot-sectors", type=str, default="0,1,2", help="comma-separated sector indices for sector plots")
    args = ap.parse_args()

    # 启动智能体博弈需从“未展开”的仿真器开始，便于精确控制 warmup 步数
    # 最小可行变更：改用 bootstrap_simulator（仅求初始均衡，不推进动态步）
    cfg = SimulationConfig(total_periods=0, conflict_start=200)
    sim = bootstrap_simulator(params_raw=create_symmetric_parameters(), theta_price=cfg.theta_price)
    n_sec = getattr(sim.params.home.alpha, "shape", [5])[0]  # type: ignore[attr-defined]

    # ---------- Warmup until stable (no agent actions) ----------
    def _stable_window(arr: np.ndarray, w: int, eps: float) -> bool:
        if arr is None:
            return False
        if arr.shape[0] < w + 1:  # 需要 w+1 个点才能计算 w 个差分
            return False
        # 检查相邻期变化而非绝对值：判断是否停稳
        diffs = np.diff(arr[-w-1:])  # 最后 w+1 个点产生 w 个差分
        return bool(np.all(np.abs(diffs) < eps))

    def warmup_until_stable(sim_obj: Any, w: int, eps: float, max_steps: int) -> Dict[str, Any]:
        steps = 0
        stable = False
        t0 = time.time()
        while steps < max_steps:
            # need at least w points to check
            summary = sim_obj.summarize_history()
            ok_all = True
            for c in ["H", "F"]:
                m = summary.get(c, {})
                ig = np.array(m.get("income_growth", []), float)
                og = np.array(m.get("output_growth", []), float)
                if not (_stable_window(ig, w, eps) and _stable_window(og, w, eps)):
                    ok_all = False
                    break
            if ok_all:
                stable = True
                break
            sim_obj.step()
            steps += 1
        dt = time.time() - t0
        return {"steps": steps, "stable": stable, "seconds": dt}

    warmup_steps = 0
    if args.warmup_window and args.warmup_max:
        info = warmup_until_stable(sim, int(args.warmup_window), float(args.warmup_eps), int(args.warmup_max))
        warmup_steps = int(info.get("steps", 0) or 0)
        print(f"[warmup] steps={info['steps']} stable={info['stable']} time={info['seconds']:.2f}s")
        if not info["stable"]:
            if bool(args.require_stable_warmup):
                print("[warmup] ERROR: did not reach stability within warmup_max; aborting.")
                sys.exit(2)
            else:
                print("[warmup] WARNING: did not reach stability within warmup_max; proceeding anyway.")

    countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    # --policy 可写成 country:spec;country:spec 或单一 spec 应用于所有国家
    def _is_country_mapping(spec: str, countries: list[str]) -> bool:
        parts = [p for p in spec.split(",") if p.strip()]
        if not parts:
            return False
        hits = 0
        total = 0
        for part in parts:
            if ":" not in part:
                return False
            total += 1
            left = part.split(":", 1)[0].strip().upper()
            if left in countries:
                hits += 1
        return hits > 0 and hits == total

    pol: Dict[str, Any] = {}
    if _is_country_mapping(args.policy, countries):
        for part in args.policy.split(","):
            c, sp = part.split(":", 1)
            pol[c.strip().upper()] = build_policy_from_spec(sp.strip(), int(n_sec))
    else:
        pol = {c: build_policy_from_spec(args.policy, int(n_sec)) for c in countries}

    rw_common = parse_reward_weights(args.rw)
    rw = {c: rw_common for c in countries}
    state = MultiCountryLoopState(
        t=0,
        sim=sim,
        countries=countries,
        k_per_step=args.k,
        max_rounds=args.rounds,
        policy=pol,
        reward_weights=rw,
        use_early_stop=bool(args.early),
        obs_full=True if not args.obs_full else True,
        obs_topk=int(args.obs_topk),
        max_sectors_per_type=(int(args.max_sectors_per_type) if args.max_sectors_per_type is not None else None),
        recursion_limit=(int(args.recursion_limit) if args.recursion_limit else None),
        fixed_import_multiplier=(float(args.fixed_import_multiplier) if args.fixed_import_multiplier is not None else None),
    )
    mode = "simultaneous" if ("sim" in args.mode) else "alternating"
    state = run_multilateral_with_graph(state, mode=mode, order=countries)

    # 统一 CSV：动态列头
    with open(args.out, "w", newline="") as f:
        import csv
        w = csv.writer(f)
        header = ["t"]
        for c in countries:
            header += [f"reward_{c}", f"income_growth_{c}", f"price_mean_{c}", f"trade_balance_{c}"]
        w.writerow(header)
        # 兼容 LangGraph 返回 dict 或对象的情况（非兜底，仅消费接口）
        log_iter = None
        if isinstance(state, dict):
            log_iter = state.get("log", [])
        else:
            log_iter = getattr(state, "log", [])
        for item in log_iter:
            row = [item.get("t")]
            for c in countries:
                m = item["obs"][c]["metrics"]
                r = item["reward"].get(c)
                row += [r, m.get("income_growth_last"), m.get("price_mean_last"), m.get("trade_balance_last")]
            w.writerow(row)

    print(f"done -> {args.out}")

    # Optional plotting
    if args.plot:
        try:
            tag = (args.out.rsplit("/", 1)[-1].rsplit(".", 1)[0] if args.out else "experiment")
            # Summary history
            plot_history(sim, save_path=f"{args.plot_dir}/{tag}_summary.png", show=False, warmup=warmup_steps)
            plot_diagnostics(sim, save_path=f"{args.plot_dir}/{tag}_diagnostics.png", show=False, warmup=warmup_steps)
            # Agent-view summary (trim warmup + annotate decisions)
            # 重用上面的 log_iter（已为 list），若不是 list，这里安全转为 list
            try:
                agent_log = list(log_iter) if not isinstance(log_iter, list) else log_iter
            except Exception:
                agent_log = []
            plot_history_agent_view(
                sim,
                agent_log=agent_log,
                k_per_step=int(getattr(state, "k_per_step", args.k)),
                save_path=f"{args.plot_dir}/{tag}_summary_agent.png",
                show=False,
                annotate_decisions=True,
                warmup=warmup_steps,
            )
            # Sector paths for both countries (output & price, relative)
            try:
                secs = [int(s) for s in (args.plot_sectors.split(',') if args.plot_sectors else []) if str(s).strip() != '']
            except Exception:
                secs = [0,1,2]
            for c in countries:
                plot_sector_paths(sim, c, 'output', sectors=secs, save_path=f"{args.plot_dir}/{tag}_sector_output_{c}.png", show=False, relative=True, warmup=warmup_steps)
                plot_sector_paths(sim, c, 'price', sectors=secs, save_path=f"{args.plot_dir}/{tag}_sector_price_{c}.png", show=False, relative=True, warmup=warmup_steps)
            print(f"plots -> {args.plot_dir}/{tag}_*.png")
        except Exception as e:
            print(f"[plot] failed: {e}")


if __name__ == "__main__":
    main()
