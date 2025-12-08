#!/usr/bin/env python3
"""
两国博弈：基于网格的交替 Best-Response 搜索。

流程（假定 COUNTRIES=H,F）：
- 先用对称参数构造并预热仿真（可选 INIT_IMPORT_TARIFFS / INIT_EXPORT_QUOTAS 作为初始冲突 / 初始决策）。
- 记 A=COUNTRIES[0], B=COUNTRIES[1]。
- 对给定部门集合和关税/配额网格，交替执行若干轮 Best-Response：
  - 在固定 A 当前静态政策下，B 在网格上搜索使自身最终 reward 最大的静态关税/配额组合；
  - 再在固定 B 当前静态政策下，A 进行同样的单国网格搜索；
  - 重复 BR_ITERATIONS 轮。
- 最后，用得到的 (A*, B*) 静态政策从预热后的基线状态重新跑一遍完整仿真，导出时间序列 CSV 与图表。

注意：
- 本脚本只负责“静态关税/配额”的网格搜索；智能体策略仍由 SEARCH_SPEC 决定（通常是 SearchPolicy / LLM 策略）。
- 为避免组合数爆炸，可通过 SEARCH_SECTORS 限定部门，并用 GRID_MAX_COMBOS 限制单国搜索的候选数。
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import itertools
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from eco_simu import create_symmetric_parameters, bootstrap_simulator
from eco_simu.sim import TwoCountryDynamicSimulator
from eco_simu.plotting import (
    plot_history,
    plot_diagnostics,
    plot_history_agent_view,
    plot_sector_paths,
)
from agent_loop import (
    MultiCountryLoopState,
    run_multilateral_with_graph,
    build_policy_from_spec,
    parse_reward_weights,
)


def main() -> None:
    env = os.environ

    ROUNDS = int(env["ROUNDS"])
    K = int(env["K"])
    COUNTRIES = [c.strip().upper() for c in env["COUNTRIES"].split(",") if c.strip()]
    if len(COUNTRIES) != 2:
        raise SystemExit("[best-response] 目前仅支持两国博弈（COUNTRIES 必须包含且仅包含两个国家，例如 H,F）")
    A, B = COUNTRIES[0], COUNTRIES[1]

    RESULTS_DIR = env["RESULTS_DIR"]
    OUT_TAG = env["OUT_TAG"]
    OUT_CSV = env["OUT_CSV"]
    SEARCH_SPEC = env["SEARCH_SPEC"]
    SEARCH_TARIFF_STR = env.get("SEARCH_TARIFFS", "")
    SEARCH_QUOTA_STR = env.get("SEARCH_QUOTAS", "")
    SEARCH_SECTORS_RAW = env.get("SEARCH_SECTORS", "")
    TARIFF_MIN = env.get("TARIFF_MIN", "")
    TARIFF_MAX = env.get("TARIFF_MAX", "")
    TARIFF_STEP = env.get("TARIFF_STEP", "")
    QUOTA_MIN = env.get("QUOTA_MIN", "")
    QUOTA_MAX = env.get("QUOTA_MAX", "")
    QUOTA_STEP = env.get("QUOTA_STEP", "")
    RW_SPEC = env["RW"]
    WARMUP_WINDOW = int(env["WARMUP_WINDOW"])
    WARMUP_EPS = float(env["WARMUP_EPS"])
    WARMUP_MAX = int(env["WARMUP_MAX"])
    REQUIRE_STABLE_WARMUP = env.get("REQUIRE_STABLE_WARMUP", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    MAX_SECTORS = env.get("MAX_SECTORS_PER_TYPE")
    REC_LIMIT = env.get("RECURSION_LIMIT")
    INIT_TARIFF = env.get("INIT_IMPORT_TARIFFS", "")
    INIT_QUOTA = env.get("INIT_EXPORT_QUOTAS", "")
    PLOT_SECTORS = [
        int(s) for s in env.get("PLOT_SECTORS", "0,1,2").split(",") if s.strip()
    ]
    GRID_MAX_COMBOS = int(env.get("GRID_MAX_COMBOS", "0") or "0")
    BR_ITERATIONS = int(env.get("BR_ITERATIONS", "1") or "1")

    # 1. 构造并预热仿真（含初始冲突）
    sim = bootstrap_simulator(params_raw=create_symmetric_parameters(), theta_price=0.05)
    n_sec = int(getattr(sim.params.home.alpha, "shape")[0])  # type: ignore[attr-defined]

    warm_info = warmup_until_stable(sim, WARMUP_WINDOW, WARMUP_EPS, WARMUP_MAX)
    warmup_steps = int(warm_info.get("steps", 0) or 0)
    print(
        f"[warmup] steps={warm_info['steps']} stable={warm_info['stable']} "
        f"time={warm_info['seconds']:.2f}s"
    )
    if not warm_info["stable"] and REQUIRE_STABLE_WARMUP:
        raise SystemExit("[warmup] ERROR: did not reach stability within warmup_max; aborting.")

    if INIT_TARIFF.strip():
        _apply_mapping(INIT_TARIFF, sim.apply_import_tariff)
        print(f"[init] import tariff -> {INIT_TARIFF}")
    if INIT_QUOTA.strip():
        _apply_mapping(INIT_QUOTA, sim.apply_export_control)
        print(f"[init] export quota -> {INIT_QUOTA}")

    # 2. 构造智能体策略与奖励权重（不改）
    policy = {c: build_policy_from_spec(SEARCH_SPEC, n_sec) for c in COUNTRIES}
    rw_common = parse_reward_weights(RW_SPEC)
    reward_weights = {c: rw_common for c in COUNTRIES}

    # 3. 关税/配额网格与部门集合
    tariff_values = _prepare_values(
        SEARCH_TARIFF_STR,
        TARIFF_MIN,
        TARIFF_MAX,
        TARIFF_STEP,
        include_default=[-0.1, 0.1],
        baseline=0.0,
    )
    quota_values = _prepare_values(
        SEARCH_QUOTA_STR,
        QUOTA_MIN,
        QUOTA_MAX,
        QUOTA_STEP,
        include_default=[0.8, 0.6],
        baseline=1.0,
    )

    target_indices = _parse_sectors(SEARCH_SECTORS_RAW, n_sec)
    if not target_indices:
        target_indices = [0]

    print(
        f"[best-response] countries={COUNTRIES}, targets={target_indices}, "
        f"tariff_values={tariff_values}, quota_values={quota_values}, "
        f"max_combos_per_actor={GRID_MAX_COMBOS}, br_iterations={BR_ITERATIONS}"
    )

    # 4. 交替 best-response：current_policy 只记录在 INIT_* 之外的额外静态政策
    base_sim = sim
    current_policy: Dict[str, Dict[str, Dict[int, float]]] = {
        c: {"import_tariff": {}, "export_quota": {}} for c in COUNTRIES
    }

    summary_rows: List[List[Any]] = []

    for br_round in range(1, BR_ITERATIONS + 1):
        print(f"[best-response] round {br_round}: actor={B} searching best response")
        best_B_policy, _, _, rows_B = search_best_for_country(
            base_sim=base_sim,
            countries=COUNTRIES,
            actor=B,
            current_policy=current_policy,
            target_indices=target_indices,
            tariff_values=tariff_values,
            quota_values=quota_values,
            rounds=ROUNDS,
            k_per_step=K,
            policy=policy,
            reward_weights=reward_weights,
            max_sectors=MAX_SECTORS,
            rec_limit=REC_LIMIT,
            max_grid=GRID_MAX_COMBOS,
        )
        current_policy[B] = best_B_policy
        for row in rows_B:
            summary_rows.append([br_round, B] + row)

        print(f"[best-response] round {br_round}: actor={A} searching best response")
        best_A_policy, _, _, rows_A = search_best_for_country(
            base_sim=base_sim,
            countries=COUNTRIES,
            actor=A,
            current_policy=current_policy,
            target_indices=target_indices,
            tariff_values=tariff_values,
            quota_values=quota_values,
            rounds=ROUNDS,
            k_per_step=K,
            policy=policy,
            reward_weights=reward_weights,
            max_sectors=MAX_SECTORS,
            rec_limit=REC_LIMIT,
            max_grid=GRID_MAX_COMBOS,
        )
        current_policy[A] = best_A_policy
        for row in rows_A:
            summary_rows.append([br_round, A] + row)

    # 5. 用最终 (A*,B*) 静态政策跑一次完整仿真并导出结果
    final_sim = base_sim.clone()
    for c in COUNTRIES:
        pol = current_policy.get(c, {})
        imp = pol.get("import_tariff") or {}
        quo = pol.get("export_quota") or {}
        if imp:
            final_sim.apply_import_tariff(c, imp)
        if quo:
            final_sim.apply_export_control(c, quo)

    final_state = MultiCountryLoopState(
        t=0,
        sim=final_sim,
        countries=COUNTRIES,
        k_per_step=K,
        max_rounds=ROUNDS,
        policy=policy,
        reward_weights=reward_weights,
        use_early_stop=False,
        obs_full=True,
        obs_topk=5,
        max_sectors_per_type=(int(MAX_SECTORS) if (MAX_SECTORS or "").strip() else None),
        recursion_limit=(int(REC_LIMIT) if (REC_LIMIT or "").strip() else None),
    )
    final_state = run_multilateral_with_graph(final_state, mode="simultaneous", order=COUNTRIES)
    if isinstance(final_state, dict):
        final_log = list(final_state.get("log", []))
    else:
        final_log = list(getattr(final_state, "log", []))

    if not final_log:
        raise SystemExit("[best-response] 最终仿真无日志输出，无法导出结果")

    # 写出交替搜索过程的网格摘要
    summary_path = os.path.join(RESULTS_DIR, f"{OUT_TAG}_grid_summary.csv")
    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        header = [
            "br_round",
            "actor",
            "combo_index",
            "score_actor",
            f"reward_{A}",
            f"reward_{B}",
            "combo_profile",
        ]
        writer.writerow(header)
        writer.writerows(summary_rows)

    # 写出最终 (A*,B*) 策略下的时间序列与图表
    _write_best_csv(final_log, OUT_CSV, COUNTRIES)
    _plot_outputs(final_sim, final_log, RESULTS_DIR, OUT_TAG, COUNTRIES, PLOT_SECTORS, K, warmup_steps=warmup_steps)

    last = final_log[-1]
    rewards = last.get("reward", {})
    print(
        f"[best-response] final rewards: "
        f"{A}={float(rewards.get(A, 0.0) or 0.0):.4f}, "
        f"{B}={float(rewards.get(B, 0.0) or 0.0):.4f}"
    )
    print(f"[best-response] final static policies: {current_policy}")
    print(f"[best-response] summary -> {summary_path}")
    print(f"[done] CSV -> {OUT_CSV}")
    print(f"[done] 图表 -> {RESULTS_DIR}/{OUT_TAG}_summary*.png")


def search_best_for_country(
    base_sim: TwoCountryDynamicSimulator,
    countries: Sequence[str],
    actor: str,
    current_policy: Dict[str, Dict[str, Dict[int, float]]],
    target_indices: Sequence[int],
    tariff_values: Sequence[float],
    quota_values: Sequence[float],
    rounds: int,
    k_per_step: int,
    policy: Dict[str, Any],
    reward_weights: Dict[str, Dict[str, float]],
    max_sectors: str | None,
    rec_limit: str | None,
    max_grid: int,
) -> Tuple[Dict[str, Dict[int, float]], List[Dict[str, Any]], TwoCountryDynamicSimulator, List[List[Any]]]:
    """在固定对手静态政策的前提下，为指定 actor 做单国网格搜索。"""

    actor = actor.upper()
    if len(countries) != 2:
        raise ValueError("search_best_for_country 仅支持两国场景")
    A, B = countries[0], countries[1]
    if actor not in (A, B):
        raise ValueError(f"actor 必须为 {A} 或 {B}")

    combos = _country_action_grid(target_indices, tariff_values, quota_values, max_grid=max_grid)

    best_score = float("-inf")
    best_policy_actor: Dict[str, Dict[int, float]] = {"import_tariff": {}, "export_quota": {}}
    best_log: List[Dict[str, Any]] = []
    best_sim: TwoCountryDynamicSimulator | None = None
    rows: List[List[Any]] = []

    for idx, candidate in enumerate(combos, start=1):
        sim_candidate = base_sim.clone()

        # 组合当前静态政策 profile：对手采用 current_policy，actor 采用当前候选
        combined_profile: Dict[str, Dict[str, Dict[int, float]]] = {}
        for c in countries:
            if c == actor:
                imp = dict(candidate.get("import_tariff") or {})
                quo = dict(candidate.get("export_quota") or {})
            else:
                pol = current_policy.get(c, {})
                imp = dict((pol.get("import_tariff") or {}))
                quo = dict((pol.get("export_quota") or {}))
            combined_profile[c] = {"import_tariff": imp, "export_quota": quo}

        for c in countries:
            imp = combined_profile[c]["import_tariff"]
            quo = combined_profile[c]["export_quota"]
            if imp:
                sim_candidate.apply_import_tariff(c, imp)
            if quo:
                sim_candidate.apply_export_control(c, quo)

        state = MultiCountryLoopState(
            t=0,
            sim=sim_candidate,
            countries=list(countries),
            k_per_step=k_per_step,
            max_rounds=rounds,
            policy=policy,
            reward_weights=reward_weights,
            use_early_stop=False,
            obs_full=True,
            obs_topk=5,
            max_sectors_per_type=(int(max_sectors) if (max_sectors or "").strip() else None),
            recursion_limit=(int(rec_limit) if (rec_limit or "").strip() else None),
        )
        state = run_multilateral_with_graph(state, mode="simultaneous", order=list(countries))
        if isinstance(state, dict):
            log_iter = list(state.get("log", []))
        else:
            log_iter = list(getattr(state, "log", []))

        if log_iter:
            final_log = log_iter[-1]
            rewards = final_log.get("reward", {})
            actor_score = float(rewards.get(actor, 0.0) or 0.0)
            reward_A = float(rewards.get(A, 0.0) or 0.0)
            reward_B = float(rewards.get(B, 0.0) or 0.0)
        else:
            actor_score = None
            reward_A = 0.0
            reward_B = 0.0

        profile_json = json.dumps(combined_profile, sort_keys=True)
        row = [
            idx,
            actor_score if actor_score is not None else "",
            reward_A if actor_score is not None else "",
            reward_B if actor_score is not None else "",
            profile_json,
        ]
        rows.append(row)

        readable_score = f"{actor_score:.4f}" if actor_score is not None else "nan"
        print(f"[best-response] actor={actor} combo {idx}/{len(combos)} score={readable_score}")

        if log_iter and actor_score is not None and actor_score > best_score:
            best_score = actor_score
            best_policy_actor = {
                "import_tariff": dict(candidate.get("import_tariff") or {}),
                "export_quota": dict(candidate.get("export_quota") or {}),
            }
            best_log = log_iter
            best_sim = sim_candidate

    if best_sim is None:
        raise SystemExit(f"[best-response] actor={actor} 没有有效组合被成功评估")

    return best_policy_actor, best_log, best_sim, rows


def warmup_until_stable(
    sim_obj: TwoCountryDynamicSimulator,
    window: int,
    eps: float,
    max_steps: int,
) -> Dict[str, Any]:
    steps = 0
    stable = False
    start = time.time()
    while steps < max_steps:
        summary = sim_obj.summarize_history()
        ok_all = True
        for c in ["H", "F"]:
            metrics = summary.get(c, {})
            ig = np.array(metrics.get("income_growth", []), float)
            og = np.array(metrics.get("output_growth", []), float)
            if ig.shape[0] < window + 1 or og.shape[0] < window + 1:
                ok_all = False
                break
            diffs_ig = np.diff(ig[-window - 1 :])
            diffs_og = np.diff(og[-window - 1 :])
            if not (np.all(np.abs(diffs_ig) < eps) and np.all(np.abs(diffs_og) < eps)):
                ok_all = False
                break
        if ok_all:
            stable = True
            break
        sim_obj.step()
        steps += 1
    return {"steps": steps, "stable": stable, "seconds": time.time() - start}


def _apply_mapping(spec: str, fn) -> None:
    for part in spec.split(","):
        if not part or ":" not in part or "=" not in part:
            continue
        left, right = part.split(":", 1)
        sec_raw, val_raw = right.split("=", 1)
        try:
            fn(left.strip().upper(), {int(sec_raw.strip()): float(val_raw.strip())})
        except Exception:
            continue


def _list_from_string(raw: str) -> List[float]:
    vals: List[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(float(part))
        except Exception:
            continue
    return vals


def _range_values(min_str: str, max_str: str, step_str: str, *, include_default: Sequence[float]) -> List[float]:
    try:
        min_val = float(min_str)
        max_val = float(max_str)
        step_val = float(step_str)
    except Exception:
        return list(include_default)
    if step_val <= 0 or max_val < min_val:
        return list(include_default)
    vals: List[float] = []
    cur = min_val
    while cur <= max_val + 1e-9:
        vals.append(round(cur, 6))
        cur += step_val
    return vals or list(include_default)


def _prepare_values(
    raw_list: str,
    min_str: str,
    max_str: str,
    step_str: str,
    *,
    include_default: Sequence[float],
    baseline: float,
) -> List[float]:
    values = _list_from_string(raw_list)
    if not values:
        values = _range_values(min_str, max_str, step_str, include_default=include_default)
    if baseline not in values:
        values.append(baseline)
    if baseline == 0.0:
        return [baseline] + [val for val in sorted(set(values)) if abs(val) > 1e-9]
    return [baseline] + [val for val in sorted(set(values)) if abs(val - baseline) > 1e-9]


def _parse_sectors(raw: str, n_sec: int) -> List[int]:
    if raw.strip():
        target_indices: List[int] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                sec = int(part)
            except Exception:
                continue
            if 0 <= sec < n_sec:
                target_indices.append(sec)
        return sorted(set(target_indices))
    return list(range(n_sec))


def _country_action_grid(
    target_indices: Sequence[int],
    tariff_values: Sequence[float],
    quota_values: Sequence[float],
    max_grid: int,
) -> List[Dict[str, Dict[int, float]]]:
    """构造单国的 (关税,配额) 网格组合，并按 max_grid 截断。"""
    sector_opts = list(itertools.product(tariff_values, quota_values))
    combos: List[Dict[str, Dict[int, float]]] = []
    for assign in itertools.product(sector_opts, repeat=len(target_indices)):
        tariffs: Dict[int, float] = {}
        quotas: Dict[int, float] = {}
        for sec, (tar, quo) in zip(target_indices, assign):
            if abs(tar) > 1e-9:
                tariffs[sec] = float(tar)
            if abs(quo - 1.0) > 1e-9:
                quotas[sec] = float(quo)
        combos.append({"import_tariff": tariffs, "export_quota": quotas})
        if max_grid > 0 and len(combos) >= max_grid:
            break
    return combos or [{"import_tariff": {}, "export_quota": {}}]


def _write_best_csv(
    best_log: List[Dict[str, Any]],
    out_csv: str,
    countries: Sequence[str],
) -> None:
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["t"]
        for c in countries:
            header += [
                f"reward_{c}",
                f"income_growth_{c}",
                f"price_mean_{c}",
                f"trade_balance_{c}",
            ]
        writer.writerow(header)
        for item in best_log:
            row = [item.get("t")]
            for c in countries:
                metrics = item["obs"][c]["metrics"]
                reward = item["reward"].get(c)
                row += [
                    reward,
                    metrics.get("income_growth_last"),
                    metrics.get("price_mean_last"),
                    metrics.get("trade_balance_last"),
                ]
            writer.writerow(row)


def _plot_outputs(
    sim_obj: TwoCountryDynamicSimulator,
    best_log: Sequence[Dict[str, Any]],
    results_dir: str,
    out_tag: str,
    countries: Sequence[str],
    plot_sectors: Sequence[int],
    k_per_step: int,
    warmup_steps: int = 0,
) -> None:
    plot_history(sim_obj, save_path=f"{results_dir}/{out_tag}_summary.png", show=False, warmup=warmup_steps)
    plot_diagnostics(sim_obj, save_path=f"{results_dir}/{out_tag}_diagnostics.png", show=False, warmup=warmup_steps)
    for c in countries:
        plot_sector_paths(
            sim_obj,
            c,
            "output",
            sectors=plot_sectors,
            save_path=f"{results_dir}/{out_tag}_sector_output_{c}.png",
            show=False,
            relative=True,
            warmup=warmup_steps,
        )
        plot_sector_paths(
            sim_obj,
            c,
            "price",
            sectors=plot_sectors,
            save_path=f"{results_dir}/{out_tag}_sector_price_{c}.png",
            show=False,
            relative=True,
            warmup=warmup_steps,
        )
    plot_history_agent_view(
        sim_obj,
        agent_log=list(best_log),
        k_per_step=k_per_step,
        save_path=f"{results_dir}/{out_tag}_summary_agent.png",
        show=False,
        annotate_decisions=True,
        warmup=warmup_steps,
    )


if __name__ == "__main__":
    main()
