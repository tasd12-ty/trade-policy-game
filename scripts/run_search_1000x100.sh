#!/usr/bin/env bash
set -euo pipefail

# 网格搜索入口：解析环境参数后调用 Python 实现。

RESULTS_DIR="${RESULTS_DIR:-results-search}"              # 搜索结果 CSV/PNG 的输出目录
AGENT_LOG_DIR="${AGENT_LOG_DIR:-output/agent_logs}"       # 智能体执行日志目录
COUNTRIES="${COUNTRIES:-H,F}"                             # 参与搜索的国家列表
ROUNDS="${ROUNDS:-1000}"                                  # 智能体循环的总轮数
K="${K:-100}"                                             # 每轮内部的步长（k per step）
PLOT_SECTORS="${PLOT_SECTORS:-0,1,2,3,4,5}"               # 绘图时展示的部门索引
WARMUP_WINDOW="${WARMUP_WINDOW:-100s}"                    # 预热稳定检测窗口
WARMUP_EPS="${WARMUP_EPS:-0.001}"                         # 稳定判定的波动阈值
WARMUP_MAX="${WARMUP_MAX:-2000}"                          # 预热最多迭代次数
REQUIRE_STABLE_WARMUP="${REQUIRE_STABLE_WARMUP:-true}"    # 若预热不稳定是否直接中止
RW="${RW:-w_income=1.0,w_price=0.2,w_trade=0.1}"         # 奖励权重配置
MAX_SECTORS_PER_TYPE="${MAX_SECTORS_PER_TYPE:-}"          # 观测裁剪：每类可保留的部门数
RECURSION_LIMIT="${RECURSION_LIMIT:-}"                    # 智能体递归深度上限

INIT_IMPORT_TARIFFS="${INIT_IMPORT_TARIFFS:-}"            # 预热后直接施加的进口关税映射
INIT_EXPORT_QUOTAS="${INIT_EXPORT_QUOTAS:-}"              # 预热后直接施加的出口配额映射

SEARCH_LOOKAHEAD="${SEARCH_LOOKAHEAD:-1}"                 # 策略 lookahead（传递给 policy spec）
SEARCH_STEPS="${SEARCH_STEPS:-100}"                       # 策略内部模拟步数
SEARCH_TARIFFS="${SEARCH_TARIFFS:-}"                      # 显式设定的关税候选值（可留空）
SEARCH_QUOTAS="${SEARCH_QUOTAS:-}"                        # 显式设定的配额候选值（可留空）
SEARCH_OBJECTIVE="${SEARCH_OBJECTIVE:-reward}"            # 策略目标（传递给 policy spec）
SEARCH_SECTORS="${SEARCH_SECTORS:-}"                      # 必填：指定要搜索的部门索引，逗号分隔
TARIFF_MIN="${TARIFF_MIN:--0.25}"                         # 关税搜索范围：最小值
TARIFF_MAX="${TARIFF_MAX:-0.25}"                          # 关税搜索范围：最大值
TARIFF_STEP="${TARIFF_STEP:-0.05}"                        # 关税搜索范围：步长
QUOTA_MIN="${QUOTA_MIN:-0.5}"                             # 配额乘数搜索：最小值
QUOTA_MAX="${QUOTA_MAX:-1.0}"                             # 配额乘数搜索：最大值
QUOTA_STEP="${QUOTA_STEP:-0.05}"                          # 配额乘数搜索：步长

OUT_TAG="${OUT_TAG:-search_${K}x${ROUNDS}}"
OUT_CSV="${OUT_CSV:-${RESULTS_DIR}/${OUT_TAG}.csv}"
GRID_MAX_COMBOS="${GRID_MAX_COMBOS:-64}"                  # 网格组合的最大评估数量

mkdir -p "${RESULTS_DIR}" "${AGENT_LOG_DIR}"

if [[ -z "${SEARCH_SECTORS}" ]]; then
  echo "ERROR: SEARCH_SECTORS must be set (e.g., 0,2,4)" >&2
  exit 1
fi
SEARCH_SPEC="search:lookahead=${SEARCH_LOOKAHEAD},steps=${SEARCH_STEPS},tariffs=${SEARCH_TARIFFS//,/;},quotas=${SEARCH_QUOTAS//,/;},objective=${SEARCH_OBJECTIVE},sectors=${SEARCH_SECTORS//,/;},method=grid"  # 传递给 policy 构建的参数字符串

export RESULTS_DIR AGENT_LOG_DIR COUNTRIES ROUNDS K PLOT_SECTORS WARMUP_WINDOW WARMUP_EPS WARMUP_MAX REQUIRE_STABLE_WARMUP
export RW MAX_SECTORS_PER_TYPE RECURSION_LIMIT SEARCH_SPEC OUT_TAG OUT_CSV GRID_MAX_COMBOS
export SEARCH_SECTORS TARIFF_MIN TARIFF_MAX TARIFF_STEP QUOTA_MIN QUOTA_MAX QUOTA_STEP
export INIT_IMPORT_TARIFFS INIT_EXPORT_QUOTAS

if [[ -z "${RECURSION_LIMIT}" ]]; then
  if [[ "${ROUNDS}" =~ ^[0-9]+$ ]]; then
    RECURSION_LIMIT=$(( ROUNDS * 120 + 200 ))
  else
    RECURSION_LIMIT=20000
  fi
fi
export RECURSION_LIMIT

echo "[search-run] rounds=${ROUNDS}, k=${K}, countries=${COUNTRIES}"
echo "[search-run] warmup window=${WARMUP_WINDOW} eps=${WARMUP_EPS} max=${WARMUP_MAX} require=${REQUIRE_STABLE_WARMUP}"
echo "[search-run] policy=${SEARCH_SPEC}"
echo "[search-run] init_tariff='${INIT_IMPORT_TARIFFS}' init_quota='${INIT_EXPORT_QUOTAS}'"
echo "[paths] results=${RESULTS_DIR} logs=${AGENT_LOG_DIR} csv=${OUT_CSV}"

PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=""
exec "${PYTHON_BIN}" scripts/search_grid_runner.py "$@"
