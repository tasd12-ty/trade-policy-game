#!/usr/bin/env bash
set -euo pipefail

# 动态搜索策略运行脚本：
# 使用 SearchPolicyAdapter 在每一步（或每 k 步）进行前瞻搜索，动态决定最优动作。
# 这与 run_search_1000x100.sh 不同，后者是寻找静态最优解的网格搜索。

# --- 基础配置 ---
RESULTS_DIR="${RESULTS_DIR:-results-dynamic-search}"  # 结果输出目录
AGENT_LOG_DIR="${AGENT_LOG_DIR:-output/agent_logs}"   # 智能体日志目录
COUNTRIES="${COUNTRIES:-H,F}"                         # 参与博弈的国家
ROUNDS="${ROUNDS:-1000}"                               # 总轮数
K="${K:-100}"                                           # 每轮步数 (k_per_step)
PLOT_SECTORS="${PLOT_SECTORS:-0,1,2,3,4,5}"           # 绘图展示的部门
MODE="${MODE:-multi-alt-graph}"                       # 运行模式

# --- 预热配置 ---
WARMUP_WINDOW="${WARMUP_WINDOW:-100}"                 # 预热稳定窗口
WARMUP_EPS="${WARMUP_EPS:-0.001}"                     # 稳定阈值
WARMUP_MAX="${WARMUP_MAX:-1000}"                      # 最大预热步数
REQUIRE_STABLE_WARMUP="${REQUIRE_STABLE_WARMUP:-true}" # 是否强制要求预热稳定

# --- 奖励权重 ---
RW="${RW:-w_income=1.0,w_price=0.2,w_trade=0.1}"

# --- 搜索策略配置 (SearchPolicyAdapter) ---
SEARCH_LOOKAHEAD="${SEARCH_LOOKAHEAD:-100}"             # 前瞻轮数 (lookahead_rounds)
SEARCH_STEPS="${SEARCH_STEPS:-50}"                    # 前瞻模拟内部步数 (lookahead_steps)
SEARCH_TARIFFS="${SEARCH_TARIFFS:-}"                  # 关税候选挡位 (逗号分隔)
SEARCH_QUOTAS="${SEARCH_QUOTAS:-}"                    # 配额候选挡位 (逗号分隔)
SEARCH_OBJECTIVE="${SEARCH_OBJECTIVE:-reward}"        # 优化目标 (reward/income/etc)
SEARCH_SECTORS="${SEARCH_SECTORS:-}"                  # 必填：静态指定要搜索的部门 (例如 "0,2,4")
SEARCH_METHOD="grid"                                  # 仅支持 grid（笛卡尔积枚举）

# --- 其他配置 ---
MAX_SECTORS_PER_TYPE="${MAX_SECTORS_PER_TYPE:-}"      # 观测裁剪
RECURSION_LIMIT="${RECURSION_LIMIT:-}"                # 递归深度限制

# --- 准备目录 ---
mkdir -p "${RESULTS_DIR}" "${AGENT_LOG_DIR}"

OUT_TAG="${OUT_TAG:-dynamic_search_${K}x${ROUNDS}}"
OUT_CSV="${OUT_CSV:-${RESULTS_DIR}/${OUT_TAG}.csv}"

export RESULTS_DIR AGENT_LOG_DIR

# --- 构建策略 Spec ---
if [[ -z "${SEARCH_SECTORS}" ]]; then
  echo "ERROR: SEARCH_SECTORS must be set (e.g., 0,2,4)" >&2
  exit 1
fi
# 将逗号分隔转换为分号，避免解析冲突
SEARCH_SPEC="search:lookahead=${SEARCH_LOOKAHEAD},steps=${SEARCH_STEPS},tariffs=${SEARCH_TARIFFS//,/;},quotas=${SEARCH_QUOTAS//,/;},objective=${SEARCH_OBJECTIVE},sectors=${SEARCH_SECTORS//,/;},method=grid"

echo "[run] rounds=${ROUNDS}, k=${K}, countries=${COUNTRIES}"
echo "[policy] ${SEARCH_SPEC}"
echo "[paths] results=${RESULTS_DIR} logs=${AGENT_LOG_DIR} out=${OUT_CSV}"

# --- 计算递归限制 ---
if [[ -z "${RECURSION_LIMIT}" ]]; then
  if [[ "${ROUNDS}" =~ ^[0-9]+$ ]]; then
    RECURSION_LIMIT=$(( ROUNDS * 100 + 200 ))
  else
    RECURSION_LIMIT=20000
  fi
fi

# --- 构造参数 ---
EXTRA_ARGS=()
if [[ -n "${MAX_SECTORS_PER_TYPE}" ]]; then
  EXTRA_ARGS+=(--max-sectors-per-type "${MAX_SECTORS_PER_TYPE}")
fi
EXTRA_ARGS+=(--recursion-limit "${RECURSION_LIMIT}")
EXTRA_ARGS+=(--warmup-window "${WARMUP_WINDOW}" --warmup-eps "${WARMUP_EPS}" --warmup-max "${WARMUP_MAX}")
if [[ "${REQUIRE_STABLE_WARMUP}" =~ ^(1|true|yes|y)$ ]]; then
  EXTRA_ARGS+=(--require-stable-warmup)
fi

# --- 执行实验 ---
python3 run_experiment.py \
  --mode "${MODE}" \
  --countries "${COUNTRIES}" \
  --rounds "${ROUNDS}" \
  --k "${K}" \
  --policy "${SEARCH_SPEC}" \
  --rw "${RW}" \
  --obs-topk 5 \
  --plot \
  --plot-dir "${RESULTS_DIR}" \
  --plot-sectors "${PLOT_SECTORS}" \
  --out "${OUT_CSV}" \
  "${EXTRA_ARGS[@]}"

echo "[done] CSV -> ${OUT_CSV}"
echo "[done] Plots -> ${RESULTS_DIR}"
