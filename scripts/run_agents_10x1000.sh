#!/usr/bin/env bash
set -euo pipefail

# 多智能体 LLM 运行脚本：先预热 200 步，再执行 10,000 次同步行动，
# 行动之间穿插 k=5 个仿真步；如需调整请通过环境变量覆盖。

# --- 配置（导出同名环境变量即可覆盖） ---
# OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8001/v1}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
# OPENAI_MODEL="${OPENAI_MODEL:-/home/u20249114/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct}"
OPENAI_MODEL="${OPENAI_MODEL:-minimax/minimax-m2:free}"
# OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
OPENAI_API_KEY="${OPENAI_API_KEY:-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}"
OPENAI_TEMPERATURE="${OPENAI_TEMPERATURE:-1.0}"
POLICY_TARIFF_STEP="${POLICY_TARIFF_STEP:-0.05}"
AGENT_LOG_DIR="${AGENT_LOG_DIR:-output/agent_logs}"
AGENT_JSON_BLOCK_MODE="${AGENT_JSON_BLOCK_MODE:-true}"
AGENT_PROGRESS_EVERY="${AGENT_PROGRESS_EVERY:-10}"
AGENT_PROGRESS_JSON="${AGENT_PROGRESS_JSON:-}"
AGENT_LLM_RETRIES="${AGENT_LLM_RETRIES:-8}"
AGENT_USE_TOOLS="${AGENT_USE_TOOLS:-false}"

RESULTS_DIR="${RESULTS_DIR:-results}"
PLOT_SECTORS="${PLOT_SECTORS:-0,1,2}"
COUNTRIES="${COUNTRIES:-H,F}"
MODE="${MODE:-multi-sim-graph}"
ROUNDS="${ROUNDS:-500}"
K="${K:-5}"
# 预热控制：在窗口与阈值内趋于稳定前持续运行，最多执行指定步数
# 兼容旧参数：若设置 WARMUP_STEPS，则 WARMUP_MAX 回退为该值
WARMUP_WINDOW="${WARMUP_WINDOW:-100}"
WARMUP_EPS="${WARMUP_EPS:-0.001}"
WARMUP_MAX="${WARMUP_MAX:-${WARMUP_STEPS:-1000}}"
REQUIRE_STABLE_WARMUP="${REQUIRE_STABLE_WARMUP:-true}"
RW="${RW:-w_income=1.0,w_price=0.5,w_trade=0.5}"
# 可选：开启预热阶段的额外追踪
# 可选：限制每类政策影响的部门数量；为空时 CLI 不施加上限
MAX_SECTORS_PER_TYPE="${MAX_SECTORS_PER_TYPE:-}"
RECURSION_LIMIT="${RECURSION_LIMIT:-}"

OUT_TAG="${OUT_TAG:-llm_sim_${K}x${ROUNDS}}"
OUT_CSV="${OUT_CSV:-${RESULTS_DIR}/${OUT_TAG}.csv}"

# --- 准备工作 ---
mkdir -p "${RESULTS_DIR}" "${AGENT_LOG_DIR}"

export OPENAI_BASE_URL OPENAI_MODEL OPENAI_API_KEY OPENAI_TEMPERATURE POLICY_TARIFF_STEP
export AGENT_LOG_DIR AGENT_JSON_BLOCK_MODE AGENT_PROGRESS_EVERY AGENT_PROGRESS_JSON AGENT_LLM_RETRIES AGENT_USE_TOOLS

echo "[run] rounds=${ROUNDS}, k=${K}, mode=${MODE}, countries=${COUNTRIES}"
echo "[model] ${OPENAI_MODEL} @ ${OPENAI_BASE_URL}"
echo "[paths] results=${RESULTS_DIR} logs=${AGENT_LOG_DIR} out=${OUT_CSV}"
echo "[llm] temp=${OPENAI_TEMPERATURE} step=${POLICY_TARIFF_STEP} retries=${AGENT_LLM_RETRIES} block=${AGENT_JSON_BLOCK_MODE} tools=${AGENT_USE_TOOLS}"
echo "[progress] every=${AGENT_PROGRESS_EVERY} json=${AGENT_PROGRESS_JSON}"
echo "[warmup] window=${WARMUP_WINDOW} eps=${WARMUP_EPS} max=${WARMUP_MAX} require_stable=${REQUIRE_STABLE_WARMUP}"
if [[ -z "${RECURSION_LIMIT}" ]]; then
  if [[ "${ROUNDS}" =~ ^[0-9]+$ ]]; then
    # 每轮大约遍历 6 个图节点；用轮数的 10 倍并适当留出余量
    RECURSION_LIMIT=$(( ROUNDS * 100 + 100 ))
  else
    RECURSION_LIMIT=20000
  fi
fi
echo "[graph] recursion_limit=${RECURSION_LIMIT}"

# 可选：快速探测模型端点（失败不会终止脚本）
if command -v curl >/dev/null 2>&1; then
  echo "[check] probing ${OPENAI_BASE_URL}/models ..."
  if ! curl -sSf "${OPENAI_BASE_URL%/}/models" >/dev/null; then
    echo "[warn] model endpoint not reachable; continuing anyway"
  fi
fi

POLICY_SPEC="llm:model=${OPENAI_MODEL},base=${OPENAI_BASE_URL},temp=${OPENAI_TEMPERATURE},step=${POLICY_TARIFF_STEP}"

EXTRA_ARGS=()
if [[ -n "${MAX_SECTORS_PER_TYPE}" ]]; then
  EXTRA_ARGS+=(--max-sectors-per-type "${MAX_SECTORS_PER_TYPE}")
fi
EXTRA_ARGS+=(--recursion-limit "${RECURSION_LIMIT}")
EXTRA_ARGS+=(--warmup-window "${WARMUP_WINDOW}" --warmup-eps "${WARMUP_EPS}" --warmup-max "${WARMUP_MAX}")
if [[ "${REQUIRE_STABLE_WARMUP}" =~ ^(1|true|yes|y)$ ]]; then
  EXTRA_ARGS+=(--require-stable-warmup)
fi


python3 run_experiment.py \
  --mode "${MODE}" \
  --countries "${COUNTRIES}" \
  --rounds "${ROUNDS}" \
  --k "${K}" \
  --policy "${POLICY_SPEC}" \
  --rw "${RW}" \
  --obs-topk 5 \
  --plot \
  --plot-dir "${RESULTS_DIR}" \
  --plot-sectors "${PLOT_SECTORS}" \
  --out "${OUT_CSV}" \
  "${EXTRA_ARGS[@]}"

echo "[done] CSV -> ${OUT_CSV}"
echo "[done] Plots -> ${RESULTS_DIR}"
echo "[done] Agent logs -> ${AGENT_LOG_DIR}"
