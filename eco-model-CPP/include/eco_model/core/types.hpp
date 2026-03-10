#pragma once
/**
 * @file types.hpp
 * @brief 基本类型定义和全局常量
 * 
 * 对应 Python: model.py 中的 EPS, TORCH_DTYPE 等
 */

#include <Eigen/Core>
#include <cstdint>

namespace eco_model {

// ============================================================================
// 类型别名
// ============================================================================

/// 标量类型（对应 Python float64）
using Scalar = double;

/// 向量类型（列向量）
using Vector = Eigen::VectorXd;

/// 矩阵类型（行优先存储以匹配 Python numpy 默认布局）
using Matrix = Eigen::MatrixXd;

/// 索引类型
using Index = Eigen::Index;

// ============================================================================
// 全局常量
// ============================================================================

/// 数值稳定性常量，避免 log(0) 或除零
/// 对应 Python: model.py::EPS = 1e-9
constexpr Scalar EPS = 1e-9;

/// 默认部门数量
constexpr int DEFAULT_NUM_SECTORS = 6;

/// 平滑函数的默认参数 k（控制 smooth_min/max 的锐度）
/// 对应 Python: model.py::SMOOTH_K = 20.0
constexpr Scalar SMOOTH_K = 20.0;

// ============================================================================
// 辅助宏
// ============================================================================

/// 确保值不小于 EPS
#define CLAMP_MIN(x) std::max((x), EPS)

}  // namespace eco_model
