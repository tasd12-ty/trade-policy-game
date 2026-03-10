#pragma once
/**
 * @file math_utils.hpp
 * @brief 数学工具函数
 * 
 * 对应 Python: model.py::safe_log 等
 */

#include "types.hpp"
#include <cmath>
#include <algorithm>

namespace eco_model {

// ============================================================================
// 数值安全函数
// ============================================================================

/**
 * @brief 数值安全的对数：log(max(x, EPS))
 * 
 * 对应 Python: model.py::safe_log
 * 
 * @param x 输入值
 * @return log(max(x, EPS))
 */
inline Scalar safe_log(Scalar x) {
    return std::log(std::max(x, EPS));
}

/**
 * @brief 向量版本的安全对数
 * 
 * @param x 输入向量
 * @return 逐元素 safe_log
 */
inline Vector safe_log(const Vector& x) {
    return x.cwiseMax(EPS).array().log().matrix();
}

/**
 * @brief 矩阵版本的安全对数
 * 
 * @param x 输入矩阵
 * @return 逐元素 safe_log
 */
inline Matrix safe_log(const Matrix& x) {
    return x.cwiseMax(EPS).array().log().matrix();
}

/**
 * @brief 安全的除法：x / max(y, EPS)
 * 
 * @param x 被除数
 * @param y 除数
 * @return x / max(y, EPS)
 */
inline Scalar safe_div(Scalar x, Scalar y) {
    return x / std::max(y, EPS);
}

/**
 * @brief 安全的幂运算：pow(max(base, EPS), exp)
 * 
 * @param base 底数
 * @param exp 指数
 * @return pow(max(base, EPS), exp)
 */
inline Scalar safe_pow(Scalar base, Scalar exp) {
    return std::pow(std::max(base, EPS), exp);
}

/**
 * @brief 将标量限制在 [min_val, max_val] 范围内
 */
inline Scalar clamp(Scalar x, Scalar min_val, Scalar max_val) {
    return std::min(std::max(x, min_val), max_val);
}

/**
 * @brief 将向量元素限制在 [min_val, max_val] 范围内
 */
inline Vector clamp(const Vector& x, Scalar min_val, Scalar max_val) {
    return x.cwiseMax(min_val).cwiseMin(max_val);
}

}  // namespace eco_model
