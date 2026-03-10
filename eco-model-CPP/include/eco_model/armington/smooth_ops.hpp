#pragma once
/**
 * @file smooth_ops.hpp
 * @brief 平滑数学算子
 * 
 * 对应 Python: model/smooth_ops.py
 * 
 * 这些函数用于替代经济模型中的非可微操作（min, max, if-else），
 * 使得整个模型能够通过自动微分进行梯度下降优化。
 */

#include "../core/types.hpp"
#include <cmath>

namespace eco_model::smooth {

/**
 * @brief 平滑最大值函数（基于 LogSumExp）
 * 
 * 公式：S_max(x, y; k) = (1/k) * log(exp(k*x) + exp(k*y))
 * 
 * 当 k 趋向无穷大时，逼近 max(x, y)。
 * 使用数值稳定的 LSE 实现避免溢出。
 * 
 * @param x 第一个输入
 * @param y 第二个输入
 * @param k 平滑参数（逆温度），默认 SMOOTH_K。值越大越"硬"
 * @return 平滑最大值
 */
inline Scalar max(Scalar x, Scalar y, Scalar k = SMOOTH_K) {
    // 数值稳定实现：max_val + (1/k) * log(exp(k*(x-max_val)) + exp(k*(y-max_val)))
    Scalar max_val = std::max(x, y);
    return max_val + (1.0 / k) * std::log(
        std::exp(k * (x - max_val)) + std::exp(k * (y - max_val))
    );
}

/**
 * @brief 平滑最小值函数
 * 
 * 公式：S_min(x, y; k) = -S_max(-x, -y; k)
 * 
 * @param x 第一个输入
 * @param y 第二个输入
 * @param k 平滑参数，默认 SMOOTH_K
 * @return 平滑最小值
 */
inline Scalar min(Scalar x, Scalar y, Scalar k = SMOOTH_K) {
    return -max(-x, -y, k);
}

/**
 * @brief 平滑阶跃函数（Sigmoid 近似）
 * 
 * 近似 Heaviside 阶跃函数：
 *   - x > 0 时，返回接近 1
 *   - x < 0 时，返回接近 0
 *   - x = 0 时，返回 0.5
 * 
 * 公式：sigmoid(k * x) = 1 / (1 + exp(-k * x))
 * 
 * @param x 输入
 * @param k 平滑参数，默认 SMOOTH_K。值越大过渡越陡峭
 * @return 平滑阶跃值 ∈ (0, 1)
 */
inline Scalar step(Scalar x, Scalar k = SMOOTH_K) {
    return 1.0 / (1.0 + std::exp(-k * x));
}

/**
 * @brief 平滑份额指示函数
 * 
 * 返回 (val_lower <= val_higher) 的平滑近似：
 *   - val_lower < val_higher 时，返回接近 1（选择 lower）
 *   - val_lower > val_higher 时，返回接近 0
 * 
 * 用途：在 Armington 模型的 ρ → 1 极限情况下，
 * 平滑地选择价格较低的来源。
 * 
 * @param val_lower 较低值（如本国价格）
 * @param val_higher 较高值（如外国价格）
 * @param k 平滑参数
 * @return 平滑份额 ∈ (0, 1)
 */
inline Scalar share_lower(Scalar val_lower, Scalar val_higher, Scalar k = SMOOTH_K) {
    return step(val_higher - val_lower, k);
}

// ============================================================================
// 向量版本
// ============================================================================

inline Vector max(const Vector& x, const Vector& y, Scalar k = SMOOTH_K) {
    Vector result(x.size());
    for (Index i = 0; i < x.size(); ++i) {
        result[i] = max(x[i], y[i], k);
    }
    return result;
}

inline Vector min(const Vector& x, const Vector& y, Scalar k = SMOOTH_K) {
    return -max(-x, -y, k);
}

inline Vector step(const Vector& x, Scalar k = SMOOTH_K) {
    return (1.0 / (1.0 + (-k * x.array()).exp())).matrix();
}

}  // namespace eco_model::smooth
