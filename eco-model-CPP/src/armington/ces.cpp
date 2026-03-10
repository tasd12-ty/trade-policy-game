/**
 * @file ces.cpp
 * @brief Armington CES 函数实现
 * 
 * 对应 Python: model.py::armington_share, armington_price, armington_quantity
 */

#include "eco_model/armington/ces.hpp"
#include "eco_model/armington/smooth_ops.hpp"
#include <cmath>

namespace eco_model::armington {

Scalar share(Scalar gamma, Scalar p_dom, Scalar p_for, Scalar rho) {
    // 确保输入有效
    Scalar g = clamp(gamma, 1e-6, 1.0 - 1e-6);
    Scalar p_d = std::max(p_dom, EPS);
    Scalar p_f = std::max(p_for, EPS);
    
    // 计算替代弹性 σ = 1/(1-ρ)
    Scalar sigma = (std::abs(1.0 - rho) < 1e-10) ? 1.0 : 1.0 / (1.0 - rho);
    
    // 边界情况：ρ → 1（完全替代）
    if (std::abs(1.0 - rho) < 0.02) {
        // 使用平滑近似：p_d < p_f 时份额趋向 1
        return clamp(smooth::share_lower(p_d, p_f), 1e-6, 1.0 - 1e-6);
    }
    
    // 边界情况：σ → 1（Cobb-Douglas）
    if (std::abs(sigma - 1.0) < 1e-8) {
        return clamp(g, 1e-6, 1.0 - 1e-6);
    }
    
    // 标准 CES 份额计算
    // θ = γ^σ p_d^{1-σ} / [γ^σ p_d^{1-σ} + (1-γ)^σ p_f^{1-σ}]
    Scalar w_d = std::pow(g, sigma) * std::pow(p_d, 1.0 - sigma);
    Scalar w_f = std::pow(1.0 - g, sigma) * std::pow(p_f, 1.0 - sigma);
    Scalar theta = w_d / std::max(w_d + w_f, EPS);
    
    return clamp(theta, 1e-6, 1.0 - 1e-6);
}

Vector share(const Vector& gamma, const Vector& p_dom, 
             const Vector& p_for, const Vector& rho) {
    Index n = gamma.size();
    Vector result(n);
    for (Index i = 0; i < n; ++i) {
        result[i] = share(gamma[i], p_dom[i], p_for[i], rho[i]);
    }
    return result;
}

Scalar price(Scalar gamma, Scalar p_dom, Scalar p_for, Scalar rho) {
    // 确保输入有效
    Scalar g = clamp(gamma, 1e-6, 1.0 - 1e-6);
    Scalar p_d = std::max(p_dom, EPS);
    Scalar p_f = std::max(p_for, EPS);
    
    // 计算替代弹性
    Scalar sigma = (std::abs(1.0 - rho) < 1e-10) ? 1.0 : 1.0 / (1.0 - rho);
    
    // 边界情况：ρ → 1（完全替代），使用 smooth_min
    if (std::abs(1.0 - rho) < 0.02) {
        return smooth::min(p_d, p_f);
    }
    
    // 边界情况：σ → 1（Cobb-Douglas），几何平均
    if (std::abs(sigma - 1.0) < 1e-8) {
        return std::exp(g * std::log(p_d) + (1.0 - g) * std::log(p_f));
    }
    
    // 标准 CES 对偶价格
    // P* = [γ^σ p_d^{1-σ} + (1-γ)^σ p_f^{1-σ}]^{1/(1-σ)}
    Scalar inner = std::pow(g, sigma) * std::pow(p_d, 1.0 - sigma) +
                   std::pow(1.0 - g, sigma) * std::pow(p_f, 1.0 - sigma);
    
    return std::pow(std::max(inner, EPS), 1.0 / (1.0 - sigma));
}

Vector price(const Vector& gamma, const Vector& p_dom,
             const Vector& p_for, const Vector& rho) {
    Index n = gamma.size();
    Vector result(n);
    for (Index i = 0; i < n; ++i) {
        result[i] = price(gamma[i], p_dom[i], p_for[i], rho[i]);
    }
    return result;
}

Scalar quantity(Scalar gamma, Scalar x_dom, Scalar x_for, 
                Scalar alpha, Scalar rho) {
    // α ≤ 0 时不使用该投入，返回 1
    if (alpha <= 0.0) {
        return 1.0;
    }
    
    Scalar g = clamp(gamma, EPS, 1.0 - EPS);
    Scalar x_d = std::max(x_dom, EPS);
    Scalar x_f = std::max(x_for, EPS);
    
    // 边界情况：ρ → 0（Cobb-Douglas）
    if (std::abs(rho) < 1e-10) {
        return std::exp(alpha * (g * std::log(x_d) + (1.0 - g) * std::log(x_f)));
    }
    
    // CES 物量合成
    // X^{CES} = [(γ x_d)^ρ + ((1-γ) x_f)^ρ]^{α/ρ}
    Scalar comp = g * std::pow(x_d, rho) + (1.0 - g) * std::pow(x_f, rho);
    
    return std::pow(std::max(comp, EPS), alpha / rho);
}

Scalar theta_from_usage(Scalar gamma, Scalar rho, 
                        Scalar x_dom, Scalar x_imp) {
    Scalar g = clamp(gamma, 1e-6, 1.0 - 1e-6);
    Scalar x_d = std::max(x_dom, EPS);
    Scalar x_f = std::max(x_imp, EPS);
    
    // ρ → 0 时 θ = γ（Cobb-Douglas 固定份额）
    if (std::abs(rho) < 1e-10) {
        return clamp(g, 1e-6, 1.0 - 1e-6);
    }
    
    // θ = (γ x_d)^ρ / [(γ x_d)^ρ + ((1-γ) x_f)^ρ]
    Scalar dom = g * std::pow(x_d, rho);
    Scalar imp = (1.0 - g) * std::pow(x_f, rho);
    Scalar theta = dom / std::max(dom + imp, EPS);
    
    return clamp(theta, 1e-6, 1.0 - 1e-6);
}

Matrix theta_from_usage(const Matrix& gamma, const Matrix& rho,
                        const Matrix& x_dom, const Matrix& x_imp) {
    Index rows = gamma.rows();
    Index cols = gamma.cols();
    Matrix result(rows, cols);
    
    for (Index i = 0; i < rows; ++i) {
        for (Index j = 0; j < cols; ++j) {
            result(i, j) = theta_from_usage(gamma(i, j), rho(i, j), 
                                            x_dom(i, j), x_imp(i, j));
        }
    }
    
    return result;
}

}  // namespace eco_model::armington
