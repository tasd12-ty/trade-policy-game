#pragma once
/**
 * @file ces.hpp
 * @brief Armington CES 模型函数
 * 
 * 对应 Python: model.py::armington_share, armington_price, armington_quantity
 * 
 * 经济学背景：
 * Armington 假设认为国内外同类商品是不完全替代品，用 CES 函数刻画替代关系。
 * 替代弹性 σ = 1/(1-ρ)：
 *   - σ → ∞ (ρ → 1): 完全替代，选择最低价
 *   - σ = 1 (ρ = 0): Cobb-Douglas，固定份额
 *   - σ → 0 (ρ → -∞): Leontief，固定比例
 */

#include "../core/types.hpp"
#include "../core/math_utils.hpp"

namespace eco_model::armington {

/**
 * @brief Armington 份额 θ(p)
 * 
 * 在给定价格下计算本国产品的需求份额。
 * 
 * 公式（对应 tex 公式 (6)）：
 *   θ = (γ X_d)^ρ / [(γ X_d)^ρ + ((1-γ) X_f)^ρ]
 * 
 * 或基于价格的对偶形式：
 *   θ = γ^σ p_d^{1-σ} / [γ^σ p_d^{1-σ} + (1-γ)^σ p_f^{1-σ}]
 * 
 * @param gamma Armington 权重 γ ∈ (0,1)
 * @param p_dom 国内价格 P_d
 * @param p_for 进口价格 P_f
 * @param rho 形状参数 ρ
 * @return θ ∈ (0,1) 本国产品份额
 */
Scalar share(Scalar gamma, Scalar p_dom, Scalar p_for, Scalar rho);

/// 向量版本
Vector share(const Vector& gamma, const Vector& p_dom, 
             const Vector& p_for, const Vector& rho);

/**
 * @brief Armington 对偶价格 P*(p)
 * 
 * 嵌套 CES 的单位成本函数（用于边际成本计算）。
 * 
 * 公式（对应 tex 中边际成本推导）：
 *   P* = [γ^σ p_d^{1-σ} + (1-γ)^σ p_f^{1-σ}]^{1/(1-σ)}
 * 
 * 边界情况：
 *   - σ → ∞ (ρ → 1): P* → min(p_d, p_f)，使用 smooth_min
 *   - σ → 1 (ρ → 0): P* = p_d^γ · p_f^{1-γ}（几何平均）
 * 
 * @param gamma Armington 权重 γ
 * @param p_dom 国内价格
 * @param p_for 进口价格
 * @param rho 形状参数 ρ
 * @return P* 对偶价格
 */
Scalar price(Scalar gamma, Scalar p_dom, Scalar p_for, Scalar rho);

/// 向量版本
Vector price(const Vector& gamma, const Vector& p_dom,
             const Vector& p_for, const Vector& rho);

/**
 * @brief Armington 物量合成
 * 
 * 作为生产函数的"有效投入"。
 * 
 * 公式（对应 tex 公式 (2) 嵌套部分）：
 *   X^{CES} = [(γ x_d)^ρ + ((1-γ) x_f)^ρ]^{α/ρ}
 * 
 * @param gamma Armington 权重 γ
 * @param x_dom 国内投入量
 * @param x_for 进口投入量
 * @param alpha 投入弹性 α（若 α ≤ 0 返回 1.0）
 * @param rho 形状参数 ρ
 * @return 有效投入量
 */
Scalar quantity(Scalar gamma, Scalar x_dom, Scalar x_for, 
                Scalar alpha, Scalar rho);

/**
 * @brief 基于用量计算份额 θ
 * 
 * 公式（对应 tex 公式 (6)）：
 *   θ_{ij} = (γ_{ij} X_{ij}^I)^{ρ_{ij}} / 
 *            [(γ_{ij} X_{ij}^I)^{ρ_{ij}} + ((1-γ_{ij}) X_{ij}^O)^{ρ_{ij}}]
 * 
 * @param gamma Armington 权重
 * @param rho 形状参数
 * @param x_dom 国内用量
 * @param x_imp 进口用量
 * @return θ ∈ (0,1)
 */
Scalar theta_from_usage(Scalar gamma, Scalar rho, 
                        Scalar x_dom, Scalar x_imp);

/// 矩阵版本（用于整个投入矩阵）
Matrix theta_from_usage(const Matrix& gamma, const Matrix& rho,
                        const Matrix& x_dom, const Matrix& x_imp);

}  // namespace eco_model::armington
