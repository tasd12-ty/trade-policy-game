#pragma once
/**
 * @file cost.hpp
 * @brief 边际成本计算
 * 
 * 对应 Python: model.py::compute_marginal_cost
 * 
 * 公式（对应 tex 公式 (7) 简化）：
 *   ln λ_i = -ln A_i + Σ_j α_{ij} ln P_j^*
 * 
 * 其中 P_j^* = P_j（不可贸易）或 Armington 对偶价格（可贸易）
 */

#include "../core/types.hpp"
#include "../core/country_params.hpp"
#include <vector>

namespace eco_model::production {

/**
 * @brief 计算各部门边际成本 λ
 * 
 * 零利润条件下 P_i = λ_i
 * 
 * @param params 国家参数
 * @param prices 国内价格 [n_sectors]
 * @param import_prices 进口价格 [n_sectors]
 * @param tradable_mask 可贸易部门掩码
 * @return Vector 各部门边际成本 λ [n_sectors]
 */
Vector compute_marginal_cost(
    const CountryParams& params,
    const Vector& prices,
    const Vector& import_prices,
    const std::vector<bool>& tradable_mask
);

}  // namespace eco_model::production
