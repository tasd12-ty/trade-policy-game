#pragma once
/**
 * @file output.hpp
 * @brief 生产函数：计算各部门产出
 * 
 * 对应 Python: model.py::compute_output
 * 
 * 公式（对应 tex 公式 (2)）：
 *   Y_i = A_i ∏_j [X_{ij}]^{α_{ij}}
 * 
 * 其中对可贸易部门 j，使用 Armington CES 嵌套：
 *   X_{ij}^{CES} = [(γ_{ij} X_{ij}^I)^{ρ_{ij}} + ((1-γ_{ij}) X_{ij}^O)^{ρ_{ij}}]^{α_{ij}/ρ_{ij}}
 */

#include "../core/types.hpp"
#include "../core/country_params.hpp"
#include <vector>

namespace eco_model::production {

/**
 * @brief 计算各部门产出 Y
 * 
 * @param params 国家参数
 * @param X_dom 国内中间品使用矩阵 [n_sectors, n_sectors]
 * @param X_imp 进口中间品使用矩阵 [n_sectors, n_sectors]
 * @param tradable_mask 可贸易部门掩码
 * @return Vector 各部门产出 Y [n_sectors]
 */
Vector compute_output(
    const CountryParams& params,
    const Matrix& X_dom,
    const Matrix& X_imp,
    const std::vector<bool>& tradable_mask
);

}  // namespace eco_model::production
