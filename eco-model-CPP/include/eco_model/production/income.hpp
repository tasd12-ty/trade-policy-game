#pragma once
/**
 * @file income.hpp
 * @brief 国民收入计算
 * 
 * 对应 Python: model.py::compute_income
 * 
 * 公式（对应 tex 公式 (4), (28)）：
 *   I = Σ_i P_i Y_i v_i
 * 
 * 其中 v_i = 1 - Σ_j α_{ij} 为增加值份额（近似要素收入）
 */

#include "../core/types.hpp"
#include "../core/country_params.hpp"

namespace eco_model::production {

/**
 * @brief 计算增加值份额
 * 
 * v_i = 1 - Σ_j α_{ij}
 * 
 * @param params 国家参数
 * @return Vector 各部门增加值份额 [n_sectors]
 */
Vector value_added_share(const CountryParams& params);

/**
 * @brief 计算国民收入（GDP 近似）
 * 
 * I = Σ_i P_i Y_i v_i
 * 
 * @param params 国家参数
 * @param prices 国内价格 [n_sectors]
 * @param outputs 各部门产出 [n_sectors]
 * @return Scalar 国民收入
 */
Scalar compute_income(
    const CountryParams& params,
    const Vector& prices,
    const Vector& outputs
);

}  // namespace eco_model::production
