/**
 * @file income.cpp
 * @brief 收入计算实现
 * 
 * 对应 Python: model.py::compute_income, value_added_share
 */

#include "eco_model/production/income.hpp"
#include "eco_model/core/math_utils.hpp"

namespace eco_model::production {

Vector value_added_share(const CountryParams& params) {
    // v_i = 1 - Σ_j α_{ij}
    Vector row_sum = params.alpha.rowwise().sum();
    Vector va = Vector::Ones(params.num_sectors()) - row_sum;
    
    // 确保正值
    return va.cwiseMax(1e-6);
}

Scalar compute_income(
    const CountryParams& params,
    const Vector& prices,
    const Vector& outputs
) {
    // I = Σ_i P_i Y_i v_i
    Vector va = value_added_share(params);
    return (prices.array() * outputs.array() * va.array()).sum();
}

}  // namespace eco_model::production
