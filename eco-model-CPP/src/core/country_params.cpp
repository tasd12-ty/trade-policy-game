/**
 * @file country_params.cpp
 * @brief CountryParams 实现
 */

#include "eco_model/core/country_params.hpp"

namespace eco_model {

CountryParams CountryParams::create_symmetric(
    Index n_sectors,
    const std::vector<Index>& tradable_sectors
) {
    CountryParams params;
    
    // 初始化矩阵和向量
    params.alpha = Matrix::Constant(n_sectors, n_sectors, 0.15);
    params.gamma = Matrix::Constant(n_sectors, n_sectors, 0.5);
    params.rho = Matrix::Constant(n_sectors, n_sectors, 0.2);
    params.beta = Vector::Constant(n_sectors, 1.0 / n_sectors);
    params.A = Vector::Ones(n_sectors);
    params.exports = Vector::Zero(n_sectors);
    params.gamma_cons = Vector::Constant(n_sectors, 0.5);
    params.rho_cons = Vector::Constant(n_sectors, 0.2);
    params.import_cost = Vector::Ones(n_sectors);
    
    // 对角线投入弹性设为 0（部门不使用自己的产品作为投入）
    params.alpha.diagonal().setZero();
    
    // 创建可贸易掩码
    std::vector<bool> tradable_mask(n_sectors, false);
    for (Index j : tradable_sectors) {
        if (j >= 0 && j < n_sectors) {
            tradable_mask[j] = true;
        }
    }
    
    // 不可贸易部门的 gamma 设为 1
    for (Index i = 0; i < n_sectors; ++i) {
        for (Index j = 0; j < n_sectors; ++j) {
            if (!tradable_mask[j] || i == j) {
                params.gamma(i, j) = 1.0;
            }
        }
        if (!tradable_mask[i]) {
            params.gamma_cons(i) = 1.0;
        }
    }
    
    return params;
}

}  // namespace eco_model
