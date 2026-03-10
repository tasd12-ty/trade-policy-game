/**
 * @file cost.cpp
 * @brief 边际成本实现
 * 
 * 对应 Python: model.py::compute_marginal_cost
 */

#include "eco_model/production/cost.hpp"
#include "eco_model/armington/ces.hpp"
#include "eco_model/core/math_utils.hpp"

namespace eco_model::production {

Vector compute_marginal_cost(
    const CountryParams& params,
    const Vector& prices,
    const Vector& import_prices,
    const std::vector<bool>& tradable_mask
) {
    Index n = params.num_sectors();
    Vector lambdas(n);
    
    for (Index i = 0; i < n; ++i) {
        // log λ_i = -log A_i + Σ_j α_{ij} log P_j^*
        Scalar log_cost = -safe_log(params.A[i]);
        
        for (Index j = 0; j < n; ++j) {
            Scalar a = params.alpha(i, j);
            
            if (a <= 0.0) {
                continue;
            }
            
            if (tradable_mask[j]) {
                // P_j^* = Armington 对偶价格
                Scalar p_idx = armington::price(
                    params.gamma(i, j),
                    prices[j],
                    import_prices[j],
                    params.rho(i, j)
                );
                log_cost += a * safe_log(p_idx);
            } else {
                // P_j^* = P_j（国内价格）
                log_cost += a * safe_log(prices[j]);
            }
        }
        
        lambdas[i] = std::exp(log_cost);
    }
    
    return lambdas;
}

}  // namespace eco_model::production
