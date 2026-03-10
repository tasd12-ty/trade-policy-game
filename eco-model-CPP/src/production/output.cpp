/**
 * @file output.cpp
 * @brief 生产函数实现
 * 
 * 对应 Python: model.py::compute_output
 */

#include "eco_model/production/output.hpp"
#include "eco_model/armington/ces.hpp"
#include "eco_model/core/math_utils.hpp"

namespace eco_model::production {

Vector compute_output(
    const CountryParams& params,
    const Matrix& X_dom,
    const Matrix& X_imp,
    const std::vector<bool>& tradable_mask
) {
    Index n = params.num_sectors();
    Vector Y(n);
    
    for (Index i = 0; i < n; ++i) {
        // 从 TFP 开始
        Scalar prod = std::max(params.A[i], EPS);
        
        for (Index j = 0; j < n; ++j) {
            Scalar a = params.alpha(i, j);
            
            // 跳过零投入弹性
            if (a <= 0.0) {
                continue;
            }
            
            if (tradable_mask[j]) {
                // 可贸易部门：使用 Armington 物量合成
                Scalar qty = armington::quantity(
                    params.gamma(i, j),
                    X_dom(i, j),
                    X_imp(i, j),
                    a,
                    params.rho(i, j)
                );
                prod *= std::max(qty, EPS);
            } else {
                // 不可贸易部门：直接使用国内投入
                Scalar comp = std::max(X_dom(i, j), EPS);
                prod *= std::pow(comp, a);
            }
        }
        
        Y[i] = std::max(prod, EPS);
    }
    
    return Y;
}

}  // namespace eco_model::production
