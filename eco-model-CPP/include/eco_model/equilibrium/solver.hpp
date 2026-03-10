#pragma once
/**
 * @file solver.hpp
 * @brief 静态均衡求解器
 * 
 * 对应 Python: model.py::solve_initial_equilibrium
 * 
 * 使用 Levenberg-Marquardt 算法求解非线性最小二乘问题。
 * 残差方程对应 tex Section 2.5 的均衡条件。
 */

#include "../core/types.hpp"
#include "../core/model_params.hpp"
#include "../simulation/state.hpp"

namespace eco_model::equilibrium {

/**
 * @brief 均衡求解结果
 */
struct EquilibriumResult {
    /// 是否收敛
    bool converged = false;
    
    /// 迭代次数
    int iterations = 0;
    
    /// 最终残差范数
    Scalar final_residual = 0.0;
    
    /// 求解器消息
    std::string message;
    
    /// 本国均衡状态
    simulation::CountryState home_state;
    
    /// 外国均衡状态
    simulation::CountryState foreign_state;
};

/**
 * @brief 求解两国静态初始均衡
 * 
 * 对应 Python: model.py::solve_initial_equilibrium
 * 
 * 均衡条件（对应 tex Section 2.5）：
 * 1. 零利润：λ_i = P_i
 * 2. 产品市场出清：D = Y
 * 3. 国际收支平衡：出口收入 = 进口支出
 * 4. 中间品需求一致：P_j X_{ij} = α_{ij} P_i Y_i
 * 5. 消费需求一致：P_j C_j = β_j I
 * 
 * @param params 模型参数
 * @param max_iterations 最大迭代次数（默认 400）
 * @param tolerance 收敛容差（默认 1e-8）
 * @return EquilibriumResult 求解结果
 */
EquilibriumResult solve_initial_equilibrium(
    const ModelParams& params,
    int max_iterations = 400,
    Scalar tolerance = 1e-8
);

}  // namespace eco_model::equilibrium
