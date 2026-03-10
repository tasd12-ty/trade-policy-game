/**
 * @file solver.cpp
 * @brief 静态均衡求解器实现
 * 
 * 对应 Python: model.py::solve_initial_equilibrium
 * 
 * 当 Ceres 可用时使用 Ceres LM 求解器，否则使用简化迭代求解。
 */

#include "eco_model/equilibrium/solver.hpp"
#include "eco_model/production/output.hpp"
#include "eco_model/production/cost.hpp"
#include "eco_model/production/income.hpp"
#include "eco_model/armington/ces.hpp"
#include "eco_model/core/math_utils.hpp"

#ifdef HAS_CERES
#include <ceres/ceres.h>
#endif

namespace eco_model::equilibrium {

// ============================================================================
// 优化初始猜测
// ============================================================================

/**
 * @brief 计算优化的初始猜测值
 * 
 * 使用 Leontief 近似和 TFP 信息来获得更好的起点。
 */
static void compute_initial_guess(
    const ModelParams& params,
    Vector& prices_H, Vector& prices_F,
    Matrix& X_dom_H, Matrix& X_imp_H,
    Matrix& X_dom_F, Matrix& X_imp_F,
    Vector& C_dom_H, Vector& C_imp_H,
    Vector& C_dom_F, Vector& C_imp_F
) {
    Index n = params.num_sectors();
    auto tradable_mask = params.tradable_mask();
    
    // 1. 初始价格：基于 TFP 的成本近似 P_i ≈ 1/A_i
    for (Index i = 0; i < n; ++i) {
        prices_H[i] = 1.0 / std::max(params.home.A[i], EPS);
        prices_F[i] = 1.0 / std::max(params.foreign.A[i], EPS);
    }
    
    // 归一化价格使均值为 1
    Scalar mean_H = prices_H.mean();
    Scalar mean_F = prices_F.mean();
    prices_H /= mean_H;
    prices_F /= mean_F;
    
    // 2. 初始产出猜测：基于对称性假设 Y_i = 1
    Vector Y_guess = Vector::Ones(n);
    
    // 3. 初始收入猜测
    Scalar income_guess = 1.0;
    
    // 4. 中间品需求：基于一阶条件
    //    X_{ij} = α_{ij} * P_i * Y_i / P_j
    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < n; ++j) {
            Scalar a_H = params.home.alpha(i, j);
            Scalar a_F = params.foreign.alpha(i, j);
            
            if (a_H > 0) {
                if (tradable_mask[j]) {
                    Scalar theta = params.home.gamma(i, j);
                    X_dom_H(i, j) = a_H * theta * prices_H[i] * Y_guess[i] / prices_H[j];
                    X_imp_H(i, j) = a_H * (1-theta) * prices_H[i] * Y_guess[i] / prices_F[j];
                } else {
                    X_dom_H(i, j) = a_H * prices_H[i] * Y_guess[i] / prices_H[j];
                    X_imp_H(i, j) = 0.0;
                }
            }
            
            if (a_F > 0) {
                if (tradable_mask[j]) {
                    Scalar theta = params.foreign.gamma(i, j);
                    X_dom_F(i, j) = a_F * theta * prices_F[i] * Y_guess[i] / prices_F[j];
                    X_imp_F(i, j) = a_F * (1-theta) * prices_F[i] * Y_guess[i] / prices_H[j];
                } else {
                    X_dom_F(i, j) = a_F * prices_F[i] * Y_guess[i] / prices_F[j];
                    X_imp_F(i, j) = 0.0;
                }
            }
        }
    }
    
    // 5. 消费需求：基于预算份额
    for (Index j = 0; j < n; ++j) {
        Scalar b_H = params.home.beta[j];
        Scalar b_F = params.foreign.beta[j];
        
        if (b_H > 0) {
            if (tradable_mask[j]) {
                Scalar theta = params.home.gamma_cons[j];
                C_dom_H[j] = b_H * theta * income_guess / prices_H[j];
                C_imp_H[j] = b_H * (1-theta) * income_guess / prices_F[j];
            } else {
                C_dom_H[j] = b_H * income_guess / prices_H[j];
                C_imp_H[j] = 0.0;
            }
        }
        
        if (b_F > 0) {
            if (tradable_mask[j]) {
                Scalar theta = params.foreign.gamma_cons[j];
                C_dom_F[j] = b_F * theta * income_guess / prices_F[j];
                C_imp_F[j] = b_F * (1-theta) * income_guess / prices_H[j];
            } else {
                C_dom_F[j] = b_F * income_guess / prices_F[j];
                C_imp_F[j] = 0.0;
            }
        }
    }
}

// ============================================================================
// Ceres 求解器
// ============================================================================

#ifdef HAS_CERES

/**
 * @brief Ceres CostFunctor: 零利润条件残差
 * 
 * 残差 r_i = ln(P_i) - ln(λ_i)
 */
struct ZeroProfitCostFunctor {
    const CountryParams& params_;
    const std::vector<bool>& tradable_mask_;
    const Vector& other_prices_;
    Index n_;
    
    ZeroProfitCostFunctor(const CountryParams& params,
                          const std::vector<bool>& tradable_mask,
                          const Vector& other_prices)
        : params_(params), tradable_mask_(tradable_mask), 
          other_prices_(other_prices), n_(params.num_sectors()) {}
    
    // DynamicNumericDiffCostFunction 要求 operator()(double const* const* parameters, double* residual)
    bool operator()(double const* const* parameters, double* residual) const {
        const double* prices = parameters[0];  // 第一个参数块
        
        Vector p(n_);
        for (Index i = 0; i < n_; ++i) {
            p[i] = prices[i];
        }
        
        // 计算进口价格
        Vector import_prices = params_.import_cost.array() * other_prices_.array();
        
        // 计算边际成本
        Vector lambdas = production::compute_marginal_cost(params_, p, import_prices, tradable_mask_);
        
        // 残差：ln(P) - ln(λ)
        for (Index i = 0; i < n_; ++i) {
            residual[i] = safe_log(p[i]) - safe_log(lambdas[i]);
        }
        
        return true;
    }
};

/**
 * @brief 使用 Ceres 求解均衡
 */
static EquilibriumResult solve_with_ceres(
    const ModelParams& params,
    int max_iterations,
    Scalar tolerance
) {
    EquilibriumResult result;
    Index n = params.num_sectors();
    auto tradable_mask = params.tradable_mask();
    
    // 初始化状态
    Vector prices_H(n), prices_F(n);
    Matrix X_dom_H(n, n), X_imp_H(n, n);
    Matrix X_dom_F(n, n), X_imp_F(n, n);
    Vector C_dom_H(n), C_imp_H(n);
    Vector C_dom_F(n), C_imp_F(n);
    
    // 优化初始猜测
    compute_initial_guess(params, 
        prices_H, prices_F,
        X_dom_H, X_imp_H,
        X_dom_F, X_imp_F,
        C_dom_H, C_imp_H,
        C_dom_F, C_imp_F);
    
    // 转换为 double 数组
    std::vector<double> p_H(n), p_F(n);
    for (Index i = 0; i < n; ++i) {
        p_H[i] = prices_H[i];
        p_F[i] = prices_F[i];
    }
    
    // 交替迭代求解（双方价格相互依赖）
    Scalar total_residual = 1.0;
    int iter = 0;
    
    while (iter < max_iterations && total_residual > tolerance) {
        // 求解 H 国价格（给定 F 国价格）
        ceres::Problem problem_H;
        auto* cost_H = new ceres::DynamicNumericDiffCostFunction<ZeroProfitCostFunctor, ceres::CENTRAL>(
            new ZeroProfitCostFunctor(params.home, tradable_mask, prices_F),
            ceres::TAKE_OWNERSHIP
        );
        cost_H->AddParameterBlock(static_cast<int>(n));
        cost_H->SetNumResiduals(static_cast<int>(n));
        problem_H.AddResidualBlock(cost_H, nullptr, p_H.data());
        
        // 添加下界约束
        for (Index i = 0; i < n; ++i) {
            problem_H.SetParameterLowerBound(p_H.data(), i, EPS);
        }
        
        ceres::Solver::Options options;
        options.max_num_iterations = 50;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        
        ceres::Solver::Summary summary_H;
        ceres::Solve(options, &problem_H, &summary_H);
        
        // 求解 F 国价格（给定 H 国价格）
        Vector new_prices_H(n);
        for (Index i = 0; i < n; ++i) {
            new_prices_H[i] = p_H[i];
        }
        
        ceres::Problem problem_F;
        auto* cost_F = new ceres::DynamicNumericDiffCostFunction<ZeroProfitCostFunctor, ceres::CENTRAL>(
            new ZeroProfitCostFunctor(params.foreign, tradable_mask, new_prices_H),
            ceres::TAKE_OWNERSHIP
        );
        cost_F->AddParameterBlock(static_cast<int>(n));
        cost_F->SetNumResiduals(static_cast<int>(n));
        problem_F.AddResidualBlock(cost_F, nullptr, p_F.data());
        
        for (Index i = 0; i < n; ++i) {
            problem_F.SetParameterLowerBound(p_F.data(), i, EPS);
        }
        
        ceres::Solver::Summary summary_F;
        ceres::Solve(options, &problem_F, &summary_F);
        
        // 更新价格
        Vector new_prices_F(n);
        for (Index i = 0; i < n; ++i) {
            new_prices_F[i] = p_F[i];
        }
        
        // 计算总残差
        total_residual = (new_prices_H - prices_H).norm() + (new_prices_F - prices_F).norm();
        
        prices_H = new_prices_H;
        prices_F = new_prices_F;
        
        ++iter;
    }
    
    // 更新中间品和消费量（使用一阶条件）
    Vector Y_H = production::compute_output(params.home, X_dom_H, X_imp_H, tradable_mask);
    Vector Y_F = production::compute_output(params.foreign, X_dom_F, X_imp_F, tradable_mask);
    Scalar income_H = production::compute_income(params.home, prices_H, Y_H);
    Scalar income_F = production::compute_income(params.foreign, prices_F, Y_F);
    
    // 重新计算中间品需求
    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < n; ++j) {
            Scalar a = params.home.alpha(i, j);
            if (a > 0) {
                if (tradable_mask[j]) {
                    Scalar theta = params.home.gamma(i, j);
                    X_dom_H(i, j) = a * theta * prices_H[i] * Y_H[i] / prices_H[j];
                    X_imp_H(i, j) = a * (1-theta) * prices_H[i] * Y_H[i] / prices_F[j];
                } else {
                    X_dom_H(i, j) = a * prices_H[i] * Y_H[i] / prices_H[j];
                }
            }
            
            Scalar a_F = params.foreign.alpha(i, j);
            if (a_F > 0) {
                if (tradable_mask[j]) {
                    Scalar theta = params.foreign.gamma(i, j);
                    X_dom_F(i, j) = a_F * theta * prices_F[i] * Y_F[i] / prices_F[j];
                    X_imp_F(i, j) = a_F * (1-theta) * prices_F[i] * Y_F[i] / prices_H[j];
                } else {
                    X_dom_F(i, j) = a_F * prices_F[i] * Y_F[i] / prices_F[j];
                }
            }
        }
    }
    
    // 重新计算消费需求
    for (Index j = 0; j < n; ++j) {
        Scalar b = params.home.beta[j];
        if (b > 0) {
            if (tradable_mask[j]) {
                Scalar theta = params.home.gamma_cons[j];
                C_dom_H[j] = b * theta * income_H / prices_H[j];
                C_imp_H[j] = b * (1-theta) * income_H / prices_F[j];
            } else {
                C_dom_H[j] = b * income_H / prices_H[j];
            }
        }
        
        Scalar b_F = params.foreign.beta[j];
        if (b_F > 0) {
            if (tradable_mask[j]) {
                Scalar theta = params.foreign.gamma_cons[j];
                C_dom_F[j] = b_F * theta * income_F / prices_F[j];
                C_imp_F[j] = b_F * (1-theta) * income_F / prices_H[j];
            } else {
                C_dom_F[j] = b_F * income_F / prices_F[j];
            }
        }
    }
    
    // 重新计算最终产出和收入
    Y_H = production::compute_output(params.home, X_dom_H, X_imp_H, tradable_mask);
    Y_F = production::compute_output(params.foreign, X_dom_F, X_imp_F, tradable_mask);
    income_H = production::compute_income(params.home, prices_H, Y_H);
    income_F = production::compute_income(params.foreign, prices_F, Y_F);
    
    // 提取出口基准
    Vector export_H = X_imp_F.colwise().sum().transpose() + C_imp_F;
    Vector export_F = X_imp_H.colwise().sum().transpose() + C_imp_H;
    
    result.converged = (total_residual <= tolerance);
    result.iterations = iter;
    result.final_residual = total_residual;
    result.message = result.converged ? "Ceres converged" : "Ceres max iterations reached";
    
    result.home_state = simulation::CountryState::from_equilibrium(
        X_dom_H, X_imp_H, C_dom_H, C_imp_H, prices_H, export_H, Y_H, income_H
    );
    result.foreign_state = simulation::CountryState::from_equilibrium(
        X_dom_F, X_imp_F, C_dom_F, C_imp_F, prices_F, export_F, Y_F, income_F
    );
    
    return result;
}

#endif  // HAS_CERES

// ============================================================================
// 简化迭代求解器（回退）
// ============================================================================

static EquilibriumResult solve_iterative(
    const ModelParams& params,
    int max_iterations,
    Scalar tolerance
) {
    EquilibriumResult result;
    Index n = params.num_sectors();
    auto tradable_mask = params.tradable_mask();
    
    // 初始化状态
    Vector prices_H(n), prices_F(n);
    Matrix X_dom_H(n, n), X_imp_H(n, n);
    Matrix X_dom_F(n, n), X_imp_F(n, n);
    Vector C_dom_H(n), C_imp_H(n);
    Vector C_dom_F(n), C_imp_F(n);
    
    // 优化初始猜测
    compute_initial_guess(params, 
        prices_H, prices_F,
        X_dom_H, X_imp_H,
        X_dom_F, X_imp_F,
        C_dom_H, C_imp_H,
        C_dom_F, C_imp_F);
    
    // 简化迭代
    Scalar residual = 1.0;
    int iter = 0;
    
    while (iter < max_iterations && residual > tolerance) {
        // 计算产出
        Vector Y_H = production::compute_output(params.home, X_dom_H, X_imp_H, tradable_mask);
        Vector Y_F = production::compute_output(params.foreign, X_dom_F, X_imp_F, tradable_mask);
        
        // 计算边际成本
        Vector lambda_H = production::compute_marginal_cost(
            params.home, prices_H, params.home.import_cost.array() * prices_F.array(), tradable_mask
        );
        Vector lambda_F = production::compute_marginal_cost(
            params.foreign, prices_F, params.foreign.import_cost.array() * prices_H.array(), tradable_mask
        );
        
        // 更新价格（向边际成本收敛）
        Vector new_prices_H = 0.9 * prices_H + 0.1 * lambda_H;
        Vector new_prices_F = 0.9 * prices_F + 0.1 * lambda_F;
        
        // 计算残差
        residual = (new_prices_H - prices_H).norm() + (new_prices_F - prices_F).norm();
        
        prices_H = new_prices_H;
        prices_F = new_prices_F;
        
        // 更新中间品需求
        Scalar income_H_val = production::compute_income(params.home, prices_H, Y_H);
        Scalar income_F_val = production::compute_income(params.foreign, prices_F, Y_F);
        
        for (Index i = 0; i < n; ++i) {
            for (Index j = 0; j < n; ++j) {
                Scalar a = params.home.alpha(i, j);
                if (a > 0) {
                    if (tradable_mask[j]) {
                        Scalar theta = params.home.gamma(i, j);
                        X_dom_H(i, j) = a * theta * prices_H[i] * Y_H[i] / prices_H[j];
                        X_imp_H(i, j) = a * (1-theta) * prices_H[i] * Y_H[i] / prices_F[j];
                    } else {
                        X_dom_H(i, j) = a * prices_H[i] * Y_H[i] / prices_H[j];
                    }
                }
                
                Scalar a_F = params.foreign.alpha(i, j);
                if (a_F > 0) {
                    if (tradable_mask[j]) {
                        Scalar theta = params.foreign.gamma(i, j);
                        X_dom_F(i, j) = a_F * theta * prices_F[i] * Y_F[i] / prices_F[j];
                        X_imp_F(i, j) = a_F * (1-theta) * prices_F[i] * Y_F[i] / prices_H[j];
                    } else {
                        X_dom_F(i, j) = a_F * prices_F[i] * Y_F[i] / prices_F[j];
                    }
                }
            }
        }
        
        for (Index j = 0; j < n; ++j) {
            Scalar b = params.home.beta[j];
            if (b > 0) {
                if (tradable_mask[j]) {
                    Scalar theta = params.home.gamma_cons[j];
                    C_dom_H[j] = b * theta * income_H_val / prices_H[j];
                    C_imp_H[j] = b * (1-theta) * income_H_val / prices_F[j];
                } else {
                    C_dom_H[j] = b * income_H_val / prices_H[j];
                }
            }
            
            Scalar b_F = params.foreign.beta[j];
            if (b_F > 0) {
                if (tradable_mask[j]) {
                    Scalar theta = params.foreign.gamma_cons[j];
                    C_dom_F[j] = b_F * theta * income_F_val / prices_F[j];
                    C_imp_F[j] = b_F * (1-theta) * income_F_val / prices_H[j];
                } else {
                    C_dom_F[j] = b_F * income_F_val / prices_F[j];
                }
            }
        }
        
        ++iter;
    }
    
    // 构建结果
    result.converged = (residual <= tolerance);
    result.iterations = iter;
    result.final_residual = residual;
    result.message = result.converged ? "Iterative converged" : "Iterative max iterations reached";
    
    // 计算最终产出和收入
    Vector Y_H = production::compute_output(params.home, X_dom_H, X_imp_H, tradable_mask);
    Vector Y_F = production::compute_output(params.foreign, X_dom_F, X_imp_F, tradable_mask);
    Scalar income_H = production::compute_income(params.home, prices_H, Y_H);
    Scalar income_F = production::compute_income(params.foreign, prices_F, Y_F);
    
    // 提取出口基准
    Vector export_H = X_imp_F.colwise().sum().transpose() + C_imp_F;
    Vector export_F = X_imp_H.colwise().sum().transpose() + C_imp_H;
    
    result.home_state = simulation::CountryState::from_equilibrium(
        X_dom_H, X_imp_H, C_dom_H, C_imp_H, prices_H, export_H, Y_H, income_H
    );
    result.foreign_state = simulation::CountryState::from_equilibrium(
        X_dom_F, X_imp_F, C_dom_F, C_imp_F, prices_F, export_F, Y_F, income_F
    );
    
    return result;
}

// ============================================================================
// 公共接口
// ============================================================================

EquilibriumResult solve_initial_equilibrium(
    const ModelParams& params,
    int max_iterations,
    Scalar tolerance
) {
#ifdef HAS_CERES
    return solve_with_ceres(params, max_iterations, tolerance);
#else
    return solve_iterative(params, max_iterations, tolerance);
#endif
}

}  // namespace eco_model::equilibrium
