/**
 * @file country_sim.cpp
 * @brief CountrySimulator 实现（骨架）
 * 
 * TODO: 完整实现动态仿真逻辑
 */

#include "eco_model/simulation/country_sim.hpp"
#include "eco_model/production/output.hpp"
#include "eco_model/production/cost.hpp"
#include "eco_model/production/income.hpp"
#include "eco_model/armington/ces.hpp"
#include "eco_model/core/math_utils.hpp"

namespace eco_model::simulation {

CountrySimulator::CountrySimulator(
    const CountryParams& params,
    const std::vector<bool>& tradable_mask,
    Scalar theta_price
) : params_(params)
  , tradable_mask_(tradable_mask)
  , tau_(Vector::Constant(params.num_sectors(), theta_price))
{
}

CountrySimulator::PlannedDemands CountrySimulator::plan_demands(
    const CountryState& state,
    const Vector& import_prices
) {
    PlannedDemands result;
    Index n = params_.num_sectors();
    
    // 计算当前产出
    result.outputs = production::compute_output(
        params_, state.X_dom, state.X_imp, tradable_mask_
    );
    
    // 计算边际成本
    Vector lambdas = production::compute_marginal_cost(
        params_, state.price, import_prices, tradable_mask_
    );
    
    // 计算 theta（国内外份额）
    Matrix theta_prod = armington::theta_from_usage(
        params_.gamma, params_.rho, state.X_dom, state.X_imp
    );
    Vector theta_cons = armington::theta_from_usage(
        params_.gamma_cons, params_.rho_cons, state.C_dom, state.C_imp
    );
    
    // 初始化计划需求
    result.planned_X_dom = Matrix::Zero(n, n);
    result.planned_X_imp = Matrix::Zero(n, n);
    result.planned_C_dom = Vector::Zero(n);
    result.planned_C_imp = Vector::Zero(n);
    
    // 中间品需求调整（简化版：直接使用当前值）
    for (Index i = 0; i < n; ++i) {
        Scalar Y_i = result.outputs[i];
        Scalar lambda_i = lambdas[i];
        
        for (Index j = 0; j < n; ++j) {
            Scalar a = params_.alpha(i, j);
            if (a <= 0.0) continue;
            
            if (tradable_mask_[j]) {
                Scalar theta = theta_prod(i, j);
                result.planned_X_dom(i, j) = a * theta * lambda_i * Y_i / 
                    std::max(state.price[j], EPS);
                result.planned_X_imp(i, j) = a * (1.0 - theta) * lambda_i * Y_i / 
                    std::max(import_prices[j], EPS);
            } else {
                result.planned_X_dom(i, j) = a * lambda_i * Y_i / 
                    std::max(state.price[j], EPS);
            }
        }
    }
    
    // 消费需求调整
    for (Index j = 0; j < n; ++j) {
        Scalar b = params_.beta[j];
        if (b <= 0.0) continue;
        
        if (tradable_mask_[j]) {
            Scalar theta = theta_cons[j];
            result.planned_C_dom[j] = b * theta * state.income / 
                std::max(state.price[j], EPS);
            result.planned_C_imp[j] = b * (1.0 - theta) * state.income / 
                std::max(import_prices[j], EPS);
        } else {
            result.planned_C_dom[j] = b * state.income / 
                std::max(state.price[j], EPS);
        }
    }
    
    return result;
}

Vector CountrySimulator::update_prices(
    const CountryState& state,
    const Vector& outputs,
    const Matrix& planned_X_dom,
    const Vector& planned_C_dom
) {
    // 计算总需求
    Vector planned_total = planned_X_dom.colwise().sum().transpose() + 
                           planned_C_dom + state.export_base;
    
    // 供需缺口
    Vector demand_gap = planned_total - outputs;
    
    // 价格更新：P' = P · exp(τ · gap)
    Vector delta = tau_.array() * demand_gap.array();
    return state.price.array() * delta.array().exp();
}

std::tuple<Matrix, Vector, Vector> CountrySimulator::allocate_goods(
    const Vector& outputs,
    const Matrix& planned_X_dom,
    const Vector& planned_C_dom,
    const Vector& planned_export
) {
    // 总需求
    Vector demand_total = (planned_X_dom.colwise().sum().transpose() + 
                          planned_C_dom + planned_export).cwiseMax(EPS);
    Vector supply = outputs.cwiseMax(EPS);
    
    // 分配比例
    Vector ratio = supply.array() / demand_total.array();
    Vector scale = ratio.cwiseMin(1.0);
    
    // 应用分配
    Matrix actual_X_dom = planned_X_dom.array().rowwise() * scale.transpose().array();
    Vector actual_C_dom = planned_C_dom.array() * scale.array();
    Vector actual_export = planned_export.array() * scale.array();
    
    return {actual_X_dom, actual_C_dom, actual_export};
}

std::tuple<Matrix, Vector> CountrySimulator::allocate_imports_fx(
    const Vector& import_prices,
    const Matrix& planned_X_imp,
    const Vector& planned_C_imp,
    Scalar export_value,
    const std::optional<Vector>& supply_cap
) {
    // 计划进口价值
    Scalar planned_value = (planned_X_imp.array() * 
        import_prices.transpose().replicate(planned_X_imp.rows(), 1).array()).sum() +
        (planned_C_imp.array() * import_prices.array()).sum();
    
    // 外汇约束缩放
    Scalar scale_fx = (planned_value > EPS) ? 
        std::min(export_value / planned_value, 1.0) : 1.0;
    
    Matrix scaled_X = planned_X_imp * scale_fx;
    Vector scaled_C = planned_C_imp * scale_fx;
    
    // 可选供给上限约束
    if (supply_cap.has_value()) {
        Vector total = scaled_X.colwise().sum().transpose() + scaled_C;
        Vector cap = supply_cap.value().cwiseMax(0.0);
        Vector scale_cap = (cap.array() / total.cwiseMax(EPS).array()).cwiseMin(1.0);
        
        scaled_X = scaled_X.array().rowwise() * scale_cap.transpose().array();
        scaled_C = scaled_C.array() * scale_cap.array();
    }
    
    return {scaled_X, scaled_C};
}

CountryState CountrySimulator::step(
    const CountryState& state,
    const Vector& import_prices,
    const std::optional<Vector>& supply_cap
) {
    // 1. 计划需求
    auto planned = plan_demands(state, import_prices);
    
    // 2. 价格更新
    Vector new_price = update_prices(
        state, planned.outputs, planned.planned_X_dom, planned.planned_C_dom
    );
    
    // 3. 国内分配
    auto [actual_X_dom, actual_C_dom, actual_export] = allocate_goods(
        planned.outputs, planned.planned_X_dom, planned.planned_C_dom, state.export_base
    );
    
    // 4. 出口价值
    Scalar export_value = (actual_export.array() * state.price.array()).sum();
    
    // 5. 外汇约束分配
    auto [actual_X_imp, actual_C_imp] = allocate_imports_fx(
        import_prices, planned.planned_X_imp, planned.planned_C_imp, 
        export_value, supply_cap
    );
    
    // 6. 重新计算产出和收入
    Vector new_output = production::compute_output(
        params_, actual_X_dom, actual_X_imp, tradable_mask_
    );
    Scalar new_income = production::compute_income(params_, new_price, new_output);
    
    // 7. 构建新状态
    return CountryState::from_equilibrium(
        actual_X_dom, actual_X_imp, actual_C_dom, actual_C_imp,
        new_price, state.export_base, new_output, new_income
    );
}

}  // namespace eco_model::simulation
