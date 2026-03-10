#pragma once
/**
 * @file country_sim.hpp
 * @brief 单国仿真器
 * 
 * 对应 Python: sim.py::CountrySimulator
 * 
 * 实现单国的计划—价格—分配—再计算一周期更新。
 */

#include "../core/types.hpp"
#include "../core/country_params.hpp"
#include "state.hpp"
#include <vector>
#include <optional>

namespace eco_model::simulation {

/**
 * @brief 单国动态仿真器
 * 
 * 核心规则（对应 tex Section 2.2）：
 * 1. _plan_demands: 给定价格，计算计划需求
 * 2. _update_prices: 基于供需缺口更新价格（公式 (9)）
 * 3. _allocate_goods: 供给不足时按比例分配（公式 (10)-(11)）
 * 4. _allocate_imports_fx: 外汇约束分配（公式 (14)-(15)）
 */
class CountrySimulator {
public:
    /**
     * @brief 构造仿真器
     * 
     * @param params 国家参数
     * @param tradable_mask 可贸易部门掩码
     * @param theta_price 价格调整速度 τ（默认 0.05）
     */
    CountrySimulator(
        const CountryParams& params,
        const std::vector<bool>& tradable_mask,
        Scalar theta_price = 0.05
    );
    
    /**
     * @brief 单期更新
     * 
     * 计划→价格→国内分配→外贸约束→重算产出/收入
     * 
     * @param state 当前状态
     * @param import_prices 进口价格
     * @param supply_cap 可选的进口供给上限
     * @return CountryState 新状态
     */
    CountryState step(
        const CountryState& state,
        const Vector& import_prices,
        const std::optional<Vector>& supply_cap = std::nullopt
    );
    
private:
    const CountryParams& params_;
    std::vector<bool> tradable_mask_;
    Vector tau_;  // 价格调整速度
    
    /**
     * @brief 计划需求
     * 
     * 对应 tex 公式 (4)-(8), (12)-(13)
     */
    struct PlannedDemands {
        Vector outputs;
        Matrix planned_X_dom;
        Matrix planned_X_imp;
        Vector planned_C_dom;
        Vector planned_C_imp;
    };
    
    PlannedDemands plan_demands(
        const CountryState& state,
        const Vector& import_prices
    );
    
    /**
     * @brief 价格更新
     * 
     * 对应 tex 公式 (9)：
     *   P' = P · exp(τ · (D - Y))
     */
    Vector update_prices(
        const CountryState& state,
        const Vector& outputs,
        const Matrix& planned_X_dom,
        const Vector& planned_C_dom
    );
    
    /**
     * @brief 国内供给分配
     * 
     * 对应 tex 公式 (10)-(11)
     */
    std::tuple<Matrix, Vector, Vector> allocate_goods(
        const Vector& outputs,
        const Matrix& planned_X_dom,
        const Vector& planned_C_dom,
        const Vector& planned_export
    );
    
    /**
     * @brief 外汇约束分配
     * 
     * 对应 tex 公式 (14)-(15)
     */
    std::tuple<Matrix, Vector> allocate_imports_fx(
        const Vector& import_prices,
        const Matrix& planned_X_imp,
        const Vector& planned_C_imp,
        Scalar export_value,
        const std::optional<Vector>& supply_cap
    );
};

}  // namespace eco_model::simulation
