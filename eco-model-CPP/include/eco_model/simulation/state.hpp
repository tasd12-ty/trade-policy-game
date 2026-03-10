#pragma once
/**
 * @file state.hpp
 * @brief 国家状态快照
 * 
 * 对应 Python: sim.py::CountryState
 */

#include "../core/types.hpp"

namespace eco_model::simulation {

/**
 * @brief 单国状态快照
 * 
 * 包含一个时期内的所有经济状态变量。
 */
struct CountryState {
    /// 国内中间品使用 [n_sectors, n_sectors]
    /// X_dom[i,j]: 部门 i 使用部门 j 的国内产品
    Matrix X_dom;
    
    /// 进口中间品使用 [n_sectors, n_sectors]
    /// X_imp[i,j]: 部门 i 进口部门 j 的产品
    Matrix X_imp;
    
    /// 国内消费 [n_sectors]
    /// C_dom[j]: 对部门 j 国内产品的消费
    Vector C_dom;
    
    /// 进口消费 [n_sectors]
    /// C_imp[j]: 对部门 j 进口产品的消费
    Vector C_imp;
    
    /// 国内价格 [n_sectors]
    Vector price;
    
    /// 进口价格（含关税）[n_sectors]
    Vector imp_price;
    
    /// 基础出口量 [n_sectors]
    Vector export_base;
    
    /// 实际出口量 [n_sectors]
    Vector export_actual;
    
    /// 各部门产出 [n_sectors]
    Vector output;
    
    /// 国民收入
    Scalar income;
    
    // ========================================================================
    // 方法
    // ========================================================================
    
    /// 获取部门数量
    [[nodiscard]] Index num_sectors() const { return price.size(); }
    
    /// 深拷贝
    [[nodiscard]] CountryState clone() const;
    
    /// 创建初始状态（从均衡解）
    static CountryState from_equilibrium(
        const Matrix& X_dom,
        const Matrix& X_imp,
        const Vector& C_dom,
        const Vector& C_imp,
        const Vector& prices,
        const Vector& export_base,
        const Vector& output,
        Scalar income
    );
};

}  // namespace eco_model::simulation
