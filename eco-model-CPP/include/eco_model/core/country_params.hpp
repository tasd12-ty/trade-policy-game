#pragma once
/**
 * @file country_params.hpp
 * @brief 国家参数结构体
 * 
 * 对应 Python: model.py::CountryParams dataclass
 * 
 * 字段含义（与 production_network_simulation0916.tex 对应）：
 * - alpha[i,j]: 生产函数中部门 i 对部门 j 中间投入的产出弹性 α_{ij}
 * - gamma[i,j]: 可贸易部门 j 的 Armington 国内权重 γ_{ij}（0..1）
 * - rho[i,j]: 可贸易部门的 Armington 形状参数 ρ_{ij}（σ = 1/(1-ρ)）
 * - beta[j]: 消费层面的品类 j 的预算份额 β_j
 * - A[i]: 部门 i 的全要素生产率 A_i
 * - exports[j]: 基础出口量 Export_j
 * - gamma_cons[j], rho_cons[j]: 消费层面 Armington 参数 γ_{cj}, ρ_{cj}
 * - import_cost[j]: 进口附加成本（含关税/运输）
 */

#include "types.hpp"

namespace eco_model {

/**
 * @brief 单国参数块
 * 
 * 所有参数在初始化后不可变（const-like 语义）。
 * 使用 create_symmetric() 构造对称基线参数。
 */
struct CountryParams {
    /// 生产函数投入弹性矩阵 [n_sectors, n_sectors]
    /// α_{ij}: 部门 i 对部门 j 中间投入的产出弹性
    Matrix alpha;
    
    /// Armington 国内权重矩阵 [n_sectors, n_sectors]
    /// γ_{ij}: 部门 i 使用部门 j 产品时的本国权重
    Matrix gamma;
    
    /// Armington 形状参数矩阵 [n_sectors, n_sectors]
    /// ρ_{ij}: 与替代弹性 σ = 1/(1-ρ) 相关
    Matrix rho;
    
    /// 消费预算份额向量 [n_sectors]
    /// β_j: 消费者在部门 j 产品上的支出份额
    Vector beta;
    
    /// 全要素生产率向量 [n_sectors]
    /// A_i: 部门 i 的技术水平
    Vector A;
    
    /// 基础出口向量 [n_sectors]
    /// Export_j: 外生的初始出口量
    Vector exports;
    
    /// 消费层面 Armington 国内权重 [n_sectors]
    /// γ_{cj}: 消费者对部门 j 产品的本国偏好权重
    Vector gamma_cons;
    
    /// 消费层面 Armington 形状参数 [n_sectors]
    /// ρ_{cj}: 消费层面的替代弹性参数
    Vector rho_cons;
    
    /// 进口附加成本系数 [n_sectors]
    /// 包含关税、运输成本等，乘以对方价格得到进口价格
    Vector import_cost;
    
    // ========================================================================
    // 构造与查询
    // ========================================================================
    
    /// 获取部门数量
    [[nodiscard]] Index num_sectors() const { return alpha.rows(); }
    
    /**
     * @brief 创建对称基线参数
     * 
     * 对应 Python: model.py::create_symmetric_parameters()
     * 
     * @param n_sectors 部门数量（默认 6）
     * @param tradable_sectors 可贸易部门索引列表
     * @return CountryParams 对称参数
     */
    static CountryParams create_symmetric(
        Index n_sectors = DEFAULT_NUM_SECTORS,
        const std::vector<Index>& tradable_sectors = {2, 3, 4, 5}
    );
};

}  // namespace eco_model
