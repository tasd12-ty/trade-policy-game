#pragma once
/**
 * @file model_params.hpp
 * @brief 模型顶层参数结构体
 * 
 * 对应 Python: model.py::ModelParams dataclass
 */

#include "country_params.hpp"
#include <vector>

namespace eco_model {

/**
 * @brief 两国模型参数
 * 
 * 包含 home/foreign 两国参数块，以及可贸易/不可贸易部门索引。
 */
struct ModelParams {
    /// 本国参数
    CountryParams home;
    
    /// 外国参数
    CountryParams foreign;
    
    /// 可贸易部门索引列表（对应 tex 中 j ∈ {M_l+1, ..., N_l}）
    std::vector<Index> tradable_idx;
    
    /// 不可贸易部门索引列表（对应 tex 中 j ∈ {1, ..., M_l}）
    std::vector<Index> non_tradable_idx;
    
    // ========================================================================
    // 查询方法
    // ========================================================================
    
    /// 获取部门数量
    [[nodiscard]] Index num_sectors() const { return home.num_sectors(); }
    
    /// 获取可贸易部门数量
    [[nodiscard]] Index num_tradable() const { 
        return static_cast<Index>(tradable_idx.size()); 
    }
    
    /// 检查部门是否可贸易
    [[nodiscard]] bool is_tradable(Index sector) const;
    
    /// 获取可贸易掩码向量 (bool)
    [[nodiscard]] std::vector<bool> tradable_mask() const;
    
    /**
     * @brief 创建对称的两国参数
     * 
     * 对应 Python: model.py::create_symmetric_parameters() 
     *            + normalize_model_params()
     * 
     * @param n_sectors 部门数量
     * @param tradable_sectors 可贸易部门索引
     * @return ModelParams 对称两国参数
     */
    static ModelParams create_symmetric(
        Index n_sectors = DEFAULT_NUM_SECTORS,
        const std::vector<Index>& tradable_sectors = {2, 3, 4, 5}
    );
};

}  // namespace eco_model
