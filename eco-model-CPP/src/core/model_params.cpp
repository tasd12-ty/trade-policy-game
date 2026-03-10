/**
 * @file model_params.cpp
 * @brief ModelParams 实现
 */

#include "eco_model/core/model_params.hpp"
#include <algorithm>

namespace eco_model {

bool ModelParams::is_tradable(Index sector) const {
    return std::find(tradable_idx.begin(), tradable_idx.end(), sector) 
           != tradable_idx.end();
}

std::vector<bool> ModelParams::tradable_mask() const {
    std::vector<bool> mask(num_sectors(), false);
    for (Index idx : tradable_idx) {
        if (idx >= 0 && idx < num_sectors()) {
            mask[idx] = true;
        }
    }
    return mask;
}

ModelParams ModelParams::create_symmetric(
    Index n_sectors,
    const std::vector<Index>& tradable_sectors
) {
    ModelParams params;
    
    // 创建对称的两国参数
    params.home = CountryParams::create_symmetric(n_sectors, tradable_sectors);
    params.foreign = CountryParams::create_symmetric(n_sectors, tradable_sectors);
    
    // 设置可贸易/不可贸易索引
    std::vector<bool> mask(n_sectors, false);
    for (Index j : tradable_sectors) {
        if (j >= 0 && j < n_sectors) {
            mask[j] = true;
            params.tradable_idx.push_back(j);
        }
    }
    
    for (Index j = 0; j < n_sectors; ++j) {
        if (!mask[j]) {
            params.non_tradable_idx.push_back(j);
        }
    }
    
    // 排序索引
    std::sort(params.tradable_idx.begin(), params.tradable_idx.end());
    std::sort(params.non_tradable_idx.begin(), params.non_tradable_idx.end());
    
    return params;
}

}  // namespace eco_model
