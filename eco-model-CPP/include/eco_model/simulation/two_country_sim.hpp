#pragma once
/**
 * @file two_country_sim.hpp
 * @brief 两国动态仿真器
 * 
 * 对应 Python: sim.py::TwoCountryDynamicSimulator
 */

#include "../core/types.hpp"
#include "../core/model_params.hpp"
#include "state.hpp"
#include "country_sim.hpp"
#include <vector>
#include <map>
#include <string>
#include <memory>

namespace eco_model::simulation {

/**
 * @brief 政策事件记录
 */
struct PolicyEvent {
    int period;
    std::string type;       // "import_tariff", "export_control", etc.
    std::string country;    // "H" or "F"
    std::map<Index, Scalar> sectors;  // sector -> rate/multiplier
    std::string note;
};

/**
 * @brief 两国动态仿真器
 * 
 * 围绕静态均衡的政策冲击与响应路径。
 */
class TwoCountryDynamicSimulator {
public:
    /**
     * @brief 从均衡解构造仿真器
     * 
     * @param params 模型参数
     * @param home_eq 本国均衡状态
     * @param foreign_eq 外国均衡状态
     * @param theta_price 价格调整速度
     */
    TwoCountryDynamicSimulator(
        const ModelParams& params,
        const CountryState& home_eq,
        const CountryState& foreign_eq,
        Scalar theta_price = 0.05
    );
    
    /// 单步仿真
    void step();
    
    /// 运行多期仿真
    void run(int periods);
    
    // ========================================================================
    // 政策接口
    // ========================================================================
    
    /**
     * @brief 应用进口关税
     * 
     * @param country "H" 或 "F"
     * @param sector_rates {部门索引 -> 关税率}
     */
    void apply_import_tariff(
        const std::string& country,
        const std::map<Index, Scalar>& sector_rates,
        const std::string& note = ""
    );
    
    /**
     * @brief 应用出口管制（配额乘子）
     * 
     * @param country "H" 或 "F"
     * @param sector_factors {部门索引 -> 乘子 ∈ [0,1]}
     */
    void apply_export_control(
        const std::string& country,
        const std::map<Index, Scalar>& sector_factors,
        const std::string& note = ""
    );
    
    /**
     * @brief 重置进口政策
     */
    void reset_import_policies(const std::string& country);
    
    /**
     * @brief 重置出口管制
     */
    void reset_export_control(const std::string& country);
    
    // ========================================================================
    // 访问器
    // ========================================================================
    
    const CountryState& home_state() const { return home_state_; }
    const CountryState& foreign_state() const { return foreign_state_; }
    
    const std::vector<CountryState>& home_history() const { return home_history_; }
    const std::vector<CountryState>& foreign_history() const { return foreign_history_; }
    
    const std::vector<PolicyEvent>& policy_events() const { return policy_events_; }
    
    int current_period() const { return static_cast<int>(home_history_.size()) - 1; }
    
    /**
     * @brief 轻量复制（仅当前状态）
     */
    std::unique_ptr<TwoCountryDynamicSimulator> fork() const;
    
private:
    const ModelParams& params_;
    std::vector<bool> tradable_mask_;
    
    CountryState home_state_;
    CountryState foreign_state_;
    
    std::unique_ptr<CountrySimulator> home_sim_;
    std::unique_ptr<CountrySimulator> foreign_sim_;
    
    std::vector<CountryState> home_history_;
    std::vector<CountryState> foreign_history_;
    
    // 政策状态
    Vector home_import_multiplier_;
    Vector foreign_import_multiplier_;
    Vector home_export_multiplier_;
    Vector foreign_export_multiplier_;
    
    // 基线值（用于重置）
    Vector baseline_home_import_;
    Vector baseline_foreign_import_;
    Vector baseline_home_export_;
    Vector baseline_foreign_export_;
    
    std::vector<PolicyEvent> policy_events_;
    
    // 计算进口价格
    std::pair<Vector, Vector> compute_import_prices() const;
    
    // 更新出口基准
    void update_export_base(const std::string& country);
    
    // 记录政策事件
    void log_policy_event(PolicyEvent event);
};

}  // namespace eco_model::simulation
