/**
 * @file two_country_sim.cpp
 * @brief TwoCountryDynamicSimulator 实现（骨架）
 * 
 * TODO: 完整实现两国动态仿真逻辑
 */

#include "eco_model/simulation/two_country_sim.hpp"

namespace eco_model::simulation {

TwoCountryDynamicSimulator::TwoCountryDynamicSimulator(
    const ModelParams& params,
    const CountryState& home_eq,
    const CountryState& foreign_eq,
    Scalar theta_price
) : params_(params)
  , tradable_mask_(params.tradable_mask())
  , home_state_(home_eq)
  , foreign_state_(foreign_eq)
{
    // 创建子仿真器
    home_sim_ = std::make_unique<CountrySimulator>(
        params_.home, tradable_mask_, theta_price
    );
    foreign_sim_ = std::make_unique<CountrySimulator>(
        params_.foreign, tradable_mask_, theta_price
    );
    
    // 初始化历史
    home_history_.push_back(home_state_.clone());
    foreign_history_.push_back(foreign_state_.clone());
    
    // 初始化政策乘子
    Index n = params.num_sectors();
    home_import_multiplier_ = params_.home.import_cost;
    foreign_import_multiplier_ = params_.foreign.import_cost;
    home_export_multiplier_ = Vector::Ones(n);
    foreign_export_multiplier_ = Vector::Ones(n);
    
    // 保存基线值
    baseline_home_import_ = home_import_multiplier_;
    baseline_foreign_import_ = foreign_import_multiplier_;
    baseline_home_export_ = home_export_multiplier_;
    baseline_foreign_export_ = foreign_export_multiplier_;
}

std::pair<Vector, Vector> TwoCountryDynamicSimulator::compute_import_prices() const {
    // H 的进口价格 = F 的国内价格 × H 的进口乘子
    Vector price_H = home_import_multiplier_.array() * foreign_state_.price.array();
    // F 的进口价格 = H 的国内价格 × F 的进口乘子
    Vector price_F = foreign_import_multiplier_.array() * home_state_.price.array();
    return {price_H, price_F};
}

void TwoCountryDynamicSimulator::step() {
    auto [import_price_H, import_price_F] = compute_import_prices();
    
    // 计算进口供给上限（基于对方出口）
    // TODO: 实现配额约束
    
    // 同步更新双方
    CountryState new_H = home_sim_->step(home_state_, import_price_H);
    CountryState new_F = foreign_sim_->step(foreign_state_, import_price_F);
    
    // 更新状态
    home_state_ = new_H;
    foreign_state_ = new_F;
    
    // 记录历史
    home_history_.push_back(home_state_.clone());
    foreign_history_.push_back(foreign_state_.clone());
}

void TwoCountryDynamicSimulator::run(int periods) {
    for (int t = 0; t < periods; ++t) {
        step();
    }
}

void TwoCountryDynamicSimulator::apply_import_tariff(
    const std::string& country,
    const std::map<Index, Scalar>& sector_rates,
    const std::string& note
) {
    Vector& multiplier = (country == "H") ? home_import_multiplier_ : foreign_import_multiplier_;
    const Vector& baseline = (country == "H") ? baseline_home_import_ : baseline_foreign_import_;
    
    for (const auto& [sector, rate] : sector_rates) {
        if (sector >= 0 && sector < multiplier.size()) {
            multiplier[sector] = baseline[sector] * (1.0 + rate);
        }
    }
    
    // 记录事件
    PolicyEvent event;
    event.period = current_period();
    event.type = "import_tariff";
    event.country = country;
    event.sectors = sector_rates;
    event.note = note;
    policy_events_.push_back(event);
}

void TwoCountryDynamicSimulator::apply_export_control(
    const std::string& country,
    const std::map<Index, Scalar>& sector_factors,
    const std::string& note
) {
    Vector& multiplier = (country == "H") ? home_export_multiplier_ : foreign_export_multiplier_;
    
    for (const auto& [sector, factor] : sector_factors) {
        if (sector >= 0 && sector < multiplier.size()) {
            multiplier[sector] = std::max(factor, 0.0);
        }
    }
    
    update_export_base(country);
    
    // 记录事件
    PolicyEvent event;
    event.period = current_period();
    event.type = "export_control";
    event.country = country;
    event.sectors = sector_factors;
    event.note = note;
    policy_events_.push_back(event);
}

void TwoCountryDynamicSimulator::update_export_base(const std::string& country) {
    if (country == "H") {
        home_state_.export_base = baseline_home_export_.array() * 
            home_export_multiplier_.array();
    } else {
        foreign_state_.export_base = baseline_foreign_export_.array() * 
            foreign_export_multiplier_.array();
    }
}

void TwoCountryDynamicSimulator::reset_import_policies(const std::string& country) {
    if (country == "H") {
        home_import_multiplier_ = baseline_home_import_;
    } else {
        foreign_import_multiplier_ = baseline_foreign_import_;
    }
}

void TwoCountryDynamicSimulator::reset_export_control(const std::string& country) {
    if (country == "H") {
        home_export_multiplier_ = Vector::Ones(params_.num_sectors());
        home_state_.export_base = baseline_home_export_;
    } else {
        foreign_export_multiplier_ = Vector::Ones(params_.num_sectors());
        foreign_state_.export_base = baseline_foreign_export_;
    }
}

void TwoCountryDynamicSimulator::log_policy_event(PolicyEvent event) {
    policy_events_.push_back(std::move(event));
}

std::unique_ptr<TwoCountryDynamicSimulator> TwoCountryDynamicSimulator::fork() const {
    auto forked = std::make_unique<TwoCountryDynamicSimulator>(
        params_, home_state_.clone(), foreign_state_.clone()
    );
    forked->home_import_multiplier_ = home_import_multiplier_;
    forked->foreign_import_multiplier_ = foreign_import_multiplier_;
    forked->home_export_multiplier_ = home_export_multiplier_;
    forked->foreign_export_multiplier_ = foreign_export_multiplier_;
    return forked;
}

}  // namespace eco_model::simulation
