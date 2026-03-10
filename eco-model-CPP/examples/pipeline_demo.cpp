/**
 * @file pipeline_demo.cpp
 * @brief 仿真流程演示程序
 * 
 * 对应 Python: analysis/pipeline_demo.py
 */

#include <iostream>
#include <iomanip>

#include "eco_model/core/model_params.hpp"
#include "eco_model/equilibrium/solver.hpp"
#include "eco_model/simulation/two_country_sim.hpp"

using namespace eco_model;
using namespace eco_model::equilibrium;
using namespace eco_model::simulation;

int main() {
    std::cout << "=== 1. 设置与静态均衡 ===\n";
    
    // 创建对称参数
    auto params = ModelParams::create_symmetric(6, {2, 3, 4, 5});
    std::cout << "部门数量: " << params.num_sectors() << "\n";
    std::cout << "可贸易部门: ";
    for (auto idx : params.tradable_idx) {
        std::cout << idx << " ";
    }
    std::cout << "\n";
    
    // 求解均衡（使用较宽松的容差，因为简化求解器精度有限）
    std::cout << "\n正在求解初始均衡...\n";
    auto eq_result = solve_initial_equilibrium(params, 1000, 1e-4);
    
    std::cout << "收敛状态: " << (eq_result.converged ? "成功" : "失败") << "\n";
    std::cout << "最终残差: " << std::scientific << eq_result.final_residual << "\n";
    std::cout << "迭代次数: " << eq_result.iterations << "\n";
    
    if (!eq_result.converged) {
        std::cerr << "错误: 初始均衡未能收敛。\n";
        return 1;
    }
    
    // 创建仿真器
    std::cout << "\n=== 2. 初始化动态仿真器 ===\n";
    TwoCountryDynamicSimulator sim(
        params, eq_result.home_state, eq_result.foreign_state, 0.05
    );
    
    std::cout << "H 国初始收入: " << std::fixed << std::setprecision(4) 
              << sim.home_state().income << "\n";
    std::cout << "F 国初始收入: " << sim.foreign_state().income << "\n";
    
    // 预热
    std::cout << "\n=== 3. 预热阶段 (100 步) ===\n";
    for (int t = 0; t < 100; ++t) {
        sim.step();
        if ((t + 1) % 20 == 0) {
            std::cout << "  [预热] 步 " << (t + 1) 
                      << ": H均价=" << sim.home_state().price.mean() << "\n";
        }
    }
    
    // 政策冲击
    std::cout << "\n=== 4. 政策冲击 ===\n";
    std::cout << "  动作: H 对 F 部门 2/3 加征 20% 关税\n";
    sim.apply_import_tariff("H", {{2, 0.2}, {3, 0.2}}, "H对F部门2/3加征20%关税");
    
    // 动态仿真
    std::cout << "\n=== 5. 动态仿真 (100 步) ===\n";
    for (int t = 0; t < 100; ++t) {
        sim.step();
        if ((t + 1) % 20 == 0) {
            Scalar h_income = sim.home_state().income;
            Scalar f_income = sim.foreign_state().income;
            
            // 计算贸易余额
            Scalar export_val = (sim.home_state().export_actual.array() * 
                                sim.home_state().price.array()).sum();
            Scalar import_val = (sim.home_state().imp_price.array() * 
                                (sim.home_state().X_imp.colwise().sum().transpose() + 
                                 sim.home_state().C_imp).array()).sum();
            
            std::cout << "  [动态] 步 " << (t + 1) << std::fixed << std::setprecision(4)
                      << ": H收入=" << h_income
                      << ", F收入=" << f_income
                      << ", H贸易余额=" << (export_val - import_val) << "\n";
        }
    }
    
    std::cout << "\n=== 6. 结果汇总 ===\n";
    std::cout << "总仿真步数: " << sim.current_period() << "\n";
    std::cout << "政策事件数: " << sim.policy_events().size() << "\n";
    
    std::cout << "\n演示结束。\n";
    return 0;
}
