/**
 * @file test_production.cpp
 * @brief 生产函数单元测试
 */

#include <gtest/gtest.h>
#include "eco_model/production/output.hpp"
#include "eco_model/production/cost.hpp"
#include "eco_model/production/income.hpp"
#include "eco_model/core/country_params.hpp"

using namespace eco_model;
using namespace eco_model::production;

class ProductionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建对称参数
        params = CountryParams::create_symmetric(6, {2, 3, 4, 5});
        tradable_mask = {false, false, true, true, true, true};
        
        // 初始化中间品矩阵
        X_dom = Matrix::Constant(6, 6, 0.1);
        X_imp = Matrix::Constant(6, 6, 0.05);
        prices = Vector::Ones(6);
    }
    
    CountryParams params;
    std::vector<bool> tradable_mask;
    Matrix X_dom;
    Matrix X_imp;
    Vector prices;
};

TEST_F(ProductionTest, ComputeOutputPositive) {
    Vector Y = compute_output(params, X_dom, X_imp, tradable_mask);
    
    // 所有产出应为正
    for (Index i = 0; i < 6; ++i) {
        EXPECT_GT(Y[i], 0.0);
    }
}

TEST_F(ProductionTest, ComputeOutputWithTFP) {
    // 提高 TFP 应增加产出
    CountryParams high_tfp = params;
    high_tfp.A = Vector::Constant(6, 2.0);
    
    Vector Y_base = compute_output(params, X_dom, X_imp, tradable_mask);
    Vector Y_high = compute_output(high_tfp, X_dom, X_imp, tradable_mask);
    
    for (Index i = 0; i < 6; ++i) {
        EXPECT_GT(Y_high[i], Y_base[i]);
    }
}

TEST_F(ProductionTest, ComputeMarginalCostPositive) {
    Vector import_prices = params.import_cost;
    Vector lambdas = compute_marginal_cost(params, prices, import_prices, tradable_mask);
    
    // 所有边际成本应为正
    for (Index i = 0; i < 6; ++i) {
        EXPECT_GT(lambdas[i], 0.0);
    }
}

TEST_F(ProductionTest, ValueAddedShareValid) {
    Vector va = value_added_share(params);
    
    // 增加值份额应在 (0, 1) 之间
    for (Index i = 0; i < 6; ++i) {
        EXPECT_GT(va[i], 0.0);
        EXPECT_LT(va[i], 1.0);
    }
}

TEST_F(ProductionTest, ComputeIncomePositive) {
    Vector Y = compute_output(params, X_dom, X_imp, tradable_mask);
    Scalar income = compute_income(params, prices, Y);
    
    EXPECT_GT(income, 0.0);
}
