/**
 * @file test_armington.cpp
 * @brief Armington CES 函数单元测试
 */

#include <gtest/gtest.h>
#include "eco_model/armington/ces.hpp"
#include "eco_model/armington/smooth_ops.hpp"

using namespace eco_model;
using namespace eco_model::armington;

TEST(ArmingtonTest, ShareBasic) {
    // 等价格时，份额等于 gamma
    Scalar theta = share(0.5, 1.0, 1.0, 0.2);
    EXPECT_NEAR(theta, 0.5, 0.01);
}

TEST(ArmingtonTest, SharePriceDominance) {
    // 国内价格更低时，份额应更高
    Scalar theta_low = share(0.5, 0.5, 1.0, 0.5);
    Scalar theta_high = share(0.5, 1.5, 1.0, 0.5);
    EXPECT_GT(theta_low, theta_high);
}

TEST(ArmingtonTest, ShareBoundaryRhoZero) {
    // ρ→0 (Cobb-Douglas) 时，份额等于 gamma
    Scalar theta = share(0.7, 0.8, 1.2, 0.001);
    EXPECT_NEAR(theta, 0.7, 0.05);
}

TEST(ArmingtonTest, ShareBoundaryRhoOne) {
    // ρ→1 (完全替代) 时，选择更低价格
    Scalar theta = share(0.5, 0.8, 1.0, 0.99);
    EXPECT_GT(theta, 0.8);  // 国内价格更低，份额更高
}

TEST(ArmingtonTest, PriceBasic) {
    // 使用 ρ=0（Cobb-Douglas），等价格时对偶价格等于该价格
    Scalar p = price(0.5, 1.0, 1.0, 0.0);
    EXPECT_NEAR(p, 1.0, 0.01);
}

TEST(ArmingtonTest, PriceBetween) {
    // 对偶价格应介于两个价格之间（使用 ρ=0 Cobb-Douglas）
    Scalar p = price(0.5, 0.5, 1.5, 0.0);
    EXPECT_GT(p, 0.5);
    EXPECT_LT(p, 1.5);
}

TEST(ArmingtonTest, QuantityBasic) {
    // 基本物量合成
    Scalar q = quantity(0.5, 1.0, 1.0, 0.3, 0.5);
    EXPECT_GT(q, 0.0);
}

TEST(ArmingtonTest, QuantityZeroAlpha) {
    // α = 0 时返回 1
    Scalar q = quantity(0.5, 1.0, 1.0, 0.0, 0.5);
    EXPECT_NEAR(q, 1.0, EPS);
}

TEST(SmoothOpsTest, SmoothMax) {
    // smooth_max 应接近 max
    Scalar result = smooth::max(1.0, 2.0, 50.0);
    EXPECT_NEAR(result, 2.0, 0.1);
}

TEST(SmoothOpsTest, SmoothMin) {
    // smooth_min 应接近 min
    Scalar result = smooth::min(1.0, 2.0, 50.0);
    EXPECT_NEAR(result, 1.0, 0.1);
}

TEST(SmoothOpsTest, SmoothStep) {
    // smooth_step 的基本性质
    EXPECT_GT(smooth::step(1.0), 0.9);
    EXPECT_LT(smooth::step(-1.0), 0.1);
    EXPECT_NEAR(smooth::step(0.0), 0.5, 0.01);
}
