/**
 * @file state.cpp
 * @brief CountryState 实现
 */

#include "eco_model/simulation/state.hpp"

namespace eco_model::simulation {

CountryState CountryState::clone() const {
    CountryState copy;
    copy.X_dom = X_dom;
    copy.X_imp = X_imp;
    copy.C_dom = C_dom;
    copy.C_imp = C_imp;
    copy.price = price;
    copy.imp_price = imp_price;
    copy.export_base = export_base;
    copy.export_actual = export_actual;
    copy.output = output;
    copy.income = income;
    return copy;
}

CountryState CountryState::from_equilibrium(
    const Matrix& X_dom,
    const Matrix& X_imp,
    const Vector& C_dom,
    const Vector& C_imp,
    const Vector& prices,
    const Vector& export_base,
    const Vector& output,
    Scalar income
) {
    CountryState state;
    state.X_dom = X_dom;
    state.X_imp = X_imp;
    state.C_dom = C_dom;
    state.C_imp = C_imp;
    state.price = prices;
    state.imp_price = Vector::Ones(prices.size());
    state.export_base = export_base;
    state.export_actual = export_base;
    state.output = output;
    state.income = income;
    return state;
}

}  // namespace eco_model::simulation
