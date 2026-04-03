#include "car_racing.hpp"

NNV_NAMESPACING_START

torch::Tensor CCarRacing::Bound() {
    return torch::tensor({{parameter_.x_bound.first, parameter_.x_bound.second},  {parameter_.y_bound.first, parameter_.y_bound.second}, {parameter_.v_bound.first, parameter_.v_bound.second}}, torch::dtype(torch::kFloat64));
}

torch::Tensor CCarRacing::InitialConstraint() {
    return torch::tensor({{parameter_.x_initial_constraint.first, parameter_.x_initial_constraint.second},  {parameter_.y_initial_constraint.first, parameter_.y_initial_constraint.second}, {parameter_.v_initial_constraint.first, parameter_.v_initial_constraint.second}}, torch::dtype(torch::kFloat64));
}

void CCarRacing::Init(int horizon) {
    // initialize variable for the horizon
    SetHorizon(horizon);
}

void CCarRacing::Reset() {}

void CCarRacing::SetRelationSDT(vector<CSDT>& soft_decision_tree_v, bool initialize) {
    int horizon = StateHorizon();
    GRBModel& convex_solver = solver_.ConvexSolver();
    for (int i = 0; i < horizon - 1; ++i) {
        vector<list<CSolver::SatVar*>>& sat_output_var_vl = soft_decision_tree_v[i].SatOutputVariable();
        vector<GRBVar>& convex_input_var_v = soft_decision_tree_v[i].ConvexInputVariable();
        assert(sat_output_var_vl.size() == 5);
        assert(convex_input_var_v.size() == 3);
        if (initialize) {
            // connest relation for state variable
            for (int j = 0; j < 3; ++j) {
                // cout << j << endl;
                convex_solver.addConstr(time_state_var_vv_[i][j] == convex_input_var_v[j]);
            }
        }
    }
}

void CCarRacing::SetRelationNN(vector<CNN>& neural_network_v) { }

int CCarRacing::CheckNextState(const torch::Tensor& state, int action, const torch::Tensor& next_state) { return {}; }

torch::Tensor CCarRacing::Forward(const torch::Tensor& state, const::torch::Tensor& action) { return {}; }

torch::Tensor CCarRacing::Forward(const torch::Tensor& state, int action) { return {}; }

torch::Tensor CCarRacing::State(int step) {
    torch::Tensor state = torch::empty({2}, torch::dtype(torch::kFloat64));
    state.index_put_({0}, time_state_var_vv_[step][0].get(GRB_DoubleAttr_X));
    state.index_put_({1}, time_state_var_vv_[step][3].get(GRB_DoubleAttr_X));
    return state;
}

int CCarRacing::Action(int step) {
    for (int i = 0; i < (int)time_action_var_vv_[step].size(); ++i) {
        if (solver_.EvalSatVar(time_action_var_vv_[step][i]) == 1) {
            return i;
        }
    }
    assert(false);
}

torch::Tensor CCarRacing::Trajectory() {
    torch::Tensor res = torch::empty({StateHorizon(), 3}, torch::dtype(torch::kFloat64));
    for (int i = 0; i < StateHorizon(); ++i) {
        res.index_put_({i, torch::indexing::Slice(0, 2)}, State(i));
        if (i != StateHorizon() - 1) {
            res.index_put_({i, 2}, Action(i));
        }
        else {
            res.index_put_({i, 2}, 0);
        }
    }
    return res;
}

string CCarRacing::Str() const {
    string res = TimeStateStr() + ActionStateStr();
    return res;
}

string CCarRacing::TimeStateStr() const {
    string res;
    res += "Time State Variable:\n";
    for (int i = 0; i < (int)time_state_var_vv_.size(); ++i) {
        for (int j = 0; j < (int)time_state_var_vv_[i].size(); ++j) {
            res += "\t";
            res += solver_.ConvexVarStr(time_state_var_vv_[i][j]) + '\n';
        }
    };
    return res;
}

string CCarRacing::ActionStateStr() const {
    string res;
    res += "Action State Variable:\n";
    for (int i = 0; i < (int)time_action_var_vv_.size(); ++i) {
        for (int j = 0; j < (int)time_action_var_vv_[i].size(); ++j) {
            res += "\t";
            res += solver_.SatVarStr(time_action_var_vv_[i][j]) + '\n';
        }
    }
    return res;
}

void CCarRacing::SetHorizon(int horizon) {
    GRBModel& model = solver_.ConvexSolver();
    time_state_var_vv_.resize(horizon);
    for (int i = 0; i < horizon; ++i) {
        time_state_var_vv_[i].reserve(3);
        time_state_var_vv_[i].emplace_back(model.addVar(parameter_.x_bound.first, parameter_.x_bound.second, 0, GRB_CONTINUOUS, "x@" + to_string(i)));

        time_state_var_vv_[i].emplace_back(model.addVar(parameter_.y_bound.first, parameter_.y_bound.second, 0, GRB_CONTINUOUS, "y@" + to_string(i)));
        time_state_var_vv_[i].emplace_back(model.addVar(parameter_.v_bound.first, parameter_.v_bound.second, 0, GRB_CONTINUOUS, "v@" + to_string(i)));
    };
    model.update();
    // printf("%s", Str().c_str());
}

void CCarRacing::SetCosineConstraintForSegment(const double& left_end, const double& right_end, bool taylor_is_over_approx, GRBVar& position, GRBVar& position_cos, CSolver::SatVar* var) {
    if (taylor_is_over_approx) {
        var->lin_expr_.emplace_back(position_cos <= CosineTaylorApproximation(position, left_end, right_end));
        var->lin_expr_.emplace_back(position_cos >= CosineTwoPointEquation(position, left_end, right_end));
    }
    else {
        var->lin_expr_.emplace_back(position_cos <= CosineTwoPointEquation(position, left_end, right_end));
        var->lin_expr_.emplace_back(position_cos >= CosineTaylorApproximation(position, left_end, right_end));
    }
}

GRBLinExpr CCarRacing::CosineTwoPointEquation(GRBVar& var, const double& left_end, const double& right_end) {
    return cos(3 * left_end) + (cos(3 * right_end) - cos(3 * left_end)) * (var - left_end) / (right_end - left_end);
}

GRBLinExpr CCarRacing::CosineTaylorApproximation(GRBVar& var, const double& left_end, const double& right_end) {
    return cos(1.5 * (left_end + right_end)) + (-3) * sin(1.5 * (left_end + right_end)) * (var - 0.5 * (left_end + right_end));
}



NNV_NAMESPACING_END
