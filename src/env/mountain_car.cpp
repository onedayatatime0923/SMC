#include "mountain_car.hpp"

NNV_NAMESPACING_START

torch::Tensor CMountainCar::Bound() {
    return torch::tensor({{parameter_.position_bound.first, parameter_.position_bound.second}, {parameter_.velocity_bound.first, parameter_.velocity_bound.second}}, torch::dtype(torch::kFloat64));
}

torch::Tensor CMountainCar::InitialConstraint() {
    return torch::tensor({{parameter_.position_initial_constraint.first, parameter_.position_initial_constraint.second}, {parameter_.velocity_initial_constraint.first, parameter_.velocity_initial_constraint.second}}, torch::dtype(torch::kFloat64));
}

void CMountainCar::Init(int horizon) {
    // initialize variable for the horizon
    SetHorizon(horizon);

    // for Sat
    SetWaveConstraint();
    // construct the relation from action to the next state
    SetStateRelationConstraint();
    SetBoundaryConstraint();

    // for convex
    // initialize initial constraint
    SetInitialConstraint();
    // construct safty constraint
    SetSafeConstraint();
}

void CMountainCar::Reset() {
    SetWaveConstraint();
    // construct the relation from action to the next state
    ResetStateRelationConstraint();
    ResetBoundaryConstraint();
}

void CMountainCar::SetInitialConstraint() {
    GRBModel& model = solver_.ConvexSolver();
    model.addConstr(time_state_var_vv_[0][0] >= parameter_.position_initial_constraint.first, "initial_constr_0");
    model.addConstr(time_state_var_vv_[0][0] <= parameter_.position_initial_constraint.second, "initial_constr_1");
    model.addConstr(time_state_var_vv_[0][3] >= parameter_.velocity_initial_constraint.first, "initial_constr_2");
    model.addConstr(time_state_var_vv_[0][3] <= parameter_.velocity_initial_constraint.second, "initial_constr_3");
    model.update();
}

void CMountainCar::ClearInitialConstraint() {
    GRBModel& model = solver_.ConvexSolver();
    model.remove(model.getConstrByName("initial_constr_0"));
    model.remove(model.getConstrByName("initial_constr_1"));
    model.remove(model.getConstrByName("initial_constr_2"));
    model.remove(model.getConstrByName("initial_constr_3"));
    model.update();
}

void CMountainCar::SetSafeConstraint() {
    GRBModel& model = solver_.ConvexSolver();
    for (int i = 0; i < (int)time_state_var_vv_.size(); ++i) {
        model.addConstr(time_state_var_vv_[i][0] <= parameter_.safe_constraint, "safe_constr_" + to_string(i));
    }
    model.update();
}

void CMountainCar::ClearSafeConstraint() {
    GRBModel& model = solver_.ConvexSolver();
    for (int i = 0; i < (int)time_state_var_vv_.size(); ++i) {
        model.remove(model.getConstrByName("safe_constr_" + to_string(i)));
    }
    model.update();
}

void CMountainCar::SetRelationSDT(vector<CSDT>& soft_decision_tree_v, bool initialize) {
    int horizon = StateHorizon();
    GRBModel& convex_solver = solver_.ConvexSolver();
    for (int i = 0; i < horizon - 1; ++i) {
        vector<list<CSolver::SatVar*>>& sat_output_var_vl = soft_decision_tree_v[i].SatOutputVariable();
        vector<GRBVar>& convex_input_var_v = soft_decision_tree_v[i].ConvexInputVariable();
        assert(time_action_var_vv_[i].size() == 3);
        assert(sat_output_var_vl.size() == 3);
        assert(convex_input_var_v.size() == 2);
        if (initialize) {
            // connest relation for state variable
            for (int j = 0; j < 2; ++j) {
                convex_solver.addConstr(time_state_var_vv_[i][j * 3] == convex_input_var_v[j]);
            }
        }
        // connest relation for action variable
        for (int j = 0; j < 3; ++j) {
            z3::expr constr = !CSolver::SatExpr(time_action_var_vv_[i][j]);
            for (auto it = sat_output_var_vl[j].begin(); it != sat_output_var_vl[j].end(); ++it) {
                constr = constr or CSolver::SatExpr(*it);
            }
            solver_.AddSatConstraint(constr);
        }
    }
}

void CMountainCar::SetRelationNN(vector<CNN>& neural_network_v) {
    int horizon = StateHorizon();
    GRBModel& convex_solver = solver_.ConvexSolver();
    for (int i = 0; i < horizon - 1; ++i) {
        vector<vector<GRBVar> >& convex_network_var_v = neural_network_v[i].ConvexNetworkVariable();
        assert(time_action_var_vv_[i].size() == 3);
        assert(convex_network_var_v[0].size() == 2);
        assert(convex_network_var_v.back().size() == 3);
        // connest relation for state variable
        for (int j = 0; j < 2; ++j) {
            convex_solver.addConstr(time_state_var_vv_[i][j * 3] == convex_network_var_v[0][j]);
        }
        // connest relation for action variable
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                if (j != k) {
                    time_action_var_vv_[i][j]->lin_expr_.emplace_back(convex_network_var_v.back()[j] - convex_network_var_v.back()[k] >= 0);
                }
            }
        }
    }
}

int CMountainCar::CheckNextState(const torch::Tensor& state, int action, const torch::Tensor& next_state) {
    torch::Tensor actual_state = Forward(state, action);
    if (torch::max(torch::abs(next_state - actual_state)).item<double>() > parameter_.perception_disturbance + solver_.parameter_.kEPS) {
        return 0;
    }
    else {
        return 1;
    }
}

torch::Tensor CMountainCar::Forward(const torch::Tensor& state, const::torch::Tensor& action) {
    int action_i = action.index({0}).item<int>();
    return Forward(state, action_i);
}

torch::Tensor CMountainCar::Forward(const torch::Tensor& state, int action) {
    torch::Tensor res = torch::empty({2}, torch::dtype(torch::kFloat64));
    torch::Tensor velocity = torch::clamp(state.index({1}) + 0.001 * (action - 1) + -0.0025 * torch::cos(3 * state.index({0})), parameter_.velocity_bound.first, parameter_.velocity_bound.second);
    torch::Tensor position = torch::clamp(state.index({0}) + velocity, parameter_.position_bound.first, parameter_.position_bound.second);
    res.index_put_({1}, velocity);
    res.index_put_({0}, position);
    return res;
}

torch::Tensor CMountainCar::State(int step) {
    torch::Tensor state = torch::empty({2}, torch::dtype(torch::kFloat64));
    state.index_put_({0}, time_state_var_vv_[step][0].get(GRB_DoubleAttr_X));
    state.index_put_({1}, time_state_var_vv_[step][3].get(GRB_DoubleAttr_X));
    return state;
}

int CMountainCar::Action(int step) {
    for (int i = 0; i < (int)time_action_var_vv_[step].size(); ++i) {
        if (solver_.EvalSatVar(time_action_var_vv_[step][i]) == 1) {
            return i;
        }
    }
    assert(false);
}

torch::Tensor CMountainCar::Trajectory() {
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

string CMountainCar::Str() const {
    string res = TimeStateStr() + ActionStateStr();
    return res;
}

string CMountainCar::TimeStateStr() const {
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

string CMountainCar::ActionStateStr() const {
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

void CMountainCar::SetHorizon(int horizon) {
    GRBModel& model = solver_.ConvexSolver();
    time_state_var_vv_.resize(horizon);
    for (int i = 0; i < horizon; ++i) {
        time_state_var_vv_[i].reserve(5);
        time_state_var_vv_[i].emplace_back(model.addVar(parameter_.position_bound.first, parameter_.position_bound.second, 0, GRB_CONTINUOUS, "pos@" + to_string(i)));
        time_state_var_vv_[i].emplace_back(model.addVar(-1, 1, 0, GRB_CONTINUOUS, "pos_cos@" + to_string(i)));
        time_state_var_vv_[i].emplace_back(model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "pos_unbound@" + to_string(i)));

        time_state_var_vv_[i].emplace_back(model.addVar(parameter_.velocity_bound.first, parameter_.velocity_bound.second, 0, GRB_CONTINUOUS, "vel@" + to_string(i)));
        time_state_var_vv_[i].emplace_back(model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "vel_unbound@" + to_string(i)));
    };
    model.update();
    // printf("%s", Str().c_str());
}

void CMountainCar::SetWaveConstraint() {
    ClearWaveConstraint();
    // use max(1, time_state_var_vv_.size() - 1) to at least set wave condition once for CheckApproximationMountainCar
    for (int i = 0; i < max(1, (int)time_state_var_vv_.size() - 1); ++i) {
        GRBVar& position = time_state_var_vv_[i][0];
        GRBVar& position_cos = time_state_var_vv_[i][1];

        double left_end, right_end;
        CSolver::SatVar* var;
        bool constr_f = false;
        z3::expr constr(solver_.SatContex());
        for (int j = 0; j < wave_approx_param_; ++j) {
            // set position_cos@t for -pi/2 < position@t < -pi/3
            left_end = -M_PI / 2 + (M_PI / 6 * j / wave_approx_param_);
            right_end = -M_PI / 2 + (M_PI / 6 * (j + 1) / wave_approx_param_);
            var = solver_.AddSatVar("cos_s" + to_string(j) + "p0@" + to_string(i));
            var->lin_expr_.emplace_back(position >= left_end);
            var->lin_expr_.emplace_back(position <= right_end);
            SetCosineConstraintForSegment(left_end, right_end, false, position, position_cos, var);
            wave_constraint_l_.emplace_back(var);
            solver_.SetSatOrConstraint(var, constr_f, constr);

            // set position_cos@t for -pi/3 < position@t < -pi/6
            left_end = -M_PI / 3 + (M_PI / 6 * j / wave_approx_param_);
            right_end = -M_PI / 3 + (M_PI / 6 * (j + 1) / wave_approx_param_);
            var = solver_.AddSatVar("cos_s" + to_string(j) + "p1@" + to_string(i));
            var->lin_expr_.emplace_back(position >= left_end);
            var->lin_expr_.emplace_back(position <= right_end);
            SetCosineConstraintForSegment(left_end, right_end, false, position, position_cos, var);
            wave_constraint_l_.emplace_back(var);
            solver_.SetSatOrConstraint(var, constr_f, constr);

            // set position_cos@t for -pi/6 < position@t < 0
            left_end = -M_PI / 6 + (M_PI / 6 * j / wave_approx_param_);
            right_end = -M_PI / 6 + (M_PI / 6 * (j + 1) / wave_approx_param_);
            var = solver_.AddSatVar("cos_s" + to_string(j) + "p2@" + to_string(i));
            var->lin_expr_.emplace_back(position >= left_end);
            var->lin_expr_.emplace_back(position <= right_end);
            SetCosineConstraintForSegment(left_end, right_end, true, position, position_cos, var);
            wave_constraint_l_.emplace_back(var);
            solver_.SetSatOrConstraint(var, constr_f, constr);

            // set position_cos@t for 0 < position@t < pi/6
            left_end = (M_PI / 6 * j / wave_approx_param_);
            right_end = (M_PI / 6 * (j + 1) / wave_approx_param_);
            var = solver_.AddSatVar("cos_s" + to_string(j) + "p3@" + to_string(i));
            var->lin_expr_.emplace_back(position >= left_end);
            var->lin_expr_.emplace_back(position <= right_end);
            SetCosineConstraintForSegment(left_end, right_end, true, position, position_cos, var);
            wave_constraint_l_.emplace_back(var);
            solver_.SetSatOrConstraint(var, constr_f, constr);

            // set position_cos@t for pi/6 < position@t < pi/3
            left_end = M_PI / 6 + (M_PI / 6 * j / wave_approx_param_);
            right_end = M_PI / 6 + (M_PI / 6 * (j + 1) / wave_approx_param_);
            var = solver_.AddSatVar("cos_s" + to_string(j) + "p4@" + to_string(i));
            var->lin_expr_.emplace_back(position >= left_end);
            var->lin_expr_.emplace_back(position <= right_end);
            SetCosineConstraintForSegment(left_end, right_end, false, position, position_cos, var);
            wave_constraint_l_.emplace_back(var);
            solver_.SetSatOrConstraint(var, constr_f, constr);
        }
        solver_.AddSatConstraint(constr);
    }
    solver_.ConvexSolver().update();
}

void CMountainCar::ClearWaveConstraint() {
    for (auto it = wave_constraint_l_.begin(); it != wave_constraint_l_.end(); ++it) {
        CSolver::SatVar* v = *it;
        solver_.RemoveSatVar(v);
    }
    wave_constraint_l_.clear();
}

void CMountainCar::SetCosineConstraintForSegment(const double& left_end, const double& right_end, bool taylor_is_over_approx, GRBVar& position, GRBVar& position_cos, CSolver::SatVar* var) {
    if (taylor_is_over_approx) {
        var->lin_expr_.emplace_back(position_cos <= CosineTaylorApproximation(position, left_end, right_end));
        var->lin_expr_.emplace_back(position_cos >= CosineTwoPointEquation(position, left_end, right_end));
    }
    else {
        var->lin_expr_.emplace_back(position_cos <= CosineTwoPointEquation(position, left_end, right_end));
        var->lin_expr_.emplace_back(position_cos >= CosineTaylorApproximation(position, left_end, right_end));
    }
}

GRBLinExpr CMountainCar::CosineTwoPointEquation(GRBVar& var, const double& left_end, const double& right_end) {
    return cos(3 * left_end) + (cos(3 * right_end) - cos(3 * left_end)) * (var - left_end) / (right_end - left_end);
}

GRBLinExpr CMountainCar::CosineTaylorApproximation(GRBVar& var, const double& left_end, const double& right_end) {
    return cos(1.5 * (left_end + right_end)) + (-3) * sin(1.5 * (left_end + right_end)) * (var - 0.5 * (left_end + right_end));
}

void CMountainCar::SetStateRelationConstraint() {
    time_action_var_vv_.resize(time_state_var_vv_.size() - 1);

    GRBModel& model = solver_.ConvexSolver();
    CSolver::SatVar* var;
    for (int i = 0; i < (int)time_state_var_vv_.size() - 1; ++i) {
        time_action_var_vv_[i].reserve(3);

        GRBVar& position = time_state_var_vv_[i][0];
        GRBVar& position_cos = time_state_var_vv_[i][1];
        GRBVar& velocity = time_state_var_vv_[i][3];
        GRBVar& next_position = time_state_var_vv_[i + 1][2];
        GRBVar& next_velocity = time_state_var_vv_[i + 1][4];

        model.addConstr(next_position >= position + next_velocity - parameter_.perception_disturbance);
        model.addConstr(next_position <= position + next_velocity + parameter_.perception_disturbance);

        var = solver_.AddSatVar("action_0@" + to_string(i));
        var->lin_expr_.emplace_back(next_velocity == velocity - 0.001 + -0.0025 * position_cos);
        time_action_var_vv_[i].emplace_back(var);
        z3::expr constr = var->var_;

        var = solver_.AddSatVar("action_1@" + to_string(i));
        var->lin_expr_.emplace_back(next_velocity == velocity + -0.0025 * position_cos);
        time_action_var_vv_[i].emplace_back(var);
        constr = constr | var->var_;

        var = solver_.AddSatVar("action_2@" + to_string(i));
        var->lin_expr_.emplace_back(next_velocity == velocity + 0.001 + -0.0025 * position_cos);
        time_action_var_vv_[i].emplace_back(var);
        constr = constr | var->var_;

        solver_.AddSatConstraint(constr);
    };
    solver_.ConvexSolver().update();
}

void CMountainCar::ResetStateRelationConstraint() {
    for (int i = 0; i < (int)time_state_var_vv_.size() - 1; ++i) {
        solver_.AddSatConstraint(time_action_var_vv_[i][0]->var_ | time_action_var_vv_[i][1]->var_ | time_action_var_vv_[i][2]->var_);
    };
    solver_.ConvexSolver().update();
}

void CMountainCar::SetBoundaryConstraint() {
    time_boundary_var_vv_.resize(time_state_var_vv_.size());

    CSolver::SatVar* var;
    for (int i = 1; i < (int)time_state_var_vv_.size(); ++i) {
        time_boundary_var_vv_[i].reserve(6);

        GRBVar& position = time_state_var_vv_[i][0];
        GRBVar& position_unbound = time_state_var_vv_[i][2];
        GRBVar& velocity = time_state_var_vv_[i][3];
        GRBVar& velocity_unbound = time_state_var_vv_[i][4];

        var = solver_.AddSatVar("unbound0@" + to_string(i));
        var->lin_expr_.emplace_back(position_unbound <= parameter_.position_bound.first);
        var->lin_expr_.emplace_back(position == parameter_.position_bound.first);
        time_boundary_var_vv_[i].emplace_back(var);
        z3::expr constr = var->var_;

        var = solver_.AddSatVar("unbound1@" + to_string(i));
        var->lin_expr_.emplace_back(position_unbound == position);
        time_boundary_var_vv_[i].emplace_back(var);
        constr = constr | var->var_;

        var = solver_.AddSatVar("unbound2@" + to_string(i));
        var->lin_expr_.emplace_back(position_unbound >= parameter_.position_bound.second);
        var->lin_expr_.emplace_back(position == parameter_.position_bound.second);
        time_boundary_var_vv_[i].emplace_back(var);
        constr = constr | var->var_;

        solver_.AddSatConstraint(constr);

        var = solver_.AddSatVar("unbound3@" + to_string(i));
        var->lin_expr_.emplace_back(velocity_unbound <= parameter_.velocity_bound.first);
        var->lin_expr_.emplace_back(velocity == parameter_.velocity_bound.first);
        time_boundary_var_vv_[i].emplace_back(var);
        constr = var->var_;

        var = solver_.AddSatVar("unbound4@" + to_string(i));
        var->lin_expr_.emplace_back(velocity_unbound == velocity);
        time_boundary_var_vv_[i].emplace_back(var);
        constr = constr | var->var_;

        var = solver_.AddSatVar("unbound5@" + to_string(i));
        var->lin_expr_.emplace_back(velocity_unbound >= parameter_.velocity_bound.second);
        var->lin_expr_.emplace_back(velocity == parameter_.velocity_bound.second);
        time_boundary_var_vv_[i].emplace_back(var);
        constr = constr | var->var_;

        solver_.AddSatConstraint(constr);
    }
    solver_.ConvexSolver().update();
}

void CMountainCar::ResetBoundaryConstraint() {
    for (int i = 1; i < (int)time_state_var_vv_.size(); ++i) {
        solver_.AddSatConstraint(time_boundary_var_vv_[i][0]->var_ | time_boundary_var_vv_[i][1]->var_ | time_boundary_var_vv_[i][2]->var_);
        solver_.AddSatConstraint(time_boundary_var_vv_[i][3]->var_ | time_boundary_var_vv_[i][4]->var_ | time_boundary_var_vv_[i][5]->var_);
    };
    solver_.ConvexSolver().update();
}

NNV_NAMESPACING_END
