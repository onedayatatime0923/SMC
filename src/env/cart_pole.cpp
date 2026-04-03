#include "cart_pole.hpp"

NNV_NAMESPACING_START

torch::Tensor CCartPole::Bound() {
    return torch::tensor({{parameter_.position_bound.first, parameter_.position_bound.second}, {-GRB_INFINITY, GRB_INFINITY}, {parameter_.angle_bound.first, parameter_.angle_bound.second}, {-GRB_INFINITY, GRB_INFINITY}}, torch::dtype(torch::kFloat64));
}

torch::Tensor CCartPole::InitialConstraint() {
    return torch::tensor({{parameter_.position_initial_constraint.first, parameter_.position_initial_constraint.second}, {0.0, 0.0}, {parameter_.angle_initial_constraint.first, parameter_.angle_initial_constraint.second}, {0.0, 0.0}}, torch::dtype(torch::kFloat64));
}

void CCartPole::Init(int horizon) {
    // initialize variable for the horizon
    SetHorizon(horizon);

    // for Sat
    SetWaveConstraint();
    // construct the relation from action to the next state
    SetStateRelationConstraint();

    // for convex
    // initialize initial constraint
    SetInitialConstraint();
    // construct safty constraint
    SetSafeConstraint();
}

void CCartPole::Reset() {
    SetWaveConstraint();
    // construct the relation from action to the next state
    ResetStateRelationConstraint();
}

void CCartPole::SetInitialConstraint() {
    GRBModel& model = solver_.ConvexSolver();
    model.addConstr(time_state_var_vv_[0][0] >= parameter_.position_initial_constraint.first, "initial_constr_0");
    model.addConstr(time_state_var_vv_[0][0] <= parameter_.position_initial_constraint.second, "initial_constr_1");
    model.addConstr(time_state_var_vv_[0][1] == 0, "initial_constr_2");
    model.addConstr(time_state_var_vv_[0][2] >= parameter_.angle_initial_constraint.first, "initial_constr_3");
    model.addConstr(time_state_var_vv_[0][2] <= parameter_.angle_initial_constraint.second, "initial_constr_4");
    model.addConstr(time_state_var_vv_[0][3] == 0, "initial_constr_5");
    model.update();
}

void CCartPole::ClearInitialConstraint() {
    GRBModel& model = solver_.ConvexSolver();
    model.remove(model.getConstrByName("initial_constr_0"));
    model.remove(model.getConstrByName("initial_constr_1"));
    model.remove(model.getConstrByName("initial_constr_2"));
    model.remove(model.getConstrByName("initial_constr_3"));
    model.remove(model.getConstrByName("initial_constr_4"));
    model.remove(model.getConstrByName("initial_constr_5"));
    model.update();
}

void CCartPole::SetSafeConstraint() {
    GRBModel& model = solver_.ConvexSolver();
    int horizon = StateHorizon();
    GRBVar abs_ang = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "absolute_ang@" + to_string(horizon - 1));
    objective_constr_ = model.addGenConstrAbs(abs_ang, time_state_var_vv_[horizon - 1][2], "safe_constr_0");
    model.addConstr(abs_ang >= parameter_.safe_constraint, "safe_constr_1");
    model.update();
}

void CCartPole::ClearSafeConstraint() {
    GRBModel& model = solver_.ConvexSolver();
    solver_.WriteConvexSolver();
    int horizon = StateHorizon();
    model.remove(model.getVarByName("absolute_ang@" + to_string(horizon - 1)));
    model.remove(objective_constr_);
    model.remove(model.getConstrByName("safe_constr_1"));
    model.update();
}

void CCartPole::SetRelationSDT(vector<CSDT>& soft_decision_tree_v, bool initialize) {
    int horizon = StateHorizon();
    GRBModel& convex_solver = solver_.ConvexSolver();
    for (int i = 0; i < horizon - 1; ++i) {
        vector<list<CSolver::SatVar*>>& sat_output_var_vl = soft_decision_tree_v[i].SatOutputVariable();
        vector<GRBVar>& convex_input_var_v = soft_decision_tree_v[i].ConvexInputVariable();
        assert(time_action_var_vv_[i].size() == 2);
        assert(sat_output_var_vl.size() == 2);
        assert(convex_input_var_v.size() == 4);
        if (initialize) {
            // connest relation for state variable
            for (int j = 0; j < 4; ++j) {
                convex_solver.addConstr(time_state_var_vv_[i][j] == convex_input_var_v[j]);
            }
        }
        // connest relation for action variable
        for (int j = 0; j < 2; ++j) {
            z3::expr constr = !CSolver::SatExpr(time_action_var_vv_[i][j]);
            for (auto it = sat_output_var_vl[j].begin(); it != sat_output_var_vl[j].end(); ++it) {
                constr = constr or CSolver::SatExpr(*it);
            }
            solver_.AddSatConstraint(constr);
        }
    }
}

void CCartPole::SetRelationNN(vector<CNN>& neural_network_v) {
    int horizon = StateHorizon();
    GRBModel& convex_solver = solver_.ConvexSolver();
    for (int i = 0; i < horizon - 1; ++i) {
        vector<vector<GRBVar> >& convex_network_var_v = neural_network_v[i].ConvexNetworkVariable();
        assert(time_action_var_vv_[i].size() == 2);
        assert(convex_network_var_v[0].size() == 4);
        assert(convex_network_var_v.back().size() == 2);
        // connest relation for state variable
        for (int j = 0; j < 4; ++j) {
            convex_solver.addConstr(time_state_var_vv_[i][j] == convex_network_var_v[0][j]);
        }
        // connest relation for action variable
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                if (j != k) {
                    time_action_var_vv_[i][j]->lin_expr_.emplace_back(convex_network_var_v.back()[j] - convex_network_var_v.back()[k] >= 0);
                }
            }
        }
    }
}

int CCartPole::CheckNextState(const torch::Tensor& state, int action, const torch::Tensor& next_state) {
    torch::Tensor actual_state = Forward(state, action);
    if (torch::max(torch::abs(next_state - actual_state)).item<double>() > solver_.parameter_.kEPS) {
        return 0;
    }
    else {
        return 1;
    }
}

torch::Tensor CCartPole::Forward(const torch::Tensor& state, const::torch::Tensor& action) {
    int action_i = action.index({0}).item<int>();
    return Forward(state, action_i);
}

torch::Tensor CCartPole::Forward(const torch::Tensor& state, int action) {
    torch::Tensor res = torch::empty({4}, torch::dtype(torch::kFloat64));

    double f;
    if (action == 1) {
        f = 10.0;
    }
    else if (action == 0) {
        f = -10.0;
    }
    else assert(false);
    double pole_mass = 0.1;
    double total_mass = 1.1;
    double len = 0.5;
    double g = 9.8;
    double tau = 0.02;

    torch::Tensor temp = (f + pole_mass * len * torch::square(state.index({3})) * torch::sin(state.index({2}))) / total_mass;
    torch::Tensor angle_dot_dot = (g * torch::sin(state.index({2})) - temp * torch::cos(state.index({2}))) / (len * (4.0 / 3.0 - pole_mass * torch::square(torch::cos(state.index({2}))) / total_mass));
    torch::Tensor position_dot_dot = temp - pole_mass * len * angle_dot_dot * torch::cos(state.index({2})) / total_mass;

    res.index_put_({0}, state.index({0}) + tau * state.index({1}));
    res.index_put_({1}, state.index({1}) + tau * position_dot_dot);
    res.index_put_({2}, state.index({2}) + tau * state.index({3}));
    res.index_put_({3}, state.index({3}) + tau * angle_dot_dot);
    return res;
}

torch::Tensor CCartPole::State(int step) {
    torch::Tensor state = torch::empty({4}, torch::dtype(torch::kFloat64));
    state.index_put_({0}, time_state_var_vv_[step][0].get(GRB_DoubleAttr_X));
    state.index_put_({1}, time_state_var_vv_[step][1].get(GRB_DoubleAttr_X));
    state.index_put_({2}, time_state_var_vv_[step][2].get(GRB_DoubleAttr_X));
    state.index_put_({3}, time_state_var_vv_[step][3].get(GRB_DoubleAttr_X));
    return state;
}

vector<GRBVar> CCartPole::StateVar(int step) {
    return {time_state_var_vv_[step][0], time_state_var_vv_[step][1], time_state_var_vv_[step][2], time_state_var_vv_[step][3]};
}

int CCartPole::Action(int step) {
    for (int i = 0; i < (int)time_action_var_vv_[step].size(); ++i) {
        if (solver_.EvalSatVar(time_action_var_vv_[step][i]) == 1) {
            return i;
        }
    }
    assert(false);
}

torch::Tensor CCartPole::Trajectory() {
    torch::Tensor res = torch::empty({StateHorizon(), 5}, torch::dtype(torch::kFloat64));
    for (int i = 0; i < StateHorizon(); ++i) {
        res.index_put_({i, torch::indexing::Slice(0, 4)}, State(i));
        if (i != StateHorizon() - 1) {
            res.index_put_({i, 4}, Action(i));
        }
        else {
            res.index_put_({i, 4}, 0);
        }
    }
    return res;
}

string CCartPole::Str() const {
    string res = TimeStateStr() + ActionStateStr();
    return res;
}

string CCartPole::TimeStateStr() const {
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

string CCartPole::ActionStateStr() const {
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

void CCartPole::SetHorizon(int horizon) {
    GRBModel& model = solver_.ConvexSolver();
    time_state_var_vv_.resize(horizon);
    for (int i = 0; i < horizon; ++i) {
        time_state_var_vv_[i].reserve(11);
        time_state_var_vv_[i].emplace_back(model.addVar(parameter_.position_bound.first, parameter_.position_bound.second, 0, GRB_CONTINUOUS, "pos@" + to_string(i)));
        time_state_var_vv_[i].emplace_back(model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "pos_dot@" + to_string(i)));
        time_state_var_vv_[i].emplace_back(model.addVar(parameter_.angle_bound.first, parameter_.angle_bound.second, 0, GRB_CONTINUOUS, "ang@" + to_string(i)));
        time_state_var_vv_[i].emplace_back(model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "ang_dot@" + to_string(i)));
        time_state_var_vv_[i].emplace_back(model.addVar(-1, 1, 0, GRB_CONTINUOUS, "ang_cos@" + to_string(i)));
        time_state_var_vv_[i].emplace_back(model.addVar(-1, 1, 0, GRB_CONTINUOUS, "ang_sin@" + to_string(i)));

        time_state_var_vv_[i].emplace_back(model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "ang_dot_square@" + to_string(i)));
        time_state_var_vv_[i].emplace_back(model.addVar(0, 1, 0, GRB_CONTINUOUS, "ang_cos_square@" + to_string(i)));
        time_state_var_vv_[i].emplace_back(model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "temp@" + to_string(i)));
        time_state_var_vv_[i].emplace_back(model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "pos_dot_dot@" + to_string(i)));
        time_state_var_vv_[i].emplace_back(model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "ang_dot_dot@" + to_string(i)));
    };
    model.update();
    // printf("%s", Str().c_str());
}

void CCartPole::SetWaveConstraint() {
    ClearWaveConstraint();
    // use max(1, time_state_var_vv_.size() - 1) to at least set wave condition once for CheckApproximationCartPole
    for (int i = 0; i < max(1, (int)time_state_var_vv_.size() - 1); ++i) {
        GRBVar& angle = time_state_var_vv_[i][2];
        GRBVar& angle_cos = time_state_var_vv_[i][4];
        GRBVar& angle_sin = time_state_var_vv_[i][5];

        double left_end, right_end;
        CSolver::SatVar* var;
        bool cos_constr_f = false;
        bool sin_constr_f = false;
        z3::expr cos_constr(solver_.SatContex());
        z3::expr sin_constr(solver_.SatContex());
        for (int j = 0; j < wave_approx_param_; ++j) {
            // set angle_cos@t for -pi/2 < angle@t < 0
            left_end = -M_PI / 2 + (M_PI / 2 * j / wave_approx_param_);
            right_end = -M_PI / 2 + (M_PI / 2 * (j + 1) / wave_approx_param_);
            var = solver_.AddSatVar("cos_s" + to_string(j) + "p0@" + to_string(i));
            var->lin_expr_.emplace_back(angle >= left_end);
            var->lin_expr_.emplace_back(angle <= right_end);
            SetCosineConstraintForSegment(left_end, right_end, true, angle, angle_cos, var);
            wave_constraint_l_.emplace_back(var);
            solver_.SetSatOrConstraint(var, cos_constr_f, cos_constr);

            // set angle_cos@t for 0 < angle@t < pi/2
            left_end = (M_PI / 2 * j / wave_approx_param_);
            right_end = (M_PI / 2 * (j + 1) / wave_approx_param_);
            var = solver_.AddSatVar("cos_s" + to_string(j) + "p1@" + to_string(i));
            var->lin_expr_.emplace_back(angle >= left_end);
            var->lin_expr_.emplace_back(angle <= right_end);
            SetCosineConstraintForSegment(left_end, right_end, true, angle, angle_cos, var);
            wave_constraint_l_.emplace_back(var);
            solver_.SetSatOrConstraint(var, cos_constr_f, cos_constr);

            // set angle_sin@t for -pi/2 < angle@t < 0
            left_end = -M_PI / 2 + (M_PI / 2 * j / wave_approx_param_);
            right_end = -M_PI / 2 + (M_PI / 2 * (j + 1) / wave_approx_param_);
            var = solver_.AddSatVar("sin_s" + to_string(j) + "p0@" + to_string(i));
            var->lin_expr_.emplace_back(angle >= left_end);
            var->lin_expr_.emplace_back(angle <= right_end);
            SetSineConstraintForSegment(left_end, right_end, false, angle, angle_sin, var);
            wave_constraint_l_.emplace_back(var);
            solver_.SetSatOrConstraint(var, sin_constr_f, sin_constr);

            // set angle_sin@t for 0 < angle@t < pi/2
            left_end = (M_PI / 2 * j / wave_approx_param_);
            right_end = (M_PI / 2 * (j + 1) / wave_approx_param_);
            var = solver_.AddSatVar("sin_s" + to_string(j) + "p1@" + to_string(i));
            var->lin_expr_.emplace_back(angle >= left_end);
            var->lin_expr_.emplace_back(angle <= right_end);
            SetSineConstraintForSegment(left_end, right_end, true, angle, angle_sin, var);
            wave_constraint_l_.emplace_back(var);
            solver_.SetSatOrConstraint(var, sin_constr_f, sin_constr);
        }
        solver_.AddSatConstraint(cos_constr);
        solver_.AddSatConstraint(sin_constr);
    }
    solver_.ConvexSolver().update();
}

void CCartPole::ClearWaveConstraint() {
    for (auto it = wave_constraint_l_.begin(); it != wave_constraint_l_.end(); ++it) {
        CSolver::SatVar* v = *it;
        solver_.RemoveSatVar(v);
    }
    wave_constraint_l_.clear();
}

void CCartPole::SetCosineConstraintForSegment(const double& left_end, const double& right_end, bool taylor_is_over_approx, GRBVar& angle, GRBVar& angle_cos, CSolver::SatVar* var) {
    if (taylor_is_over_approx) {
        var->lin_expr_.emplace_back(angle_cos <= CosineTaylorApproximation(angle, left_end, right_end));
        var->lin_expr_.emplace_back(angle_cos >= CosineTwoPointEquation(angle, left_end, right_end));
    }
    else {
        var->lin_expr_.emplace_back(angle_cos <= CosineTwoPointEquation(angle, left_end, right_end));
        var->lin_expr_.emplace_back(angle_cos >= CosineTaylorApproximation(angle, left_end, right_end));
    }
}

GRBLinExpr CCartPole::CosineTwoPointEquation(GRBVar& var, const double& left_end, const double& right_end) {
    return cos(left_end) + (cos(right_end) - cos(left_end)) * (var - left_end) / (right_end - left_end);
}

GRBLinExpr CCartPole::CosineTaylorApproximation(GRBVar& var, const double& left_end, const double& right_end) {
    return cos(0.5 * (left_end + right_end)) - sin(0.5 * (left_end + right_end)) * (var - 0.5 * (left_end + right_end));
}

void CCartPole::SetSineConstraintForSegment(const double& left_end, const double& right_end, bool taylor_is_over_approx, GRBVar& angle, GRBVar& angle_sin, CSolver::SatVar* var) {
    if (taylor_is_over_approx) {
        var->lin_expr_.emplace_back(angle_sin <= SineTaylorApproximation(angle, left_end, right_end));
        var->lin_expr_.emplace_back(angle_sin >= SineTwoPointEquation(angle, left_end, right_end));
    }
    else {
        var->lin_expr_.emplace_back(angle_sin <= SineTwoPointEquation(angle, left_end, right_end));
        var->lin_expr_.emplace_back(angle_sin >= SineTaylorApproximation(angle, left_end, right_end));
    }
}

GRBLinExpr CCartPole::SineTwoPointEquation(GRBVar& var, const double& left_end, const double& right_end) {
    return sin(left_end) + (sin(right_end) - sin(left_end)) * (var - left_end) / (right_end - left_end);
}

GRBLinExpr CCartPole::SineTaylorApproximation(GRBVar& var, const double& left_end, const double& right_end) {
    return sin(0.5 * (left_end + right_end)) + cos(0.5 * (left_end + right_end)) * (var - 0.5 * (left_end + right_end));
}

void CCartPole::SetStateRelationConstraint() {
    GRBModel& model = solver_.ConvexSolver();

    time_action_var_vv_.resize(time_state_var_vv_.size() - 1);
    CSolver::SatVar* var;
    for (int i = 0; i < (int)time_state_var_vv_.size() - 1; ++i) {
        time_action_var_vv_[i].reserve(2);

        double f = 10.0;
        double pole_mass = 0.1;
        double total_mass = 1.1;
        double len = 0.5;
        double g = 9.8;
        double tau = 0.02;

        GRBVar& position = time_state_var_vv_[i][0];
        GRBVar& position_dot = time_state_var_vv_[i][1];
        GRBVar& angle = time_state_var_vv_[i][2];
        GRBVar& angle_dot = time_state_var_vv_[i][3];
        GRBVar& angle_cos = time_state_var_vv_[i][4];
        GRBVar& angle_sin = time_state_var_vv_[i][5];
        GRBVar& angle_dot_square = time_state_var_vv_[i][6];
        GRBVar& angle_cos_square = time_state_var_vv_[i][7];
        GRBVar& temp = time_state_var_vv_[i][8];
        GRBVar& position_dot_dot = time_state_var_vv_[i][9];
        GRBVar& angle_dot_dot = time_state_var_vv_[i][10];

        GRBVar& next_position = time_state_var_vv_[i + 1][0];
        GRBVar& next_position_dot = time_state_var_vv_[i + 1][1];
        GRBVar& next_angle = time_state_var_vv_[i + 1][2];
        GRBVar& next_angle_dot = time_state_var_vv_[i + 1][3];

        model.addQConstr(angle_dot_square == angle_dot * angle_dot);
        model.addQConstr(angle_cos_square == angle_cos * angle_cos);
        model.addQConstr(position_dot_dot == temp - pole_mass * len * angle_dot_dot * angle_cos / total_mass);
        model.addQConstr(angle_dot_dot * (len * (4.0 / 3.0 - pole_mass * angle_cos_square/ total_mass)) == (g * angle_sin - temp * angle_cos));
        model.addConstr(next_position == position + tau * position_dot);
        model.addConstr(next_angle == angle + tau * angle_dot);
        model.addConstr(next_position_dot == position_dot + tau * position_dot_dot);
        model.addConstr(next_angle_dot == angle_dot + tau * angle_dot_dot);

        var = solver_.AddSatVar("action_0@" + to_string(i));
        var->quad_expr_.emplace_back(temp == (-1 * f + pole_mass * len * angle_dot_square * angle_sin) / total_mass);
        time_action_var_vv_[i].emplace_back(var);
        z3::expr constr = var->var_;

        var = solver_.AddSatVar("action_1@" + to_string(i));
        var->quad_expr_.emplace_back(temp == (f + pole_mass * len * angle_dot_square * angle_sin) / total_mass);
        time_action_var_vv_[i].emplace_back(var);
        constr = constr | var->var_;

        solver_.AddSatConstraint(constr);
    };
    model.update();
}

void CCartPole::ResetStateRelationConstraint() {
    for (int i = 0; i < (int)time_state_var_vv_.size() - 1; ++i) {
        solver_.AddSatConstraint(time_action_var_vv_[i][0]->var_ | time_action_var_vv_[i][1]->var_);
    };
    solver_.ConvexSolver().update();
}


NNV_NAMESPACING_END
