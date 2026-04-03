#include "example.hpp"

NNV_NAMESPACING_START

torch::Tensor CExample::Bound() {
    return torch::tensor({{-GRB_INFINITY, GRB_INFINITY}, {-GRB_INFINITY, GRB_INFINITY}}, torch::dtype(torch::kFloat64));
}

torch::Tensor CExample::InitialConstraint() {
    return torch::tensor({{parameter_.x0_initial_constraint.first, parameter_.x0_initial_constraint.second}, {parameter_.x1_initial_constraint.first, parameter_.x1_initial_constraint.second}}, torch::dtype(torch::kFloat64));
}

void CExample::Init(int horizon) {
    // initialize variable for the horizon
    SetHorizon(horizon);

    // construct the relation from action to the next state
    SetStateRelationConstraint();

    // for convex
    // initialize initial constraint
    SetInitialConstraint();
    // construct safty constraint
    SetSafeConstraint();
}

void CExample::Reset() {
    // construct the relation from action to the next state
    ResetStateRelationConstraint();
}

void CExample::SetInitialConstraint() {
    GRBModel& model = solver_.ConvexSolver();
    model.addConstr(time_state_var_vv_[0][0] >= parameter_.x0_initial_constraint.first, "initial_constr_0");
    model.addConstr(time_state_var_vv_[0][0] <= parameter_.x0_initial_constraint.second, "initial_constr_1");
    model.addConstr(time_state_var_vv_[0][1] >= parameter_.x1_initial_constraint.first, "initial_constr_2");
    model.addConstr(time_state_var_vv_[0][1] <= parameter_.x1_initial_constraint.second, "initial_constr_3");
    model.update();
}

void CExample::ClearInitialConstraint() {
    GRBModel& model = solver_.ConvexSolver();
    model.remove(model.getConstrByName("initial_constr_0"));
    model.remove(model.getConstrByName("initial_constr_1"));
    model.remove(model.getConstrByName("initial_constr_2"));
    model.remove(model.getConstrByName("initial_constr_3"));
    model.update();
}

void CExample::SetSafeConstraint() {
}

void CExample::ClearSafeConstraint() {
}

void CExample::SetRelationSDT(vector<CSDT>& soft_decision_tree_v, bool initialize) {
    assert(false);
}

void CExample::SetRelationNN(vector<CNN>& neural_network_v) {
    GRBModel& convex_solver = solver_.ConvexSolver();
    for (int i = 0; i < (int)time_state_var_vv_.size() - 1; ++i) {
        vector<vector<GRBVar> >& convex_network_var_v = neural_network_v[i].ConvexNetworkVariable();
        assert(convex_network_var_v[0].size() == 2);
        assert(convex_network_var_v.back().size() == 1);
        // connest relation for state variable
        for (int j = 0; j < 2; ++j) {
            convex_solver.addConstr(time_state_var_vv_[i][j] == convex_network_var_v[0][j]);
        }
        // connest relation for action variable
        convex_solver.addConstr(time_state_var_vv_[i][3 * simulation_per_control_] == convex_network_var_v.back()[0]);
    }
}

int CExample::CheckNextState(const torch::Tensor& state, int action, const torch::Tensor& next_state) {
    torch::Tensor actual_state = Forward(state, action);
    if (torch::max(torch::abs(next_state - actual_state)).item<double>() > solver_.parameter_.kEPS) {
        return 0;
    }
    else {
        return 1;
    }
}

torch::Tensor CExample::Forward(const torch::Tensor& state, const::torch::Tensor& action) {
    // double tau = 0.005;
    // res.index_put_({0}, state.index({0}) + tau * state.index({1}));
    // res.index_put_({1}, state.index({1}) + tau * (action * torch::square(state.index({1})) - state.index({0})));
    assert(false);
}

torch::Tensor CExample::Forward(const torch::Tensor& state, int action) {
    assert(false);
}

torch::Tensor CExample::State(int step) {
    torch::Tensor state = torch::empty({2 * simulation_per_control_}, torch::dtype(torch::kFloat64));
    for (int i = 0; i < simulation_per_control_; ++i) {
        state.index_put_({2 * i}, time_state_var_vv_[step][3 * i].get(GRB_DoubleAttr_X));
        state.index_put_({1 + 2 * i}, time_state_var_vv_[step][1 + 3 * i].get(GRB_DoubleAttr_X));
    }
    return state;
}

vector<GRBVar> CExample::StateVar(int step) {
    return {time_state_var_vv_[step][0], time_state_var_vv_[step][1]};
}

int CExample::Action(int step) {
    assert(false);
}

torch::Tensor CExample::Trajectory() {
    torch::Tensor res = torch::empty({StateHorizon(), 1 + 2 * simulation_per_control_}, torch::dtype(torch::kFloat64));
    for (int i = 0; i < StateHorizon(); ++i) {
        for (int j = 0; j < simulation_per_control_; ++j) {
            res.index_put_({i, torch::indexing::Slice(0, 2 * simulation_per_control_)}, State(i));
        }
        if (i != StateHorizon() - 1) {
            res.index_put_({i, 2 * simulation_per_control_}, time_state_var_vv_[i][3 * simulation_per_control_].get(GRB_DoubleAttr_X));
        }
        else {
            res.index_put_({i, 2 * simulation_per_control_}, 0);
        }
    }
    return res;
}

string CExample::Str() const {
    string res = TimeStateStr() + ActionStateStr();
    return res;
}

string CExample::TimeStateStr() const {
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

string CExample::ActionStateStr() const {
    return "";
}

void CExample::SetHorizon(int horizon) {
    GRBModel& model = solver_.ConvexSolver();
    time_state_var_vv_.resize(horizon);
    for (int i = 0; i < horizon; ++i) {
        time_state_var_vv_[i].reserve(3 * simulation_per_control_ + 1);
        for (int j = 0; j < simulation_per_control_; ++j) {
            time_state_var_vv_[i].emplace_back(model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "x0_" + to_string(j) + "@" + to_string(i)));
            time_state_var_vv_[i].emplace_back(model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "x1_" + to_string(j) + "@" + to_string(i)));
            time_state_var_vv_[i].emplace_back(model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "x1_square_" + to_string(j) + "@" + to_string(i)));
        }
        time_state_var_vv_[i].emplace_back(model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "u@" + to_string(i)));
    };
    model.update();
    // printf("%s", Str().c_str());
}

void CExample::SetStateRelationConstraint() {
    GRBModel& model = solver_.ConvexSolver();
    for (int i = 0; i < (int)time_state_var_vv_.size() - 1; ++i) {
        double tau = 0.005;

        GRBVar& u = time_state_var_vv_[i][3 * simulation_per_control_];
        for (int j = 0; j < simulation_per_control_; ++j) {
            GRBVar& x0 = time_state_var_vv_[i][3 * j];
            GRBVar& x1 = time_state_var_vv_[i][1 + 3 * j];
            GRBVar& x1_square = time_state_var_vv_[i][2 + 3 * j];

            GRBVar next_x0, next_x1;

            if (j != simulation_per_control_ - 1) {
                next_x0 = time_state_var_vv_[i][3 * (j+1)];
                next_x1 = time_state_var_vv_[i][1 + 3 * (j+1)];
            }
            else {
                next_x0 = time_state_var_vv_[i + 1][0];
                next_x1 = time_state_var_vv_[i + 1][1];
            }

            model.addQConstr(x1_square == x1 * x1);
            model.addConstr(next_x0 == x0 + tau * x1);
            model.addQConstr(next_x1 == x1 + tau * (u * (x1_square) - x0));
        }
    };
    model.update();
}

void CExample::ResetStateRelationConstraint() {
}


NNV_NAMESPACING_END
