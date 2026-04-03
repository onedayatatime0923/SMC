#include "verifier.hpp"

extern Usage usage;

NNV_NAMESPACING_START

// int CVerifier::Run(int argc, char **argv) {
//     ArgumentParser program("parse");
//
//     bool res;
//     try {
//     res = program.ParseArgs(argc, argv);
//     }
//     catch (const runtime_error& err) {
//         std::cout << err.what() << std::endl;
//         return 1;
//     }
//     if (res) return 0;
//
//     int result = Verify_Neural_Network();
//
//     if (result) {
//         cout << CounterExample() << endl;
//         cout << neural_network_v_[0].Forward(CounterExample()) << endl;
//     };
//     return 0;
// };

int CVerifier::Verify_Neural_Network(const string & model_path) {
    assert(neural_network_v_.size() == 1);
    solver_.ResetSat();

    // // Parse Specification
    z3::context property_context;
    z3::expr_vector parsed_property_v = property_context.parse_file(model_path.c_str());
    for (int i = 0; i < parsed_property_v.size(); ++i) {
        cout << parsed_property_v[i] << endl;
        getchar();
    }
    printf("here\n");
    getchar();

    // parse input specification
    vector<int>& num_neuron_v = neural_network_v_[0].NumNeuron();
    torch::Tensor bound = torch::zeros({num_neuron_v[0], 2}, torch::dtype(torch::kFloat64));
    int output_constraint_initial_id;
    for (int i = 0; i < num_neuron_v[0] * 2; ++i) {
        string var_name = parsed_property_v[i].arg(0).decl().name().str();
        int var_id = stoi(var_name.substr(2));
        assert(var_id == i / 2);
        // printf("variable id: %d\n", var_id);

        double value;
        if (parsed_property_v[i].arg(1).num_args() == 0) {
            value = parsed_property_v[i].arg(1).as_double();
        }
        else if (parsed_property_v[i].arg(1).num_args() == 1) {
            value = parsed_property_v[i].arg(1).arg(0).as_double();
        }
        // printf("value: %lf\n", value);

        if (parsed_property_v[i].decl().decl_kind() == Z3_OP_GE) {
            bound.index_put_({var_id, 0}, value);
        }
        else if (parsed_property_v[i].decl().decl_kind() == Z3_OP_LE) {
            bound.index_put_({var_id, 1}, value);
        }
        else assert(false);

        // cout << parsed_property_v[i] << endl;
        // cout << (parsed_property_v[i].decl().decl_kind() == Z3_OP_LE) << endl;
        // cout << (parsed_property_v[i].decl().decl_kind() == Z3_OP_GE) << endl;
        output_constraint_initial_id = i + 1;
    };
    // cout << bound << endl;
    // getchar();

    // parse output specification
    assert(output_constraint_initial_id < (int)parsed_property_v.size());
    auto VarId = [](const z3::expr& expr) {
        string name = expr.decl().name().str();
        assert(name[0] == 'Y');
        assert(name[1] == '_');
        return stoi(name.substr(2));
    };
    bool eq_f = false;
    torch::Tensor eq = torch::empty({0});
    auto EqConstr = [VarId, &num_neuron_v, &eq_f, &eq](const z3::expr& expr) {
        if (!eq_f) {
            eq_f = true;
            eq = torch::zeros({1, num_neuron_v[num_neuron_v.size() - 1] + 1}, torch::dtype(torch::kFloat64));
        }
        else {
            eq = torch::cat({eq, torch::zeros({1, num_neuron_v[num_neuron_v.size() - 1] + 1}, torch::dtype(torch::kFloat64))}, 0);
        }
        
        assert(expr.num_args() == 2);
        int var_id0 = VarId(expr.arg(0));
        if (expr.arg(1).decl().decl_kind() != Z3_OP_ANUM) {
            int var_id1 = VarId(expr.arg(1));
            if (expr.decl().decl_kind() == Z3_OP_GE) {
                eq.index_put_({eq.sizes()[0] - 1, var_id0}, -1);
                eq.index_put_({eq.sizes()[0] - 1, var_id1}, 1);
            }
            else if (expr.decl().decl_kind() == Z3_OP_LE) {
                eq.index_put_({eq.sizes()[0] - 1, var_id0}, 1);
                eq.index_put_({eq.sizes()[0] - 1, var_id1}, -1);
            }
            else assert(false);
        }
        else {
            double value = expr.arg(1).as_double();
            if (expr.decl().decl_kind() == Z3_OP_GE) {
                eq.index_put_({eq.sizes()[0] - 1, var_id0}, -1);
                eq.index_put_({eq.sizes()[0] - 1, num_neuron_v[num_neuron_v.size() - 1]}, value);
            }
            else if (expr.decl().decl_kind() == Z3_OP_LE) {
                eq.index_put_({eq.sizes()[0] - 1, var_id0}, 1);
                eq.index_put_({eq.sizes()[0] - 1, num_neuron_v[num_neuron_v.size() - 1]}, -1 * value);
            }
            else assert(false);
        }
    };

    bool any_or_all_constraint;
    if (output_constraint_initial_id == (int)parsed_property_v.size() - 1) {
        any_or_all_constraint = true;
    }
    else {
        any_or_all_constraint = false;
    }
    for (int i = output_constraint_initial_id; i < (int)parsed_property_v.size(); ++i) {
        if (parsed_property_v[i].num_args() == 2) {
            EqConstr(parsed_property_v[i]);
        }
        else {
            any_or_all_constraint = true;
            assert(parsed_property_v[i].decl().decl_kind() == Z3_OP_OR);
            for (int j = 0; j < (int)parsed_property_v[i].num_args(); ++j) {
                assert(parsed_property_v[i].arg(j).num_args() == 1);
                assert(parsed_property_v[i].arg(j).decl().decl_kind() == Z3_OP_AND);
                EqConstr(parsed_property_v[i].arg(j).arg(0));
            }
        };
    }
    neural_network_v_[0].AdjustWeightBias(eq);
    neural_network_v_[0].SetBound(bound.unsqueeze(0));
    // cout << eq << endl;
    // getchar();

    // // Verification
    // set checking function
    auto CheckNeuralNetworkOutput = [this, &any_or_all_constraint]() {
        if (solver_.Certificate() == CSolver::Certificate::DUAL) {
            ResolveConvexByDualCertificate();
        }
        vector<CNN>& neural_network_v = neural_network_v_;

        int input_dim = neural_network_v[0].NumNeuron()[0];
        torch::Tensor input = torch::empty({input_dim}, torch::dtype(torch::kFloat64));
        for (int i = 0; i < input_dim; ++i) {
            input.index_put_({i}, neural_network_v[0].ConvexNetworkVariable()[0][i].get(GRB_DoubleAttr_X));
        }
        // printf("input:\n");
        // cout << input << endl;
        torch::Tensor output = neural_network_v[0].Forward(input);
        #ifdef DEBUG
        printf("check output\n");
        cout << output << endl;
        #endif
        // cout << output << endl;

        if (any_or_all_constraint && torch::any(output < 0).item<bool>()) {
            counter_example_ = input;
            // printf("1\n");
            return 1;
        }
        else if (!any_or_all_constraint && torch::all(output < 0).item<bool>()) {
            counter_example_ = input;
            // printf("1\n");
            return 1;
        }
        else {
            // printf("0\n");
            return 0;
        }
    };

    torch::Tensor& input_bound = neural_network_v_[0].BoundBeforeActivation()[0];
    torch::Tensor& output_bound = neural_network_v_[0].BoundBeforeActivation().back();
    // // cout << input_bound << endl;
    // // cout << output_bound << endl;

    // verify by input splitting
    torch::Tensor mask;
    torch::Tensor domain;
    if (neural_network_v_[0].NumNeuron()[0] <= 1) {
        while (true) {
            assert(input_bound.sizes()[0] == output_bound.sizes()[0]);
            // printf("input: \n");
            // cout << input_bound << endl;
            // printf("output: \n");
            // cout << output_bound << endl;

            if (any_or_all_constraint) {
                mask = get<0>(torch::max(output_bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}) < 0, 1));
            }
            else {
                mask = get<0>(torch::min(output_bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}) < 0, 1));
            }
            // cout << mask << endl;
            // getchar();

            if (!torch::any(mask).item<bool>()) {
                WriteOutput();
                return 0;
            }
            else if (CheckNeuralNetworkOutput()) {
                WriteOutput();
                return 1;
            }
            else {
                domain = input_bound.index({mask});
                // cout << domain.sizes() << endl;
                // cout << domain << endl;
                // getchar();

                neural_network_v_[0].SplitBound(domain, 2);
                // cout << output_bound << endl;
                // getchar();
            }
        }
    }
    else {
        // printf("here\n");
        // cout << any_or_all_constraint << endl;
        // getchar();
        // assert(any_or_all_constraint);
        bool constr_f = false;
        vector<GRBVar>& output_var = neural_network_v_[0].ConvexNetworkVariable().back();
        for (int i = 0; i < eq.sizes()[0]; ++i) {
            if (output_bound.index({0, i, 0}).item<double>() < 0) {
                constr_f = true;
            }
        }
        if (constr_f) {
            neural_network_v_[0].SetBound(bound.unsqueeze(0), true);

            constr_f = false;
            z3::expr constr(solver_.SatContex());
            for (int i = 0; i < eq.sizes()[0]; ++i) {
                if (output_bound.index({0, i, 0}).item<double>() < 0) {
                    CSolver::SatVar* var = solver_.AddSatVar("contract_" + to_string(i));
                    var->lin_expr_.emplace_back(output_var[i] + solver_.parameter_.kEPS <= 0);
                    solver_.SetSatOrConstraint(var, constr_f, constr);
                }
            }
            if (constr_f) {
                solver_.AddSatConstraint(constr);

                iteration_ = 0;
                counter_example_ = torch::empty({0});

                // solution_checker_f_ = bind(CheckNeuralNetworkOutput, this);
                solution_checker_f_ = CheckNeuralNetworkOutput;
                int result = Solve_Smc();
                WriteOutput();
                return result;
            }
            else {
                WriteOutput();
                return 0;
            }
        }
        else {
            WriteOutput();
            return 0;
        }
    }
};

int CVerifier::Verify_Neural_Network(const double& eps) {
    assert(neural_network_v_.size() == 1);
    solver_.ResetSat();
    torch::Tensor bound = torch::zeros({neural_network_v_[0].NumNeuron()[0], 2}, torch::dtype(torch::kFloat64));
    bound.index_put_({torch::indexing::Slice(), 0}, torch::clamp(neural_network_v_[0].Input() - eps, 0, c10::nullopt));
    bound.index_put_({torch::indexing::Slice(), 1}, torch::clamp(neural_network_v_[0].Input() + eps, c10::nullopt, 1));
    neural_network_v_[0].SetBound(bound.unsqueeze(0));

    torch::Tensor& output_bound = neural_network_v_[0].BoundBeforeActivation().back();
    torch::Tensor idx = torch::argsort(output_bound.index({torch::indexing::Slice(), 1}), -1, true);
    assert(idx.dim() == 1);
    // cout << output_bound.index({torch::indexing::Slice(), 1}) << endl;
    // cout << output_bound.index({torch::indexing::Slice(), 0}) << endl;

    int target = neural_network_v_[0].Target();
    vector<GRBVar>& output_var = neural_network_v_[0].ConvexNetworkVariable().back();
    CSolver::SatVar* var;
    bool constr_f = false;
    z3::expr constr(solver_.SatContex());
    for (int i = 0; i < idx.sizes()[0]; ++i) {
        int output_id = idx.index({i}).item<int>();
        if (output_id == target) {}
        else if (output_bound.index({output_id, 1}).item<double>() > output_bound.index({target, 0}).item<double>()) {
            var = solver_.AddSatVar("contract_" + to_string(output_id));
            var->lin_expr_.emplace_back(output_var[output_id] >= output_var[target] + solver_.parameter_.kEPS);
            solver_.SetSatOrConstraint(var, constr_f, constr);
        }
    }
    if (constr_f) {
        solver_.AddSatConstraint(constr);

        iteration_ = 0;
        counter_example_ = torch::empty({0});
        auto CheckNeuralNetworkOutput = [](CVerifier* verifier_p) {
            if (verifier_p->solver_.Certificate() == CSolver::Certificate::DUAL) {
                verifier_p->ResolveConvexByDualCertificate();
            }
            vector<CNN>& neural_network_v = verifier_p->neural_network_v_;

            int input_dim = neural_network_v[0].NumNeuron()[0];
            torch::Tensor input = torch::empty({input_dim}, torch::dtype(torch::kFloat64));
            // cout << input << endl;
            for (int i = 0; i < input_dim; ++i) {
                input.index_put_({i}, neural_network_v[0].ConvexNetworkVariable()[0][i].get(GRB_DoubleAttr_X));
            }
            torch::Tensor output = neural_network_v[0].Forward(input);
            #ifdef DEBUG
            printf("check output\n");
            cout << output << endl;
            #endif
            if (torch::argmax(output).item<int>() != neural_network_v[0].Target()) {
                verifier_p->counter_example_ = input;
                return 1;
            }
            else {
                return 0;
            }
        };
        solution_checker_f_ = bind(CheckNeuralNetworkOutput, this);
        int result = Solve_Smc();
        return result;
    }
    else {
        return 0;
    }
}

int CVerifier::Verify_SDT_Monolith() {
    int& cos_approx_param = env_p_->WaveApproxParam();

    env_p_->SetRelationSDT(soft_decision_tree_v_, true);
    auto SetCounterExample = [](CVerifier* verifier_p) {
        verifier_p->counter_example_ = verifier_p->env_p_->Trajectory();
        return 1;
    };
    solution_checker_f_ = bind(SetCounterExample, this);
    full_search_b_ = false;
    iteration_ = 0;

    int result = 0;
    while (true) {
        solver_.ResetSat();
        env_p_->Reset();
        env_p_->SetRelationSDT(soft_decision_tree_v_);

        counter_example_ = torch::empty({0});
        result = Solve_Smc();

        // cout << counter_example_ << endl;
        if (result == 1) {
            torch::Tensor input = counter_example_.index({0, torch::indexing::Slice(0, counter_example_.sizes()[1] - 1)});
            torch::Tensor state = input.clone();

            int horizon = env_p_->StateHorizon();
            int res = 1;
            // printf("start to check\n");
            for (int i = 0; i < horizon - 1; ++i) {
                // printf("action step: %d\n", i);
                // cout << state << endl;
                torch::Tensor action = soft_decision_tree_v_[i].Forward(state);
                int action_id = counter_example_.index({i, counter_example_.sizes()[1] - 1}).item<int>();
                // cout << action << endl;
                // printf("predicted action: %d\n", action_id);
                assert(torch::abs(action.index({action_id}) - torch::max(action)).item<double>() < solver_.parameter_.kEPS);
                // minus action_id for real action
                torch::Tensor actual_state = env_p_->Forward(state, action_id);
                state = counter_example_.index({i+1, torch::indexing::Slice(0, counter_example_.sizes()[1] - 1)});
                // cout << state << endl;
                // cout << actual_state << endl;
                if (torch::max(torch::abs(state - actual_state)).item<double>() > solver_.parameter_.kEPS) {
                    if (cos_approx_param * 2 <= env_p_->parameter_.wave_approx_limit) {
                        cos_approx_param *= 2;
                        printf("find counter example and set cosine approximation to %d\n", cos_approx_param);
                    }
                    else assert(false);
                    res = 0;
                    break;
                }
            }
            // printf("end\n");
            // getchar();
            if (res == 1) {
                break;
            }
        }
        else if (result == 0) {
            // printf("can not find counter example\n");
            break;
        }
        else assert(false);
    }

    if (result == 1) {
        cout << counter_example_ << endl;
        // printf("%s", env_p_->Str().c_str());
        printf("NeuralNetwork is unsafe within %d iteration(s).\n", iteration_);
    }
    else if (result == 0) {
        printf("NeuralNetwork is safe within %d iteration(s).\n", iteration_);
    }
    return result;
}

vector<torch::Tensor> CVerifier::Verify_SDT_Composition(int horizon) {
    assert(env_p_->ActionHorizon() == (int)soft_decision_tree_v_.size());
    assert((int)env_p_->StateVar(0).size() == (int)soft_decision_tree_v_[0].ConvexInputVariable().size());
    assert(env_p_->ActionNum() == (int)soft_decision_tree_v_[0].SatOutputVariable().size());
    int& cos_approx_param = env_p_->WaveApproxParam();
    cos_approx_param = env_p_->parameter_.wave_approx_limit;

    env_p_->ClearInitialConstraint();
    env_p_->ClearSafeConstraint();
    env_p_->SetRelationSDT(soft_decision_tree_v_, true);
    auto UpdateSolutionBound = [](CVerifier* verifier_p) {
        // printf("star checking\n");
        GRBModel& convex_solver = verifier_p->solver_.ConvexSolver();
        vector<GRBVar> var_v = verifier_p->env_p_->StateVar(verifier_p->env_p_->StateHorizon() - 1);

        // cout << verifier_p->solver_.Str() << endl;
        for (int i = 0; i < (int)var_v.size(); ++i) {
            for (int j = 0; j < 2; ++j) {
                // printf("handling i: %d, j: %d\n", i, j);
                // cout << verifier_p->counter_example_ << endl;
                GRBVar var = var_v[i];
                if (j == 0) {
                    verifier_p->solver_.SetConvexObjective(var);
                }
                else if (j == 1) {
                    verifier_p->solver_.SetConvexObjective(-1 * var);
                }
                else assert(false);
                convex_solver.optimize();
                // cout << env_p_->Trajectory() << endl;
                int status = convex_solver.get(GRB_IntAttr_Status);
                // status should be optimal, but sometimes at the boundary stange thing happens and it becomes infeasible
                if (status == GRB_OPTIMAL) {
                    double prev_value = verifier_p->counter_example_.index({i, j}).item<double>();
                    double current_value;
                    if (j == 0) {
                        current_value = min(prev_value, var.get(GRB_DoubleAttr_X));
                        verifier_p->counter_example_.index_put_({i, j}, current_value);
                    }
                    else if (j == 1) {
                        current_value = max(prev_value, var.get(GRB_DoubleAttr_X));
                        verifier_p->counter_example_.index_put_({i, j}, current_value);
                    }
                    else assert(false);
                    // cout << "  previous value: " << prev_value << endl;
                    // cout << "  current value: " << current_value << endl;
                    // cout << verifier_p->solver_.ConvexVarStr(var) << endl;
                    // cout << verifier_p->counter_example_ << endl;
                }
            }
        }
        // cout << verifier_p->counter_example_ << endl;
        // printf("end checking\n");
        // getchar();
        return 1;
    };
    solution_checker_f_ = bind(UpdateSolutionBound, this);
    full_search_b_ = true;
    iteration_ = 0;
    parameter_.timeout_b = true;

    GRBModel& model = solver_.ConvexSolver();
    vector<GRBVar> var_v = env_p_->StateVar(0);

    torch::Tensor bound = env_p_->InitialConstraint();
    vector<torch::Tensor> bound_v;
    bound_v.emplace_back(torch::cat({torch::flatten(torch::transpose(bound, 0, 1)), torch::tensor({usage.GetTime().realTime}), torch::tensor({iteration_})}, 0));

    printf("step 1:\n");
    cout << bound << endl;
    // getchar();
    for (int i = 0; i < int((horizon - 1) / (env_p_->StateHorizon() - 1)); ++i) {
        solver_.ResetSat();
        env_p_->Reset();
        env_p_->SetRelationSDT(soft_decision_tree_v_);

        vector<GRBConstr> constr_v;
        for (int j = 0; j < (int)var_v.size(); ++j) {
            constr_v.emplace_back(model.addConstr(var_v[j] >= bound.index({j, 0}).item<double>()));
            constr_v.emplace_back(model.addConstr(var_v[j] <= bound.index({j, 1}).item<double>()));
        }
        model.update();

        printf("%s\n", solver_.Str().c_str());
        getchar();

        counter_example_ = torch::cat({torch::ones({(int)var_v.size(), 1}, torch::dtype(torch::kFloat64)) * DBL_MAX, torch::ones({(int)var_v.size(), 1}, torch::dtype(torch::kFloat64)) * -DBL_MAX}, 1);

        int res;
        bool except = false;

        try {
            res = Solve_Smc();
        }
        catch (const runtime_error& err) {
            cout << err.what() << endl;
            except = true;

            for (int j = 0; j < (int)constr_v.size(); ++j) {
                model.remove(constr_v[j]);
            }
            break;
        }
        if (!except) {
            assert(res == 0);

            bound = counter_example_;
            bound_v.emplace_back(torch::cat({torch::flatten(torch::transpose(bound, 0, 1)), torch::tensor({usage.GetTime().realTime}), torch::tensor({iteration_})}, 0));

            printf("step %d: iterator %d\n", (i + 1) * (env_p_->StateHorizon() - 1) + 1, iteration_);
            cout << bound << endl;
            getchar();

            for (int j = 0; j < (int)constr_v.size(); ++j) {
                model.remove(constr_v[j]);
            }
        }
    }
    return bound_v;
}

vector<torch::Tensor> CVerifier::Verify_SDT_Composition_Car_racing(int horizon) {
    assert(env_p_->ActionHorizon() == (int)soft_decision_tree_v_.size());
    int& cos_approx_param = env_p_->WaveApproxParam();
    cos_approx_param = env_p_->parameter_.wave_approx_limit;

    env_p_->ClearInitialConstraint();
    env_p_->ClearSafeConstraint();
    env_p_->SetRelationSDT(soft_decision_tree_v_, true);
    auto UpdateSolutionBound = [](CVerifier* verifier_p) {
        return 1;
    };
    solution_checker_f_ = bind(UpdateSolutionBound, this);
    full_search_b_ = true;
    iteration_ = 0;
    parameter_.timeout_b = true;

    GRBModel& model = solver_.ConvexSolver();
    vector<GRBVar> var_v = env_p_->StateVar(0);

    torch::Tensor bound = env_p_->InitialConstraint();

    vector<torch::Tensor> bound_v;
    bound_v.emplace_back(torch::cat({torch::flatten(torch::transpose(bound, 0, 1)), torch::tensor({usage.GetTime().realTime}), torch::tensor({iteration_})}, 0));

    printf("step 1:\n");
    cout << bound << endl;
    for (int i = 0; i < int((horizon) / (env_p_->StateHorizon() - 1)); ++i) {
        solver_.ResetSat();
        env_p_->Reset();
        env_p_->SetRelationSDT(soft_decision_tree_v_);

        vector<GRBConstr> constr_v;
        for (int j = 0; j < (int)var_v.size(); ++j) {
            constr_v.emplace_back(model.addConstr(var_v[j] >= bound.index({j, 0}).item<double>()));
            constr_v.emplace_back(model.addConstr(var_v[j] <= bound.index({j, 1}).item<double>()));
        }
        model.update();

        // printf("%s\n", solver_.Str().c_str());
        // getchar();

        counter_example_ = torch::cat({torch::ones({(int)var_v.size(), 1}, torch::dtype(torch::kFloat64)) * DBL_MAX, torch::ones({(int)var_v.size(), 1}, torch::dtype(torch::kFloat64)) * -DBL_MAX}, 1);

        int res;
        bool except = false;

        try {
            res = Solve_Smc();
        }
        catch (const runtime_error& err) {
            cout << err.what() << endl;
            except = true;

            for (int j = 0; j < (int)constr_v.size(); ++j) {
                model.remove(constr_v[j]);
            }
            break;
        }
        if (!except) {
            assert(res == 0);

            bound = counter_example_;
            bound_v.emplace_back(torch::cat({torch::flatten(torch::transpose(bound, 0, 1)), torch::tensor({usage.GetTime().realTime}), torch::tensor({iteration_})}, 0));

            printf("step %d: iterator %d\n", (i + 1) * (env_p_->StateHorizon() - 1) + 1, iteration_);
            cout << bound << endl;
            // getchar();

            for (int j = 0; j < (int)constr_v.size(); ++j) {
                model.remove(constr_v[j]);
            }
        }
    }
    return bound_v;
}

int CVerifier::Verify_NN_Monolith() {
    int horizon = env_p_->StateHorizon();
    int& cos_approx_param = env_p_->WaveApproxParam();

    assert((int)neural_network_v_.size() == horizon - 1);
    for (int i = 0; i < horizon - 1; ++i) {
        CNN& nn = neural_network_v_[i];
        nn.SetBound(env_p_->Bound().unsqueeze(0));
    }

    env_p_->SetRelationNN(neural_network_v_);
    auto SetCounterExample = [](CVerifier* verifier_p) {
        // printf("star checking\n");
        // cout << verifier_p->solver_.Str() << endl;
        torch::Tensor input = verifier_p->env_p_->State(0);
        torch::Tensor state = input.clone();

        int horizon = verifier_p->env_p_->StateHorizon();
        int res = 2;
        // printf("start to check\n");
        for (int i = 0; i < horizon - 1; ++i) {
            // printf("action step: %d\n", i);
            // cout << state << endl;
            torch::Tensor action = verifier_p->neural_network_v_[i].Forward(state);
            int action_id = verifier_p->env_p_->Action(i);
            // cout << action << endl;
            // printf("predicted action: %d\n", action_id);
            if (torch::abs(action.index({action_id}) - torch::max(action)).item<double>() > verifier_p->solver_.parameter_.kEPS) {
                res = 0;
                break;
            }
            // minus action_id for real action
            if (res == 2) {
                torch::Tensor next_state = verifier_p->env_p_->State(i+1);
                if (verifier_p->env_p_->CheckNextState(state, action_id, next_state) == 0) {
                    res = 1;
                }
                state = next_state;
            }
            else {
                state = verifier_p->env_p_->State(i+1);
            }
        }
        if (res == 2) {
            verifier_p->counter_example_ = verifier_p->env_p_->Trajectory();
        }
        // printf("result: %d\n", res);
        // getchar();
        return res;
    };
    solution_checker_f_ = bind(SetCounterExample, this);
    full_search_b_ = false;
    iteration_ = 0;
    // solver_.WriteConvexSolver();
    // printf("write\n");
    // getchar();

    int result = 0;
    while (true) {
        solver_.ResetSat();
        env_p_->Reset();

        counter_example_ = torch::empty({0});
        result = Solve_Smc();
        // cout << "start->" << (counter_example_.numel() != 0) << endl;
        if (result == 1) {
            if (counter_example_.numel() != 0) {
                break;
            }
            else if (cos_approx_param * 2 <= env_p_->parameter_.wave_approx_limit) {
                cos_approx_param *= 2;
                printf("find counter example and set cosine approximation to %d\n", cos_approx_param);
            }
            else assert(false);
        }
        else if (result == 0) {
            // printf("can not find counter example\n");
            break;
        }
        else assert(false);
    }
    if (result == 1) {
        cout << counter_example_ << endl;
        // printf("%s", env_p_->Str().c_str());
        printf("NeuralNetwork is unsafe within %d iteration(s).\n", iteration_);
    }
    else if (result == 0) {
        printf("NeuralNetwork is safe within %d iteration(s).\n", iteration_);
    }
    // WriteConvexSolver();
    return result;
}

vector<torch::Tensor> CVerifier::Verify_NN_Composition(int horizon) {
    assert(env_p_->ActionHorizon() == (int)neural_network_v_.size());
    assert((int)env_p_->StateVar(0).size() == neural_network_v_[0].NumNeuron()[0]);
    assert(env_p_->ActionNum() == neural_network_v_[0].NumNeuron().back());
    int& cos_approx_param = env_p_->WaveApproxParam();
    cos_approx_param = env_p_->parameter_.wave_approx_limit;

    for (int i = 0; i < (int)neural_network_v_.size(); ++i) {
        neural_network_v_[i].SetBound(env_p_->Bound().unsqueeze(0));
        neural_network_v_[i].JumpAssertion() = false;
    }

    env_p_->ClearInitialConstraint();
    env_p_->ClearSafeConstraint();
    env_p_->SetRelationNN(neural_network_v_);
    auto UpdateSolutionBound = [](CVerifier* verifier_p) {
        bool full_assign_b = true;

        for (int i = 0; i < (int)verifier_p->neural_network_v_.size(); ++i) {
            vector<int>& num_neuron_v = verifier_p->neural_network_v_[i].NumNeuron();
            vector<torch::Tensor>& sat_network_status_v = verifier_p->neural_network_v_[i].SatNetworkStatus();
            vector<torch::Tensor>& nonlinear_neuron_v = verifier_p->neural_network_v_[i].NonlinearNeuron();
            for (int l = 1; l < (int)num_neuron_v.size() - 1; ++l) {
                // printf("layer: %d\n", l);
                assert((int)sat_network_status_v[l].sizes()[0] == num_neuron_v[l]);
                for (int i = 0; i < nonlinear_neuron_v[l].sizes()[0]; ++i) {
                    int neuron_id = nonlinear_neuron_v[l].index({i}).item<int>();
                    // printf("neuron_id: %d\n", neuron_id);
                    // cout << (sat_network_status_v[l].index({neuron_id}).item<int>() == 0) << endl;
                    full_assign_b &= !(sat_network_status_v[l].index({neuron_id}).item<int>() == 0);
                }
            }
        }
        // cout << full_assign_b << endl;
        // getchar();

        if (full_assign_b) {
            // printf("star checking\n");
            GRBModel& convex_solver = verifier_p->solver_.ConvexSolver();
            vector<GRBVar> var_v = verifier_p->env_p_->StateVar(verifier_p->env_p_->StateHorizon() - 1);

            // cout << verifier_p->solver_.Str() << endl;
            // cout << verifier_p->neural_network_v_[0].ConvexStr() << endl;
            for (int i = 0; i < (int)var_v.size(); ++i) {
                for (int j = 0; j < 2; ++j) {
                    // printf("handling i: %d, j: %d\n", i, j);
                    // cout << verifier_p->counter_example_ << endl;
                    GRBVar var = var_v[i];
                    if (j == 0) {
                        verifier_p->solver_.SetConvexObjective(var);
                    }
                    else if (j == 1) {
                        verifier_p->solver_.SetConvexObjective(-1 * var);
                    }
                    else assert(false);
                    // verifier_p->solver_.WriteConvexSolver();

                    convex_solver.optimize();
                    // cout << verifier_p->env_p_->Trajectory() << endl;
                    int status = convex_solver.get(GRB_IntAttr_Status);
                    // status should be optimal, but sometimes at the boundary stange thing happens and it becomes infeasible
                    if (status == GRB_OPTIMAL) {
                        assert(status == GRB_OPTIMAL);
                        double prev_value = verifier_p->counter_example_.index({i, j}).item<double>();
                        double current_value;
                        if (j == 0) {
                            current_value = min(prev_value, var.get(GRB_DoubleAttr_X));
                            verifier_p->counter_example_.index_put_({i, j}, current_value);
                        }
                        else if (j == 1) {
                            current_value = max(prev_value, var.get(GRB_DoubleAttr_X));
                            verifier_p->counter_example_.index_put_({i, j}, current_value);
                        }
                        else assert(false);
                        // cout << "  previous value: " << prev_value << endl;
                        // cout << "  current value: " << current_value << endl;
                        // cout << verifier_p->solver_.ConvexVarStr(var) << endl;
                        // cout << verifier_p->counter_example_ << endl;
                    }
                }
            }
            // cout << verifier_p->counter_example_ << endl;
            // printf("end checking\n");
            // getchar();
            return 1;
        }
        else {
            // printf("refuse checking\n");
            return 0;
        }
    };
    solution_checker_f_ = bind(UpdateSolutionBound, this);
    full_search_b_ = true;
    iteration_ = 0;
    parameter_.timeout_b = true;

    GRBModel& model = solver_.ConvexSolver();
    vector<GRBVar> var_v = env_p_->StateVar(0);

    torch::Tensor bound = env_p_->InitialConstraint();
    vector<torch::Tensor> bound_v;
    bound_v.emplace_back(torch::cat({torch::flatten(torch::transpose(bound, 0, 1)), torch::tensor({usage.GetTime().realTime}), torch::tensor({iteration_})}, 0));

    printf("step 1:\n");
    cout << bound << endl;
    for (int i = 0; i < int((horizon - 1) / (env_p_->StateHorizon() - 1)); ++i) {
        solver_.ResetSat();
        env_p_->Reset();

        vector<GRBConstr> constr_v;
        for (int j = 0; j < (int)var_v.size(); ++j) {
            constr_v.emplace_back(model.addConstr(var_v[j] >= bound.index({j, 0}).item<double>()));
            constr_v.emplace_back(model.addConstr(var_v[j] <= bound.index({j, 1}).item<double>()));
        }
        model.update();

        // printf("%s\n", solver_.Str().c_str());
        // getchar();

        counter_example_ = torch::cat({torch::ones({(int)var_v.size(), 1}, torch::dtype(torch::kFloat64)) * DBL_MAX, torch::ones({(int)var_v.size(), 1}, torch::dtype(torch::kFloat64)) * -DBL_MAX}, 1);

        int res;
        bool except = false;
        try {
            res = Solve_Smc();
        }
        catch (const runtime_error& err) {
            cout << err.what() << endl;
            except = true;

            for (int j = 0; j < (int)constr_v.size(); ++j) {
                model.remove(constr_v[j]);
            }
            break;
        }
        if (!except) {
            assert(res == 0);

            bound = counter_example_;
            bound_v.emplace_back(torch::cat({torch::flatten(torch::transpose(bound, 0, 1)), torch::tensor({usage.GetTime().realTime}), torch::tensor({iteration_})}, 0));

            printf("step %d: iterator %d\n", (i + 1) * (env_p_->StateHorizon() - 1) + 1, iteration_);
            cout << bound << endl;

            for (int j = 0; j < (int)constr_v.size(); ++j) {
                model.remove(constr_v[j]);
            }
        }
    }
    // WriteConvexSolver();
    return bound_v;
};

vector<torch::Tensor> CVerifier::Verify_NN_Composition_Car_racing(int horizon) {
    assert(env_p_->ActionHorizon() == (int)neural_network_v_.size());
    int& cos_approx_param = env_p_->WaveApproxParam();
    cos_approx_param = env_p_->parameter_.wave_approx_limit;

    for (int i = 0; i < (int)neural_network_v_.size(); ++i) {
        neural_network_v_[i].SetBound(env_p_->Bound().unsqueeze(0));
        neural_network_v_[i].JumpAssertion() = false;
    }

    env_p_->ClearInitialConstraint();
    env_p_->ClearSafeConstraint();
    env_p_->SetRelationNN(neural_network_v_);
    auto UpdateSolutionBound = [](CVerifier* verifier_p) {
        bool full_assign_b = true;

        for (int i = 0; i < (int)verifier_p->neural_network_v_.size(); ++i) {
            vector<int>& num_neuron_v = verifier_p->neural_network_v_[i].NumNeuron();
            vector<torch::Tensor>& sat_network_status_v = verifier_p->neural_network_v_[i].SatNetworkStatus();
            vector<torch::Tensor>& nonlinear_neuron_v = verifier_p->neural_network_v_[i].NonlinearNeuron();
            for (int l = 1; l < (int)num_neuron_v.size() - 1; ++l) {
                // printf("layer: %d\n", l);
                assert((int)sat_network_status_v[l].sizes()[0] == num_neuron_v[l]);
                for (int i = 0; i < nonlinear_neuron_v[l].sizes()[0]; ++i) {
                    int neuron_id = nonlinear_neuron_v[l].index({i}).item<int>();
                    // printf("neuron_id: %d\n", neuron_id);
                    // cout << (sat_network_status_v[l].index({neuron_id}).item<int>() == 0) << endl;
                    full_assign_b &= !(sat_network_status_v[l].index({neuron_id}).item<int>() == 0);
                }
            }
        }
        // cout << full_assign_b << endl;
        // getchar();

        if (full_assign_b) {
            // printf("star checking\n");
            GRBModel& convex_solver = verifier_p->solver_.ConvexSolver();
            vector<GRBVar> var_v = verifier_p->env_p_->StateVar(verifier_p->env_p_->StateHorizon() - 1);

            // cout << verifier_p->solver_.Str() << endl;
            // cout << verifier_p->neural_network_v_[0].ConvexStr() << endl;
            for (int i = 0; i < (int)var_v.size(); ++i) {
                for (int j = 0; j < 2; ++j) {
                    // printf("handling i: %d, j: %d\n", i, j);
                    // cout << verifier_p->counter_example_ << endl;
                    GRBVar var = var_v[i];
                    if (j == 0) {
                        verifier_p->solver_.SetConvexObjective(var);
                    }
                    else if (j == 1) {
                        verifier_p->solver_.SetConvexObjective(-1 * var);
                    }
                    else assert(false);
                    // verifier_p->solver_.WriteConvexSolver();

                    convex_solver.optimize();
                    // cout << verifier_p->env_p_->Trajectory() << endl;
                    int status = convex_solver.get(GRB_IntAttr_Status);
                    // status should be optimal, but sometimes at the boundary stange thing happens and it becomes infeasible
                    if (status == GRB_OPTIMAL) {
                        assert(status == GRB_OPTIMAL);
                        double prev_value = verifier_p->counter_example_.index({i, j}).item<double>();
                        double current_value;
                        if (j == 0) {
                            current_value = min(prev_value, var.get(GRB_DoubleAttr_X));
                            verifier_p->counter_example_.index_put_({i, j}, current_value);
                        }
                        else if (j == 1) {
                            current_value = max(prev_value, var.get(GRB_DoubleAttr_X));
                            verifier_p->counter_example_.index_put_({i, j}, current_value);
                        }
                        else assert(false);
                        // cout << "  previous value: " << prev_value << endl;
                        // cout << "  current value: " << current_value << endl;
                        // cout << verifier_p->solver_.ConvexVarStr(var) << endl;
                        // cout << verifier_p->counter_example_ << endl;
                    }
                }
            }
            // cout << verifier_p->counter_example_ << endl;
            // printf("end checking\n");
            // cout << verifier_p->env_p_->Bound() << endl;
            if (torch::equal(verifier_p->env_p_->Bound(), verifier_p->counter_example_)) {
                verifier_p->full_search_b_ = false;
            }
            return 1;
        }
        else {
            // printf("refuse checking\n");
            return 0;
        }
    };
    solution_checker_f_ = bind(UpdateSolutionBound, this);
    full_search_b_ = true;
    iteration_ = 0;
    parameter_.timeout_b = true;

    GRBModel& model = solver_.ConvexSolver();
    vector<GRBVar> var_v = env_p_->StateVar(0);

    torch::Tensor bound = env_p_->InitialConstraint();
    vector<torch::Tensor> bound_v;
    bound_v.emplace_back(torch::cat({torch::flatten(torch::transpose(bound, 0, 1)), torch::tensor({usage.GetTime().realTime}), torch::tensor({iteration_})}, 0));

    printf("step 1:\n");
    cout << bound << endl;
    for (int i = 0; i < int((horizon) / (env_p_->StateHorizon() - 1)); ++i) {
        solver_.ResetSat();
        env_p_->Reset();

        vector<GRBConstr> constr_v;
        for (int j = 0; j < (int)var_v.size(); ++j) {
            constr_v.emplace_back(model.addConstr(var_v[j] >= bound.index({j, 0}).item<double>()));
            constr_v.emplace_back(model.addConstr(var_v[j] <= bound.index({j, 1}).item<double>()));
        }
        model.update();

        // printf("%s\n", solver_.Str().c_str());
        // getchar();

        counter_example_ = torch::cat({torch::ones({(int)var_v.size(), 1}, torch::dtype(torch::kFloat64)) * DBL_MAX, torch::ones({(int)var_v.size(), 1}, torch::dtype(torch::kFloat64)) * -DBL_MAX}, 1);

        int res;
        bool except = false;
        try {
            res = Solve_Smc();
        }
        catch (const runtime_error& err) {
            cout << err.what() << endl;
            except = true;

            for (int j = 0; j < (int)constr_v.size(); ++j) {
                model.remove(constr_v[j]);
            }
            break;
        }
        if (!except) {
            // assert(res == 0);

            bound = counter_example_;
            bound_v.emplace_back(torch::cat({torch::flatten(torch::transpose(bound, 0, 1)), torch::tensor({usage.GetTime().realTime}), torch::tensor({iteration_})}, 0));

            printf("step %d: iterator %d\n", (i + 1) * (env_p_->StateHorizon() - 1) + 1, iteration_);
            cout << bound << endl;

            for (int j = 0; j < (int)constr_v.size(); ++j) {
                model.remove(constr_v[j]);
            }
        }
    }
    // WriteConvexSolver();
    return bound_v;
};

void CVerifier::CheckApproximationNN() {
    assert(neural_network_v_.size() == 1);
    GRBModel& convex_solver = solver_.ConvexSolver();
    vector<vector<GRBVar> >& convex_network_var_v = neural_network_v_[0].ConvexNetworkVariable();

    torch::Tensor lb = torch::zeros({neural_network_v_[0].NumNeuron()[0]}, torch::dtype(torch::kFloat64));
    torch::Tensor ub = torch::ones({neural_network_v_[0].NumNeuron()[0]}, torch::dtype(torch::kFloat64));

    int N = 1000;
    double maximum_error = 0;
    for (int i = 0; i < N; ++i) {
        double error = 0;
        vector<GRBConstr> constr_v;
        torch::Tensor input = lb + (ub - lb) / N * i;
        neural_network_v_[0].SetBound(input.index({torch::indexing::None, torch::indexing::Slice(),torch::indexing::None}).repeat({1, 2, 1}));

        for (int j = 0; j < (int)convex_network_var_v[0].size(); ++j) {
            double x = input.index({j}).item<double>();
            constr_v.emplace_back(convex_solver.addConstr(convex_network_var_v[0][j] == x));
        }
        convex_solver.optimize();
        assert(convex_solver.get(GRB_IntAttr_Status) == GRB_OPTIMAL);

        torch::Tensor output = neural_network_v_[0].Forward(input);
        for (int j = 0; j < (int)convex_network_var_v.back().size(); ++j) {
            error += abs(output.index({j}).item<double>() - convex_network_var_v.back()[j].get(GRB_DoubleAttr_X));
            // cout << "value: " << convex_network_var_v.back()[j].get(GRB_DoubleAttr_X) << "\n";
        }
        if (error > 0.001) {
            cout << output << endl;
            for (int j = 0; j < (int)convex_network_var_v.back().size(); ++j) {
                cout << "value: " << convex_network_var_v.back()[j].get(GRB_DoubleAttr_X) << "\n";
            }
        }

        // WriteConvexSolver();
        // getchar();
        
        maximum_error = max(maximum_error, error);
        for (int j = 0; j < (int)constr_v.size(); ++j) {
            convex_solver.remove(constr_v[j]);
        }
    }
    printf("maximum_error: %f\n", maximum_error);
}

void CVerifier::CheckApproximationMountainCar() {
    assert(env_p_->StateHorizon() == 1);
    env_p_->ClearInitialConstraint();
    env_p_->ClearSafeConstraint();

    GRBModel& convex_solver = solver_.ConvexSolver();

    vector<vector<GRBVar> > & time_state_var_vv = env_p_->TimeStateVarVV();
    int& cos_approx_param = env_p_->WaveApproxParam();

    int N = 1000;
    while (cos_approx_param <= env_p_->parameter_.wave_approx_limit) {
        double maximum_error = 0;
        for (int i = 1; i < N; ++i) {
            GRBVar& p = time_state_var_vv[0][0];
            double x = p.get(GRB_DoubleAttr_LB) + p.get(GRB_DoubleAttr_UB) / N * i;

            GRBConstr c = convex_solver.addConstr(p == x);
            solver_.ResetSat();
            env_p_->Reset();

            auto SetCosineValue = [](CVerifier* verifier_p) {
                verifier_p->counter_example_ = torch::tensor({verifier_p->env_p_->TimeStateVarVV()[0][1].get(GRB_DoubleAttr_X)});
                return 1;
            };
            solution_checker_f_ = bind(SetCosineValue, this);
            int res = Solve_Smc();
            assert(res == 1);

            double value = counter_example_.item<double>();
            // printf("result: %d\n", res);
            // printf("input: %f, value: %f, real value: %f", x, value, cos(3*x));


            maximum_error = max(maximum_error, abs(value - cos(3 * x)));
            convex_solver.remove(c);
        }
        printf("cosine approximation parameter: %d, maximum_error: %f\n", cos_approx_param, maximum_error);
        // getchar();
        cos_approx_param *= 2;
    }
}

void CVerifier::CheckApproximationCartPole() {
    assert(env_p_->StateHorizon() == 1);
    env_p_->ClearInitialConstraint();
    env_p_->ClearSafeConstraint();

    GRBModel& convex_solver = solver_.ConvexSolver();

    vector<vector<GRBVar> > & time_state_var_vv = env_p_->TimeStateVarVV();
    int& cos_approx_param = env_p_->WaveApproxParam();

    int N = 10;
    while (cos_approx_param <= env_p_->parameter_.wave_approx_limit) {
        double maximum_error = 0;
        for (int i = 1; i < N; ++i) {
            GRBVar& p = time_state_var_vv[0][2];
            double x = p.get(GRB_DoubleAttr_LB) + p.get(GRB_DoubleAttr_UB) / N * i;

            GRBConstr c = convex_solver.addConstr(p == x);
            solver_.ResetSat();
            env_p_->Reset();

            auto SetCosineValue = [](CVerifier* verifier_p) {
                verifier_p->counter_example_ = torch::tensor({verifier_p->env_p_->TimeStateVarVV()[0][4].get(GRB_DoubleAttr_X), verifier_p->env_p_->TimeStateVarVV()[0][5].get(GRB_DoubleAttr_X)});
                return 1;
            };
            solution_checker_f_ = bind(SetCosineValue, this);
            int res = Solve_Smc();
            assert(res == 1);

            double cos_value = counter_example_.index({0}).item<double>();
            double sin_value = counter_example_.index({1}).item<double>();
            // printf("result: %d\n", res);
            // printf("input: %f, value: %f, real value: %f\n", x, cos_value, cos(x));
            // printf("input: %f, value: %f, real value: %f\n", x, sin_value, sin(x));
            // getchar();

            maximum_error = max(maximum_error, abs(cos_value - cos(x)));
            maximum_error = max(maximum_error, abs(sin_value - sin(x)));
            convex_solver.remove(c);
        }
        printf("cosine approximation parameter: %d, maximum_error: %f\n", cos_approx_param, maximum_error);
        // getchar();
        cos_approx_param *= 2;
    }
}

string CVerifier::Str() const {
    string res;
    if (counter_example_.numel() == 0) {
        res += "NeuralNetwork is safe within " + to_string(iteration_) + " iteration(s).\n";
    }
    else {
        res += "NeuralNetwork is not safe within " + to_string(iteration_) + " iteration(s).\n";
    }
    return res;
};

int CVerifier::Solve_Smc() {
    // cout << neural_network_v_[0].Str() << endl;
    #ifdef VERBOSE
    printf("%s", solver_.Str().c_str());
    printf("%s", neural_network_v_[0].Str().c_str());
    #endif
    int result = 0;
    while (true) {
        if (parameter_.timeout_b) {
            usage.CheckTimeout(parameter_.timeout_d);
        }

        bool sat_res = Solve_Sat();
        if (sat_res) {
            vector<z3::expr> conflict;
            result = Solve_Convex(conflict);
            // solver_.WriteConvexSolver();
            // printf("result: %d\n", result);
            // cout << full_search_b_ << endl;
            if (conflict.size() > 0 and (full_search_b_ or result == 0)) {
                z3::expr expr = conflict[0];
                for (int i = 1; i < (int)conflict.size(); ++i) {
                    expr = expr || conflict[i];
                }
                // cout << expr << endl;
                solver_.AddSatConstraint(expr);
            }
            else if (conflict.size() == 0 or (!full_search_b_ and result == 1)) {
                if (full_search_b_) {
                    result = 0;
                }
                break;
            }
            else assert(false);
        }
        else {
            result = 0;
            break;
        }
        ++iteration_;
        // printf("iteration: %d\n", iteration_);
    }
    return result;
}

int CVerifier::Solve_Sat() {
    #ifdef DEBUG
    printf("sat checking\n");
    #endif
    if (solver_.CheckSat()) {
        #ifdef DEBUG
        printf("sat result: true\n");
        cout << solver_.Str() << endl;
        getchar();
        #endif
        return 1;
    }
    else {
        #ifdef DEBUG
        printf("sat result: false\n");
        cout << solver_.Str() << endl;
        getchar();
        #endif
        return 0;
    }
}

int CVerifier::Solve_Convex(vector<z3::expr>& conflict) {
    // set constraint from sat result
    solver_.SetConvexConstrant();

    // update neuron status from sat result
    solver_.ConvexObjective() = GRBLinExpr();
    for (int i = 0; i < (int)neural_network_v_.size(); ++i) {
        neural_network_v_[i].CheckSatStatus();
        neural_network_v_[i].SetBoundAssertion();
    }
    solver_.SetConvexObjective();

    GRBModel& convex_solver = solver_.ConvexSolver();
    int result = 0;
    while (true) {
        if (parameter_.timeout_b) {
            usage.CheckTimeout(parameter_.timeout_d);
        }

        #ifdef DEBUG
        printf("write model\n");
        solver_.WriteConvexSolver();
        for (int i = 0; i < (int)neural_network_v_.size(); ++i) {
            printf("controller id: %d\n", i);
            printf("%s", neural_network_v_[i].SatStr().c_str());
        }
        getchar();
        #endif
        convex_solver.optimize();
        int status = convex_solver.get(GRB_IntAttr_Status);
        bool satisfiable = ((solver_.Certificate() != CSolver::Certificate::DUAL and status == GRB_OPTIMAL) or (solver_.Certificate() == CSolver::Certificate::DUAL and status == GRB_OPTIMAL and convex_solver.get(GRB_DoubleAttr_ObjVal) <= numeric_limits<double>::epsilon()));
        #ifdef DEBUG
        printf("result: %d\n", status);
        if (status == GRB_OPTIMAL) {
            printf("obj: %lf\n", convex_solver.get(GRB_DoubleAttr_ObjVal));
            for (int i = 0; i < (int)neural_network_v_.size(); ++i) {
                printf("controller id: %d\n", i);
                printf("%s", neural_network_v_[i].ConvexStr().c_str());
            }
        }
        else {
            convex_solver.computeIIS();
            convex_solver.write("check.ilp");
            printf("write iis\n");
        }
        cout << satisfiable << endl;
        getchar();
        #endif
        if (satisfiable) {
            // printf("%s", solver_.Str().c_str());
            // printf("%s", env_p_->Str().c_str());
            int check_res = solution_checker_f_();
            if (check_res) {
                #ifdef DEBUG
                printf("checking successful\n");
                getchar();
                #endif
                if (full_search_b_) {
                    enum CSolver::Certificate certificate = solver_.Certificate();
                    solver_.SetCertificate(CSolver::Certificate::NONE);
                    ComputeCertificate(conflict);
                    solver_.SetCertificate(certificate);
                    // getchar();
                }
                result = 1;
                break;
            }
            else {
                int num_assertion = 0;
                for (int i = 0; num_assertion < solver_.parameter_.kAssertion and i < (int)neural_network_v_.size(); ++i) {
                    #ifdef VERBOSE
                    printf("controller id: %d\n", i);
                    #endif
                    neural_network_v_[i].AssertSatStatus(num_assertion);
                }
                solver_.ConvexObjective() = GRBLinExpr();
                for (int i = 0; i < (int)neural_network_v_.size(); ++i) {
                    neural_network_v_[i].SetBoundAssertion();
                }
                solver_.SetConvexObjective();
                #ifdef DEBUG
                printf("write model after assertion\n");
                printf("number of assertion: %d\n", num_assertion);
                solver_.WriteConvexSolver();
                getchar();
                #endif
                assert(num_assertion > 0);
            }
        }
        else {
            ComputeCertificate(conflict);
            #ifdef DEBUG
            printf("return certificate\n");
            for (int i = 0; i < (int)conflict.size(); ++i) {
                cout << conflict[i] << endl;
            }
            getchar();
            #endif
            result = 0;
            break;
        };
        ++iteration_;
        // printf("iteration: %d\n", iteration_);
    }
    solver_.RemoveConvexConstrant();
    return result;
}

void CVerifier::ResolveConvexByDualCertificate() {
    GRBLinExpr expr;

    vector<int>& num_neuron_v = neural_network_v_[0].NumNeuron();
    vector<torch::Tensor>& nonlinear_neuron_v = neural_network_v_[0].NonlinearNeuron();
    vector<vector<CSolver::SatVar*>>& sat_network_var_v = neural_network_v_[0].SatNetworkVariable();
    vector<torch::Tensor>& sat_network_status_v = neural_network_v_[0].SatNetworkStatus();

    assert(sat_network_var_v.size() == num_neuron_v.size() - 1);
    assert(sat_network_status_v.size() == num_neuron_v.size() - 1);
    solver_.SetCertificate(CSolver::Certificate::NONE);
    for (int l = 1; l < (int)num_neuron_v.size() - 1; ++l) {
        assert((int)sat_network_var_v[l].size() == num_neuron_v[l]);
        assert((int)sat_network_status_v[l].sizes()[0] == num_neuron_v[l]);
        for (int i = 0; i < nonlinear_neuron_v[l].sizes()[0]; ++i) {
            int neuron_id = nonlinear_neuron_v[l].index({i}).item<int>();
            int neural_status = sat_network_status_v[l].index({neuron_id}).item<int>();
            if (neural_status == 1) {
                neural_network_v_[0].AddConvexAssignmentConstraint(l, neuron_id, neural_network_v_[0].ConvexSlackVariable()[l][neuron_id] == 0);
            }
            else if (neural_status == -1) {
                neural_network_v_[0].AddConvexAssignmentConstraint(l, neuron_id, neural_network_v_[0].ConvexReluVariable()[l][neuron_id] == 0);
            }
            else {
                expr += (neural_network_v_[0].ConvexReluVariable()[l][neuron_id] + neural_network_v_[0].ConvexSlackVariable()[l][neuron_id]) * neural_network_v_[0].ObjectiveRatio(l);
            }
        }
    }
    solver_.SetCertificate(CSolver::Certificate::DUAL);
    solver_.SetConvexObjective(expr);
    solver_.ConvexSolver().optimize();
    assert(solver_.ConvexSolver().get(GRB_IntAttr_Status) == GRB_OPTIMAL);

    #ifdef DEBUG
    printf("After resolving\n");
    solver_.WriteConvexSolver();
    for (int i = 0; i < (int)neural_network_v_.size(); ++i) {
        printf("controller id: %d\n", i);
        printf("%s", neural_network_v_[i].ConvexStr().c_str());
    //     printf("  decision process:\n");
    //     for (int j = 0; j < 6; ++j) {
    //         printf("    %s", solver_.VarStr(solver_.ConvexSolver().getVarByName("action_"+to_string(i)+"_"+to_string(j))).c_str());
    //     }
    }
    #endif
}

void CVerifier::ComputeCertificate(vector<z3::expr>& conflict) {
    bool change_certificate_iis_none = false;
    if (solver_.Certificate() == CSolver::Certificate::IIS) {
        try {
            solver_.ConvexSolver().computeIIS();
        }
        catch (GRBException e) {
            assert(e.getErrorCode() == 10015);
            change_certificate_iis_none = true;
            solver_.SetCertificate(CSolver::Certificate::NONE);
        }
    }
    for (int i = 0; i < (int)neural_network_v_.size(); ++i) {
        neural_network_v_[i].ComputeCertificate(conflict);
    }
    try {
    solver_.ComputeConvexCertificate(conflict);
    }
    catch (GRBException e) {
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
        assert(false);
    }

    if (change_certificate_iis_none) {
        solver_.SetCertificate(CSolver::Certificate::IIS);
    }
}

void CVerifier::WriteOutput() {
    FILE* file_p = fopen("out.txt", "wa");
    if (counter_example_.numel() == 0) {
        fprintf(file_p, "unsat");
    }
    else {
        fprintf(file_p, "sat");
    }
    fclose(file_p);
};

NNV_NAMESPACING_END
