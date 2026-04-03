#include "nn.hpp"

NNV_NAMESPACING_START

int CNN::Run(int argc, char **argv) {
    ArgumentParser program("parse");

    program.AddArgument("-i", "--input")
        .Help("taken the neural network from the path as the model.")
        .Nargs(1);

    program.AddArgument("-e", "--eps")
        .Help("taken the eps as the perturbation.")
        .Nargs(1)
        .Action([&](const auto & s) { return stod(s); });

    bool res;
    try {
    res = program.ParseArgs(argc, argv);
    }
    catch (const runtime_error& err) {
        std::cout << err.what() << std::endl;
        return 1;
    }
    if (res) return 0;

    if (program.Present("--input")) {
        LoadModel(program.Get<string>("--input"));
    }
    if (program.Present("--eps")) {
        double eps = program.Get<double>("--eps");
        torch::Tensor bound = torch::zeros({num_neuron_v_[0], 2}, torch::dtype(torch::kFloat64));
        bound.index_put_({torch::indexing::Slice(), 0}, torch::clamp(Input() - eps, 0, c10::nullopt));
        bound.index_put_({torch::indexing::Slice(), 1}, torch::clamp(Input() + eps, c10::nullopt, 1));
        SetBound(bound.unsqueeze(0));
    }
    return 0;
};

void CNN::LoadModel(const string & model_path) {
    if (CheckEndwith(model_path, "pt")) {
        LoadTorchModel(model_path);
    }
    else if (CheckEndwith(model_path, "onnx")) {
        LoadOnnxModel(model_path);
    }
    else assert(false);

    // if (input_.numel() != 0) {
    //     AdjustWeightBias(Target());
    // }
};

void CNN::LoadInput(const string & input_path) {
    cv::Mat input = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    input_ = torch::from_blob(input.data, { input.rows * input.cols }, torch::kByte).to(torch::kFloat64) / 255;
}

void CNN::AdjustWeightBias(int target) {
    weight_v_.back() = weight_v_.back() - weight_v_.back().index({target});
    bias_v_.back() = bias_v_.back() - bias_v_.back().index({target});
    // cout << weight_v_.back() << endl;
    // cout << bias_v_.back() << endl;
};

void CNN::AdjustWeightBias(const torch::Tensor& eq) {
    assert(eq.dim() == 2);
    assert(eq.sizes()[1] == num_neuron_v_[num_neuron_v_.size() - 1] + 1);
    num_neuron_v_.back() = eq.sizes()[0];
    weight_v_.back() = torch::matmul(eq.index({torch::indexing::Slice(), torch::indexing::Slice(0, -1)}), weight_v_.back());
    bias_v_.back() = torch::matmul(eq.index({torch::indexing::Slice(), torch::indexing::Slice(0, -1)}), bias_v_.back()) + eq.index({torch::indexing::Slice(), -1});
    // cout << weight_v_.back() << endl;
    // cout << bias_v_.back() << endl;
};

void CNN::SplitBound(const torch::Tensor& bound, int num_split, bool optimize) {
    assert(bound.dim() == 3);
    assert(bound.sizes()[0] > 0);
    assert(bound.sizes()[1] == num_neuron_v_[0]);
    assert(bound.sizes()[2] == 2);
    torch::Tensor split_idx = get<1>(torch::topk(torch::sum(bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}) - bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}), 0), num_neuron_v_[0]));
    // cout << split_idx << endl;
    // getchar();

    torch::Tensor result = bound.clone();

    for (int i = 0; i < num_split; ++i) {
        int idx = split_idx.index({i % num_neuron_v_[0]}).item<int>();
        torch::Tensor middle_point = torch::mean(result.index({torch::indexing::Slice(), idx, torch::indexing::Slice()}), 1);
        torch::Tensor lower_half = result.clone().index_put_({torch::indexing::Slice(), idx, 1}, middle_point);
        torch::Tensor upper_half = result.clone().index_put_({torch::indexing::Slice(), idx, 0}, middle_point);
        result = torch::cat({lower_half, upper_half}, 0);

        // cout << lower_half << endl;
        // cout << upper_half << endl;
        // cout << result << endl;
        // getchar();
    }
    // cout << result << endl;
    // getchar();
    SetBound(result, optimize);
};

void CNN::SetBound(const torch::Tensor& bound, bool optimize) {
    assert(bound.dim() == 3);
    assert(bound.sizes()[0] > 0);
    assert(bound.sizes()[1] == num_neuron_v_[0]);
    assert(bound.sizes()[2] == 2);

    if (bound_before_activation_v_.size() == 0) {
        InitVariable();
    }
    else assert(CheckSize());

    bound_before_activation_v_[0] = bound;

    ComputeBound();
    // printf("%s", Str().c_str());
    // cout << bound_before_activation_v_[num_neuron_v_.size() - 1] << endl;
    // getchar();

    if (optimize) {
        OptimizeBound();
    }

    ComputeNonlinearNeuron();

    UpdateConvexSolver();
};

void CNN::SetBoundAssertion() {
    assert(CheckSize());
    assert(solver_.Certificate() != CSolver::Certificate::DUAL or !solver_.BoundComputing());

    if (solver_.BoundComputing()) {
        ComputeBound();

        ComputeNetworkStatus();

        UpdateConvexSolver();
    }

    UpdateConvexSolverAssertion();
};

void CNN::CheckSatStatus() {
    assert(CheckSize());
    for (int l = 1; l < (int)num_neuron_v_.size() - 1; ++l) {
        assert((int)sat_network_var_v_[l].size() == num_neuron_v_[l]);
        assert((int)sat_network_status_v_[l].sizes()[0] == num_neuron_v_[l]);
        for (int i = 0; i < nonlinear_neuron_v_[l].sizes()[0]; ++i) {
            int neuron_id = nonlinear_neuron_v_[l].index({i}).item<int>();
            if (solver_.EvalSatVar(sat_network_var_v_[l][neuron_id]) == 1) {
                // printf("Checking-> layer: %d, neuron id: %d, True\n", l, neuron_id);
                #ifdef VERBOSE
                printf("Checking-> layer: %d, neuron id: %d, True\n", l, neuron_id);
                #endif
                sat_network_status_v_[l].index_put_({neuron_id}, 1);
            }
            else if (solver_.EvalSatVar(sat_network_var_v_[l][neuron_id]) == -1) {
                // printf("Checking-> layer: %d, neuron id: %d, False\n", l, neuron_id);
                #ifdef VERBOSE
                printf("Checking-> layer: %d, neuron id: %d, False\n", l, neuron_id);
                #endif
                sat_network_status_v_[l].index_put_({neuron_id}, -1);
            }
            else {
                sat_network_status_v_[l].index_put_({neuron_id}, 0);
            }
        }
    }
};

void CNN::AssertSatStatus(int& num_assertion) {
    assert(CheckSize());
    assert(num_assertion < solver_.parameter_.kAssertion);
    // for (int l = 1; l < (int)num_neuron_v_.size() - 1; ++l) {
    for (int l = (int)num_neuron_v_.size() - 2; l >= 1; --l) {
        // printf("layer: %d\n", l);
        assert((int)sat_network_var_v_[l].size() == num_neuron_v_[l]);
        assert((int)sat_network_status_v_[l].sizes()[0] == num_neuron_v_[l]);

        vector<pair<int, double> > indeterminate_neuron_v;
        for (int i = 0; i < nonlinear_neuron_v_[l].sizes()[0]; ++i) {
            int neuron_id = nonlinear_neuron_v_[l].index({i}).item<int>();

            double relu_value = convex_relu_var_v_[l][neuron_id].get(GRB_DoubleAttr_X);
            double slack_value = convex_slack_var_v_[l][neuron_id].get(GRB_DoubleAttr_X);
            #ifdef DEBUG
            if (sat_network_status_v_[l].index({neuron_id}).item<int>() == 0) {
                printf("neuron_id: %d\n", neuron_id);
                printf("relu_value: %lf\n", relu_value);
                printf("slack_value: %lf\n", slack_value);
            }
            #endif
            if (neural_status_v_[l].index({neuron_id}).item<int>() == 0 and sat_network_status_v_[l].index({neuron_id}).item<int>() == 0 and (!jump_assertion_b_ or min(relu_value, slack_value) > 0)) {
                assert(neural_status_v_[l].index({neuron_id}).item<int>() == 0);
                assert(sat_network_status_v_[l].index({neuron_id}).item<int>() == 0);
                indeterminate_neuron_v.emplace_back(neuron_id, min(relu_value, slack_value));
            }
        }
        sort(indeterminate_neuron_v.begin(), indeterminate_neuron_v.end(), [&](const pair<int, double>& A, const pair<int, double>& B) -> bool {
            return A.second > B.second;
        });
        #ifdef DEBUG
        for (int i = 0; i < (int)indeterminate_neuron_v.size(); ++i) {
            printf("neuron_id: %d\n", indeterminate_neuron_v[i].first);
            printf("difference: %lf\n", indeterminate_neuron_v[i].second);
        }
        #endif
        for (int i = 0; i < (int)indeterminate_neuron_v.size(); ++i) {
            int neuron_id = indeterminate_neuron_v[i].first;
            double network_value = convex_network_var_v_[l][neuron_id].get(GRB_DoubleAttr_X);
            if (network_value >= 0) {
                // printf("Asserting-> layer: %d, neuron id: %d, True\n", l, neuron_id);
                #ifdef VERBOSE
                printf("Asserting-> layer: %d, neuron id: %d, True\n", l, neuron_id);
                // printf("number of assertion: %d\n", num_assertion);
                #endif
                sat_network_status_v_[l].index_put_({neuron_id}, 1);
            }
            else {
                // printf("Asserting-> layer: %d, neuron id: %d, False\n", l, neuron_id);
                #ifdef VERBOSE
                printf("Asserting-> layer: %d, neuron id: %d, False\n", l, neuron_id);
                // printf("number of assertion: %d\n", num_assertion);
                #endif
                sat_network_status_v_[l].index_put_({neuron_id}, -1);
            }
            if (++num_assertion >= solver_.parameter_.kAssertion) {
                return;
            }
        }
    }
};

void CNN::ComputeCertificate(vector<z3::expr>& conflict) {
    assert(CheckSize());
    GRBModel& convex_solver = solver_.ConvexSolver();
    if (solver_.Certificate() == CSolver::Certificate::DUAL) {
        int status = convex_solver.get(GRB_IntAttr_Status);
        if (status == GRB_OPTIMAL) {
            list<pair<int, int>> slack_l, relu_l;
            for (int l = 1; l < (int)num_neuron_v_.size() - 1; ++l) {
                assert((int)sat_network_var_v_[l].size() == num_neuron_v_[l]);
                assert((int)sat_network_status_v_[l].sizes()[0] == num_neuron_v_[l]);
                for (int i = 0; i < nonlinear_neuron_v_[l].sizes()[0]; ++i) {
                    int neuron_id = nonlinear_neuron_v_[l].index({i}).item<int>();
                    // printf("layer: %d, neuron id: %d\n", l, neuron_id);
                    if (sat_network_status_v_[l].index({neuron_id}).item<int>() == 1) {
                        assert(convex_assertion_constraint_vm_[l][neuron_id].size() == 0);
                        if (convex_slack_var_v_[l][neuron_id].get(GRB_DoubleAttr_RC) >= ObjectiveRatio(l)) {
                            // printf("layer: %d, neuron id: %d\n", l, neuron_id);
                            slack_l.emplace_back(l, neuron_id);
                        }
                    }
                    else if (sat_network_status_v_[l].index({neuron_id}).item<int>() == -1) {
                        assert(convex_assertion_constraint_vm_[l][neuron_id].size() == 0);
                        if (convex_relu_var_v_[l][neuron_id].get(GRB_DoubleAttr_RC) >= ObjectiveRatio(l)) {
                            relu_l.emplace_back(l, neuron_id);
                        }
                    }
                    else assert(sat_network_status_v_[l].index({neuron_id}).item<int>() == 0);
                }
            }

            for (auto it = slack_l.begin(); it != slack_l.end(); ++it) {
                convex_slack_var_v_[it->first][it->second].set(GRB_DoubleAttr_Obj, 0.0);
            }
            for (auto it = relu_l.begin(); it != relu_l.end(); ++it) {
                convex_relu_var_v_[it->first][it->second].set(GRB_DoubleAttr_Obj, 0.0);
            }

            for (int l = (int)num_neuron_v_.size() - 2; l >= 1; --l) {
            // for (int l = 1; l < (int)num_neuron_v_.size() - 1; ++l) {
                assert((int)sat_network_var_v_[l].size() == num_neuron_v_[l]);
                assert((int)sat_network_status_v_[l].sizes()[0] == num_neuron_v_[l]);
                for (int i = 0; i < nonlinear_neuron_v_[l].sizes()[0]; ++i) {
                    int neuron_id = nonlinear_neuron_v_[l].index({i}).item<int>();
                    // printf("layer: %d, neuron id: %d\n", l, neuron_id);
                    if (sat_network_status_v_[l].index({neuron_id}).item<int>() == 1) {
                        assert(convex_assertion_constraint_vm_[l][neuron_id].size() == 0);
                        convex_slack_var_v_[l][neuron_id].set(GRB_DoubleAttr_Obj, 0.0);
                        convex_solver.update();
                        convex_solver.optimize();
                        assert(convex_solver.get(GRB_IntAttr_Status) == GRB_OPTIMAL);
                        bool satisfiable = ((convex_solver.get(GRB_DoubleAttr_ObjVal) <= numeric_limits<double>::epsilon()));
                        if (satisfiable) {
                            convex_slack_var_v_[l][neuron_id].set(GRB_DoubleAttr_Obj, ObjectiveRatio(l));
                            convex_solver.update();
                            conflict.emplace_back(!CSolver::SatExpr(sat_network_var_v_[l][neuron_id]));
                        }
                    }
                    else if (sat_network_status_v_[l].index({neuron_id}).item<int>() == -1) {
                        assert(convex_assertion_constraint_vm_[l][neuron_id].size() == 0);
                        convex_relu_var_v_[l][neuron_id].set(GRB_DoubleAttr_Obj, 0.0);
                        convex_solver.update();
                        convex_solver.optimize();
                        assert(convex_solver.get(GRB_IntAttr_Status) == GRB_OPTIMAL);
                        bool satisfiable = ((convex_solver.get(GRB_DoubleAttr_ObjVal) <= numeric_limits<double>::epsilon()));
                        if (satisfiable) {
                            convex_relu_var_v_[l][neuron_id].set(GRB_DoubleAttr_Obj, ObjectiveRatio(l));
                            convex_solver.update();
                            conflict.emplace_back(CSolver::SatExpr(sat_network_var_v_[l][neuron_id]));
                        }
                    }
                    else assert(sat_network_status_v_[l].index({neuron_id}).item<int>() == 0);
                }
            }
        }
    }
    else if (solver_.Certificate() == CSolver::Certificate::IIS) {
        for (int l = 1; l < (int)num_neuron_v_.size() - 1; ++l) {
            assert((int)sat_network_var_v_[l].size() == num_neuron_v_[l]);
            assert((int)sat_network_status_v_[l].sizes()[0] == num_neuron_v_[l]);
            for (int i = 0; i < nonlinear_neuron_v_[l].sizes()[0]; ++i) {
                int neuron_id = nonlinear_neuron_v_[l].index({i}).item<int>();
                if (sat_network_status_v_[l].index({neuron_id}).item<int>() == 1) {
                    if (convex_assertion_constraint_vm_[l][neuron_id].size() != 1 or convex_assertion_constraint_vm_[l][neuron_id][0].get(GRB_IntAttr_IISConstr) == 1) {
                        conflict.emplace_back(!CSolver::SatExpr(sat_network_var_v_[l][neuron_id]));
                    }
                }
                else if (sat_network_status_v_[l].index({neuron_id}).item<int>() == -1) {
                    if (convex_assertion_constraint_vm_[l][neuron_id].size() != 1 or convex_assertion_constraint_vm_[l][neuron_id][0].get(GRB_IntAttr_IISConstr) == 1) {
                        conflict.emplace_back(CSolver::SatExpr(sat_network_var_v_[l][neuron_id]));
                    }
                }
                else assert(sat_network_status_v_[l].index({neuron_id}).item<int>() == 0);
            }
        }
    }
    else if (solver_.Certificate() == CSolver::Certificate::NONE) {
        for (int l = 1; l < (int)num_neuron_v_.size() - 1; ++l) {
            assert((int)sat_network_var_v_[l].size() == num_neuron_v_[l]);
            assert((int)sat_network_status_v_[l].sizes()[0] == num_neuron_v_[l]);
            for (int i = 0; i < nonlinear_neuron_v_[l].sizes()[0]; ++i) {
                int neuron_id = nonlinear_neuron_v_[l].index({i}).item<int>();
                if (sat_network_status_v_[l].index({neuron_id}).item<int>() == 1) {
                    conflict.emplace_back(!CSolver::SatExpr(sat_network_var_v_[l][neuron_id]));
                }
                else if (sat_network_status_v_[l].index({neuron_id}).item<int>() == -1) {
                    conflict.emplace_back(CSolver::SatExpr(sat_network_var_v_[l][neuron_id]));
                }
                else assert(sat_network_status_v_[l].index({neuron_id}).item<int>() == 0);
            }
        }
    }
    else assert(false);
}

torch::Tensor CNN::Forward(const torch::Tensor& input) {
    torch::Tensor prev = input;
    for (int i = 1; i < (int)num_neuron_v_.size() - 1; ++i) {
        torch::Tensor& w = weight_v_[i];
        torch::Tensor& b = bias_v_[i];

        prev = torch::matmul(w, prev) + b;
        // printf("layer: %d\n", i);
        // for (int j = 0; j < prev.sizes()[0]; ++j) {
        //         cout << j << " " << prev.index({j}) << endl;
        // }
        prev = torch::nn::functional::relu(prev);
    };
    prev = torch::matmul(weight_v_.back(), prev) + bias_v_.back();
    return prev;
};

void CNN::AddConvexBoundConstraint(GRBTempConstr constraint) {
    // printf("size: %lu\n", convex_constraint_v_.size());
    convex_constraint_v_.emplace_back(solver_.ConvexSolver().addConstr(constraint));
};

void CNN::RemoveConvexBoundConstraint() {
    for (int i = 0; i < (int)convex_constraint_v_.size(); ++i) {
        solver_.ConvexSolver().remove(convex_constraint_v_[i]);
    }
    convex_constraint_v_.clear();
};

void CNN::AddConvexAssignmentConstraint(int layer, int neuron_id, GRBTempConstr constraint) {
    assert(layer < (int)convex_assertion_constraint_vm_.size());
    // printf("layer: %d\n", layer);
    // printf("id: %d\n", neuron_id);
    // printf("size: %lu\n", convex_assertion_constraint_vm_[layer][neuron_id].size());
    convex_assertion_constraint_vm_[layer][neuron_id].emplace_back(solver_.ConvexSolver().addConstr(constraint));
};

void CNN::RemoveConvexAssignmentConstraint() {
    for (int i = 0; i < (int)convex_assertion_constraint_vm_.size(); ++i) {
        for (auto j = convex_assertion_constraint_vm_[i].begin(); j != convex_assertion_constraint_vm_[i].end(); ++j) {
            for (int k = 0; k < (int)j->second.size(); ++k) {
                solver_.ConvexSolver().remove(j->second[k]);
            }
        }
        convex_assertion_constraint_vm_[i].clear();
    }
};

double CNN::ObjectiveRatio(int layer) {
    if (solver_.Certificate() == CSolver::Certificate::DUAL) {
        // return 1;
        return layer * parameter_.kObjectiveRatio;
    }
    else {
        return 1;
        // return 0;
    }
}

string CNN::Str(bool matrix) const {
    string res;
    ostringstream string_conversion;
    res += "Neural Network:\n";
    if (matrix) {
        res += "  Weight:\n";
        for (int i = 1; i < (int)weight_v_.size(); ++i) {
            res += "    layer " + to_string(i) + ":\n";
            string_conversion << weight_v_[i];
            res += InsertSpaceAtBeginOfLine(string_conversion.str(), 6) + "\n";
            string_conversion.str("");
        };
        res += "  Bias:\n";
        for (int i = 1; i < (int)bias_v_.size(); ++i) {
            res += "    layer " + to_string(i) + ":\n";
            ostringstream string_conversion;
            string_conversion << bias_v_[i];
            res += InsertSpaceAtBeginOfLine(string_conversion.str(), 6) + "\n";
            string_conversion.str("");
        };
    }
    res += "  Neurons: [" + to_string(num_neuron_v_[0]);
    for (int i = 1; i < (int)num_neuron_v_.size(); ++i) {
        res += ", " + to_string(num_neuron_v_[i]);
    };
    res += "]\n";

    res += "  Bound before Activation Function:\n";
    for (int i = 0; i < (int)bound_before_activation_v_.size(); ++i) {
        res += "    layer: " + to_string(i) + "\n";
        string_conversion << bound_before_activation_v_[i];
        res += InsertSpaceAtBeginOfLine(string_conversion.str(), 8) + "\n";
        string_conversion.str("");
    };
    if (matrix) {
        res += "  Equation before Activation Function:\n";
        assert(lower_equation_before_activation_v_.size() == upper_equation_before_activation_v_.size());
        for (int i = 1; i < (int)lower_equation_before_activation_v_.size(); ++i) {
            res += "    layer: " + to_string(i) + "\n";
            res += "      lower:\n";
            string_conversion << lower_equation_before_activation_v_[i];
            // string_conversion << lower_equation_before_activation_v_[i].index({torch::indexing::Slice(), -1});
            res += InsertSpaceAtBeginOfLine(string_conversion.str(), 10) + "\n";
            string_conversion.str("");

            res += "      upper:\n";
            string_conversion << upper_equation_before_activation_v_[i];
            // string_conversion << upper_equation_before_activation_v_[i].index({torch::indexing::Slice(), -1});
            res += InsertSpaceAtBeginOfLine(string_conversion.str(), 10) + "\n";
            string_conversion.str("");
        };
        res += "  Equation after Activation Function:\n";
        assert(lower_equation_after_activation_v_.size() == upper_equation_after_activation_v_.size());
        for (int i = 1; i < (int)lower_equation_after_activation_v_.size(); ++i) {
            res += "    layer: " + to_string(i) + "\n";
            res += "      lower:\n";
            string_conversion << lower_equation_after_activation_v_[i];
            // string_conversion << lower_equation_after_activation_v_[i].index({torch::indexing::Slice(), -1});
            res += InsertSpaceAtBeginOfLine(string_conversion.str(), 10) + "\n";
            string_conversion.str("");

            res += "      upper:\n";
            string_conversion << upper_equation_after_activation_v_[i];
            // string_conversion << upper_equation_after_activation_v_[i].index({torch::indexing::Slice(), -1});
            res += InsertSpaceAtBeginOfLine(string_conversion.str(), 10) + "\n";
            string_conversion.str("");
        };
    }
    res += "  Neuron Status and Nonlinear Neuron:\n";
    assert(neural_status_v_.size() == nonlinear_neuron_v_.size());
    for (int i = 1; i < (int)neural_status_v_.size(); ++i) {
        res += "    layer: " + to_string(i) + "\n";
        res += "      neuron status:\n";
        string_conversion << neural_status_v_[i];
        res += InsertSpaceAtBeginOfLine(string_conversion.str(), 10) + "\n";
        string_conversion.str("");

        res += "      nonlinear neuron:\n";
        string_conversion << nonlinear_neuron_v_[i];
        res += InsertSpaceAtBeginOfLine(string_conversion.str(), 10) + "\n";
        string_conversion.str("");
    };
    return res;
};

string CNN::SatStr() const {
    string res;
    res += "SAT Variable:\n";
    assert(sat_network_var_v_.size() == num_neuron_v_.size() - 1);
    assert(sat_network_status_v_.size() == num_neuron_v_.size() - 1);
    for (int l = 1; l < (int)num_neuron_v_.size() - 1; ++l) {
        assert((int)sat_network_var_v_[l].size() == num_neuron_v_[l]);
        assert((int)sat_network_status_v_[l].sizes()[0] == num_neuron_v_[l]);
        res += "  layer " + to_string(l) + ":\n";
        string var_str;
        for (int i = 0; i < num_neuron_v_[l]; ++i) {
            var_str += solver_.SatVarStr(sat_network_var_v_[l][i]) + ": " + to_string(sat_network_status_v_[l].index({i}).item<int>()) + "\n";
        }
        res += InsertSpaceAtBeginOfLine(var_str, 4) + "\n";
    };
    return res;
};

string CNN::ConvexStr() const {
    string res;
    ostringstream string_conversion;
    res += "Convex Variable:\n";
    assert(convex_network_var_v_.size() == num_neuron_v_.size());
    assert(convex_relu_var_v_.size() == num_neuron_v_.size() - 1);
    assert(convex_slack_var_v_.size() == num_neuron_v_.size() - 1);
    for (int l = 0; l < (int)convex_network_var_v_.size(); ++l) {
        res += "  layer " + to_string(l) + ":\n";
        for (int i = 0; i < (int)convex_network_var_v_[l].size(); ++i) {
            res += "\t" + solver_.ConvexVarStr(convex_network_var_v_[l][i]) + "\n";
            if (l > 0 and l < (int)num_neuron_v_.size() - 1) {
                res += "\t" + solver_.ConvexVarStr(convex_relu_var_v_[l][i]) + "\n";
                res += "\t" + solver_.ConvexVarStr(convex_slack_var_v_[l][i]) + "\n";
                if (sat_network_status_v_[l].index({i}).item<int>() != 0) {
                    res += "\tasserted" + to_string(sat_network_status_v_[l].index({i}).item<int>()) +  "\n";
                }
            }
        }
    };
    return res;
};

void CNN::LoadTorchModel(const string & model_path) {
    torch::jit::script::Module  module = torch::jit::load(model_path);
    module.to(torch::kFloat64);
    torch::jit::slot_list_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::ParameterPolicy> > params = module.named_parameters(true);

    weight_v_.clear();
    weight_v_.emplace_back();
    bias_v_.clear();
    bias_v_.emplace_back();
    num_neuron_v_.clear();
    for (auto p = params.begin(); p != params.end(); ++p) {
        const torch::Tensor& data = (*p).value;
        if (data.dim() == 2) {
            weight_v_.emplace_back(data.to(torch::kFloat64));
            num_neuron_v_.emplace_back(data.sizes()[1]);
        }
        else if (data.dim() == 1) {
            bias_v_.emplace_back(data.to(torch::kFloat64));
        }
        else assert(false);
    }
    num_neuron_v_.emplace_back(bias_v_.back().sizes()[0]);
    // printf("%s\n", Str(true).c_str());
};

void CNN::LoadOnnxModel(const string & model_path) {
    cv::dnn::Net network = cv::dnn::readNetFromONNX(model_path);
    vector<string> layer_name = network.getLayerNames();
    // printf("size: %lu\n", layer_name.size());

    weight_v_.clear();
    weight_v_.emplace_back();
    bias_v_.clear();
    bias_v_.emplace_back();
    num_neuron_v_.clear();
    // for (int i = 0; i < (int)layer_name.size(); ++i) {
    //     printf("i: %d\n", i);
    //     printf("name: %s\n", layer_name[i].c_str());
    // }
    for (int i = 1; i < (int)layer_name.size(); ++i) {
        // printf("i: %d\n", i);
        // printf("name: %s\n", layer_name[i].c_str());
        if (layer_name[i] == "input_Sub") {
            // printf("input layer\n");
        }
        else if (CheckEndwith(layer_name[i], "Flatten")) {
            // printf("flatten layer\n");
        }
        else if (CheckEndwith(layer_name[i], "MatMul")) {
            // printf("weight layer\n");
            cv::dnn::Layer* layer = network.getLayer(layer_name[i]);
            assert(layer->blobs.size() == 1);
            cv::Mat weight = layer->blobs[0];
            assert(weight.dims == 2);
            torch::Tensor data = torch::from_blob(weight.data, { weight.size[0], weight.size[1] }).to(torch::kFloat64);
            weight_v_.emplace_back(data);
            num_neuron_v_.emplace_back(data.sizes()[1]);
        }
        else if (CheckEndwith(layer_name[i], "Add")) {
            // printf("bias layer\n");
            cv::dnn::Layer* layer = network.getLayer(layer_name[i]);
            assert(layer->blobs.size() == 1);
            cv::Mat bias = layer->blobs[0];
            assert(bias.dims == 2);
            assert(bias.size[0] == 1);
            torch::Tensor data = torch::from_blob(bias.data, { bias.size[1] }).to(torch::kFloat64);
            bias_v_.emplace_back(data);
        }
        else if (CheckStartwith(layer_name[i], "relu")) {
            // printf("relu layer\n");
        }
        else {
            // for onnx model made export from pytorch

            // the name is numerical in most of the time
            // assert(CheckDigit(layer_name[i]));
            cv::dnn::Layer* layer = network.getLayer(layer_name[i]);
            if (layer->blobs.size() == 2) {
                cv::Mat weight = layer->blobs[0];
                assert(weight.dims == 2);
                torch::Tensor weight_t = torch::from_blob(weight.data, { weight.size[0], weight.size[1] }).to(torch::kFloat64);
                weight_v_.emplace_back(weight_t);
                num_neuron_v_.emplace_back(weight_t.sizes()[1]);

                cv::Mat bias = layer->blobs[1];
                assert(bias.dims == 2);
                assert(bias.size[0] == 1);
                torch::Tensor bias_t = torch::from_blob(bias.data, { bias.size[1] }).to(torch::kFloat64);
                bias_v_.emplace_back(bias_t);
            }
            else assert(layer->blobs.size() == 0);
        }
    }
    // printf("done\n");
    num_neuron_v_.emplace_back(bias_v_.back().sizes()[0]);
    // printf("%s\n", Str().c_str());
};

void CNN::InitVariable() {
    assert(bound_before_activation_v_.size() == 0);
    assert(lower_equation_before_activation_v_.size() == 0);
    assert(upper_equation_before_activation_v_.size() == 0);
    assert(lower_equation_after_activation_v_.size() == 0);
    assert(upper_equation_after_activation_v_.size() == 0);

    assert(lower_ratio_v_.size() == 0);
    assert(upper_ratio_v_.size() == 0);
    assert(alpha_v_.size() == 0);

    assert(neural_status_v_.size() == 0);
    assert(nonlinear_neuron_v_.size() == 0);

    assert(sat_network_var_v_.size() == 0);
    assert(sat_network_status_v_.size() == 0);

    assert(convex_network_var_v_.size() == 0);
    assert(convex_relu_var_v_.size() == 0);
    assert(convex_slack_var_v_.size() == 0);
    assert(convex_constraint_v_.size() == 0);
    assert(convex_assertion_constraint_vm_.size() == 0);

    // initialize bound variables
    bound_before_activation_v_.resize(num_neuron_v_.size(), torch::empty({0}));
    lower_equation_before_activation_v_.resize(num_neuron_v_.size(), torch::empty({0}));
    upper_equation_before_activation_v_.resize(num_neuron_v_.size(), torch::empty({0}));
    lower_equation_after_activation_v_.resize(num_neuron_v_.size() - 1, torch::empty({0}));
    upper_equation_after_activation_v_.resize(num_neuron_v_.size() - 1, torch::empty({0}));
    lower_ratio_v_.resize(num_neuron_v_.size() - 1, torch::empty({0}));
    upper_ratio_v_.resize(num_neuron_v_.size() - 1, torch::empty({0}));
    alpha_v_.resize(num_neuron_v_.size());

    // initialize neuron status variables
    neural_status_v_.resize(num_neuron_v_.size() - 1, torch::empty({0}));
    nonlinear_neuron_v_.resize(num_neuron_v_.size() - 1, torch::empty({0}));
    for (int l = 1; l < (int)num_neuron_v_.size() - 1; ++l) {
        neural_status_v_[l] = torch::zeros({num_neuron_v_[l]}, torch::dtype(torch::kInt));
        nonlinear_neuron_v_[l] = torch::arange(num_neuron_v_[l], torch::dtype(torch::kInt));
    }

    // initialize boolean variables for sat solver
    sat_network_var_v_.resize(num_neuron_v_.size() - 1);
    sat_network_status_v_.resize(num_neuron_v_.size() - 1, torch::empty({0}));

    ostringstream string_conversion;
    for (int l = 0; l < (int)num_neuron_v_.size() - 1; ++l) {
        for (int i = 0; i < (int)num_neuron_v_[l]; ++i) {
            string_conversion << l << "_" << i << "_" << (void*)this;
            sat_network_var_v_[l].emplace_back(solver_.AddSatVar(string_conversion.str().c_str()));
            string_conversion.str("");
        }
        sat_network_status_v_[l] = torch::zeros({num_neuron_v_[l]}, torch::dtype(torch::kInt));
    }

    // initialize convex variables for convex solver
    convex_network_var_v_.resize(num_neuron_v_.size());
    convex_relu_var_v_.resize(num_neuron_v_.size() - 1);
    convex_slack_var_v_.resize(num_neuron_v_.size() - 1);
    convex_assertion_constraint_vm_.resize(num_neuron_v_.size() - 1);

    GRBModel& model = solver_.ConvexSolver();
    for (int l = 0; l < (int)num_neuron_v_.size(); ++l) {
        for (int i = 0; i < (int)num_neuron_v_[l]; ++i) {
            string_conversion << l << "_" << i << "_" << (void*)this;
            convex_network_var_v_[l].emplace_back(model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, ("Net_" + string_conversion.str()).c_str()));
            if (l > 0 and l < (int)num_neuron_v_.size() - 1) {
                convex_relu_var_v_[l].emplace_back(model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, ("Relu_" + string_conversion.str()).c_str()));
                convex_slack_var_v_[l].emplace_back(model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, ("Slack_" + string_conversion.str()).c_str()));
            }
            string_conversion.str("");
        }
    }
    model.update();


    // set constraint for both solvers
    for (int l = 1; l < (int)num_neuron_v_.size(); ++l) {
        vector<GRBVar>* prev_layer_var;
        if (l == 1) {
            prev_layer_var = &convex_network_var_v_[l-1];
        }
        else {
            prev_layer_var = &convex_relu_var_v_[l-1];
        }
        for (int i = 0; i < num_neuron_v_[l]; ++i) {
            // construct relation between neurons
            torch::Tensor weight;
            double bias;
            GRBLinExpr expr;

            weight = weight_v_[l].index({i, torch::indexing::Slice()});
            bias = bias_v_[l].index({i}).item<double>();

            // neuron relations constraints for sat solver
            if (l == 1) {
                // printf("l: %d\n", l);
                // cout << weight << endl;
                // cout << bias << endl;
                z3::expr pos_constraint = CSolver::SatExpr(sat_network_var_v_[l][i]);
                z3::expr neg_constraint = !CSolver::SatExpr(sat_network_var_v_[l][i]);
                for (int j = 0; j < (int)weight.sizes()[0]; ++j) {
                    if (bias >= 0) {
                        if (weight.index({j}).item<double>() > 0) {
                            pos_constraint = !CSolver::SatExpr(sat_network_var_v_[l - 1][j]) || pos_constraint;
                        }
                        else if (weight.index({j}).item<double>() < 0) {
                            pos_constraint = CSolver::SatExpr(sat_network_var_v_[l - 1][j]) || pos_constraint;
                        }
                    }
                    if (bias <= 0) {
                        if (weight.index({j}).item<double>() > 0) {
                            neg_constraint = CSolver::SatExpr(sat_network_var_v_[l - 1][j]) || neg_constraint;
                        }
                        else if (weight.index({j}).item<double>() < 0) {
                            neg_constraint = !CSolver::SatExpr(sat_network_var_v_[l - 1][j]) || neg_constraint;
                        }
                    }
                }
                if (bias >= 0) {
                    solver_.AddSatFixedConstraint(pos_constraint);
                }
                if (bias <= 0) {
                    solver_.AddSatFixedConstraint(neg_constraint);
                }
            }
            else if (l >= 2 && l < (int)num_neuron_v_.size() - 1) {
                // printf("l: %d\n", l);
                // cout << weight << endl;
                // cout << bias << endl;
                z3::expr pos_constraint = CSolver::SatExpr(sat_network_var_v_[l][i]);
                z3::expr neg_constraint = !CSolver::SatExpr(sat_network_var_v_[l][i]);
                for (int j = 0; j < (int)weight.sizes()[0]; ++j) {
                    if (bias >= 0 && weight.index({j}).item<double>() < 0) {
                        pos_constraint = CSolver::SatExpr(sat_network_var_v_[l - 1][j]) || pos_constraint;
                    }
                    if (bias <= 0 && weight.index({j}).item<double>() > 0) {
                        neg_constraint = CSolver::SatExpr(sat_network_var_v_[l - 1][j]) || neg_constraint;
                    }
                }
                if (bias >= 0) {
                    solver_.AddSatFixedConstraint(pos_constraint);
                }
                if (bias <= 0) {
                    solver_.AddSatFixedConstraint(neg_constraint);
                }
            }
            // weight constrait for convex solver
            expr = solver_.ComputeConvexExpr(*prev_layer_var, weight) + bias;
            model.addConstr(expr == convex_network_var_v_[l][i]);

            if (l < (int)num_neuron_v_.size() - 1) {
                // construct relation between network, relu and slack variables
                model.addConstr(convex_relu_var_v_[l][i] - convex_network_var_v_[l][i] == convex_slack_var_v_[l][i]);
            }
        }
    }
    model.update();

    assert(CheckSize());
}

void CNN::OptimizeBound() {
    ComputeBound("fixed");
    for (int l = 1; l < (int)alpha_v_.size(); ++l) {
        for (int i = 1; i < (int)alpha_v_[l].size(); ++i) {
            alpha_v_[l][i] = alpha_v_[l][i].detach().requires_grad_(true);
        }
    }
    // printf("optimizing bound\n");
    // cout << bound_before_activation_v_[num_neuron_v_.size() - 1] << endl;
    // getchar();

    torch::optim::Adam opt(vector<torch::Tensor>{}, torch::optim::AdamOptions(1.0e-1));
    for (int l = 1; l < (int)alpha_v_.size(); ++l) {
        opt.add_param_group(alpha_v_[l]);
    }

    for (int iter = 0; iter < 10; ++iter) {
        ComputeBound("clipped");

        // printf("iteration: %d\n", iter);
        // cout << bound_before_activation_v_[num_neuron_v_.size() - 1] << endl;
        // getchar();

        opt.zero_grad();

        torch::Tensor loss = -1 * torch::sum(bound_before_activation_v_[num_neuron_v_.size() - 1].index({torch::indexing::Slice(), torch::indexing::Slice(), 0})) + torch::sum(bound_before_activation_v_[num_neuron_v_.size() - 1].index({torch::indexing::Slice(), torch::indexing::Slice(), 1}));
        loss.backward();
        opt.step();
    }
    ComputeBound("clipped");
    // cout << bound_before_activation_v_[num_neuron_v_.size() - 1] << endl;
    // printf("done\n");
    // getchar();
};

void CNN::ComputeBound(const string& method) {
    assert(CheckSize());

    torch::Tensor lower_eq;
    torch::Tensor upper_eq;

    torch::Tensor bound = bound_before_activation_v_[0];
    torch::Tensor lower_eq_bound;
    torch::Tensor upper_eq_bound;
    for (int layer = 1; layer < (int)num_neuron_v_.size(); ++layer) {
        // cout << "layer: " << layer << endl;
        // cout << "before linear bound: " << bound << endl;
        // cout << "before linear bound size: " << bound.sizes() << endl;
        // getchar();

        ForwardLinear(layer, bound, method);

        // cout << "layer: " << layer << endl;
        // cout << "after linear bound: " << bound << endl;
        // cout << "after linear bound size: " << bound.sizes() << endl;
        // getchar();

        if (layer < (int)num_neuron_v_.size() - 1) {
            ForwardRelu(layer, bound, method);
        };
    };
};

void CNN::ForwardLinear(int layer, torch::Tensor& bound, const string& method) {
    assert(bound.dim() == 3);
    assert(bound.sizes()[1] == num_neuron_v_[layer - 1]);
    assert(bound.sizes()[2] == 2);
    assert(method == "fixed" or method == "clipped");
    int batch_size = bound.sizes()[0];
    torch::Tensor l, u;

    // 1. forward bound
    bound = torch::permute(bound, {0, 2, 1});
    torch::Tensor equation = torch::permute(weight_v_[layer], {1, 0});
    torch::Tensor positive_eq = torch::clamp(equation, 0, c10::nullopt);
    torch::Tensor negative_eq = torch::clamp(equation, c10::nullopt, 0);
    bound = torch::matmul(bound, positive_eq) + torch::matmul(bound, negative_eq).index({torch::indexing::Slice(), torch::tensor({1, 0}), torch::indexing::Slice()});
    bound = torch::permute(bound, {0, 2, 1});
    bound += bias_v_[layer].index({torch::indexing::None, torch::indexing::Slice(), torch::indexing::None});
    // cout << "bound size: " << bound.sizes() << endl;
    // getchar();

    // 2. backward propagate equation
    torch::Tensor lower_eq = torch::cat({torch::eye(num_neuron_v_[layer]), torch::zeros({num_neuron_v_[layer], 1})}, 1).to(torch::kFloat64);
    lower_eq = lower_eq.unsqueeze(0).repeat({batch_size, 1, 1});
    torch::Tensor upper_eq = lower_eq.clone();

    BackwardLinear(layer, lower_eq, upper_eq);
    for (int l = layer - 1; l >= 1; --l) {
        BackwardRelu(layer, l, lower_eq, upper_eq, method);
        BackwardLinear(l, lower_eq, upper_eq);
    }
    // getchar();
    torch::Tensor interval = torch::cat({bound_before_activation_v_[0], torch::ones({batch_size, 1,2})}, 1);
    torch::Tensor lower_eq_bound = ComputeBound(lower_eq, interval);
    torch::Tensor upper_eq_bound = ComputeBound(upper_eq, interval);

    // 3. combine both bounds
    // l = torch::maximum(bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}), lower_eq_bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 0})).detach() - bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}).detach() - lower_eq_bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}).detach() + bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}) + lower_eq_bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 0});
    // u = torch::minimum(bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}), upper_eq_bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 1})).detach() - bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}).detach() - upper_eq_bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}).detach() + bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}) + upper_eq_bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 1});
    // bound = torch::cat({l.unsqueeze(2), u.unsqueeze(2)}, 2);

    // MODIFY
    torch::Tensor mask = torch::logical_and(bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}) < 0, bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}) > 0);
    bound.index_put_({mask}, torch::cat({lower_eq_bound.index({mask}).index({torch::indexing::Slice(), 0, torch::indexing::None}), upper_eq_bound.index({mask}).index({torch::indexing::Slice(), 1, torch::indexing::None})}, 1));

    // cout << bound.sizes() << endl;;
    // cout << lower_eq.sizes() <<  endl;
    // cout << upper_eq.sizes() <<  endl;
    // cout << lower_eq_bound <<  endl;
    // cout << upper_eq_bound <<  endl;
    // getchar();


    if (method == "fixed") {
        bound_before_activation_v_[layer] = bound;
    }
    if (method == "clipped") {
        l = torch::maximum(bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}), bound_before_activation_v_[layer].index({torch::indexing::Slice(), torch::indexing::Slice(), 0})).detach() - bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}).detach() + bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 0});
        u = torch::minimum(bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}), bound_before_activation_v_[layer].index({torch::indexing::Slice(), torch::indexing::Slice(), 1})).detach() - bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}).detach() + bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 1});
        bound = torch::cat({l.unsqueeze(2), u.unsqueeze(2)}, 2);
        bound_before_activation_v_[layer] = bound;
    }
    else assert(method == "fixed");

    lower_equation_before_activation_v_[layer] = lower_eq;
    upper_equation_before_activation_v_[layer] = upper_eq;
};

void CNN::ForwardRelu(int layer, torch::Tensor& bound, const string& method) {
    assert(bound.dim() == 3);
    assert(bound.sizes()[1] == num_neuron_v_[layer]);
    assert(bound.sizes()[2] == 2);
    assert(method == "fixed" or method == "clipped");
    int batch_size = bound.sizes()[0];

    torch::Tensor positive_mask = bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}) >= 0;
    torch::Tensor unstable_mask = torch::logical_and(bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}) < 0, bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}) > 0);
    torch::Tensor negative_mask = bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}) <= 0;
    torch::Tensor slope = bound.index({unstable_mask}).index({torch::indexing::Slice(), 1}) / (bound.index({unstable_mask}).index({torch::indexing::Slice(), 1}) - bound.index({unstable_mask}).index({torch::indexing::Slice(), 0}));
    torch::Tensor bias = -1 * bound.index({unstable_mask}).index({torch::indexing::Slice(), 0}) * slope;

    upper_ratio_v_[layer] = torch::empty({batch_size, num_neuron_v_[layer], 2}, torch::dtype(torch::kFloat64));
    upper_ratio_v_[layer].index_put_({positive_mask}, torch::tensor({{1.0, 0.0}}, torch::dtype(torch::kFloat64)));
    upper_ratio_v_[layer].index_put_({unstable_mask}, torch::cat({slope.unsqueeze(1), bias.unsqueeze(1)}, 1));
    upper_ratio_v_[layer].index_put_({negative_mask}, torch::tensor({{0.0, 0.0}}, torch::dtype(torch::kFloat64)));

    lower_ratio_v_[layer] = torch::empty({batch_size, num_neuron_v_[layer], 2}, torch::dtype(torch::kFloat64));
    torch::Tensor mask = torch::logical_or(positive_mask, unstable_mask);
    mask.index_put_({unstable_mask},  slope >= 0.5);
    lower_ratio_v_[layer].index_put_({mask}, torch::tensor({{1.0, 0.0}}, torch::dtype(torch::kFloat64)));
    lower_ratio_v_[layer].index_put_({torch::logical_not(mask)}, torch::tensor({{0.0, 0.0}}, torch::dtype(torch::kFloat64)));
    if (method == "fixed") {
        for (int l = layer; l < (int)alpha_v_.size(); ++l) {
            if (layer >= (int)alpha_v_[l].size()) {
                alpha_v_[l].resize(layer + 1, torch::empty({0}));
            }
            alpha_v_[l][layer] = lower_ratio_v_[layer].index({torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice(), 0, torch::indexing::None}).repeat({1, num_neuron_v_[l], 1, 2}).detach();
        }
        // cout << lower_ratio_v_[layer] << endl;
        // cout << upper_ratio_v_[layer] << endl;
        // cout << alpha_v_[layer] << endl;
        // getchar();
    }
    else if (method == "clipped") {
    }
    else assert(false);

    torch::Tensor lower_eq = torch::cat({torch::eye(num_neuron_v_[layer]), torch::zeros({num_neuron_v_[layer], 1})}, 1).to(torch::kFloat64);
    lower_eq = lower_eq.unsqueeze(0).repeat({batch_size, 1, 1});
    torch::Tensor upper_eq = lower_eq.clone();


    BackwardRelu(layer, layer, lower_eq, upper_eq, method);
    BackwardLinear(layer, lower_eq, upper_eq);
    for (int l = layer - 1; l >= 1; --l) {
        BackwardRelu(layer, l, lower_eq, upper_eq, method);
        BackwardLinear(l, lower_eq, upper_eq);
    }

    torch::Tensor interval = torch::cat({bound_before_activation_v_[0], torch::ones({batch_size, 1,2})}, 1);
    torch::Tensor lower_eq_bound = ComputeBound(lower_eq, interval);
    torch::Tensor upper_eq_bound = ComputeBound(upper_eq, interval);

    // 3. combine both bounds
    bound = torch::relu(bound);
    torch::Tensor l = torch::maximum(bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}), lower_eq_bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}));
    torch::Tensor u = torch::minimum(bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}), upper_eq_bound.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}));
    bound = torch::cat({l.unsqueeze(2), u.unsqueeze(2)}, 2);

    lower_equation_after_activation_v_[layer] = lower_eq;
    upper_equation_after_activation_v_[layer] = upper_eq;
};

void CNN::BackwardLinear(int layer, torch::Tensor& lower_eq, torch::Tensor& upper_eq) {
    torch::Tensor& weight = weight_v_[layer];
    torch::Tensor& bias = bias_v_[layer];
    // lower
    torch::Tensor lower_eq_new = lower_eq.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, -1)});
    lower_eq_new = torch::cat({torch::matmul(lower_eq_new, weight), torch::matmul(lower_eq_new, bias.index({torch::indexing::Slice(), torch::indexing::None}))}, 2);
    lower_eq_new.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), -1}, lower_eq_new.index({torch::indexing::Slice(), torch::indexing::Slice(), -1}) + lower_eq.index({torch::indexing::Slice(), torch::indexing::Slice(), -1}));


    // upper
    torch::Tensor upper_eq_new = upper_eq.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, -1)});
    upper_eq_new = torch::cat({torch::matmul(upper_eq_new, weight), torch::matmul(upper_eq_new, bias.index({torch::indexing::Slice(), torch::indexing::None}))}, 2);
    upper_eq_new.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), -1}, upper_eq_new.index({torch::indexing::Slice(), torch::indexing::Slice(), -1}) + upper_eq.index({torch::indexing::Slice(), torch::indexing::Slice(), -1}));

    lower_eq = lower_eq_new;
    upper_eq = upper_eq_new;
};

void CNN::BackwardRelu(int current_layer, int layer, torch::Tensor& lower_eq, torch::Tensor& upper_eq, const string& method) {
    torch::Tensor positive_eq, negative_eq;

    torch::Tensor lower_ratio = lower_ratio_v_[layer].index({torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice(), 0});
    torch::Tensor lower_bias = lower_ratio_v_[layer].index({torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice(), 1});
    torch::Tensor upper_ratio = upper_ratio_v_[layer].index({torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice(), 0});
    torch::Tensor upper_bias = upper_ratio_v_[layer].index({torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice(), 1});
    // cout << lower_ratio << endl;
    // cout << lower_bias << endl;
    // cout << upper_ratio << endl;
    // cout << upper_bias << endl;
    // getchar();


    // lower
    if (method == "clipped") {
        lower_ratio = alpha_v_[current_layer][layer].index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 0});
        lower_ratio = torch::clamp(lower_ratio, 0.0, 1.0).detach() - lower_ratio.detach() + lower_ratio;
    }
    else assert(method == "fixed");

    positive_eq = torch::clamp(lower_eq.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, -1)}), 0.0, c10::nullopt);
    negative_eq = torch::clamp(lower_eq.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, -1)}), c10::nullopt, 0.0);
    lower_eq = torch::cat({positive_eq * lower_ratio + negative_eq * upper_ratio, (lower_eq.index({torch::indexing::Slice(), torch::indexing::Slice(), -1}) + torch::sum(positive_eq * lower_bias, 2) + torch::sum(negative_eq * upper_bias, 2)).unsqueeze(2)}, 2);

    // upper
    if (method == "clipped") {
        lower_ratio = alpha_v_[current_layer][layer].index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 1});
        lower_ratio = torch::clamp(lower_ratio, 0.0, 1.0).detach() - lower_ratio.detach() + lower_ratio;
    }
    else assert(method == "fixed");
    positive_eq = torch::clamp(upper_eq.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, -1)}), 0.0, c10::nullopt);
    negative_eq = torch::clamp(upper_eq.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, -1)}), c10::nullopt, 0.0);
    upper_eq = torch::cat({positive_eq * upper_ratio + negative_eq * lower_ratio, (upper_eq.index({torch::indexing::Slice(), torch::indexing::Slice(), -1}) + torch::sum(positive_eq * upper_bias, 2) + torch::sum(negative_eq * lower_bias, 2)).unsqueeze(2)}, 2);
};

torch::Tensor CNN::ComputeBound(const torch::Tensor& equation, const torch::Tensor& interval) {
    torch::Tensor positive_eq = torch::clamp(equation, 0, c10::nullopt);
    torch::Tensor negative_eq = torch::clamp(equation, c10::nullopt, 0);
    torch::Tensor res = torch::bmm(positive_eq, interval) + torch::bmm(negative_eq, interval).index({torch::indexing::Slice(), torch::indexing::Slice(), torch::tensor({1, 0})});
    return res;
};

void CNN::ComputeNonlinearNeuron() {
    assert(neural_status_v_.size() == num_neuron_v_.size() - 1);
    assert(nonlinear_neuron_v_.size() == num_neuron_v_.size() - 1);
    for (int l = 1; l < (int)num_neuron_v_.size() - 1; ++l) {
        torch::Tensor active_neuron_mask = bound_before_activation_v_[l].index({torch::indexing::Slice(), torch::indexing::Slice(), 0}) >= 0;
        torch::Tensor inactive_neuron_mask = bound_before_activation_v_[l].index({torch::indexing::Slice(), torch::indexing::Slice(), 1}) <= 0;
        active_neuron_mask = (get<0>(torch::min(active_neuron_mask, 0)) > 0);
        inactive_neuron_mask = (get<0>(torch::min(inactive_neuron_mask, 0)) > 0);
        neural_status_v_[l] = torch::zeros({num_neuron_v_[l]}, torch::dtype(torch::kInt));
        neural_status_v_[l].index_put_({active_neuron_mask}, 1);
        neural_status_v_[l].index_put_({inactive_neuron_mask}, -1);
        nonlinear_neuron_v_[l] = torch::nonzero(torch::logical_or(active_neuron_mask, inactive_neuron_mask) == 0).squeeze(1);
    };
};

void CNN::UpdateConvexSolver() {
    RemoveConvexBoundConstraint();
    RemoveConvexAssignmentConstraint();
    GRBModel& model = solver_.ConvexSolver();
    for (int l = 0; l < (int)num_neuron_v_.size(); ++l) {

        torch::Tensor lb_batch = get<0>(torch::min(bound_before_activation_v_[l].index({torch::indexing::Slice(), torch::indexing::Slice(), 0}), 0));
        torch::Tensor ub_batch = get<0>(torch::max(bound_before_activation_v_[l].index({torch::indexing::Slice(), torch::indexing::Slice(), 1}), 0));
        // cout << lb_batch << endl;
        // cout << ub_batch << endl;
        // getchar();
        for (int i = 0; i < (int)num_neuron_v_[l]; ++i) {
            double lb = lb_batch.index({i}).item<double>();
            double ub = ub_batch.index({i}).item<double>();
            convex_network_var_v_[l][i].set(GRB_DoubleAttr_LB, lb);
            convex_network_var_v_[l][i].set(GRB_DoubleAttr_UB, ub);
            if (l > 0 and l < (int)num_neuron_v_.size() - 1) {
                lb = min(0.0, lb);
                ub = max(0.0, ub);
                convex_relu_var_v_[l][i].set(GRB_DoubleAttr_LB, 0);
                convex_relu_var_v_[l][i].set(GRB_DoubleAttr_UB, ub);

                convex_slack_var_v_[l][i].set(GRB_DoubleAttr_LB, 0);
                convex_slack_var_v_[l][i].set(GRB_DoubleAttr_UB, -lb);
            }
        }
    }
    model.update();

    GRBLinExpr& objective = solver_.ConvexObjective();
    // set constraint for convex solver
    for (int l = 1; l < (int)num_neuron_v_.size() - 1; ++l) {
        torch::Tensor lb_batch = get<0>(torch::min(bound_before_activation_v_[l].index({torch::indexing::Slice(), torch::indexing::Slice(), 0}), 0));
        torch::Tensor ub_batch = get<0>(torch::max(bound_before_activation_v_[l].index({torch::indexing::Slice(), torch::indexing::Slice(), 1}), 0));
        for (int i = 0; i < num_neuron_v_[l]; ++i) {
            // construct upper and lower equation constraints
            int neural_status = neural_status_v_[l].index({i}).item<int>();
            // printf("%s", solver_.Str().c_str());
            // getchar();
            if (neural_status == 1) {
                // printf("l, i: %d, %d\n", l, i);
                sat_network_status_v_[l].index_put_({i}, 1);
                solver_.AddSatFixedConstraint(CSolver::SatExpr(sat_network_var_v_[l][i]));
                // printf("l, i: %d, %d\n", l, i);
                // getchar();
                AddConvexBoundConstraint(convex_slack_var_v_[l][i] == 0);
            }
            else if (neural_status == -1) {
                // printf("l, i: %d, %d\n", l, i);
                sat_network_status_v_[l].index_put_({i}, -1);
                solver_.AddSatFixedConstraint(!CSolver::SatExpr(sat_network_var_v_[l][i]));
                // printf("l, i: %d, %d\n", l, i);
                // getchar();
                AddConvexBoundConstraint(convex_relu_var_v_[l][i] == 0);
            }
            else if (neural_status == 0) {
                double lb = lb_batch.index({i}).item<double>();
                double ub = ub_batch.index({i}).item<double>();
                AddConvexBoundConstraint(convex_relu_var_v_[l][i] <= ub / (ub - lb) * (convex_network_var_v_[l][i] - lb));

                if (bound_before_activation_v_[0].sizes()[0] == 1) {
                    torch::Tensor weight;
                    double bias;
                    GRBLinExpr expr;
                    weight = lower_equation_after_activation_v_[l].index({0, i, torch::indexing::Slice(torch::indexing::None, -1)}); 
                    bias = lower_equation_after_activation_v_[l].index({0, i, -1}).item<double>();
                    expr = solver_.ComputeConvexExpr(convex_network_var_v_[0], weight) + bias;
                    AddConvexBoundConstraint(convex_relu_var_v_[l][i] >= expr);

                    weight = upper_equation_after_activation_v_[l].index({0, i, torch::indexing::Slice(torch::indexing::None, -1)}); 
                    bias = upper_equation_after_activation_v_[l].index({0, i, -1}).item<double>();
                    expr = solver_.ComputeConvexExpr(convex_network_var_v_[0], weight) + bias;
                    AddConvexBoundConstraint(convex_relu_var_v_[l][i] <= expr);
                }


                if (solver_.Certificate() == CSolver::Certificate::IIS or solver_.Certificate() == CSolver::Certificate::NONE) {
                    // objective += convex_slack_var_v_[l][i] + convex_relu_var_v_[l][i];
                    objective += ObjectiveRatio(l) * (convex_slack_var_v_[l][i] + convex_relu_var_v_[l][i]);
                }
                else assert(solver_.Certificate() == CSolver::Certificate::DUAL);
            }
            else assert(false);
        }
    }
    model.update();
};

void CNN::ComputeNetworkStatus() {
    assert(neural_status_v_.size() == num_neuron_v_.size() - 1);
    assert(nonlinear_neuron_v_.size() == num_neuron_v_.size() - 1);

    // add implication relation to sat solver
    bool implication_set_b = false;
    z3::expr implication(solver_.SatContex());
    auto add_implication = [&implication_set_b, &implication] (const z3::expr& expr) {
        if (implication_set_b) {
            implication = implication || expr;
        }
        else {
            implication_set_b = true;
            implication = expr;
        }
        // cout << "add: " << implication << endl;
    };
    auto assert_implication = [this, &implication_set_b, &implication] (const z3::expr& expr) {
        assert(implication_set_b);
        solver_.AddSatConstraint((implication || expr));
        // cout << "assert: " << (implication || expr) << endl;
    };

    for (int l = 1; l < (int)num_neuron_v_.size() - 1; ++l) {
        for (int i = 0; i < nonlinear_neuron_v_[l].sizes()[0]; ++i) {
            int neuron_id = nonlinear_neuron_v_[l].index({i}).item<int>();
            if (bound_before_activation_v_[l].index({0, neuron_id, 0}).item<double>() >= 0) {
                assert_implication(CSolver::SatExpr(sat_network_var_v_[l][neuron_id]));
            }
            else if (bound_before_activation_v_[l].index({0, neuron_id, 1}).item<double>() <= 0) {
                assert_implication(!CSolver::SatExpr(sat_network_var_v_[l][neuron_id]));
            }

            int neural_assignment = sat_network_status_v_[l].index({neuron_id}).item<int>();
            if (neural_assignment == 1) {
                add_implication(!CSolver::SatExpr(sat_network_var_v_[l][neuron_id]));
            }
            else if (neural_assignment == -1) {
                add_implication(CSolver::SatExpr(sat_network_var_v_[l][neuron_id]));
            }
        }
    }

    // update assignment
    torch::Tensor mask;
    for (int l = 1; l < (int)num_neuron_v_.size() - 1; ++l) {
        mask = bound_before_activation_v_[l].index({0, nonlinear_neuron_v_[l], 0}) >= 0;
        torch::Tensor active_neuron = nonlinear_neuron_v_[l].index({mask});
        sat_network_status_v_[l].index_put_({active_neuron}, 1);

        mask = bound_before_activation_v_[l].index({0, nonlinear_neuron_v_[l], 1}) <= 0;
        torch::Tensor inactive_neuron = nonlinear_neuron_v_[l].index({mask});
        sat_network_status_v_[l].index_put_({inactive_neuron}, -1);
    };
};

void CNN::UpdateConvexSolverAssertion() {
    RemoveConvexAssignmentConstraint();

    GRBModel& model = solver_.ConvexSolver();
    GRBLinExpr& objective = solver_.ConvexObjective();
    if (solver_.Certificate() == CSolver::Certificate::DUAL) {
        assert(!solver_.BoundComputing());
        // set constraint for convex solver
        for (int l = 1; l < (int)num_neuron_v_.size() - 1; ++l) {
            for (int i = 0; i < nonlinear_neuron_v_[l].sizes()[0]; ++i) {
                int neuron_id = nonlinear_neuron_v_[l].index({i}).item<int>();
                int neural_assignment = sat_network_status_v_[l].index({neuron_id}).item<int>();
                // printf("neural assignmnet: %d, %d, %d\n", l, neuron_id, neural_assignment);
                if (neural_assignment == 1) {
                    // objective += convex_slack_var_v_[l][neuron_id];
                    objective += ObjectiveRatio(l) * (convex_slack_var_v_[l][neuron_id]);
                }
                else if (neural_assignment == -1) {
                    // objective += convex_relu_var_v_[l][neuron_id];
                    objective += ObjectiveRatio(l) * (convex_relu_var_v_[l][neuron_id]);
                }
                else assert(neural_assignment == 0);
            }
        }
        model.update();
    }
    else if (solver_.Certificate() == CSolver::Certificate::IIS or solver_.Certificate() == CSolver::Certificate::NONE) {
        // set constraint for convex solver
        for (int l = 1; l < (int)num_neuron_v_.size() - 1; ++l) {
            torch::Tensor lb_batch = get<0>(torch::min(bound_before_activation_v_[l].index({torch::indexing::Slice(), torch::indexing::Slice(), 0}), 0));
            torch::Tensor ub_batch = get<0>(torch::max(bound_before_activation_v_[l].index({torch::indexing::Slice(), torch::indexing::Slice(), 1}), 0));
            for (int i = 0; i < nonlinear_neuron_v_[l].sizes()[0]; ++i) {
                int neuron_id = nonlinear_neuron_v_[l].index({i}).item<int>();
                int neural_assignment = sat_network_status_v_[l].index({neuron_id}).item<int>();
                // printf("neural assignmnet: %d, %d, %d\n", l, neuron_id, neural_assignment);
                if (neural_assignment == 1) {
                    AddConvexAssignmentConstraint(l, neuron_id, convex_slack_var_v_[l][neuron_id] == 0);
                }
                else if (neural_assignment == -1) {
                    AddConvexAssignmentConstraint(l, neuron_id, convex_relu_var_v_[l][neuron_id] == 0);
                }
                else if (neural_assignment == 0) {
                    // construct upper and lower equation constraints
                    double lb = lb_batch.index({i}).item<double>();
                    double ub = ub_batch.index({i}).item<double>();
                    GRBLinExpr upper_bound = ub / (ub - lb) * (convex_network_var_v_[l][neuron_id] - lb);
                    AddConvexAssignmentConstraint(l, neuron_id, convex_relu_var_v_[l][neuron_id] <= upper_bound);

                    // objective += convex_slack_var_v_[l][neuron_id] + convex_relu_var_v_[l][neuron_id];
                    objective += ObjectiveRatio(l) * (convex_slack_var_v_[l][neuron_id] + convex_relu_var_v_[l][neuron_id]);
                }
                else assert(false);
            }
        }
        model.update();
    }
    else assert(false);
};

bool CNN::CheckSize() {
    // initialize bound variables
    bool res = true;
    res &= (bound_before_activation_v_.size() == num_neuron_v_.size());
    res &= (lower_equation_before_activation_v_.size() == num_neuron_v_.size());
    res &= (upper_equation_before_activation_v_.size() == num_neuron_v_.size());
    res &= (lower_equation_after_activation_v_.size() == num_neuron_v_.size() - 1);
    res &= (upper_equation_after_activation_v_.size() == num_neuron_v_.size() - 1);

    res &= (lower_ratio_v_.size() == num_neuron_v_.size() - 1);
    res &= (upper_ratio_v_.size() == num_neuron_v_.size() - 1);
    res &= (alpha_v_.size() == num_neuron_v_.size());

    res &= (neural_status_v_.size() == num_neuron_v_.size() - 1);
    res &= (nonlinear_neuron_v_.size() == num_neuron_v_.size() - 1);

    res &= (sat_network_var_v_.size() == num_neuron_v_.size() - 1);
    res &= (sat_network_status_v_.size() == num_neuron_v_.size() - 1);

    res &= (convex_network_var_v_.size() == num_neuron_v_.size());
    res &= (convex_relu_var_v_.size() == num_neuron_v_.size() - 1);
    res &= (convex_slack_var_v_.size() == num_neuron_v_.size() - 1);
    res &= (convex_assertion_constraint_vm_.size() == num_neuron_v_.size() - 1);
    return res;
};

int CNN::Test() {
    for (int i = 1; i < (int)weight_v_.size(); ++i) {
        cout << weight_v_[i].sizes() << endl;
    };
    for (int i = 1; i < (int)bias_v_.size(); ++i) {
        cout << bias_v_[i].sizes() << endl;
    };
    return 0;
};


NNV_NAMESPACING_END
