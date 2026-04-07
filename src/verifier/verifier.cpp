#include "verifier.hpp"

extern Usage usage;

NNV_NAMESPACING_START

int CVerifier::Verify(const string & model_path) {
    solver_.ResetSat();
    counter_example_ = torch::empty({0});
    iteration_ = 0;

    z3::context property_context;
    z3::expr_vector parsed_property_v = property_context.parse_file(model_path.c_str());

    map<string, CSolver::SatVar*> bool_var_m;
    map<string, GRBVar> real_var_m;
    int aux_var_id = 0;

    auto EnsureBoolVar = [this, &bool_var_m](const string& name) {
        auto it = bool_var_m.find(name);
        if (it != bool_var_m.end()) {
            return it->second;
        }
        CSolver::SatVar* var = solver_.AddSatVar(name);
        bool_var_m.emplace(name, var);
        return var;
    };

    auto EnsureRealVar = [this, &real_var_m](const string& name) -> GRBVar& {
        auto it = real_var_m.find(name);
        if (it != real_var_m.end()) {
            return it->second;
        }
        GRBVar var = solver_.ConvexSolver().addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, name);
        auto res = real_var_m.emplace(name, var);
        return res.first->second;
    };

    function<bool(const z3::expr&)> IsBoolAtom = [](const z3::expr& expr) {
        return expr.is_bool() and expr.num_args() == 0 and expr.decl().decl_kind() == Z3_OP_UNINTERPRETED;
    };

    function<bool(const z3::expr&)> ContainsArithVar = [&ContainsArithVar](const z3::expr& expr) {
        if (expr.is_arith() and expr.num_args() == 0 and expr.decl().decl_kind() == Z3_OP_UNINTERPRETED) {
            return true;
        }
        for (int i = 0; i < expr.num_args(); ++i) {
            if (ContainsArithVar(expr.arg(i))) {
                return true;
            }
        }
        return false;
    };

    function<bool(const z3::expr&)> IsLinearComparison = [&ContainsArithVar](const z3::expr& expr) {
        Z3_decl_kind kind = expr.decl().decl_kind();
        return (kind == Z3_OP_LE or kind == Z3_OP_LT or kind == Z3_OP_GE or kind == Z3_OP_GT or kind == Z3_OP_EQ) and expr.num_args() == 2 and (ContainsArithVar(expr.arg(0)) or ContainsArithVar(expr.arg(1)));
    };

    function<bool(const z3::expr&)> IsLinearConstraintFormula = [&IsLinearComparison, &IsLinearConstraintFormula](const z3::expr& expr) {
        if (IsLinearComparison(expr)) {
            return true;
        }
        if (expr.is_bool() and expr.decl().decl_kind() == Z3_OP_AND) {
            for (int i = 0; i < expr.num_args(); ++i) {
                if (!IsLinearConstraintFormula(expr.arg(i))) {
                    return false;
                }
            }
            return expr.num_args() > 0;
        }
        return false;
    };

    function<double(const z3::expr&)> ToDouble = [&ToDouble](const z3::expr& expr) -> double {
        Z3_decl_kind kind = expr.decl().decl_kind();
        if (kind == Z3_OP_ANUM) {
            return expr.as_double();
        }
        else if (kind == Z3_OP_UMINUS) {
            assert(expr.num_args() == 1);
            return -ToDouble(expr.arg(0));
        }
        else if (kind == Z3_OP_ADD) {
            double value = 0;
            for (int i = 0; i < expr.num_args(); ++i) {
                value += ToDouble(expr.arg(i));
            }
            return value;
        }
        else if (kind == Z3_OP_SUB) {
            assert(expr.num_args() >= 1);
            double value = ToDouble(expr.arg(0));
            for (int i = 1; i < expr.num_args(); ++i) {
                value -= ToDouble(expr.arg(i));
            }
            return value;
        }
        else if (kind == Z3_OP_MUL) {
            double value = 1;
            for (int i = 0; i < expr.num_args(); ++i) {
                value *= ToDouble(expr.arg(i));
            }
            return value;
        }
        else if (kind == Z3_OP_DIV) {
            assert(expr.num_args() == 2);
            return ToDouble(expr.arg(0)) / ToDouble(expr.arg(1));
        }
        assert(false);
    };

    function<GRBLinExpr(const z3::expr&)> ToLinearExpr = [&ToLinearExpr, &ToDouble, &EnsureRealVar, &ContainsArithVar](const z3::expr& expr) -> GRBLinExpr {
        Z3_decl_kind kind = expr.decl().decl_kind();
        if (expr.is_arith() and expr.num_args() == 0 and kind == Z3_OP_UNINTERPRETED) {
            return EnsureRealVar(expr.decl().name().str());
        }
        else if (kind == Z3_OP_ANUM) {
            return GRBLinExpr(expr.as_double());
        }
        else if (kind == Z3_OP_UMINUS) {
            assert(expr.num_args() == 1);
            return -ToLinearExpr(expr.arg(0));
        }
        else if (kind == Z3_OP_ADD) {
            GRBLinExpr res;
            for (int i = 0; i < expr.num_args(); ++i) {
                res += ToLinearExpr(expr.arg(i));
            }
            return res;
        }
        else if (kind == Z3_OP_SUB) {
            assert(expr.num_args() >= 1);
            GRBLinExpr res = ToLinearExpr(expr.arg(0));
            for (int i = 1; i < expr.num_args(); ++i) {
                res -= ToLinearExpr(expr.arg(i));
            }
            return res;
        }
        else if (kind == Z3_OP_MUL) {
            double coeff = 1;
            bool linear_term_set_b = false;
            GRBLinExpr linear_term;
            for (int i = 0; i < expr.num_args(); ++i) {
                if (ContainsArithVar(expr.arg(i))) {
                    assert(!linear_term_set_b);
                    linear_term = ToLinearExpr(expr.arg(i));
                    linear_term_set_b = true;
                }
                else {
                    coeff *= ToDouble(expr.arg(i));
                }
            }
            if (!linear_term_set_b) {
                return GRBLinExpr(coeff);
            }
            return coeff * linear_term;
        }
        else if (kind == Z3_OP_DIV) {
            assert(expr.num_args() == 2);
            return (1.0 / ToDouble(expr.arg(1))) * ToLinearExpr(expr.arg(0));
        }
        assert(false);
    };

    function<GRBTempConstr(const z3::expr&)> ToTempConstr = [this, &ToLinearExpr](const z3::expr& expr) -> GRBTempConstr {
        assert(expr.num_args() == 2);
        GRBLinExpr lhs = ToLinearExpr(expr.arg(0));
        GRBLinExpr rhs = ToLinearExpr(expr.arg(1));
        Z3_decl_kind kind = expr.decl().decl_kind();
        if (kind == Z3_OP_LE) {
            return lhs - rhs <= 0;
        }
        else if (kind == Z3_OP_LT) {
            return lhs - rhs + solver_.parameter_.kEPS <= 0;
        }
        else if (kind == Z3_OP_GE) {
            return lhs - rhs >= 0;
        }
        else if (kind == Z3_OP_GT) {
            return lhs - rhs - solver_.parameter_.kEPS >= 0;
        }
        else if (kind == Z3_OP_EQ) {
            return lhs - rhs == 0;
        }
        assert(false);
    };

    function<z3::expr(const z3::expr&)> ToSatExpr = [this, &ToSatExpr, &EnsureBoolVar, &IsBoolAtom](const z3::expr& expr) -> z3::expr {
        if (expr.is_true() or expr.is_false()) {
            return solver_.SatContex().bool_val(expr.is_true());
        }

        Z3_decl_kind kind = expr.decl().decl_kind();
        if (IsBoolAtom(expr)) {
            return CSolver::SatExpr(EnsureBoolVar(expr.decl().name().str()));
        }
        else if (kind == Z3_OP_NOT) {
            assert(expr.num_args() == 1);
            return !ToSatExpr(expr.arg(0));
        }
        else if (kind == Z3_OP_OR) {
            assert(expr.num_args() >= 1);
            z3::expr res = ToSatExpr(expr.arg(0));
            for (int i = 1; i < expr.num_args(); ++i) {
                res = res || ToSatExpr(expr.arg(i));
            }
            return res;
        }
        else if (kind == Z3_OP_AND) {
            assert(expr.num_args() >= 1);
            z3::expr res = ToSatExpr(expr.arg(0));
            for (int i = 1; i < expr.num_args(); ++i) {
                res = res && ToSatExpr(expr.arg(i));
            }
            return res;
        }
        else if (kind == Z3_OP_IMPLIES) {
            assert(expr.num_args() == 2);
            return z3::implies(ToSatExpr(expr.arg(0)), ToSatExpr(expr.arg(1)));
        }
        else if (kind == Z3_OP_EQ) {
            assert(expr.num_args() == 2);
            return ToSatExpr(expr.arg(0)) == ToSatExpr(expr.arg(1));
        }
        assert(false);
    };

    auto TranslateExpr = [this, &property_context](const z3::expr& expr) {
        return z3::expr(solver_.SatContex(), Z3_translate(property_context, expr, solver_.SatContex()));
    };

    function<void(CSolver::SatVar*, const z3::expr&)> AddLinearConstraintFormula =
    [&AddLinearConstraintFormula, &ToTempConstr](CSolver::SatVar* sat_var, const z3::expr& expr) {
        if (expr.decl().decl_kind() == Z3_OP_AND) {
            for (int i = 0; i < expr.num_args(); ++i) {
                AddLinearConstraintFormula(sat_var, expr.arg(i));
            }
            return;
        }
        sat_var->lin_expr_.emplace_back(ToTempConstr(expr));
    };

    auto AddGuardedConstraint = [this, &EnsureBoolVar, &AddLinearConstraintFormula, &aux_var_id](const z3::expr& guard_expr, const z3::expr& constr_expr) {
        CSolver::SatVar* gate_var = nullptr;
        // cout << guard_expr << endl;
        // cout << constr_expr << endl;
        if (guard_expr.num_args() == 0 and guard_expr.decl().decl_kind() == Z3_OP_UNINTERPRETED) {
            gate_var = EnsureBoolVar(guard_expr.decl().name().str());
        }
        else {
            gate_var = solver_.AddSatVar("aux_guard_" + to_string(aux_var_id++));
            solver_.AddSatConstraint(CSolver::SatExpr(gate_var) == guard_expr);
        }
        AddLinearConstraintFormula(gate_var, constr_expr);
    };

    for (int i = 0; i < parsed_property_v.size(); ++i) {
        z3::expr expr = TranslateExpr(parsed_property_v[i]);
        // cout << expr << endl;
        if (expr.decl().decl_kind() == Z3_OP_IMPLIES and expr.num_args() == 2 and IsLinearConstraintFormula(expr.arg(1))) {
            // printf("type 1\n");
            AddGuardedConstraint(ToSatExpr(expr.arg(0)), expr.arg(1));
        }
        else if (IsLinearConstraintFormula(expr)) {
            // printf("type 2\n");
            CSolver::SatVar* gate_var = solver_.AddSatVar("aux_true_" + to_string(aux_var_id++));
            solver_.AddSatConstraint(CSolver::SatExpr(gate_var));
            AddLinearConstraintFormula(gate_var, expr);
        }
        else {
            // printf("type 3: ");
            // cout << ToSatExpr(expr) << endl;
            solver_.AddSatConstraint(ToSatExpr(expr));
        }
        // getchar();
    }

    solver_.ConvexSolver().update();
    solution_checker_f_ = [this]() {
        counter_example_ = torch::zeros({1}, torch::dtype(torch::kFloat64));
        return 1;
    };

    // printf("%s", solver_.Str().c_str());
    // getchar();
    int result = Solve_Smc();
    if (result == 0) {
        counter_example_ = torch::empty({0});
    }
    WriteOutput();
    return result;
};

string CVerifier::Str() const {
    string res;
    if (counter_example_.numel() == 0) {
        res += "Input is UNSAT within " + to_string(iteration_) + " iteration(s).\n";
    }
    else {
        res += "Input is SAT within " + to_string(iteration_) + " iteration(s).\n";
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
