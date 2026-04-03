#include "solver.hpp"

NNV_NAMESPACING_START

void CSolver::ResetSat() {
    sat_solver_.reset();
    for (list<z3::expr>::iterator it = sat_fixed_constraint_l_.begin(); it != sat_fixed_constraint_l_.end(); ++it) {
        sat_solver_.add(*it);
    }
}

CSolver::SatVar* CSolver::AddSatVar(const string& name) {
    sat_var_l_.emplace_back(sat_contex_.bool_const(name.c_str()));
    SatVar* var = &sat_var_l_.back();
    var->it_ = prev(sat_var_l_.end());
    return var;
};

void CSolver::SetSatOrConstraint(SatVar* var, bool& f, z3::expr& constr) {
    if (f) {
        constr = constr | var->var_;
    }
    else {
        f = true;
        constr = var->var_;
    }
};

void CSolver::SetSatAndConstraint(SatVar* var, bool& f, z3::expr& constr) {
    if (f) {
        constr = constr & var->var_;
    }
    else {
        f = true;
        constr = var->var_;
    }
};

void CSolver::AddSatFixedConstraint(const z3::expr& constraint) {
    sat_solver_.add(constraint);
    sat_fixed_constraint_l_.emplace_back(constraint);
};

bool CSolver::CheckSat() {
    if (sat_solver_.check() == z3::sat) {
        sat_model_available_b_ = true;
        sat_model_ = sat_solver_.get_model();
        // cout << sat_model_ << endl;
        // cout << sat_model_.size() << endl;
        // for (int i = 0; i < sat_model_.size(); ++i) {
        //     cout << sat_model_[i]() << endl;
        // }
        // getchar();
        return true;
    }
    else {
        sat_model_available_b_ = false;
        return false;
    }
}

int CSolver::EvalSatVar(SatVar* var) const {
    if (sat_model_.eval(var->var_).is_true()) {
        return 1;
    }
    else if (sat_model_.eval(var->var_).is_false()) {
        return -1;
    }
    else {
        return 0;
    }
}

GRBLinExpr CSolver::ComputeConvexExpr(const vector<GRBVar>& var, const torch::Tensor& weight) {
    GRBLinExpr expr;
    double w;
    assert(weight.dim() == 1);
    assert((int)var.size() == weight.sizes()[0]);
    for (int i = 0; i < (int)var.size(); ++i) {
        w = weight.index({i}).item<double>();
        expr += w * var[i];
    }
    return expr;
}

void CSolver::SetConvexConstrant() {
    for (list<SatVar>::iterator it = sat_var_l_.begin(); it != sat_var_l_.end(); ++it) {
        SatVar& var = *it;
        if (EvalSatVar(&var) == 1) {
            for (int i = 0; i < (int)var.lin_expr_.size(); ++i) {
                var.lin_constr_.emplace_back(convex_solver_.addConstr(var.lin_expr_[i]));
            }
            for (int i = 0; i < (int)var.quad_expr_.size(); ++i) {
                var.quad_constr_.emplace_back(convex_solver_.addQConstr(var.quad_expr_[i]));
            }
        }
    }
    convex_solver_.update();
}

void CSolver::RemoveConvexConstrant() {
    for (list<SatVar>::iterator it = sat_var_l_.begin(); it != sat_var_l_.end(); ++it) {
        SatVar& var = *it;
        for (int i = 0; i < (int)var.lin_constr_.size(); ++i) {
            convex_solver_.remove(var.lin_constr_[i]);
        }
        var.lin_constr_.clear();
        for (int i = 0; i < (int)var.quad_constr_.size(); ++i) {
            convex_solver_.remove(var.quad_constr_[i]);
        }
        var.quad_constr_.clear();
    }
    convex_solver_.update();
}

void CSolver::SetConvexObjective(GRBLinExpr expr) {
    convex_solver_.setObjective(expr);
    convex_solver_.update();
};

void CSolver::SetConvexObjective() {
    convex_solver_.setObjective(convex_objective_);
    convex_solver_.update();
};

void CSolver::ComputeConvexCertificate(vector<z3::expr>& conflict) {
    for (list<SatVar>::iterator it = sat_var_l_.begin(); it != sat_var_l_.end(); ++it) {
        SatVar& var = *it;
        bool conflict_b = false;
        for (int i = 0; i < (int)var.lin_constr_.size(); ++i) {
            if (certificate_ == Certificate::DUAL) {
                conflict_b |= true;
            }
            else if (certificate_ == Certificate::IIS) {
                if (var.lin_constr_[i].get(GRB_IntAttr_IISConstr) == 1) {
                    conflict_b |= true;
                }
            }
            else if (certificate_ == Certificate::NONE) {
                conflict_b |= true;
            }
            else assert(false);
        }
        for (int i = 0; i < (int)var.quad_constr_.size(); ++i) {
            if (certificate_ == Certificate::DUAL) {
                conflict_b |= true;
            }
            else if (certificate_ == Certificate::IIS) {
                if (var.quad_constr_[i].get(GRB_IntAttr_IISQConstr) == 1) {
                    conflict_b |= true;
                }
            }
            else if (certificate_ == Certificate::NONE) {
                conflict_b |= true;
            }
            else assert(false);
        }
        if (conflict_b) {
            conflict.emplace_back(!var.var_);
        }
    }
}

void CSolver::WriteConvexSolver(const string& file) {
    convex_solver_.write(file + ".lp");
    if (convex_solver_.get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
        convex_solver_.computeIIS();
        convex_solver_.write(file + ".ilp");
    }
};

string CSolver::Str(const string& file) {
    string res = "Sat Solver:\n";
    res += "  Var:\n";
    for (list<SatVar>::iterator it = sat_var_l_.begin(); it != sat_var_l_.end(); ++it) {
        res += InsertSpaceAtBeginOfLine(SatVarStr(&(*it)), 4) + '\n';
    }

    ostringstream string_conversion;
    res += "  Constraint:\n";
    for (list<z3::expr>::iterator it = sat_fixed_constraint_l_.begin(); it != sat_fixed_constraint_l_.end(); ++it) {
        string_conversion << *it;
        res += InsertSpaceAtBeginOfLine(string_conversion.str(), 4) + '\n';
        string_conversion.str("");
    }

    res += "  Assertion:\n";
    string_conversion << sat_solver_.assertions();
    res += InsertSpaceAtBeginOfLine(string_conversion.str(), 4) + '\n';
    string_conversion.str("");

    res += "Convex Solver\n";
    res += "  Constraint write to " + file + ".lp \n";
    WriteConvexSolver(file);
    return res;
};

string CSolver::SatVarStr(SatVar* v) const {
    string res;
    ostringstream string_conversion;

    string_conversion << v->var_;
    if (sat_model_available_b_) {
        if (EvalSatVar(v) == 1) {
            string_conversion << " - True";
        }
        else if (EvalSatVar(v) == 0) {
            string_conversion << " - Unknown";
        }
        else if (EvalSatVar(v) == -1) {
            string_conversion << " - False";
        }
        else assert(false);
    }
    res += string_conversion.str();
    string_conversion.str("");

    return res;
};

string CSolver::ConvexVarStr(const GRBVar& v) const {
    string res = v.get(GRB_StringAttr_VarName) + ": lb: " + to_string(v.get(GRB_DoubleAttr_LB)) + ", ub: " + to_string(v.get(GRB_DoubleAttr_UB)) + ", obj: " + to_string(v.get(GRB_DoubleAttr_Obj)) + ", type: " + v.get(GRB_CharAttr_VType);
    if (convex_solver_.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
        res += ", value: " + to_string(v.get(GRB_DoubleAttr_X));
    }
    return res;
};

GRBEnv& CSolver::InitConvexSolver() {
    convex_env_.set(GRB_IntParam_OutputFlag, 0);
    convex_env_.set(GRB_IntParam_Method, 4);
    convex_env_.start();
    return convex_env_;
};

NNV_NAMESPACING_END
