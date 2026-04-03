#include "sdt.hpp"

NNV_NAMESPACING_START

string CInnerNode::Str() const {
    string res;
    ostringstream string_conversion;

    res += "-> inner node:\n";
    res += "  weight:\n";
    string_conversion << weight_;
    res += InsertSpaceAtBeginOfLine(string_conversion.str(), 4) + "\n";
    string_conversion.str("");

    res += "  bias:\n";
    string_conversion << bias_;
    res += InsertSpaceAtBeginOfLine(string_conversion.str(), 4) + "\n";
    string_conversion.str("");

    res += InsertSpaceAtBeginOfLine(left_->Str(), 2) + "\n";
    res += InsertSpaceAtBeginOfLine(right_->Str(), 2);
    return res;
};

string CLeafNode::Str() const {
    string res;
    ostringstream string_conversion;

    res += "-> leaf node:\n";
    res += "  distribution:\n";
    string_conversion << distribution_;
    res += InsertSpaceAtBeginOfLine(string_conversion.str(), 4) + "\n";
    string_conversion.str("");
    return res;
};

void CSDT::LoadModel(const string & model_path) {
    LoadTorchModel(model_path);
    InitVariable();
}

torch::Tensor CSDT::Forward(const torch::Tensor& input) {
    CNode* node_p = root_;
    torch::Tensor res;
    while (true) {
        if (node_p->IsInner()) {
            torch::Tensor weight = ((CInnerNode*)node_p)->weight_.view({input_dim_});
            double bias = ((CInnerNode*)node_p)->bias_.item<double>();
            double probability = torch::dot(weight, input).item<double>() + bias;
            // cout << weight << endl;
            // cout << bias << endl;
            // cout << probability << endl;
            if (probability < 0) {
                node_p = node_p->left_;
                // printf("left\n");
            }
            else {
                node_p = node_p->right_;
                // printf("right\n");
            }
        }
        else if (node_p->IsLeaf()) {
            res = ((CLeafNode*) node_p)->distribution_;
            break;
        }
        else assert(false);
    }
    return res;
};

string CSDT::Str(bool matrix) const {
    string res;
    ostringstream string_conversion;
    res += "Soft Decision Tree:\n";
    res += "  input dim: " + to_string(input_dim_) + "\n";
    res += "  output dim: " + to_string(output_dim_) + "\n";
    if (matrix) {
        res += InsertSpaceAtBeginOfLine(root_->Str(), 2) + "\n";
    }
    res += "  SAT var: \n";
    for (int i = 0; i < (int)sat_output_var_vl_.size(); ++i) {
        res += "    action " + to_string(i) + ":";
        for (auto it = sat_output_var_vl_[i].begin(); it != sat_output_var_vl_[i].end(); ++it) {
            res += " " + solver_.SatVarStr(*it);
        }
        res += "\n";
    }
    return res;
};

void CSDT::LoadTorchModel(const string & model_path) {
    function<CNode*(const torch::jit::script::Module& node)> construct_node;
    construct_node = [&] (const torch::jit::script::Module& node) {
        CNode* node_p;

        torch::jit::named_module_list module_list = node.named_children();
        if (module_list.size() == 3) {
            // printf("inner\n");
            inner_node_l_.emplace_back();
            CInnerNode& inner_node = inner_node_l_.back();
            for (auto m = module_list.begin(); m != module_list.end(); ++m) {
                if ((*m).name == "left") {
                    inner_node.left_ = construct_node((*m).value);
                }
                else if ((*m).name == "right") {
                    inner_node.right_ = construct_node((*m).value);
                }
                else if ((*m).name == "fc") {
                    torch::jit::slot_list_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::ParameterPolicy> > params = (*m).value.named_parameters();
                    for (auto p = params.begin(); p != params.end(); ++p) {
                        if ((*p).name == "weight") {
                            input_dim_ = (*p).value.sizes()[1];
                            inner_node.weight_ = (*p).value.to(torch::kFloat64);
                        }
                        else if ((*p).name == "bias") {
                            inner_node.bias_ = (*p).value.to(torch::kFloat64);
                        }
                        else assert(false);
                    }
                }
                else assert(false);
            }
            node_p = &inner_node;
        }
        else if (module_list.size() == 0) {
            // printf("leaf\n");
            leaf_node_l_.emplace_back();
            CLeafNode& leaf_node = leaf_node_l_.back();
            torch::jit::slot_list_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::ParameterPolicy> > params = node.named_parameters();
            for (auto p = params.begin(); p != params.end(); ++p) {
                if ((*p).name == "param") {
                    output_dim_ = (*p).value.sizes()[0];
                    leaf_node.distribution_ = (*p).value.to(torch::kFloat64);
                }
                else assert(false);
            }
            node_p = &leaf_node;
        }
        else assert(false);
        return node_p;
    };

    torch::jit::script::Module module = torch::jit::load(model_path.c_str());
    torch::jit::module_list module_list = module.children();
    const torch::jit::script::Module& root_module = *module_list.begin();
    assert(module_list.size() == 1);
    assert(root_module.children().size() == 3);

    root_ = construct_node(root_module);
}

void CSDT::InitVariable() {
    // initialize convex variables for convex solver
    GRBModel& model = solver_.ConvexSolver();
    for (int i = 0; i < input_dim_; ++i) {
        convex_input_var_v_.emplace_back(model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS));
    }
    model.update();

    // initialize sat variables for sat solver
    sat_output_var_vl_.resize(output_dim_);

    function<void(CNode*, vector<GRBTempConstr>&)> construct_leaf_constraint;
    construct_leaf_constraint = [&] (CNode* node, vector<GRBTempConstr>& constr_v) {
        if (node->IsInner()) {
            torch::Tensor weight = ((CInnerNode*)node)->weight_.view({input_dim_});
            double bias = ((CInnerNode*)node)->bias_.item<double>();
            GRBLinExpr expr = solver_.ComputeConvexExpr(convex_input_var_v_, weight) + bias;
            // cout << expr << endl;

            constr_v.emplace_back(expr <= 0);
            construct_leaf_constraint(node->left_, constr_v);
            constr_v.back() = expr >= 0;
            construct_leaf_constraint(node->right_, constr_v);
            constr_v.pop_back();
        }
        else {
            torch::Tensor distribution = ((CLeafNode*)node)->distribution_.view({output_dim_});
            int id = torch::argmax(distribution).item<int>();
            assert(id < (int)sat_output_var_vl_.size());
            // cout << distribution << endl;
            // cout << id << endl;

            ostringstream string_conversion;
            string_conversion << id << "_" << sat_output_var_vl_[id].size() << "_" << (void*)this;

            CSolver::SatVar* var = solver_.AddSatVar(string_conversion.str().c_str());
            var->lin_expr_ = constr_v;
            sat_output_var_vl_[id].emplace_back(var);
        }
    };
    vector<GRBTempConstr> constr_v;
    construct_leaf_constraint(root_, constr_v);
}
NNV_NAMESPACING_END
