/**CFile****************************************************************
  FileName    [global.hpp]
  SystemName  [NNV.]
  PackageName [Global declarations.]
  Synopsis    []
  Author      [Kevin Chiaming Chang.]
  Affiliation [USC]
  Date        [Ver. 1.0. Started - May 19, 2022.]
***********************************************************************/

#ifndef NNV__db__nn_hpp
#define NNV__db__nn_hpp

#include "dbGlobal.hpp"
#include "db/image.hpp"
#include "solver/solver.hpp"
#include "util/util.hpp"

// #define DEBUG
// #define VERBOSE
NNV_NAMESPACING_START

class CNN {
public:
    struct Parameter {
        double  kObjectiveRatio = 1;
    } parameter_;

    CNN(CSolver& solver) : solver_(solver), input_(torch::empty({0})), jump_assertion_b_(true) {};

    int             Run                 (int argc, char **argv);

    // 1. load weight, bias, and number of neuron
    void            LoadModel           (const string & model_path);
    void            LoadInput           (const string & input_path);

    // 2. adjust weight, bias and number of neuron for property
    void            AdjustWeightBias    (int target);
    void            AdjustWeightBias    (const torch::Tensor& eq);

    // 3. bound computation
    void            SplitBound          (const torch::Tensor& bound, int num_split, bool optimize = false);
    void            SetBound            (const torch::Tensor& bound, bool optimize = false);

    // 4. bound recomputation due to assertion
    void            SetBoundAssertion   ();

    void            CheckSatStatus      ();
    void            AssertSatStatus     (int& num_assertion);

    void            ComputeCertificate  (vector<z3::expr>& conflict);

    torch::Tensor   Forward             () { assert(input_.numel() != 0); return Forward(input_); };
    torch::Tensor   Forward             (const CImage& input) { return Forward(input.Data()); };
    torch::Tensor   Forward             (const torch::Tensor& input);
    int             Target              () { return torch::argmax(Forward()).item<int>(); }
    int             Target              (const torch::Tensor& input) { return torch::argmax(Forward(input)).item<int>(); }

    const torch::Tensor&        Input                   () { return input_; }

    vector<int>&                NumNeuron               () { return num_neuron_v_; }
    vector<torch::Tensor>&      BoundBeforeActivation   () { return bound_before_activation_v_; }

    vector<torch::Tensor>&      NonlinearNeuron         () { return nonlinear_neuron_v_; }

    vector<vector<CSolver::SatVar*>>&   SatNetworkVariable  () { return sat_network_var_v_; }
    vector<torch::Tensor>&              SatNetworkStatus    () { return sat_network_status_v_; }
    bool&                               JumpAssertion       () { return jump_assertion_b_; }

    vector<vector<GRBVar> >&    ConvexNetworkVariable   () { return convex_network_var_v_; }
    vector<vector<GRBVar> >&    ConvexReluVariable      () { return convex_relu_var_v_; }
    vector<vector<GRBVar> >&    ConvexSlackVariable     () { return convex_slack_var_v_; }

    void            AddConvexBoundConstraint            (GRBTempConstr constraint);
    void            RemoveConvexBoundConstraint         ();
    void            AddConvexAssignmentConstraint       (int layer, int neuron_id, GRBTempConstr constraint);
    void            RemoveConvexAssignmentConstraint    ();

    double          ObjectiveRatio                      (int layer);

    string          Str             (bool matrix = true) const;
    string          SatStr          () const;
    string          ConvexStr       () const;
private:
    // for LoadModel
    void            LoadTorchModel      (const string & model_path);
    void            LoadOnnxModel       (const string & model_path);

    // for InitVariable
    void            InitVariable        ();

    // for SetBound
    void            OptimizeBound   ();
    void            ComputeBound    (const string& method = "fixed");
    void            ForwardLinear   (int layer, torch::Tensor& bound, const string& method = "fixed");
    void            ForwardRelu     (int layer, torch::Tensor& bound, const string& method = "fixed");
    void            BackwardLinear  (int layer, torch::Tensor& lower_eq, torch::Tensor& upper_eq);
    void            BackwardRelu    (int current_layer, int layer, torch::Tensor& lower_eq, torch::Tensor& upper_eq, const string& method = "fixed");
    torch::Tensor   ComputeBound    (const torch::Tensor& equation, const torch::Tensor& interval);

    void            ComputeNonlinearNeuron          ();
    void            UpdateConvexSolver              ();

    // for ResetBound
    void            ComputeNetworkStatus            ();
    void            UpdateConvexSolverAssertion     ();

    bool            CheckSize                       ();

    int             Test                            ();


    CSolver&                    solver_;
    torch::Tensor               input_;

    vector<torch::Tensor>       weight_v_;
    vector<torch::Tensor>       bias_v_;
    vector<int>                 num_neuron_v_;

    vector<torch::Tensor>       bound_before_activation_v_;
    vector<torch::Tensor>       lower_equation_before_activation_v_;
    vector<torch::Tensor>       upper_equation_before_activation_v_;
    vector<torch::Tensor>       lower_equation_after_activation_v_;
    vector<torch::Tensor>       upper_equation_after_activation_v_;

    vector<torch::Tensor>       lower_ratio_v_;
    vector<torch::Tensor>       upper_ratio_v_;
    vector<vector<torch::Tensor>>   alpha_v_;

    vector<torch::Tensor>       neural_status_v_;
    vector<torch::Tensor>       nonlinear_neuron_v_;

    // for sat solver
    vector<vector<CSolver::SatVar*>>    sat_network_var_v_;
    vector<torch::Tensor>               sat_network_status_v_;
    bool                                jump_assertion_b_;

    // for convex solver
    vector<vector<GRBVar> >                 convex_network_var_v_;
    vector<vector<GRBVar> >                 convex_relu_var_v_;
    vector<vector<GRBVar> >                 convex_slack_var_v_;
    vector<GRBConstr>                       convex_constraint_v_;
    vector<map<int, vector<GRBConstr> > >   convex_assertion_constraint_vm_;
};

NNV_NAMESPACING_END
#endif
