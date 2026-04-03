/**CFile****************************************************************
  FileName    [env.hpp]
  SystemName  [NNV.]
  PackageName [Global declarations.]
  Synopsis    []
  Author      [Kevin Chiaming Chang.]
  Affiliation [USC]
  Date        [Ver. 1.0. Started - May 19, 2022.]
***********************************************************************/

#ifndef NNV__env__env_hpp
#define NNV__env__env_hpp

#include "envGlobal.hpp"
#include "solver/solver.hpp"
#include "db/nn.hpp"
#include "db/sdt.hpp"

// #define DEBUG
NNV_NAMESPACING_START

class CEnv {
public:
    CEnv(CSolver& solver) : solver_(solver), wave_approx_param_(16) {};

    struct Parameter {
        int     wave_approx_limit = 16;
    } parameter_;

    virtual torch::Tensor   Bound                           () = 0;
    virtual torch::Tensor   InitialConstraint               () = 0;

    virtual void            Init                            (int horizon = 200) = 0;
    virtual void            Reset                           () = 0;

    virtual void            SetInitialConstraint            () = 0;
    virtual void            ClearInitialConstraint          () = 0;

    virtual void            SetSafeConstraint               () = 0;
    virtual void            ClearSafeConstraint             () = 0;

    virtual void            SetRelationSDT                  (vector<CSDT>& soft_decision_tree_v, bool initialize = false) = 0;
    virtual void            SetRelationNN                   (vector<CNN>& neural_network_v) = 0;


    virtual int             CheckNextState                  (const torch::Tensor& state, int action, const torch::Tensor& next_state) = 0;
    virtual torch::Tensor   Forward                         (const torch::Tensor& state, const::torch::Tensor& action) = 0;
    virtual torch::Tensor   Forward                         (const torch::Tensor& state, int action) = 0;

    int                                 StateHorizon            () { return time_state_var_vv_.size(); }
    vector<vector<GRBVar>> &            TimeStateVarVV          () { return time_state_var_vv_; }
    virtual int                         ActionHorizon           () { return time_action_var_vv_.size(); }
    virtual int                         ActionNum               () { return time_action_var_vv_[0].size(); }
    int&                                WaveApproxParam         () { return wave_approx_param_; }

    virtual torch::Tensor               State           (int step) = 0;
    virtual vector<GRBVar>              StateVar        (int step) = 0;
    virtual int                         Action          (int step) = 0;
    virtual torch::Tensor               Trajectory      () = 0;

    virtual string                      Str             () const = 0;
    virtual string                      TimeStateStr    () const = 0;
    virtual string                      ActionStateStr  () const = 0;

protected:

    CSolver&                            solver_;

    vector<vector<GRBVar>>              time_state_var_vv_;
    vector<vector<CSolver::SatVar*>>    time_action_var_vv_;

    int                                 wave_approx_param_;
};

NNV_NAMESPACING_END
#endif
