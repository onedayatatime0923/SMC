/**CFile****************************************************************
  FileName    [global.hpp]
  SystemName  [NNV.]
  PackageName [Global declarations.]
  Synopsis    []
  Author      [Kevin Chiaming Chang.]
  Affiliation [USC]
  Date        [Ver. 1.0. Started - May 19, 2022.]
***********************************************************************/

#ifndef NNV__env__car_racing_hpp
#define NNV__env__car_racing_hpp

#include "envGlobal.hpp"
#include "env.hpp"
#include "solver/solver.hpp"

// #define DEBUG
NNV_NAMESPACING_START

class CCarRacing: public CEnv{
public:
    struct Parameter {
        pair<double, double> x_bound = make_pair(-1.0, 1.0);
        pair<double, double> y_bound = make_pair(-1.0, 1.0);
        pair<double, double> v_bound = make_pair(-1.0, 1.0);
        pair<double, double> x_initial_constraint = make_pair(-0.02, 0.02);
        pair<double, double> y_initial_constraint = make_pair(0.0, 0.0);
        pair<double, double> v_initial_constraint = make_pair(0.0, 0.0);
        double safe_constraint          = 0.5;
        double perception_disturbance   = 0.009;
    } parameter_;

    CCarRacing(CSolver& solver) : CEnv(solver) {};
    
    torch::Tensor   Bound                           ();
    torch::Tensor   InitialConstraint               ();

    void            Init                            (int horizon = 200);
    void            Reset                           ();

    void            SetInitialConstraint            () {};
    void            ClearInitialConstraint          () {};

    void            SetSafeConstraint               () {};
    void            ClearSafeConstraint             () {};

    void            SetRelationSDT                  (vector<CSDT>& soft_decision_tree_v, bool initialize = false);
    void            SetRelationNN                   (vector<CNN>& neural_network_v);

    int             CheckNextState                  (const torch::Tensor& state, int action, const torch::Tensor& next_state);
    torch::Tensor   Forward                         (const torch::Tensor& state, const::torch::Tensor& action);
    torch::Tensor   Forward                         (const torch::Tensor& state, int action);

    int                         ActionHorizon           () { return 1; }

    torch::Tensor               State           (int step);
    vector<GRBVar>              StateVar        (int step) { return {time_state_var_vv_[step][0], time_state_var_vv_[step][1], time_state_var_vv_[step][2]}; }
    int                         Action          (int step);
    torch::Tensor               Trajectory      ();

    string                      Str             () const;
    string                      TimeStateStr    () const;
    string                      ActionStateStr  () const;
private:

    void        SetHorizon                      (int horizon);

    void        SetCosineConstraintForSegment   (const double& left_end, const double& right_end, bool taylor_is_over_approx, GRBVar& position, GRBVar& position_cos, CSolver::SatVar* var);
    GRBLinExpr  CosineTwoPointEquation          (GRBVar& var, const double& left_end, const double& right_end);
    GRBLinExpr  CosineTaylorApproximation       (GRBVar& var, const double& left_end, const double& right_end);

    void        SetStateRelationConstraint      ();
    void        ResetStateRelationConstraint    ();




    vector<vector<CSolver::SatVar*>>    time_boundary_var_vv_;

    list<CSolver::SatVar*>              wave_constraint_l_;
};

NNV_NAMESPACING_END
#endif
