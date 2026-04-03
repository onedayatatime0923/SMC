/**CFile****************************************************************
  FileName    [global.hpp]
  SystemName  [NNV.]
  PackageName [Global declarations.]
  Synopsis    []
  Author      [Kevin Chiaming Chang.]
  Affiliation [USC]
  Date        [Ver. 1.0. Started - May 19, 2022.]
***********************************************************************/

#ifndef NNV__env__example_hpp
#define NNV__env__example_hpp

#include "envGlobal.hpp"
#include "env.hpp"
#include "solver/solver.hpp"

// #define DEBUG
NNV_NAMESPACING_START

class CExample: public CEnv{
public:
    struct Parameter {
        pair<double, double> x0_initial_constraint = make_pair(0.8, 0.9);
        pair<double, double> x1_initial_constraint = make_pair(0.5, 0.6);
        // pair<double, double> x0_initial_constraint     = make_pair(-0.2926, -0.2349);
        // pair<double, double> x1_initial_constraint     = make_pair(-0.2042, -0.1566);
        pair<double, double> x0_safe_constraint     = make_pair(0, 0.2);
        pair<double, double> x1_safe_constraint     = make_pair(0.05, 0.3);
    } parameter_;

    CExample(CSolver& solver) : CEnv(solver), simulation_per_control_(40) 
        {   solver_.ConvexSolver().set(GRB_IntParam_NonConvex, 2);
            solver_.ConvexSolver().set(GRB_DoubleParam_FeasibilityTol, 1e-4);
            solver_.ConvexSolver().set(GRB_DoubleParam_IntFeasTol, 1e-4); }
    
    torch::Tensor   Bound                           ();
    torch::Tensor   InitialConstraint               ();

    void            Init                            (int horizon = 35);
    void            Reset                           ();

    void            SetInitialConstraint            ();
    void            ClearInitialConstraint          ();

    void            SetSafeConstraint               ();
    void            ClearSafeConstraint             ();

    void            SetRelationSDT                  (vector<CSDT>& soft_decision_tree_v, bool initialize = false);
    void            SetRelationNN                   (vector<CNN>& neural_network_v);

    int             CheckNextState                  (const torch::Tensor& state, int action, const torch::Tensor& next_state);
    torch::Tensor   Forward                         (const torch::Tensor& state, const::torch::Tensor& action);
    torch::Tensor   Forward                         (const torch::Tensor& state, int action);


    torch::Tensor               State           (int step);
    vector<GRBVar>              StateVar        (int step);
    int                         Action          (int step);
    int                         ActionHorizon   () { return time_state_var_vv_.size() - 1; }
    int                         ActionNum       () { return 1; }
    torch::Tensor               Trajectory      ();

    string                      Str             () const;
    string                      TimeStateStr    () const;
    string                      ActionStateStr  () const;
private:

    void        SetHorizon                      (int horizon);

    void        SetStateRelationConstraint      ();
    void        ResetStateRelationConstraint    ();

    void        SetBoundaryConstraint           ();
    void        ResetBoundaryConstraint         ();

    int                                 simulation_per_control_;

};

NNV_NAMESPACING_END
#endif
