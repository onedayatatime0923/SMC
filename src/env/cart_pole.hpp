/**CFile****************************************************************
  FileName    [global.hpp]
  SystemName  [NNV.]
  PackageName [Global declarations.]
  Synopsis    []
  Author      [Kevin Chiaming Chang.]
  Affiliation [USC]
  Date        [Ver. 1.0. Started - May 19, 2022.]
***********************************************************************/

#ifndef NNV__env__cart_pole_hpp
#define NNV__env__cart_pole_hpp

#include "envGlobal.hpp"
#include "env.hpp"
#include "solver/solver.hpp"

// #define DEBUG
NNV_NAMESPACING_START

class CCartPole: public CEnv{
public:
    struct Parameter {
        pair<double, double> position_bound = make_pair(-4.8, 4.8);
        pair<double, double> angle_bound = make_pair(-24 * 2 * M_PI / 360, 24 * 2 * M_PI / 360);
        pair<double, double> position_initial_constraint = make_pair(-0.01, 0);
        // pair<double, double> position_initial_constraint = make_pair(-1.1575, -1.1219);
        pair<double, double> angle_initial_constraint = make_pair(-0.01, 0);
        // pair<double, double> angle_initial_constraint = make_pair(0.2, 0.2);
        double safe_constraint      = 12 * 2 * M_PI / 360;
    } parameter_;

    CCartPole(CSolver& solver) : CEnv(solver) { solver_.ConvexSolver().set(GRB_IntParam_NonConvex, 2); }
    
    torch::Tensor   Bound                           ();
    torch::Tensor   InitialConstraint               ();

    void            Init                            (int horizon = 200);
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
    torch::Tensor               Trajectory      ();

    string                      Str             () const;
    string                      TimeStateStr    () const;
    string                      ActionStateStr  () const;
private:

    void        SetHorizon                      (int horizon);
    void        SetWaveConstraint               ();
    void        ClearWaveConstraint             ();


    void        SetCosineConstraintForSegment   (const double& left_end, const double& right_end, bool taylor_is_over_approx, GRBVar& position, GRBVar& position_cos, CSolver::SatVar* var);
    GRBLinExpr  CosineTwoPointEquation          (GRBVar& var, const double& left_end, const double& right_end);
    GRBLinExpr  CosineTaylorApproximation       (GRBVar& var, const double& left_end, const double& right_end);

    void        SetSineConstraintForSegment     (const double& left_end, const double& right_end, bool taylor_is_over_approx, GRBVar& position, GRBVar& position_cos, CSolver::SatVar* var);
    GRBLinExpr  SineTwoPointEquation            (GRBVar& var, const double& left_end, const double& right_end);
    GRBLinExpr  SineTaylorApproximation         (GRBVar& var, const double& left_end, const double& right_end);

    void        SetStateRelationConstraint      ();
    void        ResetStateRelationConstraint    ();

    void        SetBoundaryConstraint           ();
    void        ResetBoundaryConstraint         ();



    GRBGenConstr                        objective_constr_;

    list<CSolver::SatVar*>              wave_constraint_l_;
};

NNV_NAMESPACING_END
#endif
