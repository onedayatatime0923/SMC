/**CFile****************************************************************
  FileName    [global.hpp]
  SystemName  [NNV.]
  PackageName [Global declarations.]
  Synopsis    []
  Author      [Kevin Chiaming Chang.]
  Affiliation [USC]
  Date        [Ver. 1.0. Started - May 19, 2022.]
***********************************************************************/

#ifndef NNV__solver__solver_hpp
#define NNV__solver__solver_hpp

#include "solverGlobal.hpp"
#include "util/util.hpp"

NNV_NAMESPACING_START

class CMountainCar;
class CNN;

class CSolver {
public:
    enum class Certificate {
        DUAL,
        IIS,
        NONE,
    };

    struct Parameter {
        int     kAssertion      = 2;
        double  kEPS            = 0.001;
    } parameter_;

    struct SatVar {
        SatVar(z3::expr var) : var_(var) {};
        z3::expr                var_;
        list<SatVar>::iterator  it_;
        vector<GRBTempConstr>   lin_expr_;
        vector<GRBTempConstr>   quad_expr_;
        vector<GRBConstr>       lin_constr_;
        vector<GRBQConstr>      quad_constr_;
    };

    CSolver() : certificate_(Certificate::NONE), bound_computing_b_(false), sat_solver_(sat_contex_), sat_model_(sat_contex_), convex_env_(true), convex_solver_(InitConvexSolver()) {};
    
    enum Certificate    Certificate         ()                              { return certificate_; }
    void                SetCertificate      (enum Certificate certificate)  { certificate_ = certificate; }

    bool                BoundComputing      ()                              { return bound_computing_b_; }
    void                SetBoundComputing   (bool bound_computing)          { bound_computing_b_ = bound_computing; }

    // for sat solver
    void            ResetSat            ();
    SatVar*         AddSatVar           (const string& name);
    SatVar*         RemoveSatVar        (SatVar* var)                   { return &(*sat_var_l_.erase(var->it_)); }
    void            SetSatOrConstraint      (SatVar* var, bool& f, z3::expr& constr);
    void            SetSatAndConstraint     (SatVar* var, bool& f, z3::expr& constr);
    void            AddSatFixedConstraint   (const z3::expr& constraint);
    void            AddSatConstraint        (const z3::expr& constraint)    { sat_solver_.add(constraint); }
    bool            CheckSat                ();
    int             EvalSatVar              (SatVar* var) const;

    // for convex solver
    GRBLinExpr      ComputeConvexExpr       (const vector<GRBVar>& var, const torch::Tensor& weight);
    void            SetConvexConstrant      ();
    void            RemoveConvexConstrant   ();
    void            SetConvexObjective      (GRBLinExpr expr);
    void            SetConvexObjective      ();
    void            ComputeConvexCertificate(vector<z3::expr>& conflict);


    void            WriteConvexSolver       (const string& file = "model");

    // getter
    z3::context&    SatContex               ()                              { return sat_contex_; }
    int             SatVarLength            ()                              { return sat_var_l_.size(); }
    static z3::expr SatExpr                 (SatVar* var)                   { return var->var_; }

    GRBModel&       ConvexSolver            () { return convex_solver_; }
    GRBLinExpr&     ConvexObjective         () { return convex_objective_; }

    string          Str                     (const string& file = "model");
    string          SatVarStr               (SatVar* v) const;
    string          ConvexVarStr            (const GRBVar& v) const;

private:

    GRBEnv&     InitConvexSolver        ();

    enum Certificate                                certificate_;
    bool                                            bound_computing_b_;

    // for sat
    z3::context                                     sat_contex_;
    z3::solver                                      sat_solver_;
    list<SatVar>                                    sat_var_l_;
    list<z3::expr>                                  sat_fixed_constraint_l_;
    bool                                            sat_model_available_b_;
    z3::model                                       sat_model_;

    // for convex
    GRBEnv                                          convex_env_;
    GRBModel                                        convex_solver_;
    GRBLinExpr                                      convex_objective_;
};

NNV_NAMESPACING_END
#endif
