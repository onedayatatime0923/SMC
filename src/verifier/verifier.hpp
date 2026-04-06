/**CFile****************************************************************
  FileName    [global.hpp]
  SystemName  [NNV.]
  PackageName [Global declarations.]
  Synopsis    []
  Author      [Kevin Chiaming Chang.]
  Affiliation [USC]
  Date        [Ver. 1.0. Started - May 19, 2022.]
***********************************************************************/

#ifndef NNV__verifier__verifier_hpp
#define NNV__verifier__verifier_hpp

#include "verifierGlobal.hpp"
#include "db/image.hpp"
#include "db/nn.hpp"
#include "db/sdt.hpp"
#include "env/mountain_car.hpp"
#include "env/cart_pole.hpp"
#include "solver/solver.hpp"

// #define DEBUG
// #define VERBOSE
NNV_NAMESPACING_START

class CVerifier {
public:
    CVerifier(CSolver& solver, vector<CNN>& neural_network_v, vector<CSDT>& soft_decision_tree_v, CEnv* env_p = nullptr) : solver_(solver), neural_network_v_(neural_network_v), soft_decision_tree_v_(soft_decision_tree_v), env_p_(env_p), full_search_b_(false), iteration_(0){};

    struct Parameter {
        bool    timeout_b   = false;
        double  timeout_d   = 60000;
    } parameter_;
    
    // int             Run                     (int argc, char **argv);

    // set safety property

    int                     Verify                              (const string & model_path);

    string          Str                     () const;
    torch::Tensor&  CounterExample          () { return counter_example_; }

    void            WriteOutput             ();


private:

    int         Solve_Smc                           ();
    int         Solve_Sat                           ();
    int         Solve_Convex                        (vector<z3::expr>& conflict);

    void        ResolveConvexByDualCertificate      ();
    void        ComputeCertificate                  (vector<z3::expr>& conflict);


    CSolver&            solver_;
    vector<CNN>&        neural_network_v_;
    vector<CSDT>&       soft_decision_tree_v_;
    CEnv*               env_p_;

    function<int()>     solution_checker_f_;
    bool                full_search_b_;
    int                 iteration_;
    torch::Tensor       counter_example_;


};

NNV_NAMESPACING_END
#endif
