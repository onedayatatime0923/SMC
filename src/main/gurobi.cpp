/**CFile****************************************************************
  FileName    [main.hpp]
  SystemName  [NNV.]
  PackageName [Global declarations.]
  Synopsis    [Global declarations.]
  Author      [Kevin Chiaming Chang.]
  Affiliation [USC]
  Date        [Ver. 1.0. Started - May 19, 2022.]
***********************************************************************/

#include "misc/global.hpp"
#include "db/nn.hpp"
#include "db/sdt.hpp"
#include "env/mountain_car.hpp"
#include "solver/solver.hpp"
#include "verifier/verifier.hpp"

USING_NNV_NAMESPACING

Usage usage;

int main(int argc, char **argv){
    // set horizon
    string file = "test";
    GRBEnv convex_env;
    convex_env.set(GRB_IntParam_OutputFlag, 0);
    convex_env.set(GRB_IntParam_Method, 4);
    convex_env.start();

    GRBModel convex_solver(convex_env);
    convex_solver.set(GRB_IntParam_NonConvex, 2);

    GRBVar x = convex_solver.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "x");
    GRBVar y = convex_solver.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "y");
    GRBVar z = convex_solver.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "z");

    GRBQConstr c = convex_solver.addQConstr(y == x * x);
    convex_solver.addQConstr(z == x * y);
    convex_solver.addConstr(x >= 2);
    convex_solver.addConstr(z - x <= 0);

    try {
        convex_solver.update();
        convex_solver.optimize();
        int status = convex_solver.get(GRB_IntAttr_Status);
        cout << status << endl;
        convex_solver.write(file + ".lp");
        if (convex_solver.get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
            convex_solver.computeIIS();
            convex_solver.write(file + ".ilp");
        }
        cout << c.get(GRB_IntAttr_IISQConstr) << endl;
    }
    catch (GRBException e) {
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
    }
    return 0;
}
