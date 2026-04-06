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
#include "db/image.hpp"
#include "db/nn.hpp"
#include "db/sdt.hpp"
#include "solver/solver.hpp"
#include "verifier/verifier.hpp"

USING_NNV_NAMESPACING

Usage usage;


int main(int argc, char **argv){
    ArgumentParser program("Verification", "1.0");

    program.AddArgument("input")
        .Help("taken the vnnlib file from the path as the input file.")
        .Nargs(1);

    if (program.ParseArgs(argc, argv)) return 0;
    printf("%s\n", program.str().c_str());

    CSolver solver;

    vector<CNN> nn_v;

    vector<CSDT> sdt_v;

    CVerifier verifier(solver, nn_v, sdt_v);
    verifier.Verify(program.Get<string>("input"));

    // printf("%s", solver.Str().c_str());
    // printf("%s", nn_v[0].Str().c_str());
    // printf("%s", nn_v[0].SatStr().c_str());
    // printf("%s", nn_v[0].ConvexStr().c_str());
    // getchar();


    // // solver.WriteConvexSolver();
    // // printf("%s", mountain_car.Str().c_str());
    printf("%s", verifier.Str().c_str());
    return 0;
};
