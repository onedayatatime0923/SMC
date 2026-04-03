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
#include "db/mountain_car.hpp"
#include "db/nn.hpp"
#include "db/sdt.hpp"
#include "solver/solver.hpp"
#include "verifier/verifier.hpp"

USING_NNV_NAMESPACING

Usage usage;

int main(int argc, char **argv){
    ArgumentParser program("NNV", "1.0");
    program.AddArgument("network")
        .Help("taken the onnx file from the path as the neural network.")
        .Nargs(1);

    if (program.ParseArgs(argc, argv)) return 0;
    printf("network: %s\n", program.Get<string>("network").c_str());


    CSolver solver;
    CSDT sdt(solver);
    sdt.LoadModel(program.Get<string>("network"));
    printf("%s\n", sdt.Str().c_str());
    printf("%s\n", solver.Str().c_str());

    return 0;
}
