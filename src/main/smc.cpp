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

    program.AddArgument("network")
        .Help("taken the onnx file from the path as the neural network.")
        .Nargs(1);
    program.AddArgument("property")
        .Help("taken the vnnlib file from the path as the safety property.")
        .Nargs(1);
    program.AddArgument("-c", "--certificate")
        .Help("taken the double value as the perturbation.")
        .Nargs(1)
        .DefaultValue(string("dual"));

    program.AddArgument("-r", "--recompute")
        .Help("synbolically analize interval of neural network in every interation [default = no].")
        .DefaultValue(false)
        .ImplicitValue(true);

    if (program.ParseArgs(argc, argv)) return 0;
    printf("%s\n", program.str().c_str());

    CSolver solver;
    if (program.Get<string>("--certificate") == "dual") {
        solver.SetCertificate(CSolver::Certificate::DUAL);
    }
    else if (program.Get<string>("--certificate") == "iis") {
        solver.SetCertificate(CSolver::Certificate::IIS);
    }
    else if (program.Get<string>("--certificate") == "none") {
        solver.SetCertificate(CSolver::Certificate::NONE);
    }
    else assert(false);
    solver.SetBoundComputing(program.Get<bool>("--recompute"));

    vector<CNN> nn_v = {solver};
    nn_v[0].LoadModel(program.Get<string>("network"));

    vector<CSDT> sdt_v;

    // printf("%s", nn_v[0].Str(true).c_str());
    // getchar();

    CVerifier verifier(solver, nn_v, sdt_v);
    verifier.Verify_Neural_Network(program.Get<string>("property"));

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
