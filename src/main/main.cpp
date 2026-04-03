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
    ArgumentParser program("NNV", "1.0");
    if (program.ParseArgs(argc, argv)) return 0;
    // printf("qdimacs: %s\n", program.Get<string>("--qdimacs").c_str());
    // printf("aiger : %s\n", program.Get<string>("--aiger").c_str());
    // printf("dimacs : %s\n", program.Get<string>("--dimacs").c_str());
    // printf("qrp : %s\n", program.Get<string>("--qrp").c_str());
    // printf("qqrp : %s\n", program.Get<string>("--qqrp").c_str());

    Command cmd;

    // construct image
    CImage image;
    cmd.AddCommand("image", bind(&CImage::Run, &image, placeholders::_1, placeholders::_2));

    // construct solver
    CSolver solver;

    vector<CNN> nn_v = {CNN(solver, &image)};
    vector<CSDT> sdt_v;
    CEnv* env_p = new CMountainCar(solver);

    cmd.AddCommand("nn", bind(&CNN::Run, &(nn_v[0]), placeholders::_1, placeholders::_2));


    CVerifier verifier(solver, nn_v, sdt_v, env_p);
    // cmd.AddCommand("verify", bind(&CVerifier::Run, &verifier, placeholders::_1, placeholders::_2));


    // verifier.CheckApproximationNN();
    // verifier.CheckApproximationError();

    if (program.Get<bool>("--interactive")) {
        cmd.Interactive();
    }


    return 0;
}
