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
#include "env/cart_pole.hpp"
#include "env/mountain_car.hpp"
#include "solver/solver.hpp"
#include "verifier/verifier.hpp"

USING_NNV_NAMESPACING

Usage usage;

int main(int argc, char **argv){
    ArgumentParser program("NNV", "1.0");

    program.AddArgument("sdt")
        .Help("taken the pytorch file from the path as the soft decision tree")
        .Nargs(1);
    program.AddArgument("--env")
        .Help("taken the string value as the environment")
        .Nargs(1)
        .DefaultValue((string)"mountain_car");
    program.AddArgument("--horizon")
        .Help("taken the integer value as the finite horizon")
        .Nargs(1)
        .DefaultValue(200)
        .Action([&](const auto & s) { return stoi(s); });
    
    if (program.ParseArgs(argc, argv)) return 0;
    printf("%s\n", program.str().c_str());
    assert(program.Get<string>("--env") == "mountain_car" or program.Get<string>("--env") == "cart_pole");

    // set horizon
    int horizon = program.Get<int>("--horizon");

    // construct solver
    CSolver solver;
    solver.SetCertificate(CSolver::Certificate::IIS);

    // construct neural network
    vector<CNN> nn_v;
    vector<CSDT> sdt_v;
    for (int i = 0; i < horizon - 1; ++i) {
        sdt_v.emplace_back(solver);
        CSDT& sdt = sdt_v[i];
        sdt.LoadModel(program.Get<string>("sdt"));
    }


    // construct mountain car dynamics
    CEnv* env_p;
    if (program.Get<string>("--env") == "mountain_car") {
        env_p = new CMountainCar(solver);
    }
    else if (program.Get<string>("--env") == "cart_pole") {
        env_p = new CCartPole(solver);
    }
    else assert(false);
    env_p->Init(horizon);

    CVerifier verifier(solver, nn_v, sdt_v, env_p);
    verifier.Verify_SDT_Monolith();


    return 0;
}
