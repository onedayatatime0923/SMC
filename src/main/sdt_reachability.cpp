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
#include "env/car_racing.hpp"
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
    program.AddArgument("output")
        .Help("taken the path as the output file")
        .Nargs(1);
    program.AddArgument("--env")
        .Help("taken the string value as the environment")
        .Nargs(1)
        .DefaultValue((string)"mountain_car");
    program.AddArgument("--step")
        .Help("taken the integer value as the finite step")
        .Nargs(1)
        .DefaultValue(1)
        .Action([&](const auto & s) { return stoi(s); });
    program.AddArgument("--horizon")
        .Help("taken the integer value as the finite horizon")
        .Nargs(1)
        .DefaultValue(200)
        .Action([&](const auto & s) { return stoi(s); });

    if (program.ParseArgs(argc, argv)) return 0;
    printf("%s\n", program.str().c_str());

    // set horizon
    int step = program.Get<int>("--step");
    int horizon = program.Get<int>("--horizon");
    string env = program.Get<string>("--env");

    // construct solver
    CSolver solver;
    solver.SetCertificate(CSolver::Certificate::IIS);

    // construct mountain car dynamics
    CEnv* env_p;
    if (env == "mountain_car") {
        env_p = new CMountainCar(solver);
    }
    else if (env == "cart_pole") {
        env_p = new CCartPole(solver);
    }
    else if (env == "car_racing") {
        env_p = new CCarRacing(solver);
    }
    else assert(false);
    env_p->Init(step + 1);

    // construct neural network
    vector<CNN> nn_v;
    vector<CSDT> sdt_v;
    for (int i = 0; i < env_p->ActionHorizon(); ++i) {
        sdt_v.emplace_back(solver);
        CSDT& sdt = sdt_v[i];
        sdt.LoadModel(program.Get<string>("sdt"));
    }

    // printf("%s\n", sdt_v[0].Str().c_str());
    // getchar();


    usage.ClearTime();
    // construct verifier
    CVerifier verifier(solver, nn_v, sdt_v, env_p);
    // vector<torch::Tensor> data_v = verifier.Verify_SDT_Composition(horizon);
    vector<torch::Tensor> data_v = verifier.Verify_SDT_Composition_Car_racing(horizon);
    // usage.ShowTime();

    WriteFile(data_v, program.Get<string>("output"));


    return 0;
}
