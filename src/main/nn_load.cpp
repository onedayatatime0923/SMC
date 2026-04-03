#include "misc/global.hpp"

#include <memory>

USING_NNV_NAMESPACING

Usage usage;

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
    // cout << module.parameters(true) << endl;
    // Torch::Jit::NAMED_PARAMETER_LIST params = module.named_parameters(true [>recurse<]);
    torch::jit::slot_list_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::ParameterPolicy> > params = module.named_parameters(true);

      for (auto p = params.begin(); p != params.end(); ++p) {
        std::cout << (*p).name << ": " << (*p).value << "\n";
      }
    // for (auto it = module.parameters(); ; ++it) {
    //     cout << it << endl;
    // }
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    std::cerr << e.what() << endl;
    return -1;
  }

  std::cout << "ok\n";
}
