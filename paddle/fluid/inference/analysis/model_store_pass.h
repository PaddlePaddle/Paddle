/*
 * This file defines ModelStorePass, which store the runtime DFG to a Paddle
 * model in the disk, and that model can be reloaded for prediction.
 */

#include "paddle/fluid/inference/analysis/pass.h"

namespace paddle {
namespace inference {
namespace analysis {

class ModelStorePass : public DataFlowGraphPass {
 public:
  bool Initialize(Argument* argument) override {
    if (!argument) {
      LOG(ERROR) << "invalid argument";
      return false;
    }
    argument_ = argument;
    return true;
  }


  void Run(DataFlowGraph *x) override;

  std::string repr() const override { return "DFG-store-pass"; }
  std::string description() const override {
    return R"DD(This file defines ModelStorePass, which store the runtime DFG to a Paddle
    model in the disk, and that model can be reloaded for prediction again.)DD";
  }

 private:
  Argument* argument_{nullptr};
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
