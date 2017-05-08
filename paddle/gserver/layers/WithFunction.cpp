#include "WithFunction.h"
#include "paddle/function/Function.h"
namespace paddle {

void WithFunction::appendFunction(std::vector<function::KernelType> *functions,
                                  const std::string &name,
                                  const FuncConfig &config,
                                  bool useGPU) {
  std::shared_ptr<FunctionBase> func;
  if (useGPU) {
    func.reset(FunctionBase::funcRegistrar_.createByType(name + "-GPU"));
  } else {
    func.reset(FunctionBase::funcRegistrar_.createByType(name + "-CPU"));
  }
  func->init(config);
  functions->push_back(
      [func](const BufferArgs &inputs, const BufferArgs &outputs) {
        func->calc(inputs, outputs);
        //! TODO(yuyang18): Make FunctionBase::calc return paddle::Error.
        return paddle::Error();
      });
}

}  // namespace paddle
