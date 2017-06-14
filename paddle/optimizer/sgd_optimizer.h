#pragma once

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

class SGDOptimizer : public ParameterOptimizer {
public:
  SGDOptimizer(Tensor* parameter, LrPolicy* lr, double m, double d, bool n)
      : ParameterOptimizer(parameter, lr),
        momentums_(nullptr),
        momentum_(m),
        decay_(d),
        nesterov_(n) {
    if (momentum_ != 0.0) {
      size_t size = parameter->size();
      // TODO: fix it with align aware allocator bind to Tensor
      momentums_ = new Tensor(size);
    }
  }
  virtual ~SGDOptimizer() {
    if (momentums_) delete momentums_;
  }
  void Update(const Tensor* gradient);
  const char* SerializeState(int* state_len);
  void DeserializeState(const std::string& state);

private:
  Tensor* momentums_;
  double momentum_;
  double decay_;
  bool nesterov_;
};

}  // namespace optimizer
}  // namespace paddle
