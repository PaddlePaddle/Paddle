#pragma once

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

class AdagradOptimizer : public ParameterOptimizer {
public:
  AdagradOptimizer(Tensor *parameter,
                   LrPolicy *lr,
                   double epsilon,
                   double decay)
      : ParameterOptimizer(parameter, lr),
        accum_gradient_(new Tensor(parameter->size())),
        epsilon_(epsilon),
        decay_(decay) {}
  ~AdagradOptimizer() {
    if (accum_gradient_) delete accum_gradient_;
  }
  void Update(const Tensor *gradient);
  const char *SerializeState(int *state_len);
  void DeserializeState(const std::string &state);

private:
  Tensor *accum_gradient_;
  double epsilon_;
  double decay_;
};

}  // namespace optimizer
}  // namespace paddle
