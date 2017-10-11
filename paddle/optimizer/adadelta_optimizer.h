#pragma once

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

class AdadeltaOptimizer : public ParameterOptimizer {
public:
  AdadeltaOptimizer(
      Tensor *parameter, LrPolicy *lr, double rho, double epsilon, double decay)
      : ParameterOptimizer(parameter, lr),
        accum_gradient_(new Tensor(parameter->size())),
        accum_delta_(new Tensor(parameter->size())),
        update_delta_(new Tensor(parameter->size())),
        rho_(rho),
        epsilon_(epsilon),
        decay_(decay) {}

  ~AdadeltaOptimizer() {
    if (accum_gradient_) delete accum_gradient_;
    if (accum_delta_) delete accum_delta_;
    if (update_delta_) delete update_delta_;
  }
  void Update(const Tensor *gradient);
  const char *SerializeState(int *state_len);
  void DeserializeState(const std::string &state);

private:
  Tensor *accum_gradient_;
  Tensor *accum_delta_;
  Tensor *update_delta_;
  double rho_;
  double epsilon_;
  double decay_;
};

}  // namespace optimizer
}  // namespace paddle
