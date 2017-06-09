#pragma once

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

class AdadeltaOptimizer : public ParameterOptimizer {
public:
  AdadeltaOptimizer(
      Tensor *parameter, LrPolicy *lr, double rho, double epsilon, double decay)
      : ParameterOptimizer(parameter, lr),
        rho_(rho),
        epsilon_(epsilon),
        decay_(decay) {
    size_t size = p->size();
    if (accum_gradient_) delete accum_gradient_;
    accum_gradient_ = new Tensor(size);
    if (accum_delta_) delete accum_delta_;
    accum_delta_ = new Tensor(size);
    if (update_delta_) delete update_delta_;
    update_delta_ = new Tensor(size);
  }
  ~AdadeltaOptimizer() {
    if (accum_gradient_) delete accum_gradient_;
    if (accum_delta_) delete accum_delta_;
    if (update_delta_) delete update_delta_;
  }
  void Update(const Tensor *gradient);
  const char *SerializeState(int *state_len);
  void DeSerializeState(const std::string &state);

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
