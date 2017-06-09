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
      : ParameterOptimizer(parameter, lr), epsilon_(epsilon), decay_(decay) {
    size_t size = p->size();
    if (accum_gradient_) delete accum_gradient_;
    accum_gradient_ = new Tensor(size);
  }
  ~AdagradOptimizer() {
    if (accum_gradient_) delete accum_gradient_;
  }
  void Update(const Tensor *gradient);

private:
  Tensor *accum_gradient_;
  double epsilon_;
  double decay_;
};

}  // namespace optimizer
}  // namespace paddle
