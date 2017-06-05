#ifndef PADDLE_ADADELTA_OPTIMIZER_H_
#define PADDLE_ADADELTA_OPTIMIZER_H_

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

class AdadeltaOptimizer : public ParameterOptimizer {
public:
  using ParameterOptimizer::parameter_;
  using ParameterOptimizer::num_sample_passed;
  using ParameterOptimizer::lr_policy;

  AdadeltaOptimizer(double rho, double epsilon, double decay, BaseLr *lr)
      : ParameterOptimizer(lr), rho(rho), epsilon(epsilon), decay(decay) {}
  ~AdadeltaOptimizer() {
    if (accum_gradient) delete accum_gradient;
    if (accum_delta) delete accum_delta;
    if (update_delta) delete update_delta;
  }
  void update(const Tensor &gradient);
  void set_weight(Tensor *p);
  real *get_weight() const;

private:
  Tensor *accum_gradient;
  Tensor *accum_delta;
  Tensor *update_delta;

  double rho;
  double epsilon;
  double decay;
};

}  // namespace optimizer
}  // namespace paddle

#endif
