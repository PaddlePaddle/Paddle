#ifndef PADDLE_ADADELTA_OPTIMIZER_H_
#define PADDLE_ADADELTA_OPTIMIZER_H_

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

class AdadeltaOptimizer : public ParameterOptimizer {
public:
  AdadeltaOptimizer(double rho, double epsilon, double decay, LrPolicy *lr)
      : ParameterOptimizer(lr), rho_(rho), epsilon_(epsilon), decay_(decay) {}
  ~AdadeltaOptimizer() {
    if (accum_gradient_) delete accum_gradient_;
    if (accum_delta_) delete accum_delta_;
    if (update_delta_) delete update_delta_;
  }
  void Update(const Tensor *gradient);
  void set_weight(Tensor *p);
  real *get_weight() const;

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

#endif
