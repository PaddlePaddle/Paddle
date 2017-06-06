#ifndef PADDLE_ADAGRAD_OPTIMIZER_H_
#define PADDLE_ADAGRAD_OPTIMIZER_H_

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

class AdagradOptimizer : public ParameterOptimizer {
public:
  AdagradOptimizer(double epsilon, double decay, LrPolicy *lr)
      : ParameterOptimizer(lr), epsilon_(epsilon), decay_(decay) {}
  ~AdagradOptimizer() {
    if (accum_gradient_) delete accum_gradient_;
  }
  void Update(const Tensor *gradient);
  void set_weight(Tensor *p);
  real *get_weight() const;

private:
  Tensor *accum_gradient_;
  double epsilon_;
  double decay_;
};

}  // namespace optimizer
}  // namespace paddle

#endif
