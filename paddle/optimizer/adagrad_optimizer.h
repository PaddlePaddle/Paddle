#ifndef PADDLE_ADAGRAD_OPTIMIZER_H_
#define PADDLE_ADAGRAD_OPTIMIZER_H_

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

class AdagradOptimizer : public ParameterOptimizer {
public:
  AdagradOptimizer(double epsilon, double decay, BaseLr *lr)
      : ParameterOptimizer(lr), epsilon(epsilon), decay(decay) {}
  ~AdagradOptimizer() {
    if (accum_gradient) delete accum_gradient;
  }
  void update(const Tensor *gradient);
  void set_weight(Tensor *p);
  real *get_weight() const;

private:
  Tensor *accum_gradient;
  double epsilon;
  double decay;
};

}  // namespace optimizer
}  // namespace paddle

#endif
