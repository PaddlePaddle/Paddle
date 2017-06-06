#ifndef PADDLE_ADAM_OPTIMIZER_H_
#define PADDLE_ADAM_OPTIMIZER_H_

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

class AdamOptimizer : public ParameterOptimizer {
public:
  AdamOptimizer(
      double beta_1, double beta_2, double epsilon, double decay, LrPolicy *lr)
      : ParameterOptimizer(lr),
        beta_1_(beta_1),
        beta_2_(beta_2),
        epsilon_(epsilon),
        decay_(decay) {}
  ~AdamOptimizer() {
    if (momentums_) delete momentums_;
    if (velocitys_) delete velocitys_;
  }
  void Update(const Tensor *gradient);
  void set_weight(Tensor *p);
  real *get_weight() const;

private:
  Tensor *momentums_;
  Tensor *velocitys_;
  double beta_1_;
  double beta_2_;
  double epsilon_;
  double decay_;
};

}  // namespace optimizer
}  // namespace paddle
#endif
