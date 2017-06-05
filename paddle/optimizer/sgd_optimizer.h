#ifndef PADDLE_SGD_OPTIMIZER_H_
#define PADDLE_SGD_OPTIMIZER_H_

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

class SGDOptimizer : public ParameterOptimizer {
public:
  using ParameterOptimizer::parameter_;
  using ParameterOptimizer::num_sample_passed;
  using ParameterOptimizer::lr_policy;

  SGDOptimizer(double m, double d, bool n, BaseLr* lr)
      : ParameterOptimizer(lr), momentum(m), decay(d), nesterov(n) {}
  virtual ~SGDOptimizer() { delete momentums_; }
  void update(const Tensor& gradient);

  void set_weight(Tensor* p);
  real* get_weight() const;

private:
  Tensor* momentums_;
  double momentum;
  double decay;
  bool nesterov;
};

}  // namespace optimizer
}  // namespace paddle

#endif
