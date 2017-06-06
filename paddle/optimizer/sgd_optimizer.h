#ifndef PADDLE_SGD_OPTIMIZER_H_
#define PADDLE_SGD_OPTIMIZER_H_

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

class SGDOptimizer : public ParameterOptimizer {
public:
  SGDOptimizer(double m, double d, bool n, LrPolicy* lr)
      : ParameterOptimizer(lr), momentum_(m), decay_(d), nesterov_(n) {}
  virtual ~SGDOptimizer() { delete momentums_; }
  void Update(const Tensor* gradient);

  void set_weight(Tensor* p);
  real* get_weight() const;

private:
  Tensor* momentums_;
  double momentum_;
  double decay_;
  bool nesterov_;
};

}  // namespace optimizer
}  // namespace paddle

#endif
