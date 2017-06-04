#ifndef PADDLE_ADAM_OPTIMIZER_H_
#define PADDLE_ADAM_OPTIMIZER_H_

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

template <class T>
class AdamOptimizer : public ParameterOptimizer<T> {
public:
  AdamOptimizer(const OptimizerConfig &config);
  ~AdamOptimizer() {}
  void update(const Tensor<T> &gradient);
  void set_weight(const Tensor<T> *p);
  T *get_weight() const;

private:
  Tensor<T> *momentums_;
  Tensor<T> *velocitys_;
  double beta_1;
  double beta_2;
  double epsilon;
  double decay;
};

}  // namespace optimizer
}  // namespace paddle
#endif
