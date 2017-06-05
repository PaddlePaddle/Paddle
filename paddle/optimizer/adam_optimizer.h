#ifndef PADDLE_ADAM_OPTIMIZER_H_
#define PADDLE_ADAM_OPTIMIZER_H_

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

template <class T>
class AdamOptimizer : public ParameterOptimizer<T> {
public:
  using ParameterOptimizer<T>::parameter_;
  using ParameterOptimizer<T>::num_sample_passed;
  using ParameterOptimizer<T>::lr_policy;
  AdamOptimizer(
      double beta_1, double beta_2, double epsilon, double decay, BaseLr *lr)
      : ParameterOptimizer<T>(lr),
        beta_1(beta_1),
        beta_2(beta_2),
        epsilon(epsilon),
        decay(decay) {}
  ~AdamOptimizer() {
    if (momentums_) delete momentums_;
    if (velocitys_) delete velocitys_;
  }
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
