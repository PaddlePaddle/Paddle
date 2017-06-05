#ifndef PADDLE_SGD_OPTIMIZER_H_
#define PADDLE_SGD_OPTIMIZER_H_

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

template <class T>
class SGDOptimizer : public ParameterOptimizer<T> {
public:
  using ParameterOptimizer<T>::parameter_;
  using ParameterOptimizer<T>::num_sample_passed;
  using ParameterOptimizer<T>::lr_policy;

  SGDOptimizer(double m,
               double d,
               bool n,
               double learning_rate,
               uint64_t num_sample_passed,
               BaseLr* lr)
      : ParameterOptimizer<T>(lr), momentum(m), decay(d), nesterov(n) {}
  virtual ~SGDOptimizer() {
    // clear memory by Tensor library
    delete momentums_;
  }
  void update(const Tensor<T>& gradient);

  void set_weight(const Tensor<T>* p);
  T* get_weight() const;

private:
  Tensor<T>* momentums_;
  double momentum;
  double decay;
  bool nesterov;
};

}  // namespace optimizer
}  // namespace paddle

#endif
