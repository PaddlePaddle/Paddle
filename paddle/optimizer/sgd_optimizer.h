#ifndef PADDLE_SGD_OPTIMIZER_H_
#define PADDLE_SGD_OPTIMIZER_H_

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

template <class T>
class SGDOptimizer : public ParameterOptimizer<T> {
public:
  SGDOptimizer(const ::paddle::OptimizerConfig &config);
  void set_weight(const Tensor<T> *p);
  T* get_weight() const;
  void update(const Tensor<T> &gradient);
  char* get_config_proto();
  ~SGDOptimizer() {
    // clear memory by Tensor library
    delete momentums_;
  }
private:
  Tensor<T>* momentums_;
  double learning_rate;
  double momentum;
  double decay;
  bool nesterov;
  double lr_decay_a;
  double lr_decay_b;
};

}
}

#endif
