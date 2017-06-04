#ifndef PADDLE_SGD_OPTIMIZER_H_
#define PADDLE_SGD_OPTIMIZER_H_

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

template <class T>
class SGDOptimizer : public ParameterOptimizer<T> {
public:
  SGDOptimizer(const ::paddle::OptimizerConfig &config);
  ~SGDOptimizer() {
    // clear memory by Tensor library
    delete momentums_;
  }
  void update(const Tensor<T> &gradient);
  
  void set_weight(const Tensor<T> *p);
  T* get_weight() const;
  char* get_config_proto();
private:
  Tensor<T>* momentums_;
  double momentum;
  double decay;
  bool nesterov;
};

}
}

#endif
