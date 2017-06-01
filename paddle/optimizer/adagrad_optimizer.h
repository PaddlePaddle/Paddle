#ifndef PADDLE_ADAGRAD_OPTIMIZER_H_
#define PADDLE_ADAGRAD_OPTIMIZER_H_

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {


template <class T>
class AdagradOptimizer : public ParameterOptimizer<T> {
public:
  void update(const Tensor<T> &gradient) {
  }
private:
  double learning_rate;
  double epsilon;
  double decay;
};

}
}

#endif
