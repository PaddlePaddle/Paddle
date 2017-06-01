#ifndef PADDLE_ADADELTA_OPTIMIZER_H_
#define PADDLE_ADADELTA_OPTIMIZER_H_

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

template <class T>
class AdadeltaOptimizer : public ParameterOptimizer<T> {
public:
  /*! \brief call the applySGD for example  */
  void update(const Tensor<T> &gradient) {
  }
private:
  double learning_rate;
  double rho;
  double epsilon;
  double decay;
};


}
}

#endif
