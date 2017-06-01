#ifndef PADDLE_ADAM_OPTIMIZER_H_
#define PADDLE_ADAM_OPTIMIZER_H_

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {


template <class T>
class AdamOptimizer : public ParameterOptimizer<T> {
public:
  /*! \brief call the applySGD for example  */
  void update(const Tensor<T> &gradient) {
  }
private:
  double learning_rate ;
  double beta_1;
  double beta_2;
  double epsilon;
};


}  // namespace optimizer
}  // namespace paddle
#endif
