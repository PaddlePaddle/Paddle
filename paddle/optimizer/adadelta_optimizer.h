#ifndef PADDLE_ADADELTA_OPTIMIZER_H_
#define PADDLE_ADADELTA_OPTIMIZER_H_

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

template <class T>
class AdadeltaOptimizer : public ParameterOptimizer<T> {
public:
  AdadeltaOptimizer(const OptimizerConfig &config);
  ~AdadeltaOptimizer(){
    if(accum_gradient) delete accum_gradient;
    if(accum_delta) delete accum_delta;
    if(update_delta) delete update_delta;
  }
  void update(const Tensor<T> &gradient);
  void set_weight(const Tensor<T> *p);
  T* get_weight() const;
  
private:
  Tensor<T> *accum_gradient;
  Tensor<T> *accum_delta;
  Tensor<T> *update_delta;
  
  double rho;
  double epsilon;
  double decay;
};

}
}

#endif
