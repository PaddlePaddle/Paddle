#ifndef PADDLE_ADAGRAD_OPTIMIZER_H_
#define PADDLE_ADAGRAD_OPTIMIZER_H_

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {


template <class T>
class AdagradOptimizer : public ParameterOptimizer<T> {
public:
  AdagradOptimizer(const OptimizerConfig &config);
  ~AdagradOptimizer(){
    if(accum_gradient) delete accum_gradient;
  }
  void update(const Tensor<T> &gradient);
  void set_weight(const Tensor<T> *p);
  T* get_weight() const;
  
private:
  Tensor<T> *accum_gradient;
  double epsilon;
  double decay;
};

}
}

#endif
