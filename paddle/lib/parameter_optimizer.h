#ifndef PADDLE_LIB_OPTIMIZER_BASE_H_
#define PADDLE_LIB_OPTIMIZER_BASE_H_

#include <string>
#include <functional>
// #include <math/Tensor.h>
#include "Tensor.h"

namespace paddle {
namespace lib {

class ParameterOptimizer {
public:
  /*! \brief update hook for algorithm need to traverse parameter more
    than once.
  */
  typedef std::function<void(Tensor&, const Tensor&)> UpdateHook;

  static ParameterOptimizer *create();
  virtual update(Tensor &parameter, const Tensor &gradient, double learning_rate) = 0;
  virtual destory() = 0 ;
  ~ParameterOptimzier(){}

private:
  double learning_rate;
  /*! \brief indicate if use L1, L2 regularizer */ 
  bool applyDecay;
  bool useGpu;
};

}
}


#endif
