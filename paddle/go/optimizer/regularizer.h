#ifndef PADDLE_OPITMIZER_REGULARIZER_H_
#define PADDLE_OPTIMIZER_REGULARIZER_H_

#include "Tensor.h"
#include "OptimizerConfig.pb.h"

namespace paddle {
namespace optimizer {

/**
 * @brief regularizer in L1, L2
 */


template<class T>
class Regularizer {
public:
  /**
   *  @brief regularizer update interface
   *  @param param need to update
   *  @return void
   */
  static Regularizer* create(const std::string& config);
  virtual void update(Tensor<T> &parameter) = 0;
private:
  std::string regularizer_name;
  OptimizerConfig config_;
};

template<class T>
class L1Regularizer {
public:
  void update(Tensor<T> &parameter);
};

template<class T>
class L2Regularizer {
public:
  void update(Tensor<T> &parameter);
};

}  // namespace optimizer
}  // namespace paddle

#endif
