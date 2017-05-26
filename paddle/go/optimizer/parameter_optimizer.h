#ifndef PADDLE_LIB_OPTIMIZER_BASE_H_
#define PADDLE_LIB_OPTIMIZER_BASE_H_

#include <functional>
#include <memory>
#include <string>
// #include <math/Tensor.h>
#include "Tensor.h"
#include "optimizer_config.pb.h"

namespace paddle {
namespace optimizer {

template <class T>
class ParameterOptimizer {
public:
  /**
   * @brief  update hook for algorithm need to traverse parameter more than
   * once.
   */
  typedef std::function<void(Tensor &, const Tensor &)> UpdateHook;
  ParameterOptimizer = default;
  ParameterOptimizer(const paddle::optimizer_config &config)
      : config_(config){};

  static ParameterOptimizer *create(const std::string &config_proto);
  virtual void update(const Tensor &gradient) = 0;
  virtual void destory() = 0;
  T *get_buffer() const;
  std::string get_config_proto() const;
  void set_buffer(const Tensor<T> *parameter);
  ~ParameterOptimzier() {}

private:
  paddle::optimizer_config config_;
  std::shared_ptr<Tensor<T>> parameter_;
  /**
   * @brief indicate if use L1, L2 regularizer
   */
};

}  // namespace optimizer
}  // namespace paddle

#endif
