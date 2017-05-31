#ifndef PADDLE_LIB_OPTIMIZER_BASE_H_
#define PADDLE_LIB_OPTIMIZER_BASE_H_

#include <functional>
#include <string>
// #include <math/Tensor.h>
#include "Tensor.h"
#include "OptimizerConfig.pb.h"

namespace paddle {
namespace optimizer {

template <class T>
class ParameterOptimizer {
public:
  /**
   * @brief  update hook for algorithm need to traverse parameter more than
   * once.
   */
  ParameterOptimizer(const OptimizerConfig &config) : config_(config){};

  static ParameterOptimizer *create(const std::string &config_proto);
  virtual void update(const Tensor &gradient) = 0;
  virtual void destroy() = 0;
  virtual T *get_weight() const;
  virtual char* get_config_proto();
  virtual void set_weight(const Tensor<T> *parameter);
  ~ParameterOptimzier() {
    delete parameter_;
  }

private:
  bool config_valid(std::string &config) const;
  OptimizerConfig config_;
  Tensor<T> *parameter_;

  uint32 num_sample_passed;
  uint32 iterations;

  ParameterOptimizer(const ParameterOptimizer&) = delete;
  ParameterOptimizer& operator=(const ParameterOptimizer&) = delete;
  /**
   * @brief indicate if use L1, L2 regularizer
   */
};



}  // namespace optimizer
}  // namespace paddle

#endif
