#ifndef PADDLE_PARAMETER_OPTIMIZER_H_
#define PADDLE_PARAMETER_OPTIMIZER_H_

#include <glog/logging.h>
#include <functional>
#include <string>
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

  static ParameterOptimizer *create(const ::std::string &config_proto);
  virtual void update(const Tensor &gradient) = 0;
  virtual void destroy() = 0;
  virtual T *get_weight() const;
  virtual char* get_config_proto();
  virtual void set_weight(const Tensor<T> *parameter);
  ~ParameterOptimzier() {
    delete parameter_;
  }
  double get_learning_rate();

private:
  bool config_valid(::std::string &config) const;
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
