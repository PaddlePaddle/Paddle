#pragma once

#include <glog/logging.h>
#include <functional>
#include <string>
#include "OptimizerConfig.pb.h"
#include "lr_policy.h"
#include "serialization.h"
#include "tensor.h"

namespace paddle {
namespace optimizer {

class ParameterOptimizer {
public:
  /**
   * @brief  update hook for algorithm need to traverse parameter more than
   * once.
   */
  ParameterOptimizer(Tensor *parameter, LrPolicy *lr)
      : parameter_(parameter), lr_policy_(lr), num_sample_passed_(0) {}
  virtual ~ParameterOptimizer() {
    delete parameter_;
    delete lr_policy_;
  }

  static ParameterOptimizer *Create(const std::string &config_proto,
                                    Tensor *parameter);
  virtual void Update(const Tensor *gradient) = 0;
  virtual float *get_weight(int *param_size) const;
  virtual const char *SerializeState(int *state_len) = 0;
  virtual void DeserializeState(const std::string &state) = 0;

protected:
  Tensor *parameter_;
  // learning rate policy
  LrPolicy *lr_policy_;
  uint64_t num_sample_passed_;
};

}  // namespace optimizer
}  // namespace paddle
