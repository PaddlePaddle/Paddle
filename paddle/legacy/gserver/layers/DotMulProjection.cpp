/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "Projection.h"

namespace paddle {

/**
 * DotMulProjection performs element-wise multiplication with weight:
 * \f[
 *   out.row[i] += in.row[i] .* weight
 * \f]
 * where \f$.*\f$ means element-wise multiplication.
 *
 * The config file api is dotmul_projection.
 */
class DotMulProjection : public Projection {
 public:
  DotMulProjection(const ProjectionConfig& config,
                   const ParameterPtr& parameter,
                   bool useGpu);
  virtual void forward();
  virtual void backward(const UpdateCallback& callback);

 protected:
  /// shared memory with parameter
  std::unique_ptr<Weight> weight_;
};

REGISTER_PROJECTION(dot_mul, DotMulProjection);

DotMulProjection::DotMulProjection(const ProjectionConfig& config,
                                   const ParameterPtr& parameter,
                                   bool useGpu)
    : Projection(config, parameter, useGpu) {
  weight_.reset(new Weight(1LU, config.output_size(), parameter));
}

void DotMulProjection::forward() {
  out_->value->addDotMulMMV(*in_->value, *(weight_->getW()));
}

void DotMulProjection::backward(const UpdateCallback& callback) {
  /* Calculate the W-gradient for the current layer */
  if (weight_->getWGrad()) {
    weight_->getWGrad()->addDotMulVMM(*out_->grad, *in_->value);
  }

  /* Calculate the input layers error */
  if (in_->grad) {
    in_->grad->addDotMulMMV(*out_->grad, *(weight_->getW()));
  }

  parameter_->incUpdate(callback);
}

}  // namespace paddle
