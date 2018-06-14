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

#include "Operator.h"

namespace paddle {

/**
 * DotMulOperator takes two inputs, performs element-wise multiplication:
 * \f[
 *   out.row[i] += scale * (in1.row[i] .* in2.row[i])
 * \f]
 * where \f$.*\f$ means element-wise multiplication,
 * and scale is a config scalar, its default value is one.
 *
 * The config file api is dotmul_operator.
 */
class DotMulOperator : public Operator {
 public:
  DotMulOperator(const OperatorConfig& config, bool useGpu);
  virtual void forward();
  virtual void backward();
};

REGISTER_OPERATOR(dot_mul, DotMulOperator);

DotMulOperator::DotMulOperator(const OperatorConfig& config, bool useGpu)
    : Operator(config, useGpu) {
  CHECK_EQ(config_.input_indices_size(), 2L);
}

void DotMulOperator::forward() {
  out_->value->addDotMul(
      *ins_[0]->value, *ins_[1]->value, 1, config_.dotmul_scale());
}

void DotMulOperator::backward() {
  const MatrixPtr& inV0 = ins_[0]->value;
  const MatrixPtr& inV1 = ins_[1]->value;
  const MatrixPtr& inG0 = ins_[0]->grad;
  const MatrixPtr& inG1 = ins_[1]->grad;

  if (inG0) {
    inG0->addDotMul(*out_->grad, *inV1, 1, config_.dotmul_scale());
  }
  if (inG1) {
    inG1->addDotMul(*out_->grad, *inV0, 1, config_.dotmul_scale());
  }
}

}  // namespace paddle
