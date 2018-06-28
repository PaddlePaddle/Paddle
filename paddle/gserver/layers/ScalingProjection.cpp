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

class ScalingProjection : public Projection {
 public:
  ScalingProjection(const ProjectionConfig& config,
                    const ParameterPtr& parameter,
                    bool useGpu)
      : Projection(config, parameter, useGpu) {
    CHECK_EQ(parameter->getSize(), 1UL);
    weight_.reset(new Weight(1, 1, parameter));
  }

  void forward() {
    CHECK(in_->value);
    out_->value->add(*in_->value, weight_->getW()->getElement(0, 0));
  }

  void backward(const UpdateCallback& callback) {
    if (weight_->getWGrad()) {
      auto sum = Matrix::create(in_->value->getHeight(), 1, false, useGpu_);
      sum->sumOfProducts(*in_->value,
                         *out_->grad,
                         /* scaleSum= */ 1,
                         /* scaleDest= */ 0);
      weight_->getWGrad()->sumCols(*sum,
                                   /* scaleSum= */ 1,
                                   /* scaleDest= */ 1);
      parameter_->incUpdate(callback);
    }
    if (in_->grad) {
      in_->grad->add(*out_->grad, weight_->getW()->getElement(0, 0));
    }
  }

 protected:
  std::unique_ptr<Weight> weight_;
};

REGISTER_PROJECTION(scaling, ScalingProjection);

}  // namespace paddle
