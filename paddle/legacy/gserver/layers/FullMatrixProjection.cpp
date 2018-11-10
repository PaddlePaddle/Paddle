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

#include "FullMatrixProjection.h"

namespace paddle {

REGISTER_PROJECTION(fc, FullMatrixProjection);

FullMatrixProjection::FullMatrixProjection(const ProjectionConfig& config,
                                           const ParameterPtr& parameter,
                                           bool useGpu)
    : Projection(config, parameter, useGpu) {
  weight_.reset(
      new Weight(config.input_size(), config.output_size(), parameter));
}

void FullMatrixProjection::forward() {
  REGISTER_TIMER_INFO("FwMulTimer", getName().c_str());
  out_->value->mul(*(in_->value), *(weight_->getW()), 1, 1);
}

void FullMatrixProjection::backward(const UpdateCallback& callback) {
  bool syncFlag = hl_get_sync_flag();

  /* Calculate the W-gradient for the current layer */
  if (weight_->getWGrad()) {
    REGISTER_TIMER_INFO("GradMulTimer", getName().c_str());
    weight_->getWGrad()->mul(
        *(in_->value->getTranspose()), *(out_->grad), 1, 1);
  }

  // If callback does not change value, backward propagation error
  // asynchronously, so that we can do the callback concurrently.
  hl_set_sync_flag(false);

  /* Calculate the input layers error */
  if (in_->grad) {
    REGISTER_TIMER_INFO("BpMulTimer", getName().c_str());
    in_->grad->mul(*(out_->grad), *(weight_->getW()->getTranspose()), 1, 1);
  }

  hl_set_sync_flag(syncFlag);
  if (weight_->getWGrad()) {
    parameter_->incUpdate(callback);
  }
}

}  // namespace paddle
