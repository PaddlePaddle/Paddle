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
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * @brief TransposedFullMatrixProjection performs full matrix multiplication:
 * out.row[i] += in.row[i] * weight.transpose
 *
 * The config file api is trans_full_matrix_projection.
 */
class TransposedFullMatrixProjection : public Projection {
 public:
  TransposedFullMatrixProjection(const ProjectionConfig& config,
                                 ParameterPtr parameter,
                                 bool useGPu);
  virtual void forward();
  virtual void backward(const UpdateCallback& callback);

 protected:
  std::unique_ptr<Weight> weight_;
};

REGISTER_PROJECTION(trans_fc, TransposedFullMatrixProjection);

TransposedFullMatrixProjection::TransposedFullMatrixProjection(
    const ProjectionConfig& config, ParameterPtr parameter, bool useGpu)
    : Projection(config, parameter, useGpu) {
  weight_.reset(
      new Weight(config.output_size(), config.input_size(), parameter));
}

void TransposedFullMatrixProjection::forward() {
  REGISTER_TIMER_INFO("FwMulTimer", getName().c_str());
  out_->value->mul(*(in_->value), *(weight_->getW()->getTranspose()), 1, 1);
}

void TransposedFullMatrixProjection::backward(const UpdateCallback& callback) {
  bool syncFlag = hl_get_sync_flag();

  /* Calculate the W-gradient for the current layer */
  if (weight_->getWGrad()) {
    REGISTER_TIMER_INFO("GradMulTimer", getName().c_str());
    weight_->getWGrad()->mul(
        *(out_->grad->getTranspose()), *(in_->value), 1, 1);
  }

  // If callback does not change value, backprop error asynchronously so that
  // we can do the callback concurrently.
  // This is still a little bit dangerous since theoretically for
  // SyncMultiGpuMachine it is possible that the value copyback can still
  // happen at the same time as the error backprop where the value is being
  // used.
  hl_set_sync_flag(false);

  /* Calculate the input layers error */
  if (in_->grad) {
    REGISTER_TIMER_INFO("BpMulTimer", getName().c_str());
    in_->grad->mul(*(out_->grad), *(weight_->getW()), 1, 1);
  }

  hl_set_sync_flag(syncFlag);
  parameter_->incUpdate(callback);
}

}  // namespace paddle
