/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "MKLDNNLayer.h"
#include "mkldnn.hpp"

namespace paddle {
typedef mkldnn::pooling_forward pool_fwd;
typedef mkldnn::pooling_backward pool_bwd;

/**
 * @brief A subclass of MKLDNNLayer pool layer.
 *
 * The config file api is mkldnn_pool
 */
class MKLDNNPoolLayer : public MKLDNNLayer {
protected:
  // save forward primitive_desc, which can be used backward
  std::shared_ptr<pool_fwd::primitive_desc> fwdPD_;

public:
  explicit MKLDNNPoolLayer(const LayerConfig& config) : MKLDNNLayer(config) {}

  ~MKLDNNPoolLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void reshape(
      int& bs, int& ic, int& ih, int& iw, int oc, int& oh, int& ow) override;

  void resetFwd(std::vector<mkldnn::primitive>& pipeline,
                MKLDNNMatrixPtr& in,
                MKLDNNMatrixPtr& wgt,
                MKLDNNMatrixPtr& bias,
                MKLDNNMatrixPtr& out) override;

  void resetBwd(std::vector<mkldnn::primitive>& pipeline,
                MKLDNNMatrixPtr& in,
                MKLDNNMatrixPtr& wgt,
                MKLDNNMatrixPtr& bias,
                MKLDNNMatrixPtr& out) override;

  void updateInputData() override;

protected:
  /**
   * Forward functions: reset buffers(input, output),
   *                    reset primitive descriptor,
   *                    reset pipeline.
   */
  void resetFwdBuffers(MKLDNNMatrixPtr& in, MKLDNNMatrixPtr& out);
  void resetInValue(MKLDNNMatrixPtr& in);
  void resetOutValue(MKLDNNMatrixPtr& out);
  void resetFwdPD(std::shared_ptr<pool_fwd::primitive_desc>& pd,
                  MKLDNNMatrixPtr in,
                  MKLDNNMatrixPtr out);
  void resetFwdPipeline(std::vector<mkldnn::primitive>& pipeline,
                        std::shared_ptr<pool_fwd::primitive_desc>& pd,
                        MKLDNNMatrixPtr& in,
                        MKLDNNMatrixPtr& out);

  /**
   * Backward functions: reset buffers(input, output),
   *                     reset primitive descriptor,
   *                     reset pipeline.
   */
  void resetBwdBuffers(MKLDNNMatrixPtr& in, MKLDNNMatrixPtr& out);
  void resetOutGrad(MKLDNNMatrixPtr& out);
  void resetInGrad(MKLDNNMatrixPtr& in);
  void resetBwdPD(std::shared_ptr<pool_bwd::primitive_desc>& pd,
                  MKLDNNMatrixPtr& in,
                  MKLDNNMatrixPtr& out);
  void resetBwdPipeline(std::vector<mkldnn::primitive>& pipeline,
                        std::shared_ptr<pool_bwd::primitive_desc>& pd,
                        MKLDNNMatrixPtr& in,
                        MKLDNNMatrixPtr& out);
};

}  // namespace paddle
