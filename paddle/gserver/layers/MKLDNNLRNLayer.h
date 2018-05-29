/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

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
typedef mkldnn::lrn_forward lrn_fwd;
typedef mkldnn::lrn_backward lrn_bwd;

/**
 * @brief A subclass of MKLDNNLayer LRN(Local Response Norm) layer.
 *
 * The config file api is mkldnn_lrn
 */
class MKLDNNLRNLayer : public MKLDNNLayer {
 protected:
  // save forward primitive_desc, which can be used in backward
  std::shared_ptr<lrn_fwd::primitive_desc> fwdPD_;
  // according to https://github.com/01org/mkl-dnn/blob/master/tests/gtests/
  // test_lrn_backward.cpp, lrn need workspace for backward
  std::shared_ptr<mkldnn::memory> workspace_;

  int localSize_;
  float alpha_, beta_;  // scale and pow in paddle

 public:
  explicit MKLDNNLRNLayer(const LayerConfig& config) : MKLDNNLayer(config) {}

  ~MKLDNNLRNLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void reshape(
      int& bs, int& ic, int& ih, int& iw, int& oc, int& oh, int& ow) override;

  void resetFwd(std::vector<mkldnn::primitive>& pipeline,
                std::vector<MKLDNNMatrixPtr>& inputs,
                MKLDNNMatrixPtr& out) override;

  void resetBwd(std::vector<mkldnn::primitive>& pipeline,
                std::vector<MKLDNNMatrixPtr>& inputs,
                MKLDNNMatrixPtr& out) override;

 protected:
  void resetFwdBuffers(MKLDNNMatrixPtr& in, MKLDNNMatrixPtr& out);
  void resetFwdPD(std::shared_ptr<lrn_fwd::primitive_desc>& pd,
                  MKLDNNMatrixPtr in,
                  MKLDNNMatrixPtr out);
  void resetFwdPipeline(std::vector<mkldnn::primitive>& pipeline,
                        std::shared_ptr<lrn_fwd::primitive_desc>& pd,
                        MKLDNNMatrixPtr& in,
                        MKLDNNMatrixPtr& out);
  void resetBwdBuffers(MKLDNNMatrixPtr& in, MKLDNNMatrixPtr& out);
  void resetBwdPD(std::shared_ptr<lrn_bwd::primitive_desc>& pd,
                  MKLDNNMatrixPtr& in,
                  MKLDNNMatrixPtr& out);
  void resetBwdPipeline(std::vector<mkldnn::primitive>& pipeline,
                        std::shared_ptr<lrn_bwd::primitive_desc>& pd,
                        MKLDNNMatrixPtr& in,
                        MKLDNNMatrixPtr& out);
};

}  // namespace paddle
