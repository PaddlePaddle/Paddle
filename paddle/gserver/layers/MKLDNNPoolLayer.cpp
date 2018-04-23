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

#include "MKLDNNPoolLayer.h"
#include "paddle/math/MathUtils.h"
#include "paddle/utils/Logging.h"

using namespace mkldnn;  // NOLINT
typedef memory::format format;

namespace paddle {

REGISTER_LAYER(mkldnn_pool, MKLDNNPoolLayer);

bool MKLDNNPoolLayer::init(const LayerMap& layerMap,
                           const ParameterMap& parameterMap) {
  if (!MKLDNNLayer::init(layerMap, parameterMap)) {
    return false;
  }

  /* the size of inputs for pool-layer is 1 */
  CHECK_EQ(config_.inputs_size(), 1);
  const PoolConfig& conf = config_.inputs(0).pool_conf();
  ic_ = conf.channels();
  ih_ = conf.img_size_y();
  iw_ = conf.img_size();
  oc_ = ic_;
  oh_ = conf.output_y();
  ow_ = conf.output_x();
  fh_ = conf.size_y();
  fw_ = conf.size_x();
  ph_ = conf.padding_y();
  pw_ = conf.padding();
  sh_ = conf.stride_y();
  sw_ = conf.stride();

  const std::string& type = conf.pool_type();
  if (type == "max-projection") {
    poolAlgo_ = algorithm::pooling_max;
  } else if (type == "avg-projection") {
    // paddle only use exclude_padding
    poolAlgo_ = algorithm::pooling_avg_exclude_padding;
  } else {
    LOG(FATAL) << "unknow pooling type!";
  }
  return true;
}

void MKLDNNPoolLayer::reshape(
    int& bs, int& ic, int& ih, int& iw, int& oc, int& oh, int& ow) {
  reshapeInput(bs, ih, iw);
  // ic_ and oc can not be changed
  CHECK_EQ((size_t)ic,
           inputLayers_[0]->getOutputValue()->getElementCnt() / bs / ih / iw)
      << "Input channel can not be changed";

  // cal output sizes
  // paddle used false caffeMode for pooling
  oh = outputSize(ih, fh_, ph_, sh_, false);
  ow = outputSize(iw, fw_, pw_, sw_, false);
  reshapeOutput(oh, ow);

  resizeOutput(bs, oc * oh * ow);
}

void MKLDNNPoolLayer::resetFwd(std::vector<primitive>& pipeline,
                               std::vector<MKLDNNMatrixPtr>& inputs,
                               MKLDNNMatrixPtr& out) {
  resetFwdBuffers(inputs[0], out);

  resetFwdPD(fwdPD_, inputs[0], out);

  resetFwdPipeline(pipeline, fwdPD_, inputs[0], out);
}

void MKLDNNPoolLayer::resetBwd(std::vector<primitive>& pipeline,
                               std::vector<MKLDNNMatrixPtr>& inputs,
                               MKLDNNMatrixPtr& out) {
  std::shared_ptr<pool_bwd::primitive_desc> pd;

  resetBwdBuffers(inputs[0], out);

  resetBwdPD(pd, inputs[0], out);

  resetBwdPipeline(pipeline, pd, inputs[0], out);
}

void MKLDNNPoolLayer::resetFwdBuffers(MKLDNNMatrixPtr& in,
                                      MKLDNNMatrixPtr& out) {
  resetInValue(in);

  memory::dims outDims = memory::dims{bs_, oc_, oh_, ow_};
  CHECK(in);
  auto outPD =
      MKLDNNMatrix::createPrimitiveDesc(outDims, in->getFormat(), engine_);
  resetOutValue(out, outPD);
}

void MKLDNNPoolLayer::resetFwdPD(std::shared_ptr<pool_fwd::primitive_desc>& pd,
                                 MKLDNNMatrixPtr in,
                                 MKLDNNMatrixPtr out) {
  memory::dims kernels = memory::dims{fh_, fw_};
  memory::dims strides = memory::dims{sh_, sw_};
  memory::dims padL = memory::dims{ph_, pw_};
  memory::dims padR = getPaddingR();
  padding_kind padKind = padding_kind::zero;
  prop_kind pk = passType_ == PASS_TEST ? prop_kind::forward_scoring
                                        : prop_kind::forward_training;
  auto fwdDesc = pool_fwd::desc(pk,
                                poolAlgo_,
                                in->getMemoryDesc(),
                                out->getMemoryDesc(),
                                strides,
                                kernels,
                                padL,
                                padR,
                                padKind);
  pd.reset(new pool_fwd::primitive_desc(fwdDesc, engine_));

  // prepare workspace if necessary
  workspace_ =
      (passType_ != PASS_TEST && poolAlgo_ == algorithm::pooling_max)
          ? std::make_shared<memory>(memory(pd->workspace_primitive_desc()))
          : nullptr;
}

void MKLDNNPoolLayer::resetFwdPipeline(
    std::vector<primitive>& pipeline,
    std::shared_ptr<pool_fwd::primitive_desc>& pd,
    MKLDNNMatrixPtr& in,
    MKLDNNMatrixPtr& out) {
  fwd_ = workspace_
             ? std::make_shared<pool_fwd>(pool_fwd(*pd, *in, *out, *workspace_))
             : std::make_shared<pool_fwd>(pool_fwd(*pd, *in, *out));
  pipeline.push_back(*fwd_);
}

void MKLDNNPoolLayer::resetBwdBuffers(MKLDNNMatrixPtr& in,
                                      MKLDNNMatrixPtr& out) {
  CHECK(inVals_[0] && outVal_);
  resetOutGrad(out, outVal_->getPrimitiveDesc());
  resetInGrad(in, inVals_[0]->getPrimitiveDesc());
}

void MKLDNNPoolLayer::resetBwdPD(std::shared_ptr<pool_bwd::primitive_desc>& pd,
                                 MKLDNNMatrixPtr& in,
                                 MKLDNNMatrixPtr& out) {
  pd = nullptr;
  if (in == nullptr) {
    return;
  }
  memory::dims kernels = memory::dims{fh_, fw_};
  memory::dims strides = memory::dims{sh_, sw_};
  memory::dims padL = memory::dims{ph_, pw_};
  memory::dims padR = getPaddingR();
  CHECK(out);
  auto bwdDesc = pool_bwd::desc(poolAlgo_,
                                in->getMemoryDesc(),
                                out->getMemoryDesc(),
                                strides,
                                kernels,
                                padL,
                                padR,
                                padding_kind::zero);
  pd.reset(new pool_bwd::primitive_desc(bwdDesc, engine_, *fwdPD_));
}

void MKLDNNPoolLayer::resetBwdPipeline(
    std::vector<primitive>& pipeline,
    std::shared_ptr<pool_bwd::primitive_desc>& pd,
    MKLDNNMatrixPtr& in,
    MKLDNNMatrixPtr& out) {
  if (pd == nullptr) {
    return;
  }

  bwdData_ =
      workspace_
          ? std::make_shared<pool_bwd>(pool_bwd(*pd, *out, *workspace_, *in))
          : std::make_shared<pool_bwd>(pool_bwd(*pd, *out, *in));
  pipeline.push_back(*bwdData_);
}

}  // namespace paddle
