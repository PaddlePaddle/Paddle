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
    int& bs, int& ic, int& ih, int& iw, int oc, int& oh, int& ow) {
  reshapeInput(bs, ih, iw);
  // ic_ and oc can not be changed
  CHECK_EQ(inputElemenCnt_ / bs / ih / iw, (size_t)ic)
      << "Input channel can not be changed";

  // cal output sizes
  // paddle used false caffeMode for pooling
  oh = outputSize(ih, fh_, ph_, sh_, false);
  ow = outputSize(iw, fw_, pw_, sw_, false);
  reshapeOutput(oh, ow);

  resizeOutput(bs, oc * oh * ow);

  printSizeInfo();
}

void MKLDNNPoolLayer::resetFwd(std::vector<primitive>& pipeline,
                               MKLDNNMatrixPtr& in,
                               MKLDNNMatrixPtr& wgt,
                               MKLDNNMatrixPtr& bias,
                               MKLDNNMatrixPtr& out) {
  resetFwdBuffers(in, out);

  resetFwdPD(fwdPD_, in, out);

  resetFwdPipeline(pipeline, fwdPD_, in, out);

  printValueFormatFlow();
}

void MKLDNNPoolLayer::resetBwd(std::vector<primitive>& pipeline,
                               MKLDNNMatrixPtr& in,
                               MKLDNNMatrixPtr& wgt,
                               MKLDNNMatrixPtr& bias,
                               MKLDNNMatrixPtr& out) {
  std::shared_ptr<pool_bwd::primitive_desc> pd;

  resetBwdBuffers(in, out);

  resetBwdPD(pd, in, out);

  resetBwdPipeline(pipeline, pd, in, out);

  printGradFormatFlow();
}

void MKLDNNPoolLayer::updateInputData() {
  inVal_->setData(getInputValue(0, CPU_DEVICE)->getData());
}

void MKLDNNPoolLayer::resetFwdBuffers(MKLDNNMatrixPtr& in,
                                      MKLDNNMatrixPtr& out) {
  resetInValue(in);

  resetOutValue(out);
}

void MKLDNNPoolLayer::resetInValue(MKLDNNMatrixPtr& in) {
  if (inputIsOnlyMKLDNN()) {
    const MatrixPtr& dnnIn = getInputValue(0);
    in = std::dynamic_pointer_cast<MKLDNNMatrix>(dnnIn);
    CHECK(in) << "Input should be MKLDNNMatrix";
  } else {
    CHECK_EQ(getPrev(0)->getDeviceId(), CPU_DEVICE) << "Only support CPU yet";
    const MatrixPtr& cpuIn = getInputValue(0, CPU_DEVICE);
    in = MKLDNNMatrix::create(
        cpuIn, {bs_, ic_, ih_, iw_}, format::nchw, engine_);
  }
}

void MKLDNNPoolLayer::resetOutValue(MKLDNNMatrixPtr& out) {
  CHECK(inVal_) << "Should reset input value first";
  memory::dims outDims = memory::dims{bs_, oc_, oh_, ow_};
  out = MKLDNNMatrix::create(
      output_.value, outDims, inVal_->getFormat(), engine_);

  // create reorder if output value has cpu device and pd do not match
  cpuOutVal_ = nullptr;
  cvtOutVal_ = nullptr;
  if (!outputIsOnlyMKLDNN()) {
    const MatrixPtr& cpuOut = getOutput(CPU_DEVICE).value;
    cpuOutVal_ = MKLDNNMatrix::create(cpuOut, outDims, format::nchw, engine_);
    if (cpuOutVal_->getPrimitiveDesc() != out->getPrimitiveDesc()) {
      cvtOutVal_ = MKLDNNMatrix::createReorder(out, cpuOutVal_);
      CHECK(cvtOutVal_) << "should not be emptry";
    } else {
      // CPU output share the same data of MKLDNN output
      cpuOut->setData(out->getData());
      cpuOutVal_ = out;
    }
  }
}

void MKLDNNPoolLayer::resetFwdPD(std::shared_ptr<pool_fwd::primitive_desc>& pd,
                                 MKLDNNMatrixPtr in,
                                 MKLDNNMatrixPtr out) {
  memory::dims inDims = memory::dims{bs_, ic_, ih_, iw_};
  memory::dims outDims = memory::dims{bs_, oc_, oh_, ow_};
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
  pipeline.clear();
  fwd_ = workspace_
             ? std::make_shared<pool_fwd>(pool_fwd(*pd, *in, *out, *workspace_))
             : std::make_shared<pool_fwd>(pool_fwd(*pd, *in, *out));
  pipeline.push_back(*fwd_);

  if (cvtOutVal_) {
    pipeline.push_back(*cvtOutVal_);
  }
}

void MKLDNNPoolLayer::resetBwdBuffers(MKLDNNMatrixPtr& in,
                                      MKLDNNMatrixPtr& out) {
  resetOutGrad(out);

  resetInGrad(in);
}
void MKLDNNPoolLayer::resetOutGrad(MKLDNNMatrixPtr& out) {
  CHECK(outVal_) << "Should have output value";
  out = MKLDNNMatrix::create(output_.grad, outVal_->getPrimitiveDesc());

  // create reorder if output value has cpu device and pd do not match
  cpuOutGrad_ = nullptr;
  cvtOutGrad_ = nullptr;
  if (!outputIsOnlyMKLDNN()) {
    const MatrixPtr& cpuOut = getOutput(CPU_DEVICE).grad;
    cpuOutGrad_ = MKLDNNMatrix::create(
        cpuOut, memory::dims{bs_, oc_, oh_, ow_}, format::nchw, engine_);
    if (cpuOutGrad_->getPrimitiveDesc() != out->getPrimitiveDesc()) {
      cvtOutGrad_ = MKLDNNMatrix::createReorder(cpuOutGrad_, out);
      CHECK(cvtOutGrad_) << "should not be emptry";
    } else {
      // share the same data of CPU output
      output_.grad->setData(cpuOut->getData());
      out = cpuOutGrad_;
    }
  }
}

void MKLDNNPoolLayer::resetInGrad(MKLDNNMatrixPtr& in) {
  in = nullptr;
  const MatrixPtr& inGrad = inputLayers_[0]->getOutput().grad;
  if (inGrad == nullptr) {
    return;
  }
  CHECK(inVal_);
  in = MKLDNNMatrix::create(inGrad, inVal_->getPrimitiveDesc());
}

void MKLDNNPoolLayer::resetBwdPD(std::shared_ptr<pool_bwd::primitive_desc>& pd,
                                 MKLDNNMatrixPtr& in,
                                 MKLDNNMatrixPtr& out) {
  memory::dims kernels = memory::dims{fh_, fw_};
  memory::dims strides = memory::dims{sh_, sw_};
  memory::dims padL = memory::dims{ph_, pw_};
  memory::dims padR = getPaddingR();
  CHECK(in);
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
  pipeline.clear();
  if (cvtOutGrad_) {
    pipeline.push_back(*cvtOutGrad_);
  }

  bwdData_ =
      workspace_
          ? std::make_shared<pool_bwd>(pool_bwd(*pd, *out, *workspace_, *in))
          : std::make_shared<pool_bwd>(pool_bwd(*pd, *out, *in));
  pipeline.push_back(*bwdData_);
}

}  // namespace paddle
