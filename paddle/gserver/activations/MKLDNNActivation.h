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
#include "ActivationFunction.h"
#include "mkldnn.hpp"
#include "paddle/gserver/layers/MKLDNNBase.h"
#include "paddle/math/MKLDNNMatrix.h"
#include "paddle/parameter/Argument.h"

namespace paddle {

/**
 * @brief Base class of MKLDNN Activation.
 * Common activation function are provieded,
 * including mkldnn_relu, mkldnn_elu, mkldnn_tanh, mkldnn_softmax
 */
class MKLDNNActivation : public ActivationFunction {
 protected:
  // input value element count
  size_t cnt_;
  // should not merge the resetBwd into resetFwd,
  // because the grad data would be changing before backward.
  bool needResetBwd_;
  // mkldnn matrix, primitive, stream and pipeline
  MKLDNNMatrixPtr val_;
  MKLDNNMatrixPtr grad_;
  std::shared_ptr<mkldnn::engine> engine_;
  std::shared_ptr<MKLDNNStream> stream_;
  std::shared_ptr<mkldnn::primitive> fwd_;
  std::shared_ptr<mkldnn::primitive> bwd_;
  std::vector<mkldnn::primitive> pipelineFwd_;
  std::vector<mkldnn::primitive> pipelineBwd_;

 public:
  MKLDNNActivation() : cnt_(0), needResetBwd_(true) {}
  ~MKLDNNActivation() {}
  static ActivationFunction* create(const std::string& type);
  static std::vector<std::string> getAllRegisteredTypes();
  virtual const std::string& getName() const = 0;
  /**
   * reset the forward primitives
   */
  virtual void resetFwd(Argument& act);
  /**
   * reset the backward primitives,
   * can not merge this functions into resetFwd as the grad data
   * would be changing before backward.
   */
  virtual void resetBwd(Argument& act) {}
  virtual Error __must_check forward(Argument& act);
  virtual Error __must_check backward(Argument& act);
};

/**
 * @brief Base class of MKLDNN Eltwise Activation,
 * includes mkldnn_relu, mkldnn_elu and mkldnn_tanh.
 */
class MKLDNNEltwiseActivation : public MKLDNNActivation {
  typedef mkldnn::eltwise_forward eltwise_fwd;
  typedef mkldnn::eltwise_backward eltwise_bwd;
  typedef mkldnn::algorithm algorithm;

 protected:
  // save the forward primitive desc, which can be used backward
  std::shared_ptr<eltwise_fwd::primitive_desc> fwdPD_;
  // eltwise_bwd need src input value
  MKLDNNMatrixPtr inVal_;
  // use for copy data
  std::shared_ptr<mkldnn::reorder> copyInVal_;

 public:
  MKLDNNEltwiseActivation() {}
  ~MKLDNNEltwiseActivation() {}
  virtual const std::string& getName() const = 0;

  // in common, the alpha of forward and backward should be equal.
  // but for relu, to avoid negative value, they should be opposite
  virtual float getAlpha() const = 0;
  virtual float getBwdAlpha() const = 0;
  virtual float getBeta() const { return 0.f; }
  virtual algorithm getAlgo(std::string type) const;
  void resetFwd(Argument& act) override;
  void resetBwd(Argument& act) override;
};

/**
 * @brief Base class of MKLDNN softmax Activation,
 * only have mkldnn forward, use cpu implement for backward.
 */
class MKLDNNSoftmaxActivation : public MKLDNNActivation {
  typedef mkldnn::softmax_forward softmax_fwd;

 private:
  // for backward
  MatrixPtr sftMaxSum_;
  MatrixPtr sftMaxDot_;

 public:
  MKLDNNSoftmaxActivation() {}
  ~MKLDNNSoftmaxActivation() {}
  virtual const std::string& getName() const = 0;
  void resetFwd(Argument& act) override;
  Error __must_check forward(Argument& act) override;
  Error __must_check backward(Argument& act) override;
};

}  // namespace paddle
