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

#include "RowConvOp.h"
#include <iostream>
#include "paddle/math/Vector.h"

namespace paddle {

template <>
void RowConv<DEVICE_TYPE_CPU>(CpuMatrix& out,
                              const CpuMatrix& in,
                              const CpuMatrix& filter,
                              const CpuIVector& seq) {
  const int* starts = seq.getData();
  const size_t numSeq = seq.getSize() - 1;
  const size_t contextLength = filter.getHeight();
  for (size_t i = 0; i < numSeq; ++i) {
    size_t begin = starts[i];
    size_t end = starts[i + 1];
    for (size_t j = begin; j < end; ++j) {
      MatrixPtr x;
      MatrixPtr w;
      if ((j + contextLength) < end) {
        x = (const_cast<CpuMatrix&>(in)).subMatrix(j, contextLength);
        w = (const_cast<CpuMatrix&>(filter)).subMatrix(0, contextLength);
      } else {
        x = (const_cast<CpuMatrix&>(in)).subMatrix(j, end - j);
        w = (const_cast<CpuMatrix&>(filter)).subMatrix(0, end - j);
      }
      MatrixPtr y = out.subMatrix(j, 1);
      y->addDotMulVMM(*x, *w);
    }
  }
}

template <>
void RowConvGrad<DEVICE_TYPE_CPU>(const CpuMatrix& outG,
                                  const CpuMatrix& in,
                                  const CpuMatrix& filter,
                                  CpuMatrix& inG,
                                  CpuMatrix& filterG,
                                  const CpuIVector& seq) {
  // gradient w.r.t filter
  const int* starts = seq.getData();
  const size_t numSeq = seq.getSize() - 1;
  const size_t contextLength = filter.getHeight();
  if (filterG) {
    for (size_t i = 0; i < numSeq; ++i) {
      size_t begin = starts[i];
      size_t end = starts[i + 1];
      size_t steps = end - begin;
      for (size_t j = 0; j < contextLength && (begin + j) < end; ++j) {
        MatrixPtr x =
            (const_cast<CpuMatrix&>(in)).subMatrix(begin + j, steps - j);
        MatrixPtr dy =
            (const_cast<CpuMatrix&>(outG)).subMatrix(begin, steps - j);
        MatrixPtr dw = filterG.subMatrix(j, 1);
        dw->addDotMulVMM(*dy, *x);
      }
    }
  }

  // gradient w.r.t input feature
  if (inG) {
    for (size_t i = 0; i < numSeq; ++i) {
      size_t begin = starts[i];
      size_t end = starts[i + 1];
      size_t steps = end - begin;
      for (size_t j = 0; j < steps; ++j) {
        MatrixPtr dx = inG.subMatrix(begin + j, 1);
        for (size_t t = 0; t < contextLength; ++t) {
          if (int(j - t) >= 0) {
            MatrixPtr dy =
                (const_cast<CpuMatrix&>(outG)).subMatrix(begin + j - t, 1);
            MatrixPtr w = (const_cast<CpuMatrix&>(filter)).subMatrix(t, 1);
            dx->addDotMul(*dy, *w, 1.0, 1.0);
          }
        }
      }
    }
  }
}

/**
 * \brief The row convolution is called lookahead convolution. It is firstly
 * introduced in deep-speech2 system. The bidirectional RNN that learns
 * representation for a sequence by performing a forward and a backward pass
 * through the entire sequence. However, unlike unidirectional RNNs,
 * bidirectional RNNs are challenging to deploy in an online and low-latency
 * setting. The lookahead convolution incorporates information from future
 * subsequences in a computationally efficient manner to improve unidirectional
 * recurrent neural networks.
 *
 * The connection of row convolution is different form the 1D sequence
 * convolution. Assumed that, the future context-length is k, that is to say,
 * it can get the output at timestep t by using the the input feature from t-th
 * timestep to (t+k)-th timestep. Assumed that the hidden dim of input
 * activations are d, the activations r_t for the new layer at time-step t are:
 *
 *
 *            -- k + 1
 *  r(t,i) =  >       W(i,j) * h(t+j-1, i),  for (1 <= i <= d)
 *            -- j = 1
 *
 *
 * The weight shape is: (k + 1) x d
 * Function Arguments:
 *
 * \param inputs[0]  The input activations.
 * \param inputs[0]  The filter (or weight) and shape is (k+1) x d.
 * \param outputs[1] The output activations.
 *
 * [1] Dario Amodei, etc. Deep Speech 2 : End-to-End Speech Recognition in
 * English
 *     and Mandarin. https://arxiv.org/abs/1512.02595
 */

template <DeviceType Device>
class RowConvFunc : public FunctionBase {
 public:
  void init(const FuncConfig& config) override {}

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    // check
    CHECK_EQ(2UL, inputs.size());
    CHECK_EQ(1UL, outputs.size());
    // TODO(qingqing): support ASSIGN_TO.
    CHECK_EQ(outputs[0].getArgType(), ADD_TO);
    CHECK(inputs[0].isSequenceArg() && outputs[0].isSequenceArg())
        << "SequenceArg required here.";
    const auto in = dynamic_cast<const SequenceArg&>(inputs[0]);
    auto out = dynamic_cast<const SequenceArg&>(outputs[0]);
    auto w = inputs[1];
    CHECK(in.data() && out.data() && in.getSequenceId().data());
    CHECK_EQ(in.shape().ndims(), 2UL);
    CHECK(in.shape() == out.shape());
    CHECK_EQ(w.shape()[1], in.shape()[1]);

    auto outMat = out.matrix<Device>();
    const auto inMat = in.matrix<Device>();
    const auto wMat = w.matrix<Device>();
    const auto seqId = in.getSequenceId().vector<int, Device>();

    RowConv<Device>(outMat, inMat, wMat, seqId);
  }
};

/**
 * \brief The backward of row convolution function. This function calculated
 * the gradient w.r.t filter and the gradient w.r.t input activations(or data).
 *
 * Argument in this Function:
 *
 * \param inputs[0]  The gradient w.r.t output activations.
 * \param inputs[1]  The input activations.
 * \param inputs[2]  The filter (or weight) and shape is (k+1) x d.
 * \param outputs[0] The gradient w.r.t input activations.
 * \param outputs[1] The gradient w.r.r filter.
 *
 * Abbreviation:
 * w.r.t: with respect to.
 */

template <DeviceType Device>
class RowConvGradFunc : public FunctionBase {
  // TODO(qingqing): split into RowConvDataFunc and RowConvWeightFunc
 public:
  void init(const FuncConfig& config) override {}

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    // check
    CHECK_EQ(3UL, inputs.size());
    CHECK_EQ(2UL, outputs.size());
    CHECK_EQ(outputs[0].getArgType(), ADD_TO);
    CHECK_EQ(outputs[1].getArgType(), ADD_TO);
    CHECK(inputs[0].isSequenceArg() && inputs[1].isSequenceArg() &&
          outputs[0].isSequenceArg())
        << "SequenceArg required here.";

    const auto outGrad = dynamic_cast<const SequenceArg&>(inputs[0]);
    const auto in = dynamic_cast<const SequenceArg&>(inputs[1]);
    const auto w = inputs[2];
    auto inGrad = dynamic_cast<const SequenceArg&>(outputs[0]);
    auto wGrad = outputs[1];

    CHECK_EQ(in.shape().ndims(), 2UL);
    CHECK(in.shape() == inGrad.shape());
    CHECK(in.shape() == outGrad.shape());
    CHECK_EQ(wGrad.shape()[1], in.shape()[1]);

    const auto outGMat = outGrad.matrix<Device>();
    const auto inMat = in.matrix<Device>();
    const auto wMat = w.matrix<Device>();
    auto inGMat = inGrad.data()
                      ? inGrad.matrix<Device>()
                      : typename Tensor<real, Device>::Matrix(nullptr, 0, 0);
    auto wGMat = wGrad.data()
                     ? wGrad.matrix<Device>()
                     : typename Tensor<real, Device>::Matrix(nullptr, 0, 0);
    const auto seqId = in.getSequenceId().vector<int, Device>();

    RowConvGrad<Device>(outGMat, inMat, wMat, inGMat, wGMat, seqId);
  }
};

REGISTER_TYPED_FUNC(RowConv, CPU, RowConvFunc);
REGISTER_TYPED_FUNC(RowConvGrad, CPU, RowConvGradFunc);
#ifdef PADDLE_WITH_CUDA
REGISTER_TYPED_FUNC(RowConv, GPU, RowConvFunc);
REGISTER_TYPED_FUNC(RowConvGrad, GPU, RowConvGradFunc);
#endif

}  // namespace paddle
