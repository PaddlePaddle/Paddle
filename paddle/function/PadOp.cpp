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

#include "PadOp.h"
#include "paddle/math/Vector.h"

namespace paddle {

template <>
void Pad<DEVICE_TYPE_CPU>(real* outputs,
                          const real* inputs,
                          const int num,
                          const int inC,
                          const int inH,
                          const int inW,
                          const PadConf& pad) {
  int cstart = pad.channel[0], cend = pad.channel[1];
  int hstart = pad.height[0], hend = pad.height[1];
  int wstart = pad.width[0], wend = pad.width[1];
  int outC = inC + cstart + cend;
  int outH = inH + hstart + hend;
  int outW = inW + wstart + wend;
  for (int i = 0; i < num; i++) {
    for (int c = 0; c < inC; c++) {
      for (int h = 0; h < inH; h++) {
        int inoff = ((i * inC + c) * inH + h) * inW;
        int outoff =
            ((i * outC + c + cstart) * outH + h + hstart) * outW + wstart;
        memcpy(outputs + outoff, inputs + inoff, inW * sizeof(real));
      }
    }
  }
}

template <>
void PadGrad<DEVICE_TYPE_CPU>(real* inGrad,
                              const real* outGrad,
                              const int num,
                              const int inC,
                              const int inH,
                              const int inW,
                              const PadConf& pad) {
  int cstart = pad.channel[0], cend = pad.channel[1];
  int hstart = pad.height[0], hend = pad.height[1];
  int wstart = pad.width[0], wend = pad.width[1];
  int outC = inC + cstart + cend;
  int outH = inH + hstart + hend;
  int outW = inW + wstart + wend;
  for (int i = 0; i < num; i++) {
    for (int c = 0; c < inC; c++) {
      for (int h = 0; h < inH; h++) {
        int inoff = ((i * inC + c) * inH + h) * inW;
        int outoff =
            ((i * outC + c + cstart) * outH + h + hstart) * outW + wstart;
        CpuVector inG = CpuVector(inW, inGrad + inoff);
        CpuVector outG = CpuVector(inW, const_cast<real*>(outGrad + outoff));
        inG += outG;
      }
    }
  }
}

static inline PadConf castToPadConf(const FuncConfig& conf) {
  return {conf.get<std::vector<uint32_t>>("channel"),
          conf.get<std::vector<uint32_t>>("height"),
          conf.get<std::vector<uint32_t>>("width")};
}

/**
 * \brief Padding zeros to input according to the specify dimension.
 *        The struct pad_ contains the padding size in each dimension.
 *        The input and output is a 4D tensor. In PadFunc, we only
 *        pad zeros to the 2nd to 4th dimension.
 *
 * Argument in this Function:
 * \param pad_    A struct object contains the padding size in each dimension.
 *                It has six integers. The channelStart and channelEnd indicate
 *                how many zeros to add before and after the input in channel
 *                dimension. And the heightStart and heightEnd indicate padding
 *                in height dimension. The widthStart and widthEnd indicate the
 *                padding in width dimension.
 * \param inputs  A 4D tensor, only one input.
 * \param outputs A 4D tensor, the output value after padding.
 *
 * For example,
 * Input(2,2,2,3) = [
 *                    [ [[1,2,3], [3,4,5]],
 *                      [[2,3,5], [1,6,7]] ],
 *                    [ [[4,3,1], [1,8,7]],
 *                      [[3,8,9], [2,3,5]] ]
 *                  ] # the shape is (1,2,2,3)
 *
 * pad_: if channelStart = channelEnd = 1, others are 0.
 * Output(2,4,2,3) = [
 *                    [ [[0,0,0], [0,0,0]],
 *                      [[1,2,3], [3,4,5]],
 *                      [[2,3,5], [1,6,7]],
 *                      [[0,0,0], [0,0,0]] ],
 *                    [ [[0,0,0], [0,0,0]],
 *                      [[4,3,1], [1,8,7]],
 *                      [[3,8,9], [2,3,5]],
 *                      [[0,0,0], [0,0,0]] ]
 *                   ] # the shape is (2,4,2,3)
 *
 * pad_: if widthStart = 1, widthEnd = 2, others are 0.
 * Output(2,2,2,6) = [
 *                     [ [[0,1,2,3,0,0], [0,3,4,5,0,0]],
 *                       [[0,2,3,5,0,0], [0,1,6,7,0,0]] ],
 *                     [ [[0,4,3,1,0,0], [0,1,8,7,0,0]],
 *                       [[0,3,8,9,0,0], [0,2,3,5,0,0]] ],
 *                   ] # the shape is (2,2,2,6)
 *
 * pad_: if heightStart = 1, heightEnd = 1, others are 0.
 * Output(2,2,4,3) = [
 *                     [ [[0,0,0], [1,2,3], [3,4,5], [0,0,0]],
 *                       [[0,0,0], [2,3,5], [1,6,7], [0,0,0]] ],
 *                     [ [[0,0,0], [4,3,1], [1,8,7], [0,0,0]],
 *                       [[0,0,0], [3,8,9], [2,3,5], [0,0,0]] ],
 *                   ] # the shape is (2,2,4,3)
 */

template <DeviceType Device>
class PadFunc : public FunctionBase {
 public:
  void init(const FuncConfig& config) override { pad_ = castToPadConf(config); }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(1UL, inputs.size());
    CHECK_EQ(1UL, outputs.size());
    CHECK_EQ(outputs[0].getArgType(), ASSIGN_TO);

    size_t num = inputs[0].shape()[0];
    size_t inC = inputs[0].shape()[1];
    size_t inH = inputs[0].shape()[2];
    size_t inW = inputs[0].shape()[3];
    typename Tensor<real, Device>::Vector vec(outputs[0].shape().getElements(),
                                              outputs[0].data<real>());
    vec.zero();

    Pad<Device>(outputs[0].data<real>(),
                inputs[0].data<real>(),
                num,
                inC,
                inH,
                inW,
                pad_);
  }

 private:
  PadConf pad_;
};

/**
 * \brief The backward propagation of padding Function. Remove the elements
 *        in the padding positions of forward.
 *
 * Argument in this Function:
 * \param pad_    The same meaning as it in PadFunc.
 * \param inputs  The gradient with respect to the output value of PadFunc.
 * \param outputs The gradient with respect to the input value of PadFunc.
 */

template <DeviceType Device>
class PadGradFunc : public FunctionBase {
 public:
  void init(const FuncConfig& config) override { pad_ = castToPadConf(config); }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(1UL, inputs.size());
    CHECK_EQ(1UL, outputs.size());

    size_t num = outputs[0].shape()[0];
    size_t inC = outputs[0].shape()[1];
    size_t inH = outputs[0].shape()[2];
    size_t inW = outputs[0].shape()[3];

    if (outputs[0].getArgType() != ADD_TO) {
      // for unit test
      typename Tensor<real, Device>::Vector tmp(
          outputs[0].shape().getElements(), outputs[0].data<real>());
      tmp.zero();
    }

    PadGrad<Device>(outputs[0].data<real>(),
                    inputs[0].data<real>(),
                    num,
                    inC,
                    inH,
                    inW,
                    pad_);
  }

 private:
  PadConf pad_;
};

REGISTER_TYPED_FUNC(Pad, CPU, PadFunc);
REGISTER_TYPED_FUNC(PadGrad, CPU, PadGradFunc);
#ifdef PADDLE_WITH_CUDA
REGISTER_TYPED_FUNC(Pad, GPU, PadFunc);
REGISTER_TYPED_FUNC(PadGrad, GPU, PadGradFunc);
#endif

}  // namespace paddle
