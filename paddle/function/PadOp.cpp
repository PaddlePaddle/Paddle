/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
                          const int padc0,
                          const int padc1,
                          const int padh0,
                          const int padh1,
                          const int padw0,
                          const int padw1) {
  int outC = inC + padc0 + padc1;
  int outH = inH + padh0 + padh1;
  int outW = inW + padw0 + padw1;
  for (int i = 0; i < num; i++) {
    for (int c = 0; c < inC; c++) {
      for (int h = 0; h < inH; h++) {
        int inoff = ((i * inC + c) * inH + h) * inW;
        int outoff = ((i * outC + c + padc0) * outH + h + padh0) * outW + padw0;
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
                              const int padc0,
                              const int padc1,
                              const int padh0,
                              const int padh1,
                              const int padw0,
                              const int padw1) {
  int outC = inC + padc0 + padc1;
  int outH = inH + padh0 + padh1;
  int outW = inW + padw0 + padw1;
  for (int i = 0; i < num; i++) {
    for (int c = 0; c < inC; c++) {
      for (int h = 0; h < inH; h++) {
        int inoff = ((i * inC + c) * inH + h) * inW;
        int outoff = ((i * outC + c + padc0) * outH + h + padh0) * outW + padw0;
        CpuVector inG = CpuVector(inW, inGrad + inoff);
        CpuVector outG = CpuVector(inW, const_cast<real*>(outGrad + outoff));
        inG += outG;
      }
    }
  }
}

template <DeviceType Device>
class PadFunc : public FunctionBase {
public:
  void init(const FuncConfig& config) override {
    padc0_ = config.get<int>("padc0");
    padc1_ = config.get<int>("padc1");
    padh0_ = config.get<int>("padh0");
    padh1_ = config.get<int>("padh1");
    padw0_ = config.get<int>("padw0");
    padw1_ = config.get<int>("padw1");
  }

  /**
   * \param inputs[0] input value.
   * \param outputs[0] output value.
   */
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
                padc0_,
                padc1_,
                padh0_,
                padh1_,
                padw0_,
                padw1_);
  }

private:
  int padc0_;
  int padc1_;
  int padh0_;
  int padh1_;
  int padw0_;
  int padw1_;
};

template <DeviceType Device>
class PadGradFunc : public FunctionBase {
public:
  void init(const FuncConfig& config) override {
    padc0_ = config.get<int>("padc0");
    padc1_ = config.get<int>("padc1");
    padh0_ = config.get<int>("padh0");
    padh1_ = config.get<int>("padh1");
    padw0_ = config.get<int>("padw0");
    padw1_ = config.get<int>("padw1");
  }

  /**
   * \param inputs[0] output grad.
   * \param inouts[0] input grad.
   */
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
                    padc0_,
                    padc1_,
                    padh0_,
                    padh1_,
                    padw0_,
                    padw1_);
  }

private:
  int padc0_;
  int padc1_;
  int padh0_;
  int padh1_;
  int padw0_;
  int padw1_;
};

REGISTER_TYPED_FUNC(Pad, CPU, PadFunc);
REGISTER_TYPED_FUNC(PadGrad, CPU, PadGradFunc);
#ifndef PADDLE_ONLY_CPU
REGISTER_TYPED_FUNC(Pad, GPU, PadFunc);
REGISTER_TYPED_FUNC(PadGrad, GPU, PadGradFunc);
#endif

}  // namespace paddle
