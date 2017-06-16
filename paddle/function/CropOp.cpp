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

#include "CropOp.h"
#include "paddle/math/Vector.h"
#include "paddle/function/TensorShape.h"
namespace paddle {

static inline CropConf castToCropConf(const FuncConfig& conf) {
  return {conf.get<std::vector<uint32_t>>("crop_corner"),
          conf.get<std::vector<uint32_t>>("crop_shape")};
}

template <>
void Crop<DEVICE_TYPE_CPU>(real* outputs,
                           const real* inputs,
                           const TensorShape inShape,
                           const CropConf& crop) {
  int cCrop = crop.corner[0];
  int hCrop = crop.corner[1];
  int wCrop = crop.corner[2];

  int num = inShape[0];
  int inC = inShape[1];
  int inH = inShape[2];
  int inW = inShape[3];

  int outC = crop.shape[0];
  int outH = crop.shape[1];
  int outW = crop.shape[2];

  for (int n = 0; n < num; n++) {
    for (int c = 0; c < outC; c++) {
      for (int h = 0; h < outH; h++) {
        int outoff = ((n * outC + c) * outH + h) * outW;
        int inoff = ((n * inC + c + cCrop) * inH + h + hCrop) * inW + wCrop;
        memcpy(outputs + outoff, inputs + inoff, outW * sizeof(real));
      }
    }
  }
}

template <>
void CropGrad<DEVICE_TYPE_CPU>(const real* inGrad,
                               real* outGrad,
                               const TensorShape outShape,
                               const CropConf& crop) {
  int cCrop = crop.corner[0];
  int hCrop = crop.corner[1];
  int wCrop = crop.corner[2];

  int num = outShape[0];
  int outC = outShape[1];
  int outH = outShape[2];
  int outW = outShape[3];

  int inC = crop.shape[0];
  int inH = crop.shape[1];
  int inW = crop.shape[2];

  for (int n = 0; n < num; n++) {
    for (int c = 0; c < inC; c++) {
      for (int h = 0; h < inH; h++) {
        int outoff = ((n * outC + c + cCrop) * outH + h + hCrop) * outW + wCrop;
        int inoff = ((n * inC + c) * inH + h) * inW;
        CpuVector inG = CpuVector(inW, const_cast<real*>(inGrad + inoff));
        CpuVector outG = CpuVector(inW, outGrad + outoff);
        outG += inG;
      }
    }
  }
}

/**
 * \brief Crop input according to the specify corner and shape.
 *        The input and output is a 4D tensor. In CropFunc, we only
 *        crop the 2nd to 4th dimension.
 *
 * Argument in this Function:
 * \param pad_    A struct object contains the cropping corner and shape.
 * \param inputs  A 4D tensor, only one input.
 * \param outputs A 4D tensor, the output value after cropping.
 *
 * For example,
 * Input(2,2,2,3) = [
 *                    [ [[1,2,3], [3,4,5]],
 *                      [[2,3,5], [1,6,7]] ],
 *                    [ [[4,3,1], [1,8,7]],
 *                      [[3,8,9], [2,3,5]] ]
 *                  ] # the input shape is (2,2,2,3)
 *
 * pad_: if corner = (0,1,1) and crop_shape = (2,1,2)
 * Output(2,2,1,2) = [
 *                    [ [[4,5]],
 *                      [[6,7]] ],
 *                    [ [[8,7]],
 *                      [[3,5]] ]
 *                  ] # the input shape is (2,2,2,3)
 */
template <DeviceType Device>
class CropFunc : public FunctionBase {
public:
  void init(const FuncConfig& config) override {
    crop_ = castToCropConf(config);
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(1UL, inputs.size());
    CHECK_EQ(1UL, outputs.size());
    CHECK_EQ(outputs[0].shape()[1], crop_.shape[0]);
    CHECK_EQ(outputs[0].shape()[2], crop_.shape[1]);
    CHECK_EQ(outputs[0].shape()[3], crop_.shape[2]);
    CHECK_EQ(outputs[0].getArgType(), ASSIGN_TO);

    TensorShape inShape = inputs[0].shape();

    Crop<Device>(
        outputs[0].data<real>(), inputs[0].data<real>(), inShape, crop_);
  }

private:
  CropConf crop_;
};

/**
 * \brief The backward propagation of cropping Function.
 *
 * Argument in this Function:
 * \param crop_    The same meaning as it in CropFunc.
 * \param inputs  The gradient with respect to the output value of CropFunc.
 * \param outputs The gradient with respect to the input value of CropFunc.
 */

template <DeviceType Device>
class CropGradFunc : public FunctionBase {
public:
  void init(const FuncConfig& config) override {
    crop_ = castToCropConf(config);
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(1UL, inputs.size());
    CHECK_EQ(1UL, outputs.size());
    CHECK_EQ(inputs[0].shape()[1], crop_.shape[0]);
    CHECK_EQ(inputs[0].shape()[2], crop_.shape[1]);
    CHECK_EQ(inputs[0].shape()[3], crop_.shape[2]);
    CHECK_EQ(outputs[0].getArgType(), ASSIGN_TO);

    TensorShape outShape = outputs[0].shape();

    CropGrad<Device>(
        inputs[0].data<real>(), outputs[0].data<real>(), outShape, crop_);
  }

private:
  CropConf crop_;
};

REGISTER_TYPED_FUNC(Crop, CPU, CropFunc);
REGISTER_TYPED_FUNC(CropGrad, CPU, CropGradFunc);
#ifndef PADDLE_ONLY_CPU
REGISTER_TYPED_FUNC(Crop, GPU, CropFunc);
REGISTER_TYPED_FUNC(CropGrad, GPU, CropGradFunc);
#endif

}  // namespace paddle
