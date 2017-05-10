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

#include "CosSimOp.h"
#include "Register.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/Vector.h"

namespace paddle {
/**
 * Cosine Similarity for CpuMatrix
 *
 * \param out_mat, output value, size: nSamples * 1.
 * \param in1_mat, input value 1, size: nSamples * dim.
 * \param in2_mat, input value 2, size: n2 * dim (n2 == 1 or n2 == nSamples).
 * \param scale, default 1.0
 *
 */
template <>
void CosSimForward<DEVICE_TYPE_CPU>(CpuMatrix& out_mat,
                                    const CpuMatrix& in1_mat,
                                    const CpuMatrix& in2_mat,
                                    real scale) {
  CHECK(out_mat.getData() && in1_mat.getData() && in2_mat.getData());
  size_t num_samples = out_mat.getHeight();
  size_t dim = in1_mat.getWidth();
  /// column vector [nSamples, 1]
  real* out = out_mat.getData();
  const real* x = in1_mat.getData();
  const real* y = in2_mat.getData();

  /// in2 might only have one row or full rows
  CHECK(in2_mat.getHeight() == 1LU || in2_mat.getHeight() == num_samples);
  size_t inc = (in2_mat.getHeight() == 1LU) ? 0 : dim;
  for (size_t i = 0; i < num_samples; ++i, x += dim, y += inc) {
    real square_sum_x = 0;
    real square_sum_y = 0;
    real xy = 0;
    for (size_t j = 0; j < dim; ++j) {
      square_sum_x += x[j] * x[j];
      square_sum_y += y[j] * y[j];
      xy += x[j] * y[j];
    }
    CHECK(square_sum_x > 0 && square_sum_y > 0);
    out[i] = scale * xy / (std::sqrt(square_sum_x) * std::sqrt(square_sum_y));
  }
}
template <DeviceType Device>
Error cosineForward(const BufferArgs& inputs,
                    const BufferArgs& outputs,
                    const std::unordered_map<std::string, any>& attrs) {
  auto out_mat = outputs[0].matrix<Device>();
  const auto in1_mat = inputs[0].matrix<Device>();
  const auto in2_mat = inputs[1].matrix<Device>();

  CosSimForward<Device>(
      out_mat, in1_mat, in2_mat, any_cast<double>(attrs.at("scale")));
  return Error();
}

BEGIN_REGISTER_FUNCTION(cosFwd, cosineForward)
func->addAttribute<double>("scale", "The scale of cosine operator")
    .defaultValue(1.0)
    .largerThan(0.0);

func->addInput()                                // first input
    ->addDataType({topology::DataType::DENSE})  // only support dense as input
    .addSequenceType()                          // could be any sequence type
    .addShape(2);                               // dimension is 2

func->addInput()                                // second input
    ->addDataType({topology::DataType::DENSE})  // only support dense as input
    .addSequenceType()                          // could be any sequence type
    .addShape(2);                               // dimension is 2

func->addOutput()
    ->addDataType({topology::DataType::DENSE})
    .addSequenceType()
    .addShape(2)
    .addArgType(ASSIGN_TO);

func->setShapeInferer([](std::vector<topology::TensorPtr>& ins,
                         std::vector<topology::TensorPtr>& outs) {
  auto& shape0 = ins[0]->shape();
  auto& shape1 = ins[1]->shape();

  if (shape0 != shape1 && (shape0[1] != shape1[1] || shape1[0] != 1))
    return Error(
        "Input shape should be same, or the second height should be 1");
  if (ins[0]->sequenceType() != ins[1]->sequenceType())
    return Error("Input sequence type should be same");
  outs[0]->setShape({ins[0]->shape()[0], 1});
  outs[0]->setSequenceType(ins[0]->sequenceType());
  outs[0]->setDataType(ins[0]->dataType());
  return Error();
});

END_REGISTER_FUNCTION()

/**
 * Cosine Similarity Derivative for CpuMatrix
 *
 * \param in1_grad  forward input grad 1, size: nSamples * dim.
 * \param in2_grad  forward input grad 2,
 *                  size: n2 * dim (n2 == 1 or n2 == nSamples).
 *
 * \param out_grad  backward loss output grad, size : nSamples * 1.
 * \param out_val   forward output value, size: nSamples * 1.
 * \param in1_val   forward input value 1, size: nSamples * dim.
 * \param in2_val   forward input value 2,
 *                  size: n2 * dim (n2 == 1 or n2 == nSamples).
 * \param scale,    default 1.0
 */
template <>
void CosSimBackward<DEVICE_TYPE_CPU>(const CpuMatrix& out_grad,
                                     const CpuMatrix& out_val,
                                     const CpuMatrix& in1_val,
                                     const CpuMatrix& in2_val,
                                     CpuMatrix& in1_grad,
                                     CpuMatrix& in2_grad,
                                     real scale) {
  CHECK(out_grad.getData() && out_val.getData() && in1_val.getData() &&
        in2_val.getData() && in1_grad.getData() && in2_grad.getData());
  CHECK_EQ(out_val.useGpu_, false) << "Matrix type are GPU, CPU required";

  const real* grad = out_grad.getData();
  const real* out = out_val.getData();
  const real* prev_out_x = in1_val.getData();
  const real* prev_out_y = in2_val.getData();
  real* prev_grad_x = in1_grad.getData();
  real* prev_grad_y = in2_grad.getData();

  size_t num_samples = out_grad.getHeight();
  size_t dim = in1_val.getWidth();
  CHECK_EQ(in2_val.getHeight(), in2_grad.getHeight());
  CHECK(in2_val.getHeight() == 1LU || in2_val.getHeight() == num_samples);
  size_t inc = (in2_val.getHeight() == 1LU) ? 0 : dim;
  for (size_t i = 0; i < num_samples; ++i,
              prev_out_x += dim,
              prev_out_y += inc,
              prev_grad_x += dim,
              prev_grad_y += inc) {
    real square_sum_x = 0;
    real square_sum_y = 0;
    real xy = 0;
    for (size_t j = 0; j < dim; ++j) {
      square_sum_x += prev_out_x[j] * prev_out_x[j];
      square_sum_y += prev_out_y[j] * prev_out_y[j];
      xy += prev_out_x[j] * prev_out_y[j];
    }
    CHECK(square_sum_x > 0 && square_sum_y > 0);
    if (xy == 0) {
      real reciprocal =
          1.0f / (std::sqrt(square_sum_x) * std::sqrt(square_sum_y));
      for (size_t j = 0; j < dim; ++j) {
        prev_grad_x[j] += scale * grad[i] * prev_out_y[j] * reciprocal;
        prev_grad_y[j] += scale * grad[i] * prev_out_x[j] * reciprocal;
      }
    } else {
      real reciprocal_xy = 1.0f / xy;
      real reciprocal_square_sum_x = 1.0f / square_sum_x;
      real reciprocal_square_sum_y = 1.0f / square_sum_y;
      for (size_t j = 0; j < dim; ++j) {
        prev_grad_x[j] +=
            out[i] * grad[i] * (prev_out_y[j] * reciprocal_xy -
                                prev_out_x[j] * reciprocal_square_sum_x);
        prev_grad_y[j] +=
            out[i] * grad[i] * (prev_out_x[j] * reciprocal_xy -
                                prev_out_y[j] * reciprocal_square_sum_y);
      }
    }
  }
}

template <DeviceType Device>
Error cosBackward(const BufferArgs& ins,
                  const BufferArgs& outs,
                  const std::unordered_map<std::string, any>& attrs) {
  const auto out_grad = ins[0].matrix<Device>();
  const auto out_val = ins[1].matrix<Device>();
  const auto in1_val = ins[2].matrix<Device>();
  const auto in2_val = ins[3].matrix<Device>();
  auto in1_grad = outs[0].matrix<Device>();
  auto in2_grad = outs[1].matrix<Device>();

  CosSimBackward<Device>(out_grad,
                         out_val,
                         in1_val,
                         in2_val,
                         in1_grad,
                         in2_grad,
                         any_cast<double>(attrs.at("scale")));
  return Error();
}

BEGIN_REGISTER_FUNCTION(cosBwd, cosBackward)
func->addAttribute<double>("scale", "the scale of cosine operator")
    .defaultValue(1.0)
    .largerThan(0.0);

auto widthShouldBeOne = [](std::vector<int>* attr, bool) {
  if (attr->at(1) != 1) return Error("width should be 1");
  return Error();
};

topology::meta::Constraints<std::vector<int>>* shapeConstraints;
func->addInput()
    ->addDataType({topology::DataType::DENSE})
    .addShape(2, &shapeConstraints)
    .addSequenceType();

shapeConstraints->addConstraint(widthShouldBeOne);

func->addInput()
    ->addDataType({topology::DataType::DENSE})
    .addShape(2, &shapeConstraints)
    .addSequenceType();
shapeConstraints->addConstraint(widthShouldBeOne);

func->addInput()
    ->addDataType({topology::DataType::DENSE})
    .addShape(2)
    .addSequenceType();
func->addInput()
    ->addDataType({topology::DataType::DENSE})
    .addShape(2)
    .addSequenceType();

func->addOutput()
    ->addDataType({topology::DataType::DENSE})
    .addShape(2)
    .addSequenceType()
    .addArgType(ADD_TO);
func->addOutput()
    ->addDataType({topology::DataType::DENSE})
    .addShape(2)
    .addSequenceType()
    .addArgType(ADD_TO);

func->setShapeInferer([](std::vector<topology::TensorPtr>& ins,
                         std::vector<topology::TensorPtr>& outs) -> Error {
  if (ins[0]->shape() != ins[1]->shape() ||
      ins[2]->shape()[1] != ins[3]->shape()[1]) {
    return Error("Input shape mismatch");
  }

  if (ins[0]->shape()[0] != ins[2]->shape()[0]) {
    return Error("Input shape mismatch, height should be same.");
  }

  for (size_t i = 0; i < outs.size(); ++i) {
    auto& out = outs[i];
    out->setShape(ins[2 + i]->shape());
    out->setSequenceType(ins[2 + i]->sequenceType());
    out->setDataType(ins[2 + i]->dataType());
  }

  return Error();
});

END_REGISTER_FUNCTION()
}  // namespace paddle
