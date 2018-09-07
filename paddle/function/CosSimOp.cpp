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

#include "CosSimOp.h"
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

/**
 * Cosine Similarity
 * for each row i,
 *   out[i] = scale * cos(input1[i], input2[i])
 *      = scale * <input1[i], input2[i]>/sqrt(|input1[i]|^2 * |input2[i]|^2)
 * when input2 only has one row, then for each row i,
 *   out[i] = cos(input1[i], input2[0])
 *
 * \param inputs[0] input matrix 1, size: nSamples * dim.
 * \param inputs[1] input matrix 2, size: n2 * dim (n2 == 1 or n2 == nSamples).
 * \param outputs[0] output matrix, size : nSamples * 1.
 */

template <DeviceType Device>
class CosSimForwardFunc : public FunctionBase {
  void init(const FuncConfig& config) override {
    scale_ = config.get<real>("scale");
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(inputs.size(), 2UL);
    CHECK_EQ(outputs.size(), 1UL);

    CHECK_EQ(inputs[0].shape().ndims(), 2UL);
    CHECK_EQ(inputs[1].shape().ndims(), 2UL);
    CHECK_EQ(outputs[0].shape().ndims(), 2UL);

    CHECK_EQ(inputs[0].shape()[0], outputs[0].shape()[0]);
    CHECK_EQ(inputs[0].shape()[1], inputs[1].shape()[1]);
    CHECK_EQ(outputs[0].shape()[1], 1UL);

    CHECK(outputs[0].data() && inputs[0].data() && inputs[1].data());

    CHECK_EQ(outputs[0].getArgType(), ASSIGN_TO);
    auto out_mat = outputs[0].matrix<Device>();
    const auto in1_mat = inputs[0].matrix<Device>();
    const auto in2_mat = inputs[1].matrix<Device>();

    CosSimForward<Device>(out_mat, in1_mat, in2_mat, scale_);
  }

 private:
  real scale_;
};

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

/**
 * Cosine Similarity backward Derivative
 *
 * \param outputs[0] forward input grad 1, size: nSamples * dim.
 * \param outputs[1] forward input grad 2,
 *                  size: n2 * dim (n2 == 1 or n2 == nSamples).
 *
 * \param inputs[0] backward loss output grad, size : nSamples * 1.
 * \param inputs[1] forward output value, size: nSamples * 1.
 * \param inputs[2] forward input value 1, size: nSamples * dim.
 * \param inputs[3] forward input value 2,
 *                  size: n2 * dim (n2 == 1 or n2 == nSamples).
 */
template <DeviceType Device>
class CosSimBackwardFunc : public FunctionBase {
  void init(const FuncConfig& config) override {
    scale_ = config.get<real>("scale");
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(inputs.size(), 4UL);
    CHECK_EQ(outputs.size(), 2UL);
    /// dim of out_grad and out_val == 1, column vector
    CHECK_EQ(inputs[0].shape()[1], 1UL);
    CHECK_EQ(inputs[1].shape()[1], 1UL);
    /// nSamples of out_grad == out_val == in_val1 == in_grad1
    CHECK_EQ(inputs[1].shape()[0], inputs[0].shape()[0]);
    CHECK_EQ(inputs[0].shape()[0], inputs[0].shape()[0]);
    CHECK_EQ(outputs[0].shape()[0], inputs[0].shape()[0]);
    /// dim of in1_val1 == in_val2 == in_grad1 == in_grad2
    CHECK_EQ(inputs[3].shape()[1], inputs[2].shape()[1]);
    CHECK_EQ(outputs[0].shape()[1], inputs[2].shape()[1]);
    CHECK_EQ(outputs[1].shape()[1], inputs[2].shape()[1]);

    CHECK(inputs[0].data() && inputs[1].data() && inputs[2].data() &&
          inputs[3].data() && outputs[0].data() && outputs[1].data());

    CHECK_EQ(outputs[0].getArgType(), ADD_TO);
    CHECK_EQ(outputs[1].getArgType(), ADD_TO);

    const auto out_grad = inputs[0].matrix<Device>();
    const auto out_val = inputs[1].matrix<Device>();
    const auto in1_val = inputs[2].matrix<Device>();
    const auto in2_val = inputs[3].matrix<Device>();
    auto in1_grad = outputs[0].matrix<Device>();
    auto in2_grad = outputs[1].matrix<Device>();

    CosSimBackward<Device>(
        out_grad, out_val, in1_val, in2_val, in1_grad, in2_grad, scale_);
  }

 private:
  real scale_;
};

REGISTER_TYPED_FUNC(CosSimForward, CPU, CosSimForwardFunc);
REGISTER_TYPED_FUNC(CosSimBackward, CPU, CosSimBackwardFunc);
#ifdef PADDLE_WITH_CUDA
REGISTER_TYPED_FUNC(CosSimForward, GPU, CosSimForwardFunc);
REGISTER_TYPED_FUNC(CosSimBackward, GPU, CosSimBackwardFunc);
#endif
}  // namespace paddle
