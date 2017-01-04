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
#include "paddle/math/Matrix.h"
#include "paddle/math/Vector.h"

namespace paddle {
template <>
void CosSimForward<DEVICE_TYPE_CPU>(CpuMatrix* out_mat,
                                    const CpuMatrix* in1_mat,
                                    const CpuMatrix* in2_mat,
                                    real scale) {
  CHECK(out_mat && in1_mat && in2_mat);
  size_t num_samples = out_mat->getHeight();
  size_t dim = in1_mat->getWidth();
  /// column vector [nSamples, 1]
  real* out = out_mat->getData();
  const real* x = in1_mat->getData();
  const real* y = in2_mat->getData();

  /// in2 might only have one row or full rows
  CHECK(in2_mat->getHeight() == 1LU || in2_mat->getHeight() == num_samples);
  size_t inc = (in2_mat->getHeight() == 1LU) ? 0 : dim;
  for (size_t i = 0; i < num_samples; ++i, x += dim, y += inc) {
    /// for each row, todo(tianbing), use TensorExpression square2 ?
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
 * \param inputs[0] input matrix 1, size: nSamples * dim.
 * \param inputs[1] input matrix 2, size: n2 * dim (n2 == 1 or n2 == nSamples).
 * \param outputs[0] output matrix, size : nSamples * 1.
 */

template <DeviceType Device>
class CosSimForwardFunc : public FunctionBase {
  void init(const FuncConfig& config) override {
    scale_ = config.get<real>("scale");
  }

  void calc(const Arguments& inputs,
            const Arguments& outputs,
            const Arguments& inouts) override {
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    CHECK_EQ(inouts.size(), 0);

    CHECK_EQ(inputs[0].dims_[0], outputs[0].dims_[0]);
    CHECK_EQ(inputs[0].dims_[1], inputs[1].dims_[1]);
    CHECK_EQ(outputs[0].dims_[1], 1UL);

    CHECK(outputs[0].getData() && inputs[0].getData() && inputs[1].getData());
    auto out_mat = std::make_shared<typename MatrixT<Device>::type>(
        outputs[0].getData(), outputs[0].dims_[0], outputs[0].dims_[1]);
    const auto in1_mat = std::make_shared<typename MatrixT<Device>::type>(
        inputs[0].getData(), inputs[0].dims_[0], inputs[0].dims_[1]);
    const auto in2_mat = std::make_shared<typename MatrixT<Device>::type>(
        inputs[1].getData(), inputs[1].dims_[0], inputs[1].dims_[1]);

    CosSimForward<Device>(out_mat.get(), in1_mat.get(), in2_mat.get(), scale_);
  }

private:
  real scale_;
};

template <>
void CosSimBackward<DEVICE_TYPE_CPU>(const CpuMatrix* out_grad,
                                     const CpuMatrix* out_val,
                                     const CpuMatrix* in1_val,
                                     const CpuMatrix* in2_val,
                                     CpuMatrix* in1_grad,
                                     CpuMatrix* in2_grad,
                                     real scale) {
  CHECK(out_grad && out_val && in1_val && in2_val && in1_grad && in2_grad);
  CHECK_EQ(out_val->useGpu_, false) << "Matrix type are GPU, CPU required";

  const real* grad = out_grad->getData();
  const real* out = out_val->getData();
  const real* prev_out_x = in1_val->getData();
  const real* prev_out_y = in2_val->getData();
  real* prev_grad_x = in1_grad->getData();
  real* prev_grad_y = in2_grad->getData();

  size_t num_samples = out_grad->getHeight();
  size_t dim = in1_val->getWidth();
  CHECK_EQ(in2_val->getHeight(), in2_grad->getHeight());
  CHECK(in2_val->getHeight() == 1LU || in2_val->getHeight() == num_samples);
  size_t inc = (in2_val->getHeight() == 1LU) ? 0 : dim;
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
 * \param inputs[0] output value 1, size: nSamples * 1.
 * \param inputs[1] input value 1, size: nSamples * dim.
 * \param inputs[2] input value 2, size: n2 * dim (n2 == 1 or n2 == nSamples).
 * \param inputs[3] input grad 1, size: nSamples * dim.
 * \param inputs[4] input grad 2, size: n2 * dim (n2 == 1 or n2 == nSamples).
 * \param outputs[0] output grad, size : nSamples * 1.
 */
template <DeviceType Device>
class CosSimBackwardFunc : public FunctionBase {
  void init(const FuncConfig& config) override {
    scale_ = config.get<real>("scale");
  }

  void calc(const Arguments& inputs,
            const Arguments& outputs,
            const Arguments& inouts) override {
    CHECK_EQ(inputs.size(), 5);
    CHECK_EQ(outputs.size(), 1);
    CHECK_EQ(inouts.size(), 0);
    /// dim of out_grad and out_val == 1, column vector
    CHECK_EQ(outputs[0].dims_[1], 1UL);
    CHECK_EQ(inputs[0].dims_[1], 1UL);
    /// nSamples of out_grad == out_val == in_val1 == in_grad1
    CHECK_EQ(inputs[0].dims_[0], outputs[0].dims_[0]);
    CHECK_EQ(inputs[1].dims_[0], outputs[0].dims_[0]);
    CHECK_EQ(inputs[3].dims_[0], outputs[0].dims_[0]);
    /// dim of in1_val1 == in_val2 == in_grad1 == in_grad2
    CHECK_EQ(inputs[2].dims_[1], inputs[1].dims_[1]);
    CHECK_EQ(inputs[3].dims_[1], inputs[1].dims_[1]);
    CHECK_EQ(inputs[4].dims_[1], inputs[1].dims_[1]);

    CHECK(outputs[0].getData() && inputs[0].getData() && inputs[1].getData() &&
          inputs[2].getData() && inputs[3].getData() && inputs[4].getData());
    const auto out_grad = std::make_shared<typename MatrixT<Device>::type>(
        outputs[0].getData(), outputs[0].dims_[0], outputs[0].dims_[1]);
    const auto out_val = std::make_shared<typename MatrixT<Device>::type>(
        inputs[0].getData(), inputs[0].dims_[0], inputs[0].dims_[1]);
    const auto in1_val = std::make_shared<typename MatrixT<Device>::type>(
        inputs[1].getData(), inputs[1].dims_[0], inputs[1].dims_[1]);
    const auto in2_val = std::make_shared<typename MatrixT<Device>::type>(
        inputs[2].getData(), inputs[2].dims_[0], inputs[2].dims_[1]);
    auto in1_grad = std::make_shared<typename MatrixT<Device>::type>(
        inputs[3].getData(), inputs[3].dims_[0], inputs[3].dims_[1]);
    auto in2_grad = std::make_shared<typename MatrixT<Device>::type>(
        inputs[4].getData(), inputs[4].dims_[0], inputs[4].dims_[1]);

    CosSimBackward<Device>(out_grad.get(),
                           out_val.get(),
                           in1_val.get(),
                           in2_val.get(),
                           in1_grad.get(),
                           in2_grad.get(),
                           scale_);
  }

private:
  real scale_;
};

REGISTER_TYPED_FUNC(CosSimForward, CPU, CosSimForwardFunc);
REGISTER_TYPED_FUNC(CosSimBackward, CPU, CosSimBackwardFunc);
#ifndef PADDLE_ONLY_CPU
REGISTER_TYPED_FUNC(CosSimForward, GPU, CosSimForwardFunc);
REGISTER_TYPED_FUNC(CosSimBackward, GPU, CosSimBackwardFunc);
#endif
}  // namespace paddle
