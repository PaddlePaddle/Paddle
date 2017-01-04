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

REGISTER_TYPED_FUNC(CosSimForward, CPU, CosSimForwardFunc);
#ifndef PADDLE_ONLY_CPU
REGISTER_TYPED_FUNC(CosSimForward, GPU, CosSimForwardFunc);
#endif
}  // namespace paddle
