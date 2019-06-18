// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/types.h"
#include "paddle/fluid/lite/operators/conv_op.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/depthwise_conv.h"
#include "paddle/fluid/operators/math/im2col.h"
#include "paddle/fluid/operators/math/vol2col.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

inline bool IsExpand(const std::vector<int64_t>& filter_dim,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations) {
  bool filter_1 = true, strides_1 = true, padding_0 = true, dilation_1 = true;
  for (size_t j = 0; j < strides.size(); ++j) {
    filter_1 = filter_1 && (static_cast<int>(filter_dim[j + 2]) == 1);
    strides_1 = strides_1 && (strides[j] == 1);
    padding_0 = padding_0 && (paddings[j] == 0);
    dilation_1 = dilation_1 && (dilations[j] == 1);
  }
  return !(filter_1 && strides_1 && padding_0 && dilation_1);
}

template <typename T>
class Conv2dCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ConvParam;
  void Run() override {
    auto& param = *param_.get_mutable<operators::ConvParam>();
    lite::Tensor filter = *param.filter;
    param.output->template mutable_data<T>();

    const int batch_size = static_cast<int>(param.x->dims()[0]);

    std::vector<int64_t> filter_shape_vec(filter.dims().Vectorize());
    std::vector<int64_t> output_shape_vec(param.output->dims().Vectorize());

    size_t data_dim = filter_shape_vec.size() - 2;
    std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
    col_shape_vec[0] = param.x->dims()[1] / param.groups;
    for (size_t j = 0; j < data_dim; ++j) {
      col_shape_vec[j + 1] = filter_shape_vec[j + 2];
      col_shape_vec[j + 1 + data_dim] = output_shape_vec[j + 2];
    }
    lite::DDim col_shape(col_shape_vec);
    lite::DDim col_matrix_shape = col_shape.Flattern2D(data_dim + 1);
    bool is_expand = IsExpand(filter_shape_vec, param.strides, param.paddings,
                              param.dilations);

    lite::Tensor col;
    lite::Tensor col_matrix;
    if (is_expand) {
      col.Resize(col_shape);
      col.mutable_data<T>();
      col_matrix.ShareDataWith(col);
      col_matrix.Resize(col_matrix_shape);
    }
    lite::DDim input_shape = param.x->dims().Slice(1, param.x->dims().size());

    lite::DDim filter_matrix_shape(std::vector<int64_t>{
        filter.dims()[0], filter.dims().production() / filter.dims()[0]});
    filter.Resize(filter_matrix_shape);

    lite::DDim output_matrix_shape(std::vector<int64_t>{
        param.output->dims()[1],
        param.output->dims().production() /
            (param.output->dims()[0] * param.output->dims()[1])});

    int in_step = static_cast<int>(param.x->dims()[1]) / param.groups;
    int out_step = static_cast<int>(param.output->dims()[1]) / param.groups;

    paddle::operators::math::Vol2ColFunctor<platform::CPUDeviceContext, T>
        vol2col;
    paddle::operators::math::Im2ColFunctor<
        paddle::operators::math::ColFormat::kCFO, platform::CPUDeviceContext, T>
        im2col;
    auto blas = paddle::operators::math::GetBlas<platform::CPUDeviceContext, T>(
        platform::CPUDeviceContext());
    for (int i = 0; i < batch_size; i++) {
      lite::Tensor in_batch;
      in_batch.ShareDataWith(
          param.x->raw_tensor().Slice(i, i + 1).Resize(input_shape.data()));
      lite::Tensor out_batch;
      out_batch.ShareDataWith(param.output->raw_tensor().Slice(i, i + 1).Resize(
          output_matrix_shape.data()));

      for (int g = 0; g < param.groups; g++) {
        lite::Tensor in_slice;
        in_slice.ShareDataWith(
            in_batch.raw_tensor().Slice(g * in_step, (g + 1) * in_step));

        if (!is_expand) {
          col.ShareDataWith(in_slice);
          col_matrix.ShareDataWith(col);
          col_matrix.Resize(col_matrix_shape);
        } else if (data_dim == 2U) {
          // im2col
          im2col(platform::CPUDeviceContext(), in_slice.raw_tensor(),
                 param.dilations, param.strides,
                 std::vector<int>{param.paddings[0], param.paddings[1],
                                  param.paddings[0], param.paddings[1]},
                 &(col.raw_tensor()));
        } else if (data_dim == 3U) {
          // vol2col
          vol2col(platform::CPUDeviceContext(), in_slice.raw_tensor(),
                  param.dilations, param.strides, param.paddings,
                  &(col.raw_tensor()));
        }

        // gemm
        lite::Tensor out_slice;
        out_slice.ShareDataWith(
            out_batch.raw_tensor().Slice(g * out_step, (g + 1) * out_step));
        lite::Tensor filter_slice;
        filter_slice.ShareDataWith(
            filter.raw_tensor().Slice(g * out_step, (g + 1) * out_step));
        blas.MatMul(filter_slice.raw_tensor(), false, col_matrix.raw_tensor(),
                    false, T(1.0), &(out_slice.raw_tensor()), T(0.0));
      }
    }
  }

  virtual ~Conv2dCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
