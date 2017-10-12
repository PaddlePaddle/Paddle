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

#pragma once
#include "paddle/framework/tensor.h"
#include "paddle/memory/memcpy.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace framework {

class SelectedRows {
 public:
  SelectedRows(const std::vector<int64_t>& rows, const int64_t& height)
      : rows_(rows), height_(height) {
    value_.reset(new Tensor());
  }

  SelectedRows() {}

  platform::Place place() const { return value_->place(); }

  Tensor& value() const { return *value_; }

  int64_t height() const { return height_; }

  void set_height(int64_t height) { height_ = height; }

  const std::vector<int64_t>& rows() const { return rows_; }

  void set_rows(const std::vector<int64_t>& rows) { rows_ = rows; }

  DDim GetCompleteDims() const {
    std::vector<int64_t> dims = vectorize(value_->dims());
    dims[0] = height_;
    return make_ddim(dims);
  }

 private:
  std::vector<int64_t> rows_;
  std::unique_ptr<Tensor> value_{nullptr};
  int64_t height_;
};

template <typename T>
void SelectedRowsToTensor(const SelectedRows& input,
                          const platform::Place& dst_place,
                          const platform::DeviceContext& ctx, Tensor* output) {
  std::vector<int64_t> input_dims = vectorize(input.value().dims());
  input_dims[0] = input.height();
  output->mutable_data<T>(make_ddim(input_dims), dst_place);
  auto src_place = input.place();
  auto rows = input.rows();

  auto row_numel = input.value().numel() / input.rows().size();

  if (platform::is_cpu_place(src_place) && platform::is_cpu_place(dst_place)) {
    operators::math::SetConstant<platform::CPUPlace, T>(ctx, output,
                                                        static_cast<T>(0.0));
    auto src_cpu_place = boost::get<platform::CPUPlace>(src_place);
    auto dst_cpu_place = boost::get<platform::CPUPlace>(dst_place);

    for (size_t i = 0; i < rows.size(); i++) {
      memory::Copy(dst_cpu_place, output->data<T>() + rows[i] * row_numel,
                   src_cpu_place, input.value().data<T>() + i * row_numel,
                   row_numel * sizeof(T));
    }
#ifdef PADDLE_WITH_CUDA
  } else if (platform::is_gpu_place(src_place) &&
             platform::is_cpu_place(dst_place)) {
    auto src_gpu_place = boost::get<platform::GPUPlace>(src_place);
    auto dst_cpu_place = boost::get<platform::CPUPlace>(dst_place);
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx_place));
    auto ctx_gpu_place = boost::get<platform::GPUPlace>(ctx_place);
    PADDLE_ENFORCE_EQ(src_gpu_place, ctx_gpu_place);

    operators::math::SetConstant<platform::CPUPlace, T>(ctx, output,
                                                        static_cast<T>(0.0));

    for (size_t i = 0; i < rows.size(); i++) {
      memory::Copy(
          dst_cpu_place, output->data<T>() + rows[i] * row_numel, src_gpu_place,
          input.value().data<T>() + i * row_numel, row_numel * sizeof(T),
          reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
    }
  } else if (platform::is_cpu_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    auto src_cpu_place = boost::get<platform::CPUPlace>(src_place);
    auto dst_gpu_place = boost::get<platform::GPUPlace>(dst_place);
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx_place));
    auto ctx_gpu_place = boost::get<platform::GPUPlace>(ctx_place);
    PADDLE_ENFORCE_EQ(dst_gpu_place, ctx_gpu_place);
    operators::math::SetConstant<platform::GPUPlace, T>(ctx, output,
                                                        static_cast<T>(0.0));

    for (size_t i = 0; i < rows.size(); i++) {
      memory::Copy(
          dst_gpu_place, output->data<T>() + rows[i] * row_numel, src_cpu_place,
          input.value().data<T>() + i * row_numel, row_numel * sizeof(T),
          reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
    }
  } else if (platform::is_gpu_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    auto src_gpu_place = boost::get<platform::GPUPlace>(src_place);
    auto dst_gpu_place = boost::get<platform::GPUPlace>(dst_place);
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx_place));
    auto ctx_gpu_place = boost::get<platform::GPUPlace>(ctx_place);
    PADDLE_ENFORCE_EQ(src_gpu_place, ctx_gpu_place);
    operators::math::SetConstant<platform::GPUPlace, T>(ctx, output,
                                                        static_cast<T>(0.0));
    for (size_t i = 0; i < rows.size(); i++) {
      memory::Copy(
          dst_gpu_place, output->data<T>() + rows[i] * row_numel, src_gpu_place,
          input.value().data<T>() + i * row_numel, row_numel * sizeof(T),
          reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
    }
#endif
  }
}

}  // namespace framework
}  // namespace paddle
