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

#include <paddle/fluid/platform/device_context.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/sum_op.h"
#include "paddle/fluid/platform/float16.h"

namespace plat = paddle::platform;

namespace paddle {
namespace operators {

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

using LoDTensor = framework::LoDTensor;

template <class T>
__global__ void Sum2CUDAKernel(const T *in_0, const T *in_1, T *out,
                               int64_t N) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < N) {
    out[id] = in_0[id] + in_1[id];
    id += blockDim.x * gridDim.x;
  }
}

template <class T>
__global__ void SumArrayCUDAKernel(T **in, T *out, int64_t N, size_t in_size,
                                   bool read_dst) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < N) {
    T total(read_dst ? out[id] : static_cast<T>(0));
    for (int i = 0; i < in_size; ++i) {
      const T *tmp = in[i];
      if (tmp) {
        total += tmp[id];
      }
    }
    out[id] = total;
    id += blockDim.x * gridDim.x;
  }
}

template <class T>
__global__ void SumSelectedRowsCUDAKernel(T **sr_in_out, int64_t N,
                                          size_t rows) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < N) {
    for (int i = 0; i < 2 * rows; i += 2) {
      const T *tmp = sr_in_out[i];
      T *tmp_out = sr_in_out[i + 1];
      if (tmp && tmp_out) {
        tmp_out[id] += tmp[id];
      }
    }
    id += blockDim.x * gridDim.x;
  }
}

template <class T>
void SumToLoDTensor(const framework::ExecutionContext &context) {
  auto in_vars = context.MultiInputVar("X");
  const size_t in_num = in_vars.size();

  constexpr size_t theory_sm_threads = 1024;
  auto &dev_ctx =
      context.template device_context<platform::CUDADeviceContext>();
  auto stream = dev_ctx.stream();

  auto max_threads = dev_ctx.GetMaxPhysicalThreadCount();
  auto sm_count = max_threads / theory_sm_threads;
  size_t tile_size = 0;
  dim3 grids;
  dim3 blocks;

  auto ComputeKernelParameter = [&](size_t length) {
    if (length >= max_threads)
      tile_size = 1024;
    else if (length < max_threads && length > sm_count * 128)
      tile_size = 512;
    else if (length <= sm_count * 128)
      tile_size = 256;
    grids = dim3(CEIL_DIV(length, tile_size), 1, 1);
    blocks = dim3(tile_size, 1, 1);
  };

  auto *out = context.Output<LoDTensor>("Out");
  bool in_place = in_vars[0] == context.OutputVar("Out");

  if (!in_place) {
    auto *out_ptr = out->mutable_data<T>(context.GetPlace());
    if (in_num >= 1 && in_vars[0]->IsType<framework::LoDTensor>()) {
      auto &in_0_tensor = in_vars[0]->Get<framework::LoDTensor>();
      if (in_0_tensor.numel() > 0) {
        in_place = (in_0_tensor.data<T>() == out_ptr);
      }
    }
  }

  // Sum of two tensors
  if (in_num == 2 && in_vars[0]->IsType<framework::LoDTensor>() &&
      in_vars[1]->IsType<framework::LoDTensor>()) {
    auto &in_0 = in_vars[0]->Get<framework::LoDTensor>();
    auto &in_1 = in_vars[1]->Get<framework::LoDTensor>();
    int64_t length_0 = in_0.numel();
    int64_t length_1 = in_1.numel();
    if (length_0 && length_1 && in_0.IsInitialized() && in_1.IsInitialized()) {
      auto result = EigenVector<T>::Flatten(*out);
      auto &place = *dev_ctx.eigen_device();
      auto in_0_e = EigenVector<T>::Flatten(in_0);
      auto in_1_e = EigenVector<T>::Flatten(in_1);
      result.device(place) = in_0_e + in_1_e;
    } else if (length_0 && in_0.IsInitialized()) {
      auto result = EigenVector<T>::Flatten(*out);
      auto &place = *dev_ctx.eigen_device();
      result.device(place) = EigenVector<T>::Flatten(in_0);
    } else if (length_1 && in_1.IsInitialized()) {
      auto result = EigenVector<T>::Flatten(*out);
      auto &place = *dev_ctx.eigen_device();
      result.device(place) = EigenVector<T>::Flatten(in_1);
    }
    return;
  }

  int start = in_place ? 1 : 0;
  if (!in_place) {
    math::SetConstant<platform::CUDADeviceContext, T> constant_functor;
    constant_functor(
        context.template device_context<platform::CUDADeviceContext>(), out,
        static_cast<T>(0));
  }

  std::vector<const T *> in_data;
  std::vector<int> selectrow_index;
  int64_t lod_length = 0;
  bool dst_write = false;
  for (int i = start; i < in_num; ++i) {
    if (in_vars[i]->IsType<framework::LoDTensor>()) {
      auto &in_i = in_vars[i]->Get<framework::LoDTensor>();
      lod_length = in_i.numel();
      if (lod_length && in_i.IsInitialized()) {
        in_data.emplace_back(in_i.data<T>());
      }
    } else if (in_vars[i]->IsType<framework::SelectedRows>()) {
      selectrow_index.push_back(i);
    }
  }

  // compute select rows seperately.
  if (!selectrow_index.empty()) {
    std::vector<const T *> sr_in_out_data;
    size_t rows = 0;
    int64_t length = 0;
    for (auto index : selectrow_index) {
      auto &sr = in_vars[index]->Get<framework::SelectedRows>();
      auto &sr_value = sr.value();
      auto &sr_rows = sr.rows();

      auto row_numel = sr_value.numel() / sr_rows.size();
      auto out_dims = out->dims();

      PADDLE_ENFORCE_EQ(sr.height(), out_dims[0],
                        platform::errors::InvalidArgument(
                            "The table height of input must be same as output, "
                            "but received input height is %d"
                            ", output height is %d",
                            sr.height(), out_dims[0]));
      PADDLE_ENFORCE_EQ(row_numel, out->numel() / sr.height(),
                        platform::errors::InvalidArgument(
                            "The table width of input must be same as output, "
                            "but received input width is %d"
                            ", output width is %d",
                            row_numel, out->numel() / sr.height()));

      auto *sr_data = sr_value.data<T>();
      auto *sr_out_data = out->data<T>();
      rows += sr_rows.size();
      length = row_numel;

      for (size_t i = 0; i < sr_rows.size(); ++i) {
        sr_in_out_data.emplace_back(&sr_data[i * row_numel]);
        sr_in_out_data.emplace_back(&sr_out_data[sr_rows[i] * row_numel]);
      }
    }
    if (!sr_in_out_data.empty()) {
      auto tmp_sr_in_out_array =
          memory::Alloc(dev_ctx, sr_in_out_data.size() * sizeof(T *));

      memory::Copy(dev_ctx.GetPlace(), tmp_sr_in_out_array->ptr(),
                   platform::CPUPlace(),
                   reinterpret_cast<void *>(sr_in_out_data.data()),
                   sr_in_out_data.size() * sizeof(T *), dev_ctx.stream());

      T **sr_in_out_array_data =
          reinterpret_cast<T **>(tmp_sr_in_out_array->ptr());

      ComputeKernelParameter(length);
      SumSelectedRowsCUDAKernel<T><<<grids, blocks, 0, stream>>>(
          sr_in_out_array_data, length, rows);
      dst_write = true;
    }
  }
  // if indata not null, merge into one kernel call.
  if (!in_data.empty()) {
    auto tmp_in_array = memory::Alloc(dev_ctx, in_data.size() * sizeof(T *));

    memory::Copy(dev_ctx.GetPlace(), tmp_in_array->ptr(), platform::CPUPlace(),
                 reinterpret_cast<void *>(in_data.data()),
                 in_data.size() * sizeof(T *), dev_ctx.stream());

    T **in_array_data = reinterpret_cast<T **>(tmp_in_array->ptr());
    ComputeKernelParameter(lod_length);
    SumArrayCUDAKernel<T><<<grids, blocks, 0, stream>>>(
        in_array_data, out->data<T>(), lod_length, in_data.size(),
        dst_write | in_place);
  }
}

template <typename T>
class SumKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto out_var = context.OutputVar("Out");

    if (out_var->IsType<framework::LoDTensor>()) {
      SumToLoDTensor<T>(context);
    } else if (out_var->IsType<framework::SelectedRows>()) {
      SelectedRowsCompute<platform::CUDADeviceContext, T>(context);
    } else if (out_var->IsType<framework::LoDTensorArray>()) {
      LodTensorArrayCompute<platform::CUDADeviceContext, T>(context);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Expected type of Ouput(out) must be Tensor,  SelectedRows or "
          "LodTensorArray. But got "
          "unsupport type: %s.",
          framework::ToTypeName(out_var->Type())));
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    sum, ops::SumKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SumKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SumKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SumKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::SumKernel<paddle::platform::CUDADeviceContext, plat::float16>);
