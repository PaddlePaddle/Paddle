// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/add_n_kernel.h"

#include "paddle/phi/kernels/impl/add_n_kernel_impl.h"

#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"

namespace phi {

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

template <class T>
__global__ void Sum2CUDAKernel(const T *in_0,
                               const T *in_1,
                               T *out,
                               int64_t N) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < N) {
    out[id] = in_0[id] + in_1[id];
    id += blockDim.x * gridDim.x;
  }
}

template <class T>
__global__ void SumArrayCUDAKernel(
    T **in, T *out, int64_t N, size_t in_size, bool read_dst) {
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
__global__ void SumSelectedRowsCUDAKernel(T **sr_in_out,
                                          int64_t N,
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

template <typename T, typename Context>
void AddNKernel(const Context &dev_ctx,
                const std::vector<const TensorBase *> &x,
                DenseTensor *out) {
  const size_t in_num = x.size();

  constexpr size_t theory_sm_threads = 1024;
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
  auto *out_ptr = dev_ctx.template Alloc<T>(out);
  bool in_place = false;
  if (x.size() > 0 && x[0]->initialized() && DenseTensor::classof(x[0])) {
    if ((static_cast<const DenseTensor *>(x[0]))->data() == out->data()) {
      in_place = true;
    }
  }

  if (!in_place && in_num >= 1 && DenseTensor::classof(x[0])) {
    auto &in_0_tensor = *(static_cast<const DenseTensor *>(x[0]));
    if (in_0_tensor.numel() > 0) {
      in_place = (in_0_tensor.data<T>() == out_ptr);
    }
  }

  // Sum of two tensors
  if (in_num == 2 && DenseTensor::classof(x[0]) && DenseTensor::classof(x[1])) {
    auto &in_0 = *(static_cast<const DenseTensor *>(x[0]));
    auto &in_1 = *(static_cast<const DenseTensor *>(x[1]));
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
    phi::funcs::SetConstant<phi::GPUContext, T> constant_functor;
    constant_functor(dev_ctx, out, static_cast<T>(0));
  }

  std::vector<const T *> in_data;
  std::vector<int> selectrow_index;
  int64_t lod_length = 0;
  bool dst_write = false;
  for (int i = start; i < in_num; ++i) {
    if (DenseTensor::classof(x[i])) {
      auto &in_i = *(static_cast<const DenseTensor *>(x[i]));
      lod_length = in_i.numel();
      if (lod_length && in_i.IsInitialized()) {
        in_data.emplace_back(in_i.data<T>());
      }
    } else if (SelectedRows::classof(x[i])) {
      selectrow_index.push_back(i);
    }
  }

  // compute select rows separately.
  if (!selectrow_index.empty()) {
    std::vector<const T *> sr_in_out_data;
    size_t rows = 0;
    int64_t length = 0;
    for (auto index : selectrow_index) {
      auto &sr = *(static_cast<const SelectedRows *>(x[index]));
      auto &sr_value = sr.value();
      auto &sr_rows = sr.rows();

      auto row_numel = sr_value.numel() / sr_rows.size();
      auto out_dims = out->dims();

      PADDLE_ENFORCE_EQ(sr.height(),
                        out_dims[0],
                        errors::InvalidArgument(
                            "The table height of input must be same as output, "
                            "but received input height is %d"
                            ", output height is %d",
                            sr.height(),
                            out_dims[0]));
      PADDLE_ENFORCE_EQ(row_numel,
                        out->numel() / sr.height(),
                        errors::InvalidArgument(
                            "The table width of input must be same as output, "
                            "but received input width is %d"
                            ", output width is %d",
                            row_numel,
                            out->numel() / sr.height()));

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
      auto tmp_sr_in_out_array = paddle::memory::Alloc(
          dev_ctx.GetPlace(), sr_in_out_data.size() * sizeof(T *));

      paddle::memory::Copy(dev_ctx.GetPlace(),
                           tmp_sr_in_out_array->ptr(),
                           phi::CPUPlace(),
                           reinterpret_cast<void *>(sr_in_out_data.data()),
                           sr_in_out_data.size() * sizeof(T *),
                           dev_ctx.stream());

      T **sr_in_out_array_data =
          reinterpret_cast<T **>(tmp_sr_in_out_array->ptr());

      ComputeKernelParameter(length);
      SumSelectedRowsCUDAKernel<T>
          <<<grids, blocks, 0, stream>>>(sr_in_out_array_data, length, rows);
      dst_write = true;
    }
  }
  // if indata not null, merge into one kernel call.
  if (!in_data.empty()) {
    auto tmp_in_array =
        paddle::memory::Alloc(dev_ctx.GetPlace(), in_data.size() * sizeof(T *));

    paddle::memory::Copy(dev_ctx.GetPlace(),
                         tmp_in_array->ptr(),
                         phi::CPUPlace(),
                         reinterpret_cast<void *>(in_data.data()),
                         in_data.size() * sizeof(T *),
                         dev_ctx.stream());

    T **in_array_data = reinterpret_cast<T **>(tmp_in_array->ptr());
    ComputeKernelParameter(lod_length);
    SumArrayCUDAKernel<T><<<grids, blocks, 0, stream>>>(in_array_data,
                                                        out->data<T>(),
                                                        lod_length,
                                                        in_data.size(),
                                                        dst_write | in_place);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(add_n,
                   GPU,
                   ALL_LAYOUT,
                   phi::AddNKernel,
                   float,
                   double,
                   int,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   int64_t) {}

PD_REGISTER_KERNEL(add_n_array,
                   GPU,
                   ALL_LAYOUT,
                   phi::AddNArrayKernel,
                   float,
                   double,
                   int,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   int64_t) {}
