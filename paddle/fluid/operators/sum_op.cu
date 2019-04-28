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
#include "paddle/fluid/operators/sum_op.h"
#include "paddle/fluid/platform/float16.h"

namespace plat = paddle::platform;

namespace paddle {
namespace operators {

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <class T>
__global__ void sum_gpu(const T *in_0, const T *in_1, T *out, int64_t N) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < N) {
    out[id] = in_0[id] + in_1[id];
    id += blockDim.x * gridDim.x;
  }
}

template <class T>
__global__ void sum_gpu_array(T **in, T *out, int64_t N, size_t in_size,
                              bool read_dst) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < N) {
    T total(0);
    for (int i = 0; i < in_size; ++i) {
      const T *tmp = in[i];
      if (tmp != nullptr) {
        total += tmp[id];
      }
    }
    if (read_dst) {
      out[id] += total;
    } else {
      out[id] = total;
    }
    id += blockDim.x * gridDim.x;
  }
}

template <class T>
__global__ void sum_gpu_sr(T **sr_in, T **sr_out, int64_t N, size_t rows) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < N) {
    for (int i = 0; i < rows; ++i) {
      const T *tmp = sr_in[i];
      T *tmp_out = sr_out[i];
      if (tmp != nullptr && tmp_out != nullptr) {
        tmp_out[id] += tmp[id];
      }
    }
    id += blockDim.x * gridDim.x;
  }
}

template <class T>
__global__ void sum_gpu4(const T *in_0, const T *in_1, T *out, int64_t N) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = id; i < N / 4; i += blockDim.x * gridDim.x) {
    float4 *in0_4 = reinterpret_cast<float4 *>(const_cast<T *>(in_0));
    float4 *in1_4 = reinterpret_cast<float4 *>(const_cast<T *>(in_1));
    float4 tmp;
    tmp.x = in0_4[i].x + in1_4[i].x;
    tmp.y = in0_4[i].y + in1_4[i].y;
    tmp.z = in0_4[i].z + in1_4[i].z;
    tmp.w = in0_4[i].w + in1_4[i].w;
    reinterpret_cast<float4 *>(out)[i] = tmp;
  }
}

template <class T>
void FuseSumCompute(const framework::ExecutionContext &context) {
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

  auto KeCompute = [&](size_t length) {
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

  auto out_var = context.OutputVar("Out");
  bool in_place = in_vars[0] == out_var;

  if (!in_place) {
    out->mutable_data<T>(context.GetPlace());
  }
  int start = in_place ? 1 : 0;
  if (!in_place) {
    if (in_num == 2 && in_vars[0]->IsType<framework::LoDTensor>() &&
        in_vars[1]->IsType<framework::LoDTensor>()) {
      auto &in_0 = in_vars[0]->Get<framework::LoDTensor>();
      auto &in_1 = in_vars[1]->Get<framework::LoDTensor>();

      auto length = in_0.numel();
      if (length) {
        KeCompute(length);
        sum_gpu<T><<<grids, blocks, 0, stream>>>(in_0.data<T>(), in_1.data<T>(),
                                                 out->data<T>(), length);
      } else {
        math::SetConstant<platform::CUDADeviceContext, T> constant_functor;
        constant_functor(
            context.template device_context<platform::CUDADeviceContext>(), out,
            static_cast<T>(0));
      }
      return;
    }
  }

  std::vector<const T *> in_data;
  std::vector<int> selectrow_index;
  int64_t lod_length = 0;
  bool dst_write = false;
  for (int i = start; i < in_num; ++i) {
    if (in_vars[i]->IsType<framework::LoDTensor>()) {
      auto &in_i = in_vars[i]->Get<framework::LoDTensor>();
      in_data.emplace_back(in_i.data<T>());
      lod_length = in_i.numel();
    } else if (in_vars[i]->IsType<framework::SelectedRows>()) {
      selectrow_index.push_back(i);
    }
  }

  // compute select rows seperately.
  if (!selectrow_index.empty()) {
    std::vector<const T *> out_data;
    std::vector<const T *> sr_in_data;
    size_t rows = 0;
    int64_t length = 0;
    for (auto index : selectrow_index) {
      auto &sr = in_vars[index]->Get<framework::SelectedRows>();
      auto &sr_value = sr.value();
      auto &sr_rows = sr.rows();

      auto row_numel = sr_value.numel() / sr_rows.size();
      auto out_dims = out->dims();

      PADDLE_ENFORCE_EQ(sr.height(), out_dims[0]);
      PADDLE_ENFORCE_EQ(row_numel, out->numel() / sr.height());

      auto *sr_data = sr_value.data<T>();
      auto *sr_out_data = out->data<T>();
      rows += sr_rows.size();
      length = row_numel;

      for (size_t i = 0; i < sr_rows.size(); ++i) {
        sr_in_data.emplace_back(&sr_data[i * row_numel]);
        out_data.emplace_back(&sr_out_data[sr_rows[i] * row_numel]);
      }
    }
    if (!sr_in_data.empty() && !out_data.empty()) {
      auto tmp_sr_in_array =
          platform::DeviceTemporaryAllocator::Instance().Get(dev_ctx).Allocate(
              sr_in_data.size() * sizeof(T *));

      memory::Copy(boost::get<platform::CUDAPlace>(dev_ctx.GetPlace()),
                   tmp_sr_in_array->ptr(), platform::CPUPlace(),
                   reinterpret_cast<void *>(sr_in_data.data()),
                   sr_in_data.size() * sizeof(T *), dev_ctx.stream());

      T **sr_in_array_data = reinterpret_cast<T **>(tmp_sr_in_array->ptr());

      auto tmp_out_array =
          platform::DeviceTemporaryAllocator::Instance().Get(dev_ctx).Allocate(
              out_data.size() * sizeof(T *));

      memory::Copy(boost::get<platform::CUDAPlace>(dev_ctx.GetPlace()),
                   tmp_out_array->ptr(), platform::CPUPlace(),
                   reinterpret_cast<void *>(out_data.data()),
                   out_data.size() * sizeof(T *), dev_ctx.stream());

      T **out_array_data = reinterpret_cast<T **>(tmp_out_array->ptr());
      KeCompute(length);
      sum_gpu_sr<T><<<grids, blocks, 0, stream>>>(sr_in_array_data,
                                                  out_array_data, length, rows);
      dst_write = true;
    }
  }
  // if indata not null, merge into one kernel call.
  if (!in_data.empty()) {
    auto tmp_in_array =
        platform::DeviceTemporaryAllocator::Instance().Get(dev_ctx).Allocate(
            in_data.size() * sizeof(T *));

    memory::Copy(boost::get<platform::CUDAPlace>(dev_ctx.GetPlace()),
                 tmp_in_array->ptr(), platform::CPUPlace(),
                 reinterpret_cast<void *>(in_data.data()),
                 in_data.size() * sizeof(T *), dev_ctx.stream());

    T **in_array_data = reinterpret_cast<T **>(tmp_in_array->ptr());
    KeCompute(lod_length);
    sum_gpu_array<T><<<grids, blocks, 0, stream>>>(
        in_array_data, out->data<T>(), lod_length, in_data.size(), dst_write);
  }
}

template <typename T>
class SumKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto in_vars = context.MultiInputVar("X");
    const size_t in_num = in_vars.size();
    auto out_var = context.OutputVar("Out");

    bool in_place = out_var == in_vars[0];
    if (out_var->IsType<framework::LoDTensor>()) {
      FuseSumCompute<T>(context);
    } else if (out_var->IsType<framework::SelectedRows>()) {
      if (in_place && in_vars.size() < 2) {
        return;
      }

      std::vector<const paddle::framework::SelectedRows *> inputs;
      SelectedRows temp_in0;

      if (in_place) {
        auto &in0 = in_vars[0]->Get<SelectedRows>();
        temp_in0.set_height(in0.height());
        temp_in0.set_rows(in0.rows());
        framework::TensorCopy(in0.value(), in0.place(),
                              context.device_context(),
                              temp_in0.mutable_value());
        inputs.push_back(&temp_in0);
        for (size_t i = 1; i < in_vars.size(); ++i) {
          auto &in = in_vars[i]->Get<SelectedRows>();
          if (in.rows().size() > 0) {
            inputs.push_back(&in);
          }
        }
      } else {
        for (auto &in_var : in_vars) {
          auto &in = in_var->Get<SelectedRows>();
          if (in.rows().size() > 0) {
            inputs.push_back(&in_var->Get<SelectedRows>());
          }
        }
      }

      auto *out = context.Output<SelectedRows>("Out");
      out->mutable_rows()->clear();

      bool has_data = false;
      for (auto &in : inputs) {
        if (in->rows().size() > 0) {
          has_data = true;
          break;
        }
      }
      if (has_data) {
        math::scatter::MergeAdd<platform::CUDADeviceContext, T> merge_add;
        merge_add(
            context.template device_context<platform::CUDADeviceContext>(),
            inputs, out);

        out->SyncIndex();

      } else {
        // no data, just set a empty out tensor.
        out->mutable_value()->mutable_data<T>(framework::make_ddim({0}),
                                              context.GetPlace());
      }
    } else if (out_var->IsType<framework::LoDTensorArray>()) {
      auto &out_array = *out_var->GetMutable<framework::LoDTensorArray>();
      for (size_t i = in_place ? 1 : 0; i < in_vars.size(); ++i) {
        PADDLE_ENFORCE(in_vars[i]->IsType<framework::LoDTensorArray>(),
                       "Only support all inputs are TensorArray");
        auto &in_array = in_vars[i]->Get<framework::LoDTensorArray>();

        for (size_t i = 0; i < in_array.size(); ++i) {
          if (in_array[i].numel() != 0) {
            if (i >= out_array.size()) {
              out_array.resize(i + 1);
            }
            if (out_array[i].numel() == 0) {
              framework::TensorCopy(in_array[i], in_array[i].place(),
                                    context.device_context(), &out_array[i]);
              out_array[i].set_lod(in_array[i].lod());
            } else {
              PADDLE_ENFORCE(out_array[i].lod() == in_array[i].lod());
              auto in = EigenVector<T>::Flatten(in_array[i]);
              auto result = EigenVector<T>::Flatten(out_array[i]);
              result.device(
                  *context
                       .template device_context<platform::CUDADeviceContext>()
                       .eigen_device()) = result + in;
            }
          }
        }
      }
    } else {
      PADDLE_THROW("Unexpected branch, output variable type is %s",
                   framework::ToTypeName(out_var->Type()));
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
