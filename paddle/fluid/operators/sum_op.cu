/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/math/sum.h"
#include "paddle/fluid/operators/sum_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

using LoDTensor = framework::LoDTensor;

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
__global__ void SumAlign4CUDAKernel(const T *in_0, const T *in_1, T *out,
                                    int64_t N) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = id; i < N / 4; i += blockDim.x * gridDim.x) {
    const float4 *in0_4 = reinterpret_cast<float4 *>(in_0);
    const float4 *in1_4 = reinterpret_cast<float4 *>(in_1);
    float4 tmp;
    tmp.x = in0_4[i].x + in1_4[i].x;
    tmp.y = in0_4[i].y + in1_4[i].y;
    tmp.z = in0_4[i].z + in1_4[i].z;
    tmp.w = in0_4[i].w + in1_4[i].w;
    reinterpret_cast<float4 *>(out)[i] = tmp;
  }
}

template <class T>
void SumToLoDTensor(const framework::ExecutionContext &context) {
  auto in_vars = context.MultiInputVar("X");
  auto *out = context.Output<LoDTensor>("Out");
  bool in_place = in_vars[0] == context.OutputVar("Out");

  if (!in_place) {
    out->mutable_data<T>(context.GetPlace());
  }

  std::vector<const framework::Tensor *> inputs;
  std::vector<int> selectrow_index;
  for (size_t i = 0; i < in_vars.size(); ++i) {
    if (in_vars[i]->IsType<framework::LoDTensor>()) {
      auto &in_i = in_vars[i]->Get<framework::LoDTensor>();
      if (in_i.numel() > 0 && in_i.IsInitialized()) {
        inputs.push_back(&in_i);
      }
    } else if (in_vars[i]->IsType<framework::SelectedRows>()) {
      selectrow_index.push_back(i);
    }
  }

  auto &dev_ctx =
      context.template device_context<platform::CUDADeviceContext>();

  // Compute the sum of LoDTensor inputs.
  if (inputs.size() > 0U) {
    math::SumLoDTensorFunctor<platform::CUDADeviceContext, T> sum_functor;
    sum_functor(dev_ctx, inputs, out);
  } else {
    math::SetConstant<platform::CUDADeviceContext, T> constant_functor;
    constant_functor(dev_ctx, out, static_cast<T>(0));
  }

  // Compute the sum of SelectRows inputs.
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

      PADDLE_ENFORCE_EQ(sr.height(), out_dims[0]);
      PADDLE_ENFORCE_EQ(row_numel, out->numel() / sr.height());

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
          platform::DeviceTemporaryAllocator::Instance().Get(dev_ctx).Allocate(
              sr_in_out_data.size() * sizeof(T *));

      memory::Copy(boost::get<platform::CUDAPlace>(dev_ctx.GetPlace()),
                   tmp_sr_in_out_array->ptr(), platform::CPUPlace(),
                   reinterpret_cast<void *>(sr_in_out_data.data()),
                   sr_in_out_data.size() * sizeof(T *), dev_ctx.stream());

      T **sr_in_out_array_data =
          reinterpret_cast<T **>(tmp_sr_in_out_array->ptr());

      constexpr size_t theory_sm_threads = 1024;
      auto max_threads = dev_ctx.GetMaxPhysicalThreadCount();
      auto sm_count = max_threads / theory_sm_threads;
      int64_t tile_size = 0;

      if (length >= max_threads)
        tile_size = 1024;
      else if (length < max_threads && length > sm_count * 128)
        tile_size = 512;
      else if (length <= sm_count * 128)
        tile_size = 256;
      dim3 grids = dim3(CEIL_DIV(length, tile_size), 1, 1);
      dim3 blocks = dim3(tile_size, 1, 1);

      SumSelectedRowsCUDAKernel<T><<<grids, blocks, 0, dev_ctx.stream()>>>(
          sr_in_out_array_data, length, rows);
    }
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
      PADDLE_THROW("Unexpected branch, output variable type is %s",
                   framework::ToTypeName(out_var->Type()));
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    sum, ops::SumKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SumKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SumKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SumKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::SumKernel<paddle::platform::CUDADeviceContext,
                   paddle::platform::float16>);
