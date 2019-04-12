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

#include "sum_op_gpu_kernel.h"
#include "paddle/fluid/operators/sum_op.h"
#include <paddle/fluid/platform/device_context.h>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

inline uint64_t GetTimeInNsec() {
  using clock = std::conditional<std::chrono::high_resolution_clock::is_steady,
                                 std::chrono::high_resolution_clock,
                                 std::chrono::steady_clock>::type;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             clock::now().time_since_epoch()).count();
}

template <typename DeviceContext, typename T>
class SumCUDAKernel: public framework::OpKernel<T> {
public:
void Compute(const framework::ExecutionContext &context) const override {
    auto in_vars = context.MultiInputVar("X");
    const size_t in_num = in_vars.size();
    auto out_var = context.OutputVar("Out");

    bool in_place = out_var == in_vars[0];
    constexpr size_t theory_sm_threads = 1024;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto stream = dev_ctx.stream();

    auto max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    auto sm_count = max_threads / theory_sm_threads;
    size_t tile_size = 0;
    dim3 grids;
    dim3 blocks;

    auto KeCompute = [&] (size_t length) {
      if (length > max_threads)
        tile_size = 512;
      else if (length < max_threads && length > sm_count * 128)
        tile_size = 256;
      else if (length < sm_count * 128)
        tile_size = 128;
      grids = dim3(CEIL_DIV(length, tile_size), 1, 1);
      blocks = dim3(tile_size, 1, 1);
    };

    if (out_var->IsType<framework::LoDTensor>()) {
      auto *out = context.Output<LoDTensor>("Out");

      if (!in_place) {
        out->mutable_data<T>(context.GetPlace());
      }
      int start = in_place ? 1 : 0;
      if (!in_place) {
        if (in_num == 2) {
          auto &in_0 = in_vars[0]->Get<framework::LoDTensor>();
          auto &in_1 = in_vars[1]->Get<framework::LoDTensor>();

	  auto length = in_0.numel();
	  KeCompute(length);
          sum_gpu<T><<<grids, blocks, 0, stream>>>(in_0.data<T>(), in_1.data<T>(), out->data<T>(), length);
	  return;
        }
      }

      std::vector<const T*> in_data;
      std::vector<int> selectrow_index;
      int64_t lod_length = 0;
      for (int i = 0; i < in_num; ++i) {
        if(in_vars[i]->IsType<framework::LoDTensor>()) {
          auto &in_i = in_vars[i]->Get<framework::LoDTensor>();
          in_data.emplace_back(in_i.data<T>());
	  lod_length = in_i.numel();
	}
	else if (in_vars[i]->IsType<framework::SelectedRows>()) {
          selectrow_index.push_back(i);
	}
      }
 
      // compute select rows seperately.
      if (!selectrow_index.empty()) {
        std::vector<const T*>out_data;
        std::vector<const T*>sr_in_data;
	size_t rows = 0;
	int64_t length = 0;
        for(auto index : selectrow_index) {
          auto &sr = in_vars[index]->Get<framework::SelectedRows>();
	  auto &sr_value = sr.value();
	  auto &sr_rows = sr.rows();

          auto row_numel = sr_value.numel() / sr_rows.size();
	  auto out_dims = out->dims();

	  PADDLE_ENFORCE_EQ(sr.height(), out_dims[0]);
	  PADDLE_ENFORCE_EQ(row_numel, out->numel() / sr.height());

	  auto* sr_data = sr_value.data<T>();
	  auto* sr_out_data = out->data<T>();
	  rows += sr_rows.size();
	  length = row_numel;

	  for(size_t i = 0; i < sr_rows.size(); ++i) {
            sr_in_data.emplace_back(&sr_data[i * row_numel]);
	    out_data.emplace_back(&sr_out_data[sr_rows[i] * row_numel]);
	  }
        }
        if (!sr_in_data.empty() && !out_data.empty()) {

          auto tmp_sr_in_array =
          platform::DeviceTemporaryAllocator::Instance().Get(dev_ctx).Allocate(
              sr_in_data.size() * sizeof(T*));

          memory::Copy(boost::get<platform::CUDAPlace>(dev_ctx.GetPlace()),
                 tmp_sr_in_array->ptr(), platform::CPUPlace(),
                 reinterpret_cast<void*>(sr_in_data.data()),
                 sr_in_data.size() * sizeof(T*), dev_ctx.stream());

          T** sr_in_array_data = reinterpret_cast<T**>(tmp_sr_in_array->ptr());


          auto tmp_out_array =
          platform::DeviceTemporaryAllocator::Instance().Get(dev_ctx).Allocate(
              out_data.size() * sizeof(T*));

          memory::Copy(boost::get<platform::CUDAPlace>(dev_ctx.GetPlace()),
                 tmp_out_array->ptr(), platform::CPUPlace(),
                 reinterpret_cast<void*>(out_data.data()),
                 out_data.size() * sizeof(T*), dev_ctx.stream());

          T** out_array_data = reinterpret_cast<T**>(tmp_out_array->ptr());
          KeCompute(length);
          sum_gpu_sr<T><<<grids, blocks, 0, stream>>>(sr_in_array_data, out_array_data, length, rows);
        }
      }
	  // if case scale too large, we choose big block.
      if (!in_data.empty()) {
        int length = in_vars[0]->Get<framework::LoDTensor>().numel();
          auto tmp_in_array =
          platform::DeviceTemporaryAllocator::Instance().Get(dev_ctx).Allocate(
              in_num * sizeof(T*));

          memory::Copy(boost::get<platform::CUDAPlace>(dev_ctx.GetPlace()),
                 tmp_in_array->ptr(), platform::CPUPlace(),
                 reinterpret_cast<void*>(in_data.data()),
                 in_data.size() * sizeof(T*), dev_ctx.stream());

          T** in_array_data = reinterpret_cast<T**>(tmp_in_array->ptr());
          KeCompute(lod_length);
          sum_gpu_array<T><<<grids, blocks, 0, stream>>>(in_array_data, out->data<T>(), length, in_data.size()); 
      }
    }
  }
};
}
}
namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    sum, ops::SumCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SumCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SumCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SumCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>);
    //ops::SumCUDAKernel<paddle::platform::CUDADeviceContext, plat::float16>);
