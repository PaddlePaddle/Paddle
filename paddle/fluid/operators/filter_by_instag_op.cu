// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

#include "paddle/fluid/operators/filter_by_instag_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using SelectedRows = framework::SelectedRows;
using LoDTensor = framework::LoDTensor;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename T>
using Vector = framework::Vector<T>;
#else
template <typename T>
using Vector = framework::CPUVector<T>;
#endif

using CUDADeviceContext = paddle::platform::CUDADeviceContext;

#define THREADS 256

__global__ void filter_by_instag_cuda_kernel(
    const int N, const int64_t* x2_data, const size_t* x2_lods_data,
    const int64_t* x3_data, int64_t filter_tag_size, int* flag_data) {
  // N is instance num
  // one threads for one instance
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }
  int ins_tag_start = x2_lods_data[idx];
  int ins_tag_end = x2_lods_data[idx + 1];

  // fileter logic
  int i = ins_tag_start;
  for (; i < ins_tag_end; i++) {
    int64_t ins_tag = x2_data[i];
    int j = 0;
    for (; j < filter_tag_size; j++) {
      if (x3_data[j] == ins_tag) break;
    }
    // if ins_tag in filter tag
    if (j < filter_tag_size) {
      flag_data[idx] = 1;
      break;
    }
  }
}

template <typename T>
class FilterByInstagGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto gpu_place =
        BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace());
    gpuStream_t current_stream = context.cuda_device_context().stream();

    // auto cpu_place = platform::CPUPlace();
    // auto& dev_ctx = ctx.template device_context<CUDADeviceContext>();

    // X1 is global FC output
    // Dim [batch size, embedding size]
    auto* x1 = context.Input<LoDTensor>("Ins");
    bool is_lod = context.Attr<bool>("is_lod");

    int is_x1_lod = -1;
    if (is_lod)
      is_x1_lod = 1;
    else
      is_x1_lod = 0;

    int64_t out_val_if_empty = context.Attr<int64_t>("out_val_if_empty");
    size_t x1_embed_size = x1->dims()[1];
    // X2 is ins tag list
    // LoD [[0, Sum(ins1), Sum(ins1, ins2), ... ]]

    auto* x2 = context.Input<LoDTensor>("Ins_tag");
    // expected auto = const int64_t
    auto* x2_data = x2->data<int64_t>();

    // X3 is local fc tag list
    // LoD [[0, Sum(fc1), Sum(fc1, fc2) ...]]
    auto* x3 = context.Input<Tensor>("Filter_tag");
    const int64_t* x3_data = x3->data<int64_t>();

    // Vector, in GPU
    auto x2_lods = x2->lod()[0];
    const size_t* x2_lods_data = x2_lods.CUDAData(context.GetPlace());
    const int N = static_cast<int>(x2_lods.size()) - 1;

    // Vector, in GPU
    Vector<size_t> x1_lods(1, 0);
    if (!is_x1_lod) {
      for (int i = 0; i < x1->dims()[0]; i++) {
        x1_lods.push_back(i + 1);
      }
    } else {
      x1_lods = context.Input<LoDTensor>("Ins")->lod()[0];
    }

    const size_t* x1_lods_data = x1_lods.CUDAData(context.GetPlace());
    auto* x1_data = x1->data<T>();

    // set output value
    // for those whose ins been dropout, set 0 for whole lines.
    // otherwise, copy whole line
    // Dim [local fc count, batch size, embedding size]
    LoDTensor* out = context.Output<LoDTensor>("Out");
    LoDTensor* map = context.Output<LoDTensor>("IndexMap");
    LoDTensor* loss_weight = context.Output<LoDTensor>("LossWeight");

    Vector<int> flag(N, 0);
    int* flag_data = flag.CUDAMutableData(context.GetPlace());

    // check configuration
    // int block_size = 512;
    int block_size = THREADS;
    dim3 block_dim(block_size);
    dim3 grid_dim((N + block_size - 1) / block_size);

    // fileter_logic
    filter_by_instag_cuda_kernel<<<grid_dim, block_dim, 0, current_stream>>>(
        N, x2_data, x2_lods_data, x3_data, x3->numel(), flag_data);

    platform::GpuStreamSync(current_stream);
    std::unordered_map<int64_t, int64_t> mmap_aux;
    Vector<size_t> out_lods(1, 0);

    int cnt = 0;
    for (auto it = flag.begin(); it != flag.end(); cnt++, it++) {
      if ((*it) == 1) {
        size_t batch_len = x1_lods[cnt + 1] - x1_lods[cnt];
        mmap_aux[out_lods.back()] = x1_lods[cnt];
        out_lods.push_back(out_lods.back() + batch_len);
      }
    }

    if (out_lods.size() - 1 > 0) {
      out->Resize(framework::make_ddim(
          {(int64_t)out_lods.back(), (int64_t)x1_embed_size}));
      map->Resize(framework::make_ddim({(int64_t)out_lods.size() - 1, 3}));
      loss_weight->Resize(
          framework::make_ddim({(int64_t)out_lods.size() - 1, 1}));
    } else {
      out->Resize(framework::make_ddim({1, (int64_t)x1_embed_size}));
      map->Resize(framework::make_ddim({1, 3}));
      loss_weight->Resize(framework::make_ddim({1, 1}));
    }
    auto* out_data = out->mutable_data<T>(context.GetPlace());
    auto* map_data = map->mutable_data<int64_t>(context.GetPlace());
    auto* loss_weight_data =
        loss_weight->mutable_data<float>(context.GetPlace());

    if (out_lods.size() - 1 > 0) {
      Vector<size_t> map_lods;
      map_lods.reserve(out_lods.size());
      thrust::device_ptr<int64_t> map_data_ptr(map_data);
      for (size_t i = 0; i < out_lods.size() - 1; i++) {
        map_data_ptr[i * 3] = (int64_t)out_lods[i];
        map_data_ptr[i * 3 + 1] = mmap_aux[(int64_t)out_lods[i]];
        map_data_ptr[i * 3 + 2] = out_lods[i + 1] - out_lods[i];
        map_lods.push_back(i);
      }
      map_lods.push_back(out_lods.size() - 1);

      std::vector<Vector<size_t>> map_lod_info;
      map_lod_info.push_back(map_lods);

      map->set_lod(map_lod_info);
      loss_weight->set_lod(map_lod_info);

      std::vector<Vector<size_t>> out_lod_info;
      out_lod_info.push_back(out_lods);
      out->set_lod(out_lod_info);

      thrust::device_ptr<T> out_data_ptr(out_data);
      thrust::device_ptr<const T> x1_data_ptr(x1_data);

      thrust::device_ptr<float> loss_weight_data_ptr(loss_weight_data);

      thrust::fill(out_data_ptr, out_data_ptr + out->numel(), 0);
      thrust::fill(loss_weight_data_ptr,
                   loss_weight_data_ptr + loss_weight->numel(), 1.0);

      for (size_t i = 0; i < out_lods.size() - 1; i++) {
        size_t pos = out_lods[i];
        thrust::copy(x1_data_ptr + map_data_ptr[i * 3 + 1] * x1_embed_size,
                     x1_data_ptr +
                         (map_data_ptr[i * 3 + 1] + map_data_ptr[i * 3 + 2]) *
                             x1_embed_size,
                     out_data_ptr + pos * x1_embed_size);
        // for (int k = map_data[i * 3 + 1];
        //      k < map_data[i * 3 + 1] + map_data[i * 3 + 2]; k++) {

        //   GpuMemcpyAsync(out_data + pos * x1_embed_size, x1_data + k *
        //   x1_embed_size,
        //          x1_embed_size * sizeof(T), cudaMemcpyDeviceToDevice,
        //          current_stream);
        //   ++pos;
        // }
      }
      // GpuStreamSync(current_stream);
    } else {
      Vector<size_t> map_lods;
      thrust::device_ptr<int64_t> map_data_ptr(map_data);
      map_data_ptr[0] = 0;
      map_data_ptr[1] = 1;
      map_data_ptr[2] = 1;
      map_lods.push_back(0);
      map_lods.push_back(1);
      out_lods.push_back(1);
      std::vector<Vector<size_t>> map_lod_info;
      map_lod_info.push_back(map_lods);
      map->set_lod(map_lod_info);
      loss_weight->set_lod(map_lod_info);
      std::vector<Vector<size_t>> out_lod_info;
      out_lod_info.push_back(out_lods);
      out->set_lod(out_lod_info);

      thrust::device_ptr<T> out_data_ptr(out_data);
      // gpu kernel
      if (std::is_same<T, int32_t>::value) {
        // thrust::device_ptr<int32_t> out_data_ptr(out_data);
        thrust::fill(out_data_ptr, out_data_ptr + out->numel(),
                     static_cast<int32_t>(out_val_if_empty));
      } else if (std::is_same<T, int64_t>::value) {
        // thrust::device_ptr<int64_t> out_data_ptr(out_data);
        thrust::fill(out_data_ptr, out_data_ptr + out->numel(),
                     static_cast<int64_t>(out_val_if_empty));
      } else if (std::is_same<T, float>::value) {
        // thrust::device_ptr<double> out_data_ptr(out_data);
        thrust::fill(out_data_ptr, out_data_ptr + out->numel(),
                     static_cast<float>(out_val_if_empty));
      } else {
        thrust::fill(out_data_ptr, out_data_ptr + out->numel(),
                     static_cast<double>(out_val_if_empty));
      }

      thrust::device_ptr<float> loss_weight_data_ptr(loss_weight_data);
      loss_weight_data_ptr[0] = 0;
    }
  }
};

template <typename T>
class FilterByInstagGradGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto gpu_place =
        BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace());
    auto* output_grad = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* x1_grad = context.Output<LoDTensor>(framework::GradVarName("Ins"));
    auto* loss_weight = context.Input<LoDTensor>("LossWeight");
    auto* mmap = context.Input<LoDTensor>("IndexMap");
    auto* x1 = context.Input<LoDTensor>("Ins");

    x1_grad->set_lod(context.Input<LoDTensor>("Ins")->lod());
    x1_grad->Resize(x1->dims());

    auto* mmap_data = mmap->data<int64_t>();

    // expected auto = T
    auto* output_grad_data = output_grad->data<T>();
    auto* loss_weight_data = loss_weight->data<float>();

    // expected auto = T
    auto* x1_grad_data = x1_grad->mutable_data<T>(context.GetPlace());

    thrust::device_ptr<const float> loss_weight_data_ptr(loss_weight_data);
    thrust::device_ptr<T> x1_grad_data_ptr(x1_grad_data);
    thrust::device_ptr<const T> output_grad_data_ptr(output_grad_data);
    thrust::device_ptr<const int64_t> mmap_data_ptr(mmap_data);
    thrust::fill(x1_grad_data_ptr,
                 x1_grad_data_ptr + x1->dims()[0] * x1->dims()[1], 0);
    if (loss_weight->numel() != 1 || loss_weight_data_ptr[0] != 0) {
      auto output_dims = output_grad->dims();
      for (int i = 0; i < mmap->dims()[0]; i++) {
        int src_ln = mmap_data_ptr[i * 3], dst_ln = mmap_data_ptr[i * 3 + 1];
        int line_cnt = mmap_data_ptr[i * 3 + 2];
        thrust::copy(
            output_grad_data_ptr + src_ln * output_dims[1],
            output_grad_data_ptr + (src_ln + line_cnt) * output_dims[1],
            x1_grad_data_ptr + dst_ln * output_dims[1]);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(filter_by_instag, ops::FilterByInstagGPUKernel<float>,
                        ops::FilterByInstagGPUKernel<double>,
                        ops::FilterByInstagGPUKernel<int32_t>,
                        ops::FilterByInstagGPUKernel<int64_t>);

REGISTER_OP_CUDA_KERNEL(filter_by_instag_grad,
                        ops::FilterByInstagGradGPUKernel<float>,
                        ops::FilterByInstagGradGPUKernel<double>,
                        ops::FilterByInstagGradGPUKernel<int32_t>,
                        ops::FilterByInstagGradGPUKernel<int64_t>);
