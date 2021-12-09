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

#include "paddle/fluid/operators/filter_by_instag_op.h"

namespace cg = cooperative_groups;

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

#define MAX_THREADS 1024
#define THREADS 256

#define MAX_THREAD_STRIDE 32
#define TILE_DIM 32

// Maximum sequence-length support based on the number of threads (2048) allowed
// in each block and
// this MAX is 8K For higher sequence length we need to use higher Max, like for
// 64K : 32
#define MAX_THREAD_ITERATIONS 8  // Maximum 8K

#define MAX_WARP_NUM 32

#define MAX_REGISTERS 256

// test real performance to decide one threads for ? ins

template <typename T>
__global__ void filter_by_instag_cuda_kernel(const int N, int64_t* x2_data,
                                             size_t* x2_lods_data,
                                             int64_t* x3_data,
                                             int filter_tag_size,
                                             int* pass_data) {
  // N is instance num
  // one threads for one ins
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }
  int ins_tag_start = x2_loads_data[idx];
  int ins_tag_end = x2_loads_data[idx + 1];

  // fileter logic
  int i = ins_tag_start;
  for (; i < ins_tag_end; i++) {
    int64_t ins_tag = x2_data[i];
    int j = 0;
    for (; j < filter_tag_size; j++) {
      if (x3_data[j] == ins_tag) break;
    }
    // ins_tag in filter tag
    if (j < filter_tag_size) {
      pass_data[idx] = 1;
      break;
    }
  }
  // copy to output logic
  // if (idx == 0) {
  //}
}

template <typename T>
class FilterByInstagGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto gpu_place = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());
    auto cpu_place = platform::CPUPlace();
    auto& dev_ctx = ctx.template device_context<CUDADeviceContext>();

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
    auto* x3_data = x3->data<int64_t>();

    // Vector, in GPU
    auto x2_lods = x2->lod()[0];
    const size_t* x2_lods_data = x2_loads.CUDAData();
    int N = static_cast<int>(x2_lods.size()) - 1;

    // Vector, in GPU
    Vector<size_t> x1_lods(1, 0);
    if (!is_x1_lod) {
      for (int i = 0; i < x1->dims()[0]; i++) {
        x1_lods.push_back(i + 1);
      }
    } else {
      x1_lods = context.Input<LoDTensor>("Ins")->lod()[0];
    }
    const size_t* x1_lods_data = x1_lods.CUDAData();

    // set output value
    // for those whose ins been dropout, set 0 for whole lines.
    // otherwise, copy whole line
    // Dim [local fc count, batch size, embedding size]
    LoDTensor* out = context.Output<LoDTensor>("Out");
    LoDTensor* map = context.Output<LoDTensor>("IndexMap");
    LoDTensor* loss_weight = context.Output<LoDTensor>("LossWeight");

    auto* out_data = out->mutable_data<T>(context.GetPlace());
    auto* map_data = map->mutable_data<int64_t>(context.GetPlace());
    auto* loss_weight_data =
        loss_weight->mutable_data<float>(context.GetPlace());

    std::unordered_map<int64_t, int64_t> mmap_aux;

    // check configuration
    // int block_size = 512;
    int block_size = THREADS dim3 block_dim(block_size);
    dim3 grid_dim((N + block_size - 1) / block_size);

    filter_by_instag_cuda_kernel<<<grid_dim, block_dim, 0,
                                   ctx.cuda_device_context().stream()>>>(
        N, x1_data, x1_lods_data, out_data, loss_weight_data, x2_data,
        x2_loads_data, x3_data, is_x1_lod, x1_embed_size, out_val_if_empty);

    Vector<size_t> out_lods(1, 0);

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

    // auto* out_data = out->mutable_data<T>(context.GetPlace());
    // auto* map_data = map->mutable_data<int64_t>(context.GetPlace());
    // auto* loss_weight_data =
    //     loss_weight->mutable_data<float>(context.GetPlace());

    if (out_lods.size() - 1 > 0) {
      Vector<size_t> map_lods;
      for (size_t i = 0; i < out_lods.size() - 1; i++) {
        map_data[i * 3] = (int64_t)out_lods[i];
        map_data[i * 3 + 1] = mmap_aux[map_data[i * 3]];
        map_data[i * 3 + 2] = out_lods[i + 1] - out_lods[i];
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
      memset(out_data, 0, out->numel() * sizeof(T));
      for (int i = 0; i < loss_weight->numel(); i++) {
        loss_weight_data[i] = 1;
      }

      for (size_t i = 0; i < out_lods.size() - 1; i++) {
        size_t pos = out_lods[i];
        for (int k = map_data[i * 3 + 1];
             k < map_data[i * 3 + 1] + map_data[i * 3 + 2]; k++) {
          memcpy(out_data + pos * x1_embed_size, x1_data + k * x1_embed_size,
                 x1_embed_size * sizeof(T));
          ++pos;
        }
      }

    } else {
      Vector<size_t> map_lods;
      map_data[0] = 0;
      map_data[1] = 1;
      map_data[2] = 1;
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
      for (int64_t oi = 0; oi < out->numel(); ++oi) {
        if (std::is_same<T, int32_t>::value) {
          out_data[oi] = (int32_t)out_val_if_empty;
        } else if (std::is_same<T, int64_t>::value) {
          out_data[oi] = (int64_t)out_val_if_empty;
        } else {
          out_data[oi] = static_cast<double>(out_val_if_empty);
        }
      }
      loss_weight_data[0] = 0;
    }
  }
};

template <typename T>
class FilterByInstagGradGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* output_grad = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* x1_grad = context.Output<LoDTensor>(framework::GradVarName("Ins"));
    auto* loss_weight = context.Input<LoDTensor>("LossWeight");
    auto* mmap = context.Input<LoDTensor>("IndexMap");
    auto* x1 = context.Input<LoDTensor>("Ins");

    x1_grad->set_lod(context.Input<LoDTensor>("Ins")->lod());
    x1_grad->Resize(x1->dims());
    auto mmap_data = mmap->data<int64_t>();

    // expected auto = T
    auto* output_grad_data = output_grad->data<T>();

    auto* loss_weight_data = loss_weight->data<float>();
    // expected auto = T
    auto* x1_grad_data = x1_grad->mutable_data<T>(context.GetPlace());
    memset(x1_grad_data, 0, x1->dims()[0] * x1->dims()[1] * sizeof(T));
    if (loss_weight->numel() != 1 || loss_weight_data[0] != 0) {
      auto output_dims = output_grad->dims();
      for (int i = 0; i < mmap->dims()[0]; i++) {
        int src_ln = mmap_data[i * 3], dst_ln = mmap_data[i * 3 + 1];
        int line_cnt = mmap_data[i * 3 + 2];
        for (int l = 0; l < line_cnt; l++) {
          for (int j = 0; j < output_dims[1]; j++) {
            x1_grad_data[(dst_ln + l) * output_dims[1] + j] =
                output_grad_data[(src_ln + l) * output_dims[1] + j];
          }
        }
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
