// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <stack>
#include "paddle/fluid/operators/math/tree2col.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {
namespace math {
using Tensor = framework::Tensor;
using Node = paddle::operators::math::TreeNode;
template <typename T>
__global__ void tree2col(const T* eta, const int* node, const int* index,
                         const T* vectors, T* result, int feature_size, int n) {
  const int thread_id =
      (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  const int patch_id = thread_id / feature_size;
  const int j = thread_id % feature_size;
  if (patch_id < n) {
    const int begin_o = patch_id * 3 * feature_size;
    const int begin = index[patch_id * 2], end = index[patch_id * 2 + 1];
    T res_l = 0, res_r = 0, res_t = 0;
    for (int i = begin; i < end; i++) {
      const int id = node[i];
      const T vec = vectors[id * feature_size + j];
      res_l += eta[i * 3] * vec;
      res_r += eta[i * 3 + 1] * vec;
      res_t += eta[i * 3 + 2] * vec;
    }
    result[begin_o + j * 3] = res_l;
    result[begin_o + j * 3 + 1] = res_r;
    result[begin_o + j * 3 + 2] = res_t;
  }
}
template <typename T>
class Tree2ColFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const paddle::platform::CUDADeviceContext& context,
                  const framework::Tensor& EdgeSet,
                  const framework::Tensor& node_features,
                  framework::Tensor* patch, int max_depth) {
    std::vector<std::vector<int>> tr;
    auto gpu_place = context.GetPlace();
    auto cpu_place = platform::CPUPlace();
    auto stream = context.stream();
    auto feature_dims = node_features.dims();
    pten::funcs::SetConstant<platform::CUDADeviceContext, T> constant;

    Tensor EdgeSet_cpu;
    framework::TensorCopy(EdgeSet, cpu_place, &EdgeSet_cpu);
    int64_t feature_size = feature_dims[1];
    size_t patch_elem_size = 3 * static_cast<size_t>(feature_size);
    size_t node_count = 0, patch_count = 0, total_size = 0;
    size_t max_size = feature_dims[0];
    Tree2ColUtil::construct_tree(EdgeSet_cpu, &tr, &node_count);

    std::vector<std::vector<Node>> processing_list;
    for (size_t u = 1; u <= node_count; u++) {
      std::vector<Node> tmp = Tree2ColUtil::construct_patch(u, max_depth, tr);
      if (!tmp.empty()) {
        processing_list.push_back(tmp);
        total_size += tmp.size();
      }
    }

    size_t patch_size = processing_list.size();
    Tensor node_cpu, node_gpu, eta_cpu, eta_gpu, index_cpu, index_gpu;
    int* node = node_cpu.mutable_data<int>({static_cast<int64_t>(total_size)},
                                           cpu_place);
    T* eta = eta_cpu.mutable_data<T>({static_cast<int64_t>(total_size * 3)},
                                     cpu_place);
    int* index = index_cpu.mutable_data<int>(
        {static_cast<int64_t>(patch_size * 2)}, cpu_place);

    int idx = 0, index_idx = 0;
    for (auto& tmp : processing_list) {
      index[index_idx++] = idx;
      for (auto& v : tmp) {
        node[idx] = static_cast<int>(v.node - 1);
        eta[idx * 3] = v.eta_l<T>(max_depth);
        eta[idx * 3 + 1] = v.eta_r<T>(max_depth);
        eta[idx * 3 + 2] = v.eta_t<T>(max_depth);
        idx++;
      }
      index[index_idx++] = idx;
    }
    framework::TensorCopy(node_cpu, gpu_place, context, &node_gpu);
    framework::TensorCopy(eta_cpu, gpu_place, context, &eta_gpu);
    framework::TensorCopy(index_cpu, gpu_place, context, &index_gpu);

    int elem_size = patch_size * feature_size;
    int blocks = (elem_size + 1024 - 1) / 1024;
    int block_x = 512;
    int block_y = (blocks + 512 - 1) / 512;
    dim3 threads(1024, 1);
    dim3 grid(block_x, block_y);

    patch->mutable_data<T>(
        {static_cast<int64_t>(max_size), static_cast<int64_t>(patch_elem_size)},
        gpu_place);
    constant(context, patch, 0);
    tree2col<T><<<grid, threads, 0, stream>>>(
        eta_gpu.data<T>(), node_gpu.data<int>(), index_gpu.data<int>(),
        node_features.data<T>(), patch->data<T>(), feature_size, patch_size);
  }
};
template <typename T>
class Col2TreeFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& EdgeSet,
                  const framework::Tensor& patch_grad,
                  framework::Tensor* embedding_grad, int max_depth) {
    std::vector<std::vector<int>> tr;
    auto gpu_place = context.GetPlace();
    auto cpu_place = platform::CPUPlace();
    auto stream = context.stream();
    auto output_dims = patch_grad.dims();
    pten::funcs::SetConstant<platform::CUDADeviceContext, T> constant;

    Tensor EdgeSet_cpu;
    framework::TensorCopy(EdgeSet, cpu_place, &EdgeSet_cpu);
    int64_t output_size = output_dims[1];
    size_t patch_elem_size = 3 * static_cast<size_t>(output_size);
    size_t node_count = 0, patch_count = 0;
    size_t max_size = output_dims[0];
    Tree2ColUtil::construct_tree(EdgeSet_cpu, &tr, &node_count);
    std::vector<std::vector<Node>> processing_list;
    std::vector<std::vector<Node>> grad_list;
    grad_list.resize(node_count);
    size_t total_size = 0, grad_size = node_count;
    for (size_t u = 1; u <= node_count; u++) {
      std::vector<Node> tmp = Tree2ColUtil::construct_patch(u, max_depth, tr);
      if (!tmp.empty()) {
        processing_list.push_back(tmp);
      }
    }
    for (size_t patch_id = 0; patch_id < processing_list.size(); patch_id++) {
      for (auto v : processing_list[patch_id]) {
        grad_list[v.get_node() - 1].push_back(v.change_node(patch_id + 1));
      }
    }
    for (auto& tmp : grad_list) {
      total_size += tmp.size();
    }

    Tensor node_cpu, node_gpu, eta_cpu, eta_gpu, index_cpu, index_gpu;
    int* node = node_cpu.mutable_data<int>({static_cast<int64_t>(total_size)},
                                           cpu_place);
    T* eta = eta_cpu.mutable_data<T>({static_cast<int64_t>(total_size * 3)},
                                     cpu_place);
    int* index = index_cpu.mutable_data<int>(
        {static_cast<int64_t>(grad_size * 2)}, cpu_place);

    size_t idx = 0, index_idx = 0;
    for (auto& tmp : grad_list) {
      index[index_idx++] = idx;
      for (auto& v : tmp) {
        node[idx] = static_cast<int>(v.node - 1);
        eta[idx * 3] = v.eta_l<T>(max_depth);
        eta[idx * 3 + 1] = v.eta_r<T>(max_depth);
        eta[idx * 3 + 2] = v.eta_t<T>(max_depth);
        idx++;
      }
      index[index_idx++] = idx;
    }
    framework::TensorCopy(node_cpu, gpu_place, &node_gpu);
    framework::TensorCopy(eta_cpu, gpu_place, &eta_gpu);
    framework::TensorCopy(index_cpu, gpu_place, &index_gpu);

    int elem_size = output_size * grad_size;
    int blocks = (elem_size + 1024 - 1) / 1024;
    int block_x = 512;
    int block_y = (blocks + 512 - 1) / 512;
    dim3 threads(1024, 1);
    dim3 grid(block_x, block_y);

    embedding_grad->mutable_data<T>(
        {static_cast<int64_t>(max_size), static_cast<int64_t>(patch_elem_size)},
        gpu_place);

    constant(context, embedding_grad, 0);
    tree2col<T><<<grid, threads, 0, stream>>>(
        eta_gpu.data<T>(), node_gpu.data<int>(), index_gpu.data<int>(),
        patch_grad.data<T>(), embedding_grad->data<T>(), output_size,
        grad_size);
  }
};

template class Tree2ColFunctor<platform::CUDADeviceContext, float>;
template class Tree2ColFunctor<platform::CUDADeviceContext, double>;
template class Col2TreeFunctor<platform::CUDADeviceContext, float>;
template class Col2TreeFunctor<platform::CUDADeviceContext, double>;
}  // namespace math
}  // namespace operators
}  // namespace paddle
