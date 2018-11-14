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
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/tree2col.h"

namespace paddle {
namespace operators {
namespace math {
using Tensor = framework::Tensor;
using Node = paddle::operators::math::TreeNode;
template <typename T>
__global__ void tree2col(Node *patchs, int *indexes, const T *vectors,
                         T *result, int max_depth, int feature_size, int n) {
  const int thread_id =
      (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  const int block_id = thread_id / feature_size;
  const int thread_idx = thread_id % feature_size;
  if (block_id < n) {
    const int begin = indexes[block_id * 2], end = indexes[block_id * 2 + 1];
    const int begin_o = block_id * 3 * feature_size;
    const int j = thread_idx;
    for (int i = begin; i < end; i++) {
      T eta_l = patchs[i].eta_l<T>(max_depth);
      T eta_r = patchs[i].eta_r<T>(max_depth);
      T eta_t = patchs[i].eta_t<T>(max_depth);
      int id = patchs[i].node - 1;
      result[begin_o + j * 3] += eta_l * vectors[id * feature_size + j];
      result[begin_o + j * 3 + 1] += eta_r * vectors[id * feature_size + j];
      result[begin_o + j * 3 + 2] += eta_t * vectors[id * feature_size + j];
    }
  }
}
template <typename T>
class Tree2ColFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const paddle::platform::CUDADeviceContext &context,
                  const framework::Tensor &EdgeSet,
                  const framework::Tensor &node_features,
                  framework::Tensor *patch, int max_depth) {
    std::vector<std::vector<int>> tr;
    auto gpu_place = boost::get<platform::CUDAPlace>(context.GetPlace());
    auto cpu_place = platform::CPUPlace();
    auto stream = context.stream();
    auto feature_dims = node_features.dims();
    math::SetConstant<platform::CUDADeviceContext, T> constant;

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
    Node *nodes_gpu = reinterpret_cast<Node *>(
        memory::Alloc(gpu_place, total_size * sizeof(Node)));
    Node *nodes_cpu = reinterpret_cast<Node *>(
        memory::Alloc(cpu_place, total_size * sizeof(Node)));
    int *index_gpu = reinterpret_cast<int *>(
        memory::Alloc(gpu_place, patch_size * 2 * sizeof(int)));
    int *index_cpu = reinterpret_cast<int *>(
        memory::Alloc(cpu_place, patch_size * 2 * sizeof(int)));

    PADDLE_ENFORCE_NOT_NULL(nodes_cpu);
    PADDLE_ENFORCE_NOT_NULL(nodes_gpu);
    PADDLE_ENFORCE_NOT_NULL(index_cpu);
    PADDLE_ENFORCE_NOT_NULL(nodes_gpu);

    size_t nodes_idx = 0, index_idx = 0, idx = 0, patch_idx = 0;
    for (auto &tmp : processing_list) {
      index_cpu[index_idx++] = idx;
      for (auto v : tmp) {
        nodes_cpu[idx++] = v;
      }
      index_cpu[index_idx++] = idx;
      patch_idx++;
    }

    memory::Copy(gpu_place, nodes_gpu, cpu_place, nodes_cpu,
                 total_size * sizeof(Node), stream);
    memory::Copy(gpu_place, index_gpu, cpu_place, index_cpu,
                 patch_size * 2 * sizeof(int), stream);

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
        nodes_gpu, index_gpu, node_features.data<T>(), patch->data<T>(),
        max_depth, feature_size, patch_size);
    memory::Free(cpu_place, nodes_cpu);
    memory::Free(cpu_place, index_cpu);
    memory::Free(gpu_place, index_gpu);
    memory::Free(gpu_place, nodes_gpu);
  }
};
template <typename T>
class Col2TreeFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext &context,
                  const framework::Tensor &EdgeSet,
                  const framework::Tensor &patch_grad,
                  framework::Tensor *embedding_grad, int max_depth) {
    std::vector<std::vector<int>> tr;
    auto gpu_place = boost::get<platform::CUDAPlace>(context.GetPlace());
    auto cpu_place = platform::CPUPlace();
    auto stream = context.stream();
    auto output_dims = patch_grad.dims();
    math::SetConstant<platform::CUDADeviceContext, T> constant;

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
        total_size += processing_list.size();
      }
    }
    for (size_t patch_id = 0; patch_id < processing_list.size(); patch_id++) {
      for (auto v : processing_list[patch_id]) {
        grad_list[v.get_node() - 1].push_back(v.change_node(patch_id + 1));
      }
    }

    Node *nodes_gpu = reinterpret_cast<Node *>(
        memory::Alloc(gpu_place, total_size * sizeof(Node)));
    Node *nodes_cpu = reinterpret_cast<Node *>(
        memory::Alloc(cpu_place, total_size * sizeof(Node)));
    int *index_gpu = reinterpret_cast<int *>(
        memory::Alloc(gpu_place, grad_size * 2 * sizeof(int)));
    int *index_cpu = reinterpret_cast<int *>(
        memory::Alloc(cpu_place, grad_size * 2 * sizeof(int)));

    size_t nodes_idx = 0, index_idx = 0, idx = 0, patch_idx = 0;
    for (auto &tmp : grad_list) {
      index_cpu[index_idx++] = idx;
      for (auto v : tmp) {
        nodes_cpu[idx++] = v;
      }
      index_cpu[index_idx++] = idx;
      patch_idx++;
    }
    memory::Copy(gpu_place, nodes_gpu, cpu_place, nodes_cpu,
                 total_size * sizeof(Node), stream);
    memory::Copy(gpu_place, index_gpu, cpu_place, index_cpu,
                 grad_size * 2 * sizeof(int), stream);

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
        nodes_gpu, index_gpu, patch_grad.data<T>(), embedding_grad->data<T>(),
        max_depth, output_size, grad_size);

    memory::Free(cpu_place, index_cpu);
    memory::Free(cpu_place, nodes_cpu);
    memory::Free(gpu_place, index_gpu);
    memory::Free(gpu_place, nodes_gpu);
  }
};

template class Tree2ColFunctor<platform::CUDADeviceContext, float>;
template class Tree2ColFunctor<platform::CUDADeviceContext, double>;
template class Col2TreeFunctor<platform::CUDADeviceContext, float>;
template class Col2TreeFunctor<platform::CUDADeviceContext, double>;
}  // namespace math
}  // namespace operators
}  // namespace paddle
