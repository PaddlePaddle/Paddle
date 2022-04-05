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

#include <unordered_map>
#include <vector>

#include "paddle/phi/kernels/graph_reindex_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void GraphReindexKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& neighbors,
                        const DenseTensor& count,
                        paddle::optional<const DenseTensor&> hashtable_value,
                        paddle::optional<const DenseTensor&> hashtable_index,
                        bool flag_buffer_hashtable,
                        DenseTensor* reindex_src,
                        DenseTensor* reindex_dst,
                        DenseTensor* out_nodes) {
  const T* x_data = x.data<T>();
  const T* neighbors_data = neighbors.data<T>();
  const int* count_data = count.data<int>();
  const int bs = x.dims()[0];
  const int num_edges = neighbors.dims()[0];

  std::unordered_map<T, T> node_map;
  std::vector<T> unique_nodes;
  int reindex_id = 0;
  for (int i = 0; i < bs; i++) {
    T node = x_data[i];
    unique_nodes.emplace_back(node);
    node_map[node] = reindex_id++;
  }
  // Reindex Src
  std::vector<T> src(num_edges);
  std::vector<T> dst(num_edges);
  for (int i = 0; i < num_edges; i++) {
    T node = neighbors_data[i];
    if (node_map.find(node) == node_map.end()) {
      unique_nodes.emplace_back(node);
      node_map[node] = reindex_id++;
    }
    src[i] = node_map[node];
  }
  // Reindex Dst
  int cnt = 0;
  for (int i = 0; i < bs; i++) {
    for (int j = 0; j < count_data[i]; j++) {
      T node = x_data[i];
      dst[cnt++] = node_map[node];
    }
  }

  reindex_src->Resize({num_edges});
  T* reindex_src_data = dev_ctx.template Alloc<T>(reindex_src);
  std::copy(src.begin(), src.end(), reindex_src_data);
  reindex_dst->Resize({num_edges});
  T* reindex_dst_data = dev_ctx.template Alloc<T>(reindex_dst);
  std::copy(dst.begin(), dst.end(), reindex_dst_data);
  out_nodes->Resize({static_cast<int>(unique_nodes.size())});
  T* out_nodes_data = dev_ctx.template Alloc<T>(out_nodes);
  std::copy(unique_nodes.begin(), unique_nodes.end(), out_nodes_data);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    graph_reindex, CPU, ALL_LAYOUT, phi::GraphReindexKernel, int, int64_t) {}
