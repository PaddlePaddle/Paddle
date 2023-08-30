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

#include "paddle/phi/kernels/graph_reindex_kernel.h"

#include <unordered_map>
#include <vector>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void GraphReindexKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& neighbors,
                        const DenseTensor& count,
                        const paddle::optional<DenseTensor>& hashtable_value,
                        const paddle::optional<DenseTensor>& hashtable_index,
                        DenseTensor* reindex_src,
                        DenseTensor* reindex_dst,
                        DenseTensor* out_nodes) {
  const T* x_data = x.data<T>();
  const T* neighbors_data = neighbors.data<T>();
  const int* count_data = count.data<int>();
  const int bs = static_cast<int>(x.dims()[0]);
  const int num_edges = static_cast<int>(neighbors.dims()[0]);

  std::unordered_map<T, T> node_map;
  std::vector<T> unique_nodes;
  int reindex_id = 0;
  PADDLE_ENFORCE_NE(
      0,
      bs,
      errors::InvalidArgument("The first of dims should not be equal to 0."));
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
  // Add support for multi-type edges reindex
  int num_edge_types = static_cast<int>(count.dims()[0] / bs);
  int cnt = 0;
  for (int i = 0; i < num_edge_types; i++) {
    for (int j = 0; j < bs; j++) {
      for (int k = 0; k < count_data[i * bs + j]; k++) {
        T node = x_data[j];
        dst[cnt++] = node_map[node];
      }
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
