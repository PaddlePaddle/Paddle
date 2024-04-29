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

#pragma once

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
std::shared_ptr<phi::Allocation> FillHashTable(const Context& dev_ctx,
                                               const T* input,
                                               int num_input,
                                               int64_t len_hashtable,
                                               T* keys,
                                               int* values,
                                               int* key_index,
                                               int* final_nodes_len);

template <typename T, typename Context>
void ReindexSrc(const Context& dev_ctx,
                T* edges_src,
                T* keys,
                int* values,
                int64_t num_edges,
                int64_t table_size);

template <typename T, typename Context>
void ReindexDst(const Context& dev_ctx,
                T* reindex_dst_data,
                int* scan_dst_data,
                const int* count_data,
                int num_edge_types,
                int node_len);

template <typename T, typename Context>
void GraphReindexKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& neighbors,
                        const DenseTensor& count,
                        const paddle::optional<DenseTensor>& hashtable_value,
                        const paddle::optional<DenseTensor>& hashtable_index,
                        DenseTensor* reindex_src,
                        DenseTensor* reindex_dst,
                        DenseTensor* out_nodes);

}  // namespace phi
