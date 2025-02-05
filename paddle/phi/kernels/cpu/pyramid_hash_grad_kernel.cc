// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <xxhash.h>

#include <algorithm>
#include <cmath>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/search_compute.h"

namespace phi {

template <typename T>
void hash_embedding_bp(const T* hash_id,
                       int len,
                       const T* top_pos,
                       T* weights,
                       T mlr,
                       int _num_emb,
                       int _rand_len,
                       int _space_len) {
  for (int j = 0; j != _num_emb; j += _rand_len) {
    unsigned int pos = XXH32(hash_id, len * sizeof(T), j) % _space_len;
    phi::funcs::axpy(top_pos + j, weights + pos, _rand_len, mlr);
  }
}

template <typename T, typename Context>
void CPUPyramidHashOPGradKernel(const Context& dev_ctx,
                                const DenseTensor& x,
                                const DenseTensor& w,
                                const DenseTensor& drop_pos,
                                const DenseTensor& x_temp_out,
                                const DenseTensor& out_grad,
                                int num_emb,
                                int space_len,
                                int pyramid_layer,
                                int rand_len,
                                float drop_out_percent UNUSED,
                                int is_training,
                                bool use_filter,
                                int white_list_len UNUSED,
                                int black_list_len UNUSED,
                                int seed UNUSED,
                                float lr,
                                const std::string& distribute_update_vars
                                    UNUSED,
                                DenseTensor* x_grad) {
  auto* bottom = &x;
  auto* _blobs = &w;
  auto* drop_pos_p = &drop_pos;
  auto* top = &out_grad;

  int _num_emb = num_emb;
  float _lr = lr;
  int _rand_len = rand_len;
  int _space_len = space_len;
  int _pyramid_layer = pyramid_layer;

  auto* buff = &x_temp_out;
  auto* bottom_data = buff->data<T>();

  int _slot_len = static_cast<int>(bottom->dims()[0]);
  if (static_cast<size_t>(_slot_len) == bottom->lod()[0].size() - 1 &&
      std::count(bottom_data, bottom_data + _slot_len, -1) == _slot_len) {
    return;
  }

  auto& offset = bottom->lod()[0];
  auto& drop_pos_offset = drop_pos_p->lod()[0];

  const auto* top_diff = top->data<T>();
  // in-place update weight, so need const_cast
  T* weights = const_cast<T*>(_blobs->data<T>());
  T mlr = -1.0 * _lr;

  const int* iter = drop_pos_p->data<int>();
  int top_counter = 0;
  for (size_t i = 0; i < offset.size() - 1; ++i) {
    int w = static_cast<int>(offset[i + 1] - offset[i]);
    int w_drop = static_cast<int>(drop_pos_offset[i + 1] - drop_pos_offset[i]);
    if (w_drop == 0) {
      top_counter++;
    }
    if (w > 1) {
      for (int ilayer = 1; ilayer < _pyramid_layer && ilayer < w; ++ilayer) {
        for (int l = 0; l < w - ilayer; ++l) {
          if (*(iter++) == 0) {
            // do nothing
          } else {
            const T* top_pos = top_diff + top_counter++ * _num_emb;
            hash_embedding_bp<T>((const T*)(bottom_data + offset[i] + l),
                                 ilayer + 1,
                                 top_pos,
                                 weights,
                                 mlr,
                                 _num_emb,
                                 _rand_len,
                                 _space_len);
          }
        }
      }
    } else {
      // do nothing
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(pyramid_hash_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::CPUPyramidHashOPGradKernel,
                   float) {}
