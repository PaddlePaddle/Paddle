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
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math/bloomfilter.h"
#include "paddle/phi/kernels/funcs/search_compute.h"

namespace phi {

#ifndef _WIN32
bool should_use_term(phi::math::bloomfilter* _filter,
                     phi::math::bloomfilter* _black_filter,
                     const float* word_repr,
                     int len) {
  return (!_filter || 1 == phi::math::bloomfilter_get(
                               _filter, word_repr, len * sizeof(float))) &&
         (!_black_filter ||
          0 == phi::math::bloomfilter_get(
                   _black_filter, word_repr, len * sizeof(float)));
}

template <typename T>
void hash_embedding_ff(const float* hash_id,
                       int len,
                       T* top_pos,
                       const T* weights,
                       int _num_emb,
                       int _rand_len,
                       int _space_len) {
  unsigned int pos1 = XXH32(hash_id, len * sizeof(float), 0) % _space_len;
  unsigned int pos2 =
      XXH32(hash_id, len * sizeof(float), _rand_len) % _space_len;

  for (int j = 0; j != _num_emb; j += _rand_len) {
    if (j + _rand_len < _num_emb) {
      __builtin_prefetch(weights + pos2);
      __builtin_prefetch(top_pos + j + _rand_len);
    }

    unsigned int pos3 =
        XXH32(hash_id, len * sizeof(float), j + 2 * _rand_len) % _space_len;
    memcpy(top_pos + j, const_cast<T*>(weights + pos1), _rand_len * sizeof(T));
    pos1 = pos2;
    pos2 = pos3;
  }
}

template <typename T, typename Context>
void CPUPyramidHashOPKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& w,
                            const DenseTensor& white_list,
                            const DenseTensor& black_list,
                            int num_emb,
                            int space_len,
                            int pyramid_layer,
                            int rand_len,
                            float drop_out_percent,
                            int is_training,
                            bool use_filter,
                            int white_list_len,
                            int black_list_len,
                            int seed,
                            float lr,
                            const std::string& distribute_update_vars,
                            DenseTensor* out,
                            DenseTensor* drop_pos,
                            DenseTensor* x_temp_out) {
  auto* bottom = &x;
  auto* _blobs_0 = &w;
  auto* _blobs_1 = &white_list;
  auto* _blobs_2 = &black_list;
  auto* top = out;

  int _num_emb = num_emb;
  int _pyramid_layer = pyramid_layer;
  int _is_training = is_training;
  unsigned int _seed = (unsigned int)seed;
  int _rand_len = rand_len;
  int _space_len = space_len;
  float _drop_out_percent = drop_out_percent;

  const auto& offset = bottom->lod()[0];
  const auto* bottom_data_ori = bottom->data<int32_t>();
  auto* buff = x_temp_out;
  buff->Resize(common::make_ddim({bottom->dims()[0], bottom->dims()[1]}));
  float* bottom_data = dev_ctx.template Alloc<float>(buff);
  for (int i = 0; i < bottom->dims()[0]; i++) {
    bottom_data[i] = bottom_data_ori[i];  // NOLINT
  }

  const auto* weights = _blobs_0->data<T>();

  std::vector<size_t> top_offset;
  top_offset.resize(offset.size());
  top_offset[0] = 0;

  phi::math::bloomfilter* _filter = nullptr;
  phi::math::bloomfilter* _black_filter = nullptr;
  if (use_filter) {
    if (white_list_len != 0) {
      _filter = (phi::math::bloomfilter*)_blobs_1->data<float>();
      PADDLE_ENFORCE_EQ(
          phi::math::bloomfilter_check(_filter),
          1,
          common::errors::PreconditionNotMet(
              "The white filter is not loaded successfully, please make sure "
              "'white_list_len': %d is valid for Input(WhiteList).",
              white_list_len));
    }
    if (black_list_len != 0) {
      _black_filter = (phi::math::bloomfilter*)_blobs_2->data<float>();
      PADDLE_ENFORCE_EQ(
          phi::math::bloomfilter_check(_black_filter),
          1,
          common::errors::PreconditionNotMet(
              "The black filter is not loaded successfully, please make sure "
              "'black_list_len': %d is valid for Input(BlackList).",
              black_list_len));
    }
  }

  drop_pos->Resize(common::make_ddim(
      {bottom->dims()[0] * bottom->dims()[1] * _pyramid_layer, 1}));
  std::vector<size_t> drop_pos_offset;
  drop_pos_offset.resize(offset.size());
  drop_pos_offset[0] = 0;
  int* iter = dev_ctx.template Alloc<int>(drop_pos);
  int* iter_end = iter;

  for (size_t i = 0; i < top_offset.size() - 1; ++i) {
    int w = static_cast<int>(offset[i + 1] - offset[i]);
    int nsentense_with_pyramid = 0;
    if (w < 2) {
      nsentense_with_pyramid = 0;
    } else {
      for (int ilayer = 1; ilayer < _pyramid_layer && ilayer < w; ++ilayer) {
        for (int l = 0; l < w - ilayer; ++l) {
          if (should_use_term(_filter,
                              _black_filter,
                              (const float*)(bottom_data + offset[i] + l),
                              ilayer + 1)) {
            if (_is_training != 0) {
              unsigned int rand_val = rand_r(&_seed);
              double rate = static_cast<double>(rand_val) / (RAND_MAX);
              *(iter_end++) = (rate < _drop_out_percent ? 0 : 1);
            } else {
              *(iter_end++) = 1;
            }
          } else {
            *(iter_end++) = 0;
          }
        }
      }
      nsentense_with_pyramid = static_cast<int>(std::count(iter, iter_end, 1));
      iter = iter_end;
    }
    drop_pos_offset[i + 1] = drop_pos_offset[i] + nsentense_with_pyramid;
    top_offset[i + 1] =
        top_offset[i] +
        (nsentense_with_pyramid == 0 ? 1 : nsentense_with_pyramid);
  }

  int top_l = static_cast<int>(top_offset[top_offset.size() - 1]);

  phi::LegacyLoD top_lod;
  top_lod.push_back(top_offset);
  top->set_lod(top_lod);
  top->Resize(common::make_ddim({top_l, _num_emb}));
  auto* top_data = dev_ctx.template Alloc<T>(top);

  phi::LegacyLoD drop_pos_lod;
  drop_pos_lod.push_back(drop_pos_offset);
  drop_pos->set_lod(drop_pos_lod);

  iter = dev_ctx.template Alloc<int>(drop_pos);
  int top_counter = 0;
  for (size_t i = 0; i < offset.size() - 1; ++i) {
    int w_drop = static_cast<int>(drop_pos_offset[i + 1] - drop_pos_offset[i]);
    int w = static_cast<int>(offset[i + 1] - offset[i]);
    if (w_drop == 0) {
      if (w >= 2) {
        for (int ilayer = 1; ilayer < _pyramid_layer && ilayer < w; ++ilayer) {
          for (int l = 0; l < w - ilayer; ++l) {
            iter++;
          }
        }
      }
      auto* top_pos = top_data + top_counter++ * _num_emb;
      memset(top_pos, 0, _num_emb * sizeof(T));
      continue;
    }
    if (w >= 2) {
      for (int ilayer = 1; ilayer < _pyramid_layer && ilayer < w; ++ilayer) {
        for (int l = 0; l < w - ilayer; ++l) {
          if (*(iter++) == 0) {
            // do nothing
          } else {
            auto* top_pos = top_data + top_counter++ * _num_emb;
            hash_embedding_ff<T>((const float*)(bottom_data + offset[i] + l),
                                 ilayer + 1,
                                 top_pos,
                                 weights,
                                 _num_emb,
                                 _rand_len,
                                 _space_len);
          }
        }
      }
    }
  }
  if (iter != iter_end) {
    exit(1);
  }
  auto weight_type = phi::TransToProtoVarType(_blobs_0->dtype());
  if (_is_training == 0 && weight_type != phi::ProtoDataType::INT8) {
    phi::funcs::axpy_noadd(
        top_data, top_data, top->dims()[0] * top->dims()[1], _drop_out_percent);
  }
}
#endif

#ifdef _WIN32
template <typename T, typename Context>
void CPUPyramidHashOPKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& w,
                            const DenseTensor& white_list,
                            const DenseTensor& black_list,
                            int num_emb,
                            int space_len,
                            int pyramid_layer,
                            int rand_len,
                            float drop_out_percent,
                            int is_training,
                            bool use_filter,
                            int white_list_len,
                            int black_list_len,
                            int seed,
                            float lr,
                            const std::string& distribute_update_vars,
                            DenseTensor* out,
                            DenseTensor* drop_pos,
                            DenseTensor* x_temp_out) {}
#endif
}  // namespace phi

PD_REGISTER_KERNEL(
    pyramid_hash, CPU, ALL_LAYOUT, phi::CPUPyramidHashOPKernel, float, int8_t) {
  kernel->InputAt(0).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
}
