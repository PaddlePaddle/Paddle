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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {
namespace funcs {

// This code is referenced from apex's multi_tensor_apply.cuh.
// https://github.com/NVIDIA/apex

template <int N, int MaxTensorSize, int MaxBlockSize>
struct TensorAndBlockInfo {
  void *tensor_addrs[N - 1][MaxTensorSize];
  const void *grads[MaxTensorSize];
  int sizes[MaxTensorSize];
  uint8_t tensor_ids[MaxBlockSize];
  // int16
  uint16_t chunk_ids[MaxBlockSize];
  int start_chunk_id;

  DEVICE void GetChunkIdAndTensorId(int *chunk_id, int *tensor_id) const {
    int block_id = blockIdx.x;
    int tmp_tensor_id = tensor_ids[block_id];
    *chunk_id = static_cast<int>(chunk_ids[block_id]) +
                (tmp_tensor_id == 0) * start_chunk_id;
    *tensor_id = tmp_tensor_id;
  }
};

template <int N, int MaxTensorSize, int MaxBlockSize>
struct TensorAndBlockInfoList {
  using Info = TensorAndBlockInfo<N, MaxTensorSize, MaxBlockSize>;

  std::vector<Info> infos;
  std::vector<int> chunk_nums;

  struct TensorMeta {
    const void *ptr = nullptr;
    int numel = 0;
  };

  struct BlockMeta {
    const void *ptrs[N];
    int size;
    int tensor_id;
    int chunk_id;
  };

  std::array<std::vector<TensorMeta>, N> metas;

  int block_size = 0;
  int chunk_size = 0;

  void Validate() const {
    LOG(INFO) << "Begin to validate";
    int n = metas[0].size();
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < n; ++j) {
        PADDLE_ENFORCE_EQ(metas[i][j].numel, metas[0][j].numel);
      }
    }

    PADDLE_ENFORCE_EQ(infos.size(), chunk_nums.size());

    std::vector<BlockMeta> block_metas;
    for (int i = 0; i < n; ++i) {
      int numel = metas[0][i].numel;
      int chunk_num = (metas[0][i].numel + chunk_size - 1) / chunk_size;
      for (int j = 0; j < chunk_num; ++j) {
        block_metas.emplace_back();
        auto &new_meta = block_metas.back();
        new_meta.size = numel;
        new_meta.tensor_id = i;
        new_meta.chunk_id = j;
        for (int k = 0; k < N; ++k) {
          new_meta.ptrs[k] = metas[k][i].ptr;
        }
      }
    }

    int cur_tensor_id = -1, tensor_num = 0, block_num = 0;
    int start = 0;
    std::vector<std::pair<int, int>> segments;
    for (size_t i = 0; i < block_metas.size(); ++i) {
      if (cur_tensor_id < 0) {
        cur_tensor_id = block_metas[i].tensor_id;
        tensor_num = 1;
      } else if (block_metas[i].tensor_id != cur_tensor_id) {
        ++tensor_num;
        cur_tensor_id = block_metas[i].tensor_id;
      }

      ++block_num;
      if (tensor_num > MaxTensorSize || block_num > MaxBlockSize) {
        tensor_num = 1;
        block_num = 1;
        cur_tensor_id = block_metas[i].tensor_id;
        segments.emplace_back(start, i);
        start = i;
      }
    }

    if (start != block_metas.size()) {
      segments.emplace_back(start, block_metas.size());
    }

    PADDLE_ENFORCE_EQ(segments.size(), infos.size());
    PADDLE_ENFORCE_EQ(segments.size(), chunk_nums.size());
    LOG(INFO) << "--------------------------- Segment num: " << segments.size();
    for (size_t i = 0; i < segments.size(); ++i) {
      PADDLE_ENFORCE_LT(segments[i].first, segments[i].second);
      PADDLE_ENFORCE_EQ(segments[i].second - segments[i].first, chunk_nums[i]);
      if (i > 0) {
        PADDLE_ENFORCE_EQ(segments[i - 1].second, segments[i].first);
      }

      LOG(INFO) << "Segment id=" << i << " range=[" << segments[i].first << ","
                << segments[i].second
                << ") start_chunk_id=" << infos[i].start_chunk_id;
      int start = segments[i].first;
      int end = segments[i].second;
      PADDLE_ENFORCE_EQ(block_metas[start].chunk_id, infos[i].start_chunk_id);
      int start_tensor_id = block_metas[start].tensor_id;
      for (int j = start; j < end; ++j) {
        int idx = j - start;
        int tensor_id = block_metas[j].tensor_id;
        int chunk_id = block_metas[j].chunk_id;

        LOG(INFO) << "tensor_id=" << tensor_id << " chunk_id=" << chunk_id
                  << " size=" << metas[0][tensor_id].numel
                  << " chunk_size=" << chunk_size;

        if (start_tensor_id == block_metas[j].tensor_id) {
          PADDLE_ENFORCE_EQ(chunk_id,
                            infos[i].start_chunk_id + infos[i].chunk_ids[idx]);
        } else {
          PADDLE_ENFORCE_EQ(chunk_id, infos[i].chunk_ids[idx]);
        }

        PADDLE_ENFORCE_EQ(tensor_id - start_tensor_id,
                          infos[i].tensor_ids[idx]);
        tensor_id -= start_tensor_id;
        for (int k = 0; k < N; ++k) {
          const void *ptr = (k + 1 == N) ? infos[i].grads[tensor_id]
                                         : infos[i].tensor_addrs[k][tensor_id];
          int tmp1 = FindPtr(k, ptr);
          int tmp2 = FindPtr(k, block_metas[j].ptrs[k]);
          PADDLE_ENFORCE_EQ(tmp1,
                            tmp2,
                            phi::errors::InvalidArgument(
                                "Segment %d, j = %d, k = %d, tensor_id = %d, "
                                "start_tensor_id = %d",
                                i,
                                j,
                                k,
                                tensor_id,
                                start_tensor_id));
          PADDLE_ENFORCE_EQ(start_tensor_id + tensor_id, tmp1);
          PADDLE_ENFORCE_EQ(ptr, block_metas[j].ptrs[k]);
        }
      }
    }
  }

  void BuildMetas(const std::vector<std::vector<DenseTensor *>> &input_vector,
                  const std::vector<const DenseTensor *> &grads) {
    CheckNum(grads.size());
    PADDLE_ENFORCE_EQ(input_vector.size(), N - 1);
    for (int i = 0; i < N - 1; ++i) {
      PADDLE_ENFORCE_EQ(input_vector[i].size(), grads.size());
    }

    for (int i = 0; i < N - 1; ++i) {
      metas[i].resize(input_vector[i].size());
      for (size_t j = 0; j < metas[i].size(); ++j) {
        metas[i][j].ptr = input_vector[i][j]->data();
        auto numel = input_vector[i][j]->numel();
        CheckNum(numel);
        metas[i][j].numel = static_cast<int>(numel);
      }
    }

    metas[N - 1].resize(grads.size());
    for (size_t j = 0; j < metas[N - 1].size(); ++j) {
      metas[N - 1][j].ptr = grads[j]->data();
      auto numel = grads[j]->numel();
      CheckNum(numel);
      metas[N - 1][j].numel = static_cast<int>(numel);
    }
  }

 private:
  size_t FindPtr(int idx, const void *ptr) const {
    CheckIdx(idx);
    int n = metas[0].size();
    int ret = -1;
    for (int i = 0; i < n; ++i) {
      if (metas[idx][i].ptr == ptr) {
        PADDLE_ENFORCE_EQ(ret, -1);
        ret = i;
      }
    }
    PADDLE_ENFORCE_GE(ret, 0);
    return ret;
  }

  static void CheckIdx(int idx) {
    PADDLE_ENFORCE_GE(idx, 0);
    PADDLE_ENFORCE_LT(idx, N);
  }

  static void CheckNum(size_t n) {
    size_t limit = std::numeric_limits<int>::max();
    PADDLE_ENFORCE_LE(n, limit);
  }
};

template <int N,
          int MaxTensorSize,
          int MaxBlockSize,
          typename Functor,
          typename... ArgTypes>
__global__ void MultiTensorApplyCudaKernel(
    int chunk_size,
    TensorAndBlockInfo<N, MaxTensorSize, MaxBlockSize> t_info,
    Functor functor,
    ArgTypes... args) {
  functor(chunk_size, t_info, args...);
}

template <int InputNum,
          int MaxTensorSize,
          int MaxBlockSize,
          typename Functor,
          typename Context,
          typename... ArgTypes>
void LaunchMultiTensorApplyKernel(
    const Context &dev_ctx,
    int block_size,
    int chunk_size,
    const std::vector<std::vector<DenseTensor *>> &input_vector,
    const std::vector<const DenseTensor *> &grads,
    Functor functor,
    ArgTypes... args) {
  PADDLE_ENFORCE_EQ(
      input_vector.size(),
      InputNum - 1,
      errors::InvalidArgument(
          "input_vector.size() != InputNum - 1, the input vector's size is "
          "unequal to InputNum - 1, please cheack grads, params, momemts1, "
          "moments2, and, master_params."));
  size_t length = input_vector[0].size();
  PADDLE_ENFORCE_GT(
      length,
      0,
      errors::InvalidArgument(
          "input_vector[0].size() is not > 0, please cheack params."));
  auto place = input_vector[0][0]->place();
  PADDLE_ENFORCE_EQ(
      place,
      GPUPlace(),
      errors::InvalidArgument(
          "expected input to be on gpu, but input is on cpu now."));
  for (size_t i = 0; i < input_vector.size(); i++) {
    PADDLE_ENFORCE_EQ(
        input_vector[i].size(),
        length,
        errors::InvalidArgument(
            "some input vectors' size mismatch other input vector."));
    for (size_t j = 0; j < input_vector[i].size(); j++) {
      PADDLE_ENFORCE_EQ(
          input_vector[i][j]->place(),
          place,
          errors::InvalidArgument(
              "A tensor was not on the same device as the first tensor"));
      PADDLE_ENFORCE_EQ(input_vector[i][j]->numel(),
                        input_vector[0][j]->numel(),
                        errors::InvalidArgument(
                            "The number of elements of Inputs must be equal."));
    }
  }

  size_t tensors_size = input_vector[0].size();

  TensorAndBlockInfo<InputNum, MaxTensorSize, MaxBlockSize> t_info;
  t_info.start_chunk_id = 0;

  TensorAndBlockInfoList<InputNum, MaxTensorSize, MaxBlockSize> infos;
  infos.block_size = block_size;
  infos.chunk_size = chunk_size;
  infos.BuildMetas(input_vector, grads);

  auto stream = dev_ctx.stream();
  int block_id = 0;
  int tensor_id = 0;
  for (int t = 0; t < tensors_size; t++) {
    t_info.sizes[tensor_id] = input_vector[0][t]->numel();
    t_info.grads[tensor_id] = grads[t]->data();
    for (int d = 0; d < InputNum - 1; d++) {
      t_info.tensor_addrs[d][tensor_id] = input_vector[d][t]->data();
    }
    tensor_id++;
    int chunks_this_tensor =
        (input_vector[0][t]->numel() + chunk_size - 1) / chunk_size;

    constexpr auto kMaxChunkId = std::numeric_limits<uint16_t>::max();
    for (int chunk = 0; chunk < chunks_this_tensor; chunk++) {
      t_info.tensor_ids[block_id] = tensor_id - 1;
      auto saved_chunk_id =
          (tensor_id == 1 ? chunk - t_info.start_chunk_id : chunk);
      PADDLE_ENFORCE_GE(saved_chunk_id,
                        0,
                        errors::InvalidArgument(
                            "The chunk id is less than 0 in "
                            "MultiTensorApplyKernel. This may be a bug."));
      PADDLE_ENFORCE_LE(
          saved_chunk_id,
          kMaxChunkId,
          errors::InvalidArgument(
              "The chunk id exceeds maximum value %d. This may be a bug.",
              kMaxChunkId));
      t_info.chunk_ids[block_id] = saved_chunk_id;
      block_id++;
      bool reach_tensors_limit =
          (tensor_id == MaxTensorSize && chunk == chunks_this_tensor - 1);
      bool reach_blocks_limit = (block_id == MaxBlockSize);
      bool finish_compute =
          (t == tensors_size - 1 && chunk == chunks_this_tensor - 1);
      if (reach_tensors_limit || reach_blocks_limit || finish_compute) {
        MultiTensorApplyCudaKernel<InputNum,
                                   MaxTensorSize,
                                   MaxBlockSize,
                                   Functor,
                                   ArgTypes...>
            <<<block_id, block_size, 0, stream>>>(
                chunk_size, t_info, functor, args...);
        infos.infos.push_back(t_info);
        infos.chunk_nums.push_back(block_id);

        block_id = 0;
        if (chunk == chunks_this_tensor - 1) {
          tensor_id = 0;
          t_info.start_chunk_id = 0;
        } else {
          t_info.sizes[0] = t_info.sizes[tensor_id - 1];
          t_info.grads[0] = t_info.grads[tensor_id - 1];
          for (int d = 0; d < InputNum - 1; d++) {
            t_info.tensor_addrs[d][0] = t_info.tensor_addrs[d][tensor_id - 1];
          }
          tensor_id = 1;
          t_info.start_chunk_id = chunk + 1;
        }
      }
    }
  }

  infos.Validate();
}

}  // namespace funcs
}  // namespace phi
