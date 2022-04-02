/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <algorithm>
#include <cfloat>
#include <string>
#include <vector>
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

int const kThreadsPerBlock = sizeof(uint64_t) * 8;

static const double kBBoxClipDefault = std::log(1000.0 / 16.0);

struct RangeInitFunctor {
  int start_;
  int delta_;
  int *out_;
  __device__ void operator()(size_t i) { out_[i] = start_ + i * delta_; }
};

template <typename T>
static void SortDescending(const platform::CUDADeviceContext &ctx,
                           const Tensor &value, Tensor *value_out,
                           Tensor *index_out) {
  int num = static_cast<int>(value.numel());
  Tensor index_in_t;
  int *idx_in = index_in_t.mutable_data<int>({num}, ctx.GetPlace());
  platform::ForRange<platform::CUDADeviceContext> for_range(ctx, num);
  for_range(RangeInitFunctor{0, 1, idx_in});

  int *idx_out = index_out->mutable_data<int>({num}, ctx.GetPlace());

  const T *keys_in = value.data<T>();
  T *keys_out = value_out->mutable_data<T>({num}, ctx.GetPlace());

  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending<T, int>(
      nullptr, temp_storage_bytes, keys_in, keys_out, idx_in, idx_out, num, 0,
      sizeof(T) * 8, ctx.stream());
  // Allocate temporary storage
  auto place = ctx.GetPlace();
  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);

  // Run sorting operation
  cub::DeviceRadixSort::SortPairsDescending<T, int>(
      d_temp_storage->ptr(), temp_storage_bytes, keys_in, keys_out, idx_in,
      idx_out, num, 0, sizeof(T) * 8, ctx.stream());
}

template <typename T>
struct BoxDecodeAndClipFunctor {
  const T *anchor;
  const T *deltas;
  const T *var;
  const int *index;
  const T *im_info;
  const bool pixel_offset;

  T *proposals;

  BoxDecodeAndClipFunctor(const T *anchor, const T *deltas, const T *var,
                          const int *index, const T *im_info, T *proposals,
                          bool pixel_offset = true)
      : anchor(anchor),
        deltas(deltas),
        var(var),
        index(index),
        im_info(im_info),
        proposals(proposals),
        pixel_offset(pixel_offset) {}

  T bbox_clip_default{static_cast<T>(kBBoxClipDefault)};

  __device__ void operator()(size_t i) {
    int k = index[i] * 4;
    T axmin = anchor[k];
    T aymin = anchor[k + 1];
    T axmax = anchor[k + 2];
    T aymax = anchor[k + 3];

    T offset = pixel_offset ? static_cast<T>(1.0) : 0;
    T w = axmax - axmin + offset;
    T h = aymax - aymin + offset;
    T cx = axmin + 0.5 * w;
    T cy = aymin + 0.5 * h;

    T dxmin = deltas[k];
    T dymin = deltas[k + 1];
    T dxmax = deltas[k + 2];
    T dymax = deltas[k + 3];

    T d_cx, d_cy, d_w, d_h;
    if (var) {
      d_cx = cx + dxmin * w * var[k];
      d_cy = cy + dymin * h * var[k + 1];
      d_w = exp(Min(dxmax * var[k + 2], bbox_clip_default)) * w;
      d_h = exp(Min(dymax * var[k + 3], bbox_clip_default)) * h;
    } else {
      d_cx = cx + dxmin * w;
      d_cy = cy + dymin * h;
      d_w = exp(Min(dxmax, bbox_clip_default)) * w;
      d_h = exp(Min(dymax, bbox_clip_default)) * h;
    }

    T oxmin = d_cx - d_w * 0.5;
    T oymin = d_cy - d_h * 0.5;
    T oxmax = d_cx + d_w * 0.5 - offset;
    T oymax = d_cy + d_h * 0.5 - offset;

    proposals[i * 4] = Max(Min(oxmin, im_info[1] - offset), 0.);
    proposals[i * 4 + 1] = Max(Min(oymin, im_info[0] - offset), 0.);
    proposals[i * 4 + 2] = Max(Min(oxmax, im_info[1] - offset), 0.);
    proposals[i * 4 + 3] = Max(Min(oymax, im_info[0] - offset), 0.);
  }

  __device__ __forceinline__ T Min(T a, T b) const { return a > b ? b : a; }

  __device__ __forceinline__ T Max(T a, T b) const { return a > b ? a : b; }
};

template <typename T, int BlockSize>
static __global__ void FilterBBoxes(const T *bboxes, const T *im_info,
                                    const T min_size, const int num,
                                    int *keep_num, int *keep,
                                    bool is_scale = true,
                                    bool pixel_offset = true) {
  T im_h = im_info[0];
  T im_w = im_info[1];

  int cnt = 0;
  __shared__ int keep_index[BlockSize];

  CUDA_KERNEL_LOOP(i, num) {
    keep_index[threadIdx.x] = -1;
    __syncthreads();

    int k = i * 4;
    T xmin = bboxes[k];
    T ymin = bboxes[k + 1];
    T xmax = bboxes[k + 2];
    T ymax = bboxes[k + 3];
    T offset = pixel_offset ? static_cast<T>(1.0) : 0;
    T w = xmax - xmin + offset;
    T h = ymax - ymin + offset;
    if (pixel_offset) {
      T cx = xmin + w / 2.;
      T cy = ymin + h / 2.;

      if (is_scale) {
        w = (xmax - xmin) / im_info[2] + 1.;
        h = (ymax - ymin) / im_info[2] + 1.;
      }

      if (w >= min_size && h >= min_size && cx <= im_w && cy <= im_h) {
        keep_index[threadIdx.x] = i;
      }
    } else {
      if (w >= min_size && h >= min_size) {
        keep_index[threadIdx.x] = i;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int size = (num - i) < BlockSize ? num - i : BlockSize;
      for (int j = 0; j < size; ++j) {
        if (keep_index[j] > -1) {
          keep[cnt++] = keep_index[j];
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    keep_num[0] = cnt;
  }
}

static __device__ float IoU(const float *a, const float *b,
                            const bool pixel_offset = true) {
  float offset = pixel_offset ? static_cast<float>(1.0) : 0;
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + offset, 0.f),
        height = max(bottom - top + offset, 0.f);
  float inter_s = width * height;
  float s_a = (a[2] - a[0] + offset) * (a[3] - a[1] + offset);
  float s_b = (b[2] - b[0] + offset) * (b[3] - b[1] + offset);
  return inter_s / (s_a + s_b - inter_s);
}

static __global__ void NMSKernel(const int n_boxes,
                                 const float nms_overlap_thresh,
                                 const float *dev_boxes, uint64_t *dev_mask,
                                 bool pixel_offset = true) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  const int row_size =
      min(n_boxes - row_start * kThreadsPerBlock, kThreadsPerBlock);
  const int col_size =
      min(n_boxes - col_start * kThreadsPerBlock, kThreadsPerBlock);

  __shared__ float block_boxes[kThreadsPerBlock * 4];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 4 + 0] =
        dev_boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_boxes[threadIdx.x * 4 + 1] =
        dev_boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_boxes[threadIdx.x * 4 + 2] =
        dev_boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_boxes[threadIdx.x * 4 + 3] =
        dev_boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = kThreadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 4;
    int i = 0;
    uint64_t t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (IoU(cur_box, block_boxes + i * 4, pixel_offset) >
          nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, kThreadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

template <typename T>
static void NMS(const platform::CUDADeviceContext &ctx, const Tensor &proposals,
                const Tensor &sorted_indices, const T nms_threshold,
                Tensor *keep_out, bool pixel_offset = true) {
  int boxes_num = proposals.dims()[0];
  const int col_blocks = DIVUP(boxes_num, kThreadsPerBlock);
  dim3 blocks(DIVUP(boxes_num, kThreadsPerBlock),
              DIVUP(boxes_num, kThreadsPerBlock));
  dim3 threads(kThreadsPerBlock);

  const T *boxes = proposals.data<T>();
  auto place = ctx.GetPlace();
  auto mask_ptr = memory::Alloc(ctx, boxes_num * col_blocks * sizeof(uint64_t));
  uint64_t *mask_dev = reinterpret_cast<uint64_t *>(mask_ptr->ptr());

  NMSKernel<<<blocks, threads, 0, ctx.stream()>>>(
      boxes_num, nms_threshold, boxes, mask_dev, pixel_offset);

  std::vector<uint64_t> remv(col_blocks);
  memset(&remv[0], 0, sizeof(uint64_t) * col_blocks);

  std::vector<uint64_t> mask_host(boxes_num * col_blocks);
  memory::Copy(platform::CPUPlace(), mask_host.data(), place, mask_dev,
               boxes_num * col_blocks * sizeof(uint64_t), ctx.stream());

  std::vector<int> keep_vec;
  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / kThreadsPerBlock;
    int inblock = i % kThreadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      ++num_to_keep;
      keep_vec.push_back(i);
      uint64_t *p = mask_host.data() + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  int *keep = keep_out->mutable_data<int>({num_to_keep}, ctx.GetPlace());
  memory::Copy(place, keep, platform::CPUPlace(), keep_vec.data(),
               sizeof(int) * num_to_keep, ctx.stream());
  ctx.Wait();
}

}  // namespace operators
}  // namespace paddle
