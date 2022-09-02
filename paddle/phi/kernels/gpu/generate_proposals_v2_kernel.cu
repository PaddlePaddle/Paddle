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

#include "paddle/phi/kernels/generate_proposals_v2_kernel.h"

#include <algorithm>
#include <vector>
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/gather.cu.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

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
static void SortDescending(const phi::GPUContext &ctx,
                           const DenseTensor &value,
                           DenseTensor *value_out,
                           DenseTensor *index_out) {
  int num = static_cast<int>(value.numel());
  DenseTensor index_in_t;
  index_in_t.Resize(phi::make_ddim({num}));
  int *idx_in = ctx.template Alloc<int>(&index_in_t);
  phi::funcs::ForRange<phi::GPUContext> for_range(ctx, num);
  for_range(RangeInitFunctor{0, 1, idx_in});

  index_out->Resize(phi::make_ddim({num}));
  int *idx_out = ctx.template Alloc<int>(index_out);

  const T *keys_in = value.data<T>();
  value_out->Resize(phi::make_ddim({num}));
  T *keys_out = ctx.template Alloc<T>(value_out);

  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending<T, int>(nullptr,
                                                    temp_storage_bytes,
                                                    keys_in,
                                                    keys_out,
                                                    idx_in,
                                                    idx_out,
                                                    num,
                                                    0,
                                                    sizeof(T) * 8,
                                                    ctx.stream());
  // Allocate temporary storage
  auto place = ctx.GetPlace();
  auto d_temp_storage = paddle::memory::Alloc(place, temp_storage_bytes);

  // Run sorting operation
  cub::DeviceRadixSort::SortPairsDescending<T, int>(d_temp_storage->ptr(),
                                                    temp_storage_bytes,
                                                    keys_in,
                                                    keys_out,
                                                    idx_in,
                                                    idx_out,
                                                    num,
                                                    0,
                                                    sizeof(T) * 8,
                                                    ctx.stream());
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

  BoxDecodeAndClipFunctor(const T *anchor,
                          const T *deltas,
                          const T *var,
                          const int *index,
                          const T *im_info,
                          T *proposals,
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
static __global__ void FilterBBoxes(const T *bboxes,
                                    const T *im_info,
                                    const T min_size,
                                    const int num,
                                    int *keep_num,
                                    int *keep,
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

static __device__ float IoU(const float *a,
                            const float *b,
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
                                 const float *dev_boxes,
                                 uint64_t *dev_mask,
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
static void NMS(const phi::GPUContext &ctx,
                const DenseTensor &proposals,
                const DenseTensor &sorted_indices,
                const T nms_threshold,
                DenseTensor *keep_out,
                bool pixel_offset = true) {
  int boxes_num = proposals.dims()[0];
  const int col_blocks = DIVUP(boxes_num, kThreadsPerBlock);
  dim3 blocks(DIVUP(boxes_num, kThreadsPerBlock),
              DIVUP(boxes_num, kThreadsPerBlock));
  dim3 threads(kThreadsPerBlock);

  const T *boxes = proposals.data<T>();
  auto place = ctx.GetPlace();
  auto mask_ptr = paddle::memory::Alloc(
      place,
      boxes_num * col_blocks * sizeof(uint64_t),
      phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  uint64_t *mask_dev = reinterpret_cast<uint64_t *>(mask_ptr->ptr());

  NMSKernel<<<blocks, threads, 0, ctx.stream()>>>(
      boxes_num, nms_threshold, boxes, mask_dev, pixel_offset);

  std::vector<uint64_t> remv(col_blocks);
  memset(&remv[0], 0, sizeof(uint64_t) * col_blocks);

  std::vector<uint64_t> mask_host(boxes_num * col_blocks);
  paddle::memory::Copy(CPUPlace(),
                       mask_host.data(),
                       place,
                       mask_dev,
                       boxes_num * col_blocks * sizeof(uint64_t),
                       ctx.stream());

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
  keep_out->Resize(phi::make_ddim({num_to_keep}));
  int *keep = ctx.template Alloc<int>(keep_out);
  paddle::memory::Copy(place,
                       keep,
                       CPUPlace(),
                       keep_vec.data(),
                       sizeof(int) * num_to_keep,
                       ctx.stream());
  ctx.Wait();
}

template <typename T>
static std::pair<DenseTensor, DenseTensor> ProposalForOneImage(
    const phi::GPUContext &ctx,
    const DenseTensor &im_shape,
    const DenseTensor &anchors,
    const DenseTensor &variances,
    const DenseTensor &bbox_deltas,  // [M, 4]
    const DenseTensor &scores,       // [N, 1]
    int pre_nms_top_n,
    int post_nms_top_n,
    float nms_thresh,
    float min_size,
    float eta,
    bool pixel_offset) {
  // 1. pre nms
  DenseTensor scores_sort, index_sort;
  SortDescending<T>(ctx, scores, &scores_sort, &index_sort);
  int num = scores.numel();
  int pre_nms_num = (pre_nms_top_n <= 0 || pre_nms_top_n > num) ? scores.numel()
                                                                : pre_nms_top_n;
  scores_sort.Resize(phi::make_ddim({pre_nms_num, 1}));
  index_sort.Resize(phi::make_ddim({pre_nms_num, 1}));

  // 2. box decode and clipping
  DenseTensor proposals;
  proposals.Resize(phi::make_ddim({pre_nms_num, 4}));
  ctx.template Alloc<T>(&proposals);

  {
    phi::funcs::ForRange<phi::GPUContext> for_range(ctx, pre_nms_num);
    for_range(BoxDecodeAndClipFunctor<T>{anchors.data<T>(),
                                         bbox_deltas.data<T>(),
                                         variances.data<T>(),
                                         index_sort.data<int>(),
                                         im_shape.data<T>(),
                                         proposals.data<T>(),
                                         pixel_offset});
  }

  // 3. filter
  DenseTensor keep_index, keep_num_t;
  keep_index.Resize(phi::make_ddim({pre_nms_num}));
  ctx.template Alloc<int>(&keep_index);
  keep_num_t.Resize(phi::make_ddim({1}));
  ctx.template Alloc<int>(&keep_num_t);
  min_size = std::max(min_size, 1.0f);
  auto stream = ctx.stream();
  FilterBBoxes<T, 512><<<1, 512, 0, stream>>>(proposals.data<T>(),
                                              im_shape.data<T>(),
                                              min_size,
                                              pre_nms_num,
                                              keep_num_t.data<int>(),
                                              keep_index.data<int>(),
                                              false,
                                              pixel_offset);
  int keep_num;
  const auto gpu_place = ctx.GetPlace();
  paddle::memory::Copy(CPUPlace(),
                       &keep_num,
                       gpu_place,
                       keep_num_t.data<int>(),
                       sizeof(int),
                       ctx.stream());
  ctx.Wait();
  keep_index.Resize(phi::make_ddim({keep_num}));

  DenseTensor scores_filter, proposals_filter;
  // Handle the case when there is no keep index left
  if (keep_num == 0) {
    phi::funcs::SetConstant<phi::GPUContext, T> set_zero;
    proposals_filter.Resize(phi::make_ddim({1, 4}));
    ctx.template Alloc<T>(&proposals_filter);
    scores_filter.Resize(phi::make_ddim({1, 1}));
    ctx.template Alloc<T>(&scores_filter);
    set_zero(ctx, &proposals_filter, static_cast<T>(0));
    set_zero(ctx, &scores_filter, static_cast<T>(0));
    return std::make_pair(proposals_filter, scores_filter);
  }
  proposals_filter.Resize(phi::make_ddim({keep_num, 4}));
  ctx.template Alloc<T>(&proposals_filter);
  scores_filter.Resize(phi::make_ddim({keep_num, 1}));
  ctx.template Alloc<T>(&scores_filter);
  phi::funcs::GPUGather<T>(ctx, proposals, keep_index, &proposals_filter);
  phi::funcs::GPUGather<T>(ctx, scores_sort, keep_index, &scores_filter);

  if (nms_thresh <= 0) {
    return std::make_pair(proposals_filter, scores_filter);
  }

  // 4. nms
  DenseTensor keep_nms;
  NMS<T>(
      ctx, proposals_filter, keep_index, nms_thresh, &keep_nms, pixel_offset);
  if (post_nms_top_n > 0 && post_nms_top_n < keep_nms.numel()) {
    keep_nms.Resize(phi::make_ddim({post_nms_top_n}));
  }

  DenseTensor scores_nms, proposals_nms;
  proposals_nms.Resize(phi::make_ddim({keep_nms.numel(), 4}));
  ctx.template Alloc<T>(&proposals_nms);
  scores_nms.Resize(phi::make_ddim({keep_nms.numel(), 1}));
  ctx.template Alloc<T>(&scores_nms);
  phi::funcs::GPUGather<T>(ctx, proposals_filter, keep_nms, &proposals_nms);
  phi::funcs::GPUGather<T>(ctx, scores_filter, keep_nms, &scores_nms);

  return std::make_pair(proposals_nms, scores_nms);
}

template <typename T, typename Context>
void GenerateProposalsV2Kernel(const Context &ctx,
                               const DenseTensor &scores,
                               const DenseTensor &bbox_deltas,
                               const DenseTensor &im_shape,
                               const DenseTensor &anchors,
                               const DenseTensor &variances,
                               int pre_nms_top_n,
                               int post_nms_top_n,
                               float nms_thresh,
                               float min_size,
                               float eta,
                               bool pixel_offset,
                               DenseTensor *rpn_rois,
                               DenseTensor *rpn_roi_probs,
                               DenseTensor *rpn_rois_num) {
  PADDLE_ENFORCE_GE(
      eta,
      1.,
      errors::InvalidArgument("Not support adaptive NMS. The attribute 'eta' "
                              "should not less than 1. But received eta=[%d]",
                              eta));

  auto scores_dim = scores.dims();
  int64_t num = scores_dim[0];
  int64_t c_score = scores_dim[1];
  int64_t h_score = scores_dim[2];
  int64_t w_score = scores_dim[3];

  auto bbox_dim = bbox_deltas.dims();
  int64_t c_bbox = bbox_dim[1];
  int64_t h_bbox = bbox_dim[2];
  int64_t w_bbox = bbox_dim[3];

  DenseTensor bbox_deltas_swap, scores_swap;
  bbox_deltas_swap.Resize(phi::make_ddim({num, h_bbox, w_bbox, c_bbox}));
  ctx.template Alloc<T>(&bbox_deltas_swap);
  scores_swap.Resize(phi::make_ddim({num, h_score, w_score, c_score}));
  ctx.template Alloc<T>(&scores_swap);

  phi::funcs::Transpose<phi::GPUContext, T, 4> trans;
  std::vector<int> axis = {0, 2, 3, 1};
  trans(ctx, bbox_deltas, &bbox_deltas_swap, axis);
  trans(ctx, scores, &scores_swap, axis);

  DenseTensor tmp_anchors = anchors;
  DenseTensor tmp_variances = variances;
  tmp_anchors.Resize(phi::make_ddim({tmp_anchors.numel() / 4, 4}));
  tmp_variances.Resize(phi::make_ddim({tmp_variances.numel() / 4, 4}));

  rpn_rois->Resize(phi::make_ddim({bbox_deltas.numel() / 4, 4}));
  ctx.template Alloc<T>(rpn_rois);
  rpn_roi_probs->Resize(phi::make_ddim({scores.numel(), 1}));
  ctx.template Alloc<T>(rpn_roi_probs);

  T *rpn_rois_data = rpn_rois->data<T>();
  T *rpn_roi_probs_data = rpn_roi_probs->data<T>();

  auto place = ctx.GetPlace();
  auto cpu_place = phi::CPUPlace();

  int64_t num_proposals = 0;
  std::vector<size_t> offset(1, 0);
  std::vector<int> tmp_num;

  for (int64_t i = 0; i < num; ++i) {
    DenseTensor im_shape_slice = im_shape.Slice(i, i + 1);
    DenseTensor bbox_deltas_slice = bbox_deltas_swap.Slice(i, i + 1);
    DenseTensor scores_slice = scores_swap.Slice(i, i + 1);

    bbox_deltas_slice.Resize(phi::make_ddim({h_bbox * w_bbox * c_bbox / 4, 4}));
    scores_slice.Resize(phi::make_ddim({h_score * w_score * c_score, 1}));

    std::pair<DenseTensor, DenseTensor> box_score_pair =
        ProposalForOneImage<T>(ctx,
                               im_shape_slice,
                               tmp_anchors,
                               tmp_variances,
                               bbox_deltas_slice,
                               scores_slice,
                               pre_nms_top_n,
                               post_nms_top_n,
                               nms_thresh,
                               min_size,
                               eta,
                               pixel_offset);

    DenseTensor &proposals = box_score_pair.first;
    DenseTensor &nscores = box_score_pair.second;

    paddle::memory::Copy(place,
                         rpn_rois_data + num_proposals * 4,
                         place,
                         proposals.data<T>(),
                         sizeof(T) * proposals.numel(),
                         ctx.stream());
    paddle::memory::Copy(place,
                         rpn_roi_probs_data + num_proposals,
                         place,
                         nscores.data<T>(),
                         sizeof(T) * nscores.numel(),
                         ctx.stream());
    ctx.Wait();
    num_proposals += proposals.dims()[0];
    offset.emplace_back(num_proposals);
    tmp_num.push_back(proposals.dims()[0]);
  }
  if (rpn_rois_num != nullptr) {
    rpn_rois_num->Resize(phi::make_ddim({num}));
    ctx.template Alloc<int>(rpn_rois_num);
    int *num_data = rpn_rois_num->data<int>();
    paddle::memory::Copy(place,
                         num_data,
                         cpu_place,
                         &tmp_num[0],
                         sizeof(int) * num,
                         ctx.stream());
    rpn_rois_num->Resize(phi::make_ddim({num}));
  }
  phi::LoD lod;
  lod.emplace_back(offset);
  rpn_rois->Resize(phi::make_ddim({num_proposals, 4}));
  rpn_roi_probs->Resize(phi::make_ddim({num_proposals, 1}));
}

}  // namespace phi

PD_REGISTER_KERNEL(generate_proposals_v2,
                   GPU,
                   ALL_LAYOUT,
                   phi::GenerateProposalsV2Kernel,
                   float) {}
