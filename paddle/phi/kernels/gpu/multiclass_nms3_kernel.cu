/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/multiclass_nms3_kernel.h"

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#include "cuda.h"  // NOLINT
#endif

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/gather.cu.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/nonzero_kernel.h"

#define CUDA_MEM_ALIGN 256

namespace phi {

template <typename T>
struct Bbox {
  T xmin, ymin, xmax, ymax;
  Bbox(T xmin, T ymin, T xmax, T ymax)
      : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax) {}
  Bbox() = default;
};

template <typename KeyT, typename ValueT>
size_t CalcCubSortPairsWorkspaceSize(int num_items, int num_segments) {
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      reinterpret_cast<void*>(NULL),
      temp_storage_bytes,
      reinterpret_cast<KeyT*>(NULL),
      reinterpret_cast<KeyT*>(NULL),
      reinterpret_cast<ValueT*>(NULL),
      reinterpret_cast<ValueT*>(NULL),
      num_items,     // # items
      num_segments,  // # segments
      reinterpret_cast<int*>(NULL),
      reinterpret_cast<int*>(NULL));
  return temp_storage_bytes;
}

template <typename T>
size_t CalcDetectionForwardBBoxDataSize(int N, int C1) {
  return N * C1 * sizeof(T);
}

template <typename T>
size_t CalcDetectionForwardBBoxPermuteSize(bool share_location, int N, int C1) {
  return share_location ? 0 : N * C1 * sizeof(T);
}

template <typename T>
size_t CalcDetectionForwardPreNMSSize(int N, int C2) {
  return N * C2 * sizeof(T);
}

template <typename T>
size_t CalcDetectionForwardPostNMSSize(int N, int num_classes, int top_k) {
  return N * num_classes * top_k * sizeof(T);
}

size_t CalcTotalWorkspaceSize(size_t* workspaces, int count) {
  size_t total = 0;
  for (int i = 0; i < count; i++) {
    total += workspaces[i];
    if (workspaces[i] % CUDA_MEM_ALIGN) {
      total += CUDA_MEM_ALIGN - (workspaces[i] % CUDA_MEM_ALIGN);
    }
  }
  return total;
}

template <typename T>
size_t CalcSortScoresPerClassWorkspaceSize(const int num,
                                           const int num_classes,
                                           const int num_preds_per_class) {
  size_t wss[4];
  const int array_len = num * num_classes * num_preds_per_class;
  wss[0] = array_len * sizeof(T);                  // temp scores
  wss[1] = array_len * sizeof(int);                // temp indices
  wss[2] = (num * num_classes + 1) * sizeof(int);  // offsets
  wss[3] = CalcCubSortPairsWorkspaceSize<T, int>(
      array_len, num * num_classes);  // cub workspace

  return CalcTotalWorkspaceSize(wss, 4);
}

template <typename T>
size_t CalcSortScoresPerImageWorkspaceSize(const int num_images,
                                           const int num_items_per_image) {
  const int array_len = num_images * num_items_per_image;
  size_t wss[2];
  wss[0] = (num_images + 1) * sizeof(int);  // offsets
  wss[1] = CalcCubSortPairsWorkspaceSize<T, int>(array_len,
                                                 num_images);  // cub workspace

  return CalcTotalWorkspaceSize(wss, 2);
}

template <typename T>
size_t CalcDetectionInferenceWorkspaceSize(bool share_location,
                                           int N,
                                           int C1,
                                           int C2,
                                           int num_classes,
                                           int num_preds_per_class,
                                           int top_k) {
  size_t wss[6];
  wss[0] = CalcDetectionForwardBBoxDataSize<T>(N, C1);
  wss[1] = CalcDetectionForwardPreNMSSize<T>(N, C2);
  wss[2] = CalcDetectionForwardPreNMSSize<int>(N, C2);
  wss[3] = CalcDetectionForwardPostNMSSize<T>(N, num_classes, top_k);
  wss[4] = CalcDetectionForwardPostNMSSize<int>(N, num_classes, top_k);
  wss[5] =
      std::max(CalcSortScoresPerClassWorkspaceSize<T>(
                   N, num_classes, num_preds_per_class),
               CalcSortScoresPerImageWorkspaceSize<T>(N, num_classes * top_k));
  return CalcTotalWorkspaceSize(wss, 6);
}

// ALIGNPTR
int8_t* AlignPtr(int8_t* ptr, uintptr_t to) {
  uintptr_t addr = (uintptr_t)ptr;
  if (addr % to) {
    addr += to - addr % to;
  }
  return reinterpret_cast<int8_t*>(addr);
}

// GetNEXTWORKSPACEPTR
int8_t* GetNextWorkspacePtr(int8_t* ptr, uintptr_t previous_workspace_size) {
  uintptr_t addr = (uintptr_t)ptr;
  addr += previous_workspace_size;
  return AlignPtr(reinterpret_cast<int8_t*>(addr), CUDA_MEM_ALIGN);
}

/* ==================
 * sortScoresPerClass
 * ================== */
template <typename T_SCORE, unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta) __global__
    void PrepareSortData(const int num,
                         const int num_classes,
                         const int num_preds_per_class,
                         const int background_label_id,
                         const float confidence_threshold,
                         T_SCORE* conf_scores_gpu,
                         T_SCORE* temp_scores,
                         T_SCORE score_shift,
                         int* temp_idx,
                         int* d_offsets) {
  // Prepare scores data for sort
  const int cur_idx = blockIdx.x * nthds_per_cta + threadIdx.x;
  const int num_preds_per_batch = num_classes * num_preds_per_class;
  T_SCORE clip_val =
      T_SCORE(static_cast<float>(score_shift) + 1.f - 1.f / 1024.f);
  if (cur_idx < num_preds_per_batch) {
    const int class_idx = cur_idx / num_preds_per_class;
    for (int i = 0; i < num; i++) {
      const int target_idx = i * num_preds_per_batch + cur_idx;
      const T_SCORE score = conf_scores_gpu[target_idx];

      // "Clear" background labeled score and index
      // Because we do not care about background
      if (class_idx == background_label_id) {
        // Set scores to 0
        // Set label = -1
        // add shift of 1.0 to normalize the score values
        // to the range [1, 2).
        // add a constant shift to scores will not change the sort
        // result, but will help reduce the computation because
        // we only need to sort the mantissa part of the floating-point
        // numbers
        temp_scores[target_idx] = score_shift;
        temp_idx[target_idx] = -1;
        conf_scores_gpu[target_idx] = score_shift;
      } else {  // "Clear" scores lower than threshold
        if (static_cast<float>(score) > confidence_threshold) {
          // add shift of 1.0 to normalize the score values
          // to the range [1, 2).
          // add a constant shift to scores will not change the sort
          // result, but will help reduce the computation because
          // we only need to sort the mantissa part of the floating-point
          // numbers
          temp_scores[target_idx] = score + score_shift;
          if (static_cast<float>(score_shift) > 0.f &&
              (temp_scores[target_idx] >= clip_val))
            temp_scores[target_idx] = clip_val;
          temp_idx[target_idx] = cur_idx + i * num_preds_per_batch;
        } else {
          // Set scores to 0
          // Set label = -1
          // add shift of 1.0 to normalize the score values
          // to the range [1, 2).
          // add a constant shift to scores will not change the sort
          // result, but will help reduce the computation because
          // we only need to sort the mantissa part of the floating-point
          // numbers
          temp_scores[target_idx] = score_shift;
          temp_idx[target_idx] = -1;
          conf_scores_gpu[target_idx] = score_shift;
          // TODO(tizheng): HERE writing memory too many times
        }
      }

      if ((cur_idx % num_preds_per_class) == 0) {
        const int offset_ct = i * num_classes + cur_idx / num_preds_per_class;
        d_offsets[offset_ct] = offset_ct * num_preds_per_class;
        // set the last element in d_offset
        if (blockIdx.x == 0 && threadIdx.x == 0)
          d_offsets[num * num_classes] = num * num_preds_per_batch;
      }
    }
  }
}

template <typename T_SCORE>
void SortScoresPerClassGPU(gpuStream_t stream,
                           const int num,
                           const int num_classes,
                           const int num_preds_per_class,
                           const int background_label_id,
                           const float confidence_threshold,
                           void* conf_scores_gpu,
                           void* index_array_gpu,
                           void* workspace,
                           const int score_bits,
                           const float score_shift) {
  const int num_segments = num * num_classes;
  void* temp_scores = workspace;
  const int array_len = num * num_classes * num_preds_per_class;
  void* temp_idx = GetNextWorkspacePtr(reinterpret_cast<int8_t*>(temp_scores),
                                       array_len * sizeof(T_SCORE));
  void* d_offsets = GetNextWorkspacePtr(reinterpret_cast<int8_t*>(temp_idx),
                                        array_len * sizeof(int));
  size_t cubOffsetSize = (num_segments + 1) * sizeof(int);
  void* cubWorkspace =
      GetNextWorkspacePtr(reinterpret_cast<int8_t*>(d_offsets), cubOffsetSize);

  const int BS = 512;
  const int GS = (num_classes * num_preds_per_class + BS - 1) / BS;
  // prepare the score, index, and offsets for CUB radix sort
  // also normalize the scores to the range [1, 2)
  // so we only need to sort the mantissa of floating-point numbers
  // since their sign bit and exponential bits are identical
  // we will subtract the 1.0 shift in gatherTopDetections()
  PrepareSortData<T_SCORE, BS>
      <<<GS, BS, 0, stream>>>(num,
                              num_classes,
                              num_preds_per_class,
                              background_label_id,
                              confidence_threshold,
                              reinterpret_cast<T_SCORE*>(conf_scores_gpu),
                              reinterpret_cast<T_SCORE*>(temp_scores),
                              T_SCORE(score_shift),
                              reinterpret_cast<int*>(temp_idx),
                              reinterpret_cast<int*>(d_offsets));

  size_t temp_storage_bytes =
      CalcCubSortPairsWorkspaceSize<T_SCORE, int>(array_len, num_segments);
  size_t begin_bit = 0;
  size_t end_bit = sizeof(T_SCORE) * 8;
  if (sizeof(T_SCORE) == 2 && score_bits > 0 && score_bits <= 10) {
    // only sort score_bits in 10 mantissa bits.
    end_bit = 10;
    begin_bit = end_bit - score_bits;
  }
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      cubWorkspace,
      temp_storage_bytes,
      reinterpret_cast<T_SCORE*>(temp_scores),
      reinterpret_cast<T_SCORE*>(conf_scores_gpu),
      reinterpret_cast<int*>(temp_idx),
      reinterpret_cast<int*>(index_array_gpu),
      array_len,
      num_segments,
      reinterpret_cast<int*>(d_offsets),
      reinterpret_cast<int*>(d_offsets) + 1,
      begin_bit,
      end_bit,
      stream);
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipGetLastError());
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaGetLastError());
#endif
}

/* ===========
 * allClassNMS
 * =========== */
template <typename T_BBOX>
__device__ float CalcBboxSize(const Bbox<T_BBOX>& bbox, const bool normalized) {
  if (static_cast<float>(bbox.xmax) < static_cast<float>(bbox.xmin) ||
      static_cast<float>(bbox.ymax) < static_cast<float>(bbox.ymin)) {
    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
    return 0;
  } else {
    float width = static_cast<float>(bbox.xmax) - static_cast<float>(bbox.xmin);
    float height =
        static_cast<float>(bbox.ymax) - static_cast<float>(bbox.ymin);
    if (normalized) {
      return width * height;
    } else {
      // If bbox is not within range [0, 1].
      return (width + 1.f) * (height + 1.f);
    }
  }
}

template <typename T_BBOX>
__device__ void CalcIntersectBbox(const Bbox<T_BBOX>& bbox1,
                                  const Bbox<T_BBOX>& bbox2,
                                  Bbox<T_BBOX>* intersect_bbox) {
  if (bbox2.xmin > bbox1.xmax || bbox2.xmax < bbox1.xmin ||
      bbox2.ymin > bbox1.ymax || bbox2.ymax < bbox1.ymin) {
    // Return [0, 0, 0, 0] if there is no intersection.
    intersect_bbox->xmin = T_BBOX(0);
    intersect_bbox->ymin = T_BBOX(0);
    intersect_bbox->xmax = T_BBOX(0);
    intersect_bbox->ymax = T_BBOX(0);
  } else {
    intersect_bbox->xmin = max(bbox1.xmin, bbox2.xmin);
    intersect_bbox->ymin = max(bbox1.ymin, bbox2.ymin);
    intersect_bbox->xmax = min(bbox1.xmax, bbox2.xmax);
    intersect_bbox->ymax = min(bbox1.ymax, bbox2.ymax);
  }
}

template <typename T_BBOX>
__device__ Bbox<T_BBOX> GetDiagonalMinMaxSortedBox(const Bbox<T_BBOX>& bbox1) {
  Bbox<T_BBOX> result;
  result.xmin = min(bbox1.xmin, bbox1.xmax);
  result.xmax = max(bbox1.xmin, bbox1.xmax);

  result.ymin = min(bbox1.ymin, bbox1.ymax);
  result.ymax = max(bbox1.ymin, bbox1.ymax);
  return result;
}

template <typename T_BBOX>
__device__ void GetFlippedBox(const T_BBOX* bbox1,
                              bool flip_xy,
                              Bbox<T_BBOX>* result) {
  result->xmin = flip_xy ? bbox1[1] : bbox1[0];
  result->ymin = flip_xy ? bbox1[0] : bbox1[1];
  result->xmax = flip_xy ? bbox1[3] : bbox1[2];
  result->ymax = flip_xy ? bbox1[2] : bbox1[3];
}

template <typename T_BBOX>
__device__ float CalcJaccardOverlap(const Bbox<T_BBOX>& bbox1,
                                    const Bbox<T_BBOX>& bbox2,
                                    const bool normalized,
                                    const bool caffe_semantics) {
  Bbox<T_BBOX> intersect_bbox;

  Bbox<T_BBOX> localbbox1 = GetDiagonalMinMaxSortedBox(bbox1);
  Bbox<T_BBOX> localbbox2 = GetDiagonalMinMaxSortedBox(bbox2);

  CalcIntersectBbox(localbbox1, localbbox2, &intersect_bbox);

  float intersect_width, intersect_height;
  // Only when using Caffe semantics, IOU calculation adds "1" to width and
  // height if bbox is not normalized.
  // https://github.com/weiliu89/caffe/blob/ssd/src/caffe/util/bbox_util.cpp#L92-L97
  if (normalized || !caffe_semantics) {
    intersect_width = static_cast<float>(intersect_bbox.xmax) -
                      static_cast<float>(intersect_bbox.xmin);
    intersect_height = static_cast<float>(intersect_bbox.ymax) -
                       static_cast<float>(intersect_bbox.ymin);
  } else {
    intersect_width = static_cast<float>(intersect_bbox.xmax) -
                      static_cast<float>(intersect_bbox.xmin) +
                      static_cast<float>(T_BBOX(1));
    intersect_height = static_cast<float>(intersect_bbox.ymax) -
                       static_cast<float>(intersect_bbox.ymin) +
                       static_cast<float>(T_BBOX(1));
  }
  if (intersect_width > 0 && intersect_height > 0) {
    float intersect_size = intersect_width * intersect_height;
    float bbox1_size = CalcBboxSize(localbbox1, normalized);
    float bbox2_size = CalcBboxSize(localbbox2, normalized);
    return intersect_size / (bbox1_size + bbox2_size - intersect_size);
  } else {
    return 0.;
  }
}

template <typename T_SCORE, typename T_BBOX, int TSIZE>
__global__ void AllClassNMSKernel(
    const int num,
    const int num_classes,
    const int num_preds_per_class,
    const int top_k,
    const float nms_threshold,
    const bool share_location,
    const bool is_normalized,
    T_BBOX* bbox_data,  // bbox_data should be float to preserve location
                        // information
    T_SCORE* before_nms_scores,
    int* before_nms_index_array,
    T_SCORE* after_nms_scores,
    int* after_nms_index_array,
    bool flip_xy,
    const float score_shift,
    bool caffe_semantics) {
  // __shared__ bool kept_bboxinfo_flag[CAFFE_CUDA_NUM_THREADS * TSIZE];
  extern __shared__ bool kept_bboxinfo_flag[];

  for (int i = 0; i < num; i++) {
    int32_t const offset = i * num_classes * num_preds_per_class +
                           blockIdx.x * num_preds_per_class;
    // Should not write data beyond [offset, top_k).
    int32_t const max_idx = offset + top_k;
    // Should not read beyond [offset, num_preds_per_class).
    int32_t const max_read_idx = offset + min(top_k, num_preds_per_class);
    int32_t const bbox_idx_offset =
        i * num_preds_per_class * (share_location ? 1 : num_classes);

    // local thread data
    int loc_bboxIndex[TSIZE];
    Bbox<T_BBOX> loc_bbox[TSIZE];

    // initialize Bbox, Bboxinfo, kept_bboxinfo_flag
    // Eliminate shared memory RAW hazard
    __syncthreads();
#pragma unroll
    for (int t = 0; t < TSIZE; t++) {
      const int cur_idx = threadIdx.x + blockDim.x * t;
      const int item_idx = offset + cur_idx;
      // Init all output data
      if (item_idx < max_idx) {
        // Do not access data if it exceeds read boundary
        if (item_idx < max_read_idx) {
          loc_bboxIndex[t] = before_nms_index_array[item_idx];
        } else {
          loc_bboxIndex[t] = -1;
        }

        if (loc_bboxIndex[t] != -1) {
          const int bbox_data_idx =
              share_location
                  ? (loc_bboxIndex[t] % num_preds_per_class + bbox_idx_offset)
                  : loc_bboxIndex[t];
          GetFlippedBox(&bbox_data[bbox_data_idx * 4], flip_xy, &loc_bbox[t]);
          kept_bboxinfo_flag[cur_idx] = true;
        } else {
          kept_bboxinfo_flag[cur_idx] = false;
        }
      } else {
        kept_bboxinfo_flag[cur_idx] = false;
      }
    }

    // filter out overlapped boxes with lower scores
    int ref_item_idx = offset;

    int32_t ref_bbox_idx = -1;
    if (ref_item_idx < max_read_idx) {
      ref_bbox_idx =
          share_location
              ? (before_nms_index_array[ref_item_idx] % num_preds_per_class +
                 bbox_idx_offset)
              : before_nms_index_array[ref_item_idx];
    }
    while ((ref_bbox_idx != -1) && ref_item_idx < max_read_idx) {
      Bbox<T_BBOX> ref_bbox;
      GetFlippedBox(&bbox_data[ref_bbox_idx * 4], flip_xy, &ref_bbox);

      // Eliminate shared memory RAW hazard
      __syncthreads();

      for (int t = 0; t < TSIZE; t++) {
        const int cur_idx = threadIdx.x + blockDim.x * t;
        const int item_idx = offset + cur_idx;

        if ((kept_bboxinfo_flag[cur_idx]) && (item_idx > ref_item_idx)) {
          if (CalcJaccardOverlap(
                  ref_bbox, loc_bbox[t], is_normalized, caffe_semantics) >
              nms_threshold) {
            kept_bboxinfo_flag[cur_idx] = false;
          }
        }
      }
      __syncthreads();

      do {
        ref_item_idx++;
      } while (ref_item_idx < max_read_idx &&
               !kept_bboxinfo_flag[ref_item_idx - offset]);

      // Move to next valid point
      if (ref_item_idx < max_read_idx) {
        ref_bbox_idx =
            share_location
                ? (before_nms_index_array[ref_item_idx] % num_preds_per_class +
                   bbox_idx_offset)
                : before_nms_index_array[ref_item_idx];
      }
    }

    // store data
    for (int t = 0; t < TSIZE; t++) {
      const int cur_idx = threadIdx.x + blockDim.x * t;
      const int read_item_idx = offset + cur_idx;
      const int write_item_idx =
          (i * num_classes * top_k + blockIdx.x * top_k) + cur_idx;
      /*
       * If not not keeping the bbox
       * Set the score to 0
       * Set the bounding box index to -1
       */
      if (read_item_idx < max_idx) {
        after_nms_scores[write_item_idx] =
            kept_bboxinfo_flag[cur_idx]
                ? T_SCORE(before_nms_scores[read_item_idx])
                : T_SCORE(score_shift);
        after_nms_index_array[write_item_idx] =
            kept_bboxinfo_flag[cur_idx] ? loc_bboxIndex[t] : -1;
      }
    }
  }
}

template <typename T_SCORE, typename T_BBOX>
void AllClassNMSGPU(gpuStream_t stream,
                    const int num,
                    const int num_classes,
                    const int num_preds_per_class,
                    const int top_k,
                    const float nms_threshold,
                    const bool share_location,
                    const bool is_normalized,
                    void* bbox_data,
                    void* before_nms_scores,
                    void* before_nms_index_array,
                    void* after_nms_scores,
                    void* after_nms_index_array,
                    bool flip_xy,
                    const float score_shift,
                    bool caffe_semantics) {
#define P(tsize) AllClassNMSKernel<T_SCORE, T_BBOX, (tsize)>

  void (*kernel[8])(const int,
                    const int,
                    const int,
                    const int,
                    const float,
                    const bool,
                    const bool,
                    T_BBOX*,
                    T_SCORE*,
                    int*,
                    T_SCORE*,
                    int*,
                    bool,
                    const float,
                    bool) = {
      P(1),
      P(2),
      P(3),
      P(4),
      P(5),
      P(6),
      P(7),
      P(8),
  };

  const int BS = 512;
  const int GS = num_classes;
  const int t_size = (top_k + BS - 1) / BS;

  kernel[t_size - 1]<<<GS, BS, BS * t_size * sizeof(bool), stream>>>(
      num,
      num_classes,
      num_preds_per_class,
      top_k,
      nms_threshold,
      share_location,
      is_normalized,
      reinterpret_cast<T_BBOX*>(bbox_data),
      reinterpret_cast<T_SCORE*>(before_nms_scores),
      reinterpret_cast<int*>(before_nms_index_array),
      reinterpret_cast<T_SCORE*>(after_nms_scores),
      reinterpret_cast<int*>(after_nms_index_array),
      flip_xy,
      score_shift,
      caffe_semantics);

#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipGetLastError());
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaGetLastError());
#endif
}

/* ==================
 * sortScoresPerImage
 * ================== */
template <unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta) __global__
    void SetUniformOffsetsKernel(const int num_segments,
                                 const int offset,
                                 int* d_offsets) {
  const int idx = blockIdx.x * nthds_per_cta + threadIdx.x;
  if (idx <= num_segments) d_offsets[idx] = idx * offset;
}

void SetUniformOffsets(gpuStream_t stream,
                       const int num_segments,
                       const int offset,
                       int* d_offsets) {
#ifdef PADDLE_WITH_HIP
  const int BS = 256;
#else
  const int BS = 32;
#endif
  const int GS = (num_segments + 1 + BS - 1) / BS;
  SetUniformOffsetsKernel<BS>
      <<<GS, BS, 0, stream>>>(num_segments, offset, d_offsets);
}

/* ================
 * gatherNMSOutputs
 * ================ */
template <typename T_BBOX>
__device__ T_BBOX saturate(T_BBOX v) {
  return max(min(v, T_BBOX(1)), T_BBOX(0));
}

template <typename T_BBOX, typename T_SCORE, unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta) __global__
    void GatherNMSOutputsKernel(const bool share_location,
                                const int num_images,
                                const int num_preds_per_class,
                                const int num_classes,
                                const int top_k,
                                const int keep_top_k,
                                const int* indices,
                                const T_SCORE* scores,
                                const T_BBOX* bbox_data,
                                int* num_detections,
                                T_BBOX* nmsed_boxes,
                                T_BBOX* nmsed_scores,
                                T_BBOX* nmsed_classes,
                                int* nmsed_indices,
                                int* nmsed_valid_mask,
                                bool clip_boxes,
                                const T_SCORE score_shift) {
  if (keep_top_k > top_k) return;
  for (int i = blockIdx.x * nthds_per_cta + threadIdx.x;
       i < num_images * keep_top_k;
       i += gridDim.x * nthds_per_cta) {
    const int imgId = i / keep_top_k;
    const int detId = i % keep_top_k;
    const int offset = imgId * num_classes * top_k;
    const int index = indices[offset + detId];
    const T_SCORE score = scores[offset + detId];
    if (index == -1) {
      nmsed_classes[i] = -1;
      nmsed_scores[i] = 0;
      nmsed_boxes[i * 4] = 0;
      nmsed_boxes[i * 4 + 1] = 0;
      nmsed_boxes[i * 4 + 2] = 0;
      nmsed_boxes[i * 4 + 3] = 0;
      nmsed_indices[i] = -1;
      nmsed_valid_mask[i] = 0;
    } else {
      const int bbox_offset =
          imgId * (share_location ? num_preds_per_class
                                  : (num_classes * num_preds_per_class));
      const int bbox_id =
          ((share_location ? (index % num_preds_per_class)
                           : index % (num_classes * num_preds_per_class)) +
           bbox_offset) *
          4;
      nmsed_classes[i] = (index % (num_classes * num_preds_per_class)) /
                         num_preds_per_class;  // label
      nmsed_scores[i] = score;                 // confidence score
      nmsed_scores[i] = nmsed_scores[i] - score_shift;
      const T_BBOX xMin = bbox_data[bbox_id];
      const T_BBOX yMin = bbox_data[bbox_id + 1];
      const T_BBOX xMax = bbox_data[bbox_id + 2];
      const T_BBOX yMax = bbox_data[bbox_id + 3];
      // clipped bbox xmin
      nmsed_boxes[i * 4] = clip_boxes ? saturate(xMin) : xMin;
      // clipped bbox ymin
      nmsed_boxes[i * 4 + 1] = clip_boxes ? saturate(yMin) : yMin;
      // clipped bbox xmax
      nmsed_boxes[i * 4 + 2] = clip_boxes ? saturate(xMax) : xMax;
      // clipped bbox ymax
      nmsed_boxes[i * 4 + 3] = clip_boxes ? saturate(yMax) : yMax;
      nmsed_indices[i] = bbox_id >> 2;
      nmsed_valid_mask[i] = 1;
      atomicAdd(&num_detections[i / keep_top_k], 1);
    }
  }
}

template <typename T_BBOX, typename T_SCORE>
void GatherNMSOutputsGPU(gpuStream_t stream,
                         const bool share_location,
                         const int num_images,
                         const int num_preds_per_class,
                         const int num_classes,
                         const int top_k,
                         const int keep_top_k,
                         const void* indices,
                         const void* scores,
                         const void* bbox_data,
                         void* num_detections,
                         void* nmsed_boxes,
                         void* nmsed_scores,
                         void* nmsed_classes,
                         void* nmsed_indices,
                         void* nmsed_valid_mask,
                         bool clip_boxes,
                         const float score_shift) {
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(
      hipMemsetAsync(num_detections, 0, num_images * sizeof(int), stream));
  const int BS = 256;
#else
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(num_detections, 0, num_images * sizeof(int), stream));
  const int BS = 32;
#endif
  const int GS = 32;
  GatherNMSOutputsKernel<T_BBOX, T_SCORE, BS>
      <<<GS, BS, 0, stream>>>(share_location,
                              num_images,
                              num_preds_per_class,
                              num_classes,
                              top_k,
                              keep_top_k,
                              reinterpret_cast<const int*>(indices),
                              reinterpret_cast<const T_SCORE*>(scores),
                              reinterpret_cast<const T_BBOX*>(bbox_data),
                              reinterpret_cast<int*>(num_detections),
                              reinterpret_cast<T_BBOX*>(nmsed_boxes),
                              reinterpret_cast<T_BBOX*>(nmsed_scores),
                              reinterpret_cast<T_BBOX*>(nmsed_classes),
                              reinterpret_cast<int*>(nmsed_indices),
                              reinterpret_cast<int*>(nmsed_valid_mask),
                              clip_boxes,
                              T_SCORE(score_shift));
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipGetLastError());
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaGetLastError());
#endif
}

template <typename T_SCORE>
void SortScoresPerImageGPU(gpuStream_t stream,
                           const int num_images,
                           const int num_items_per_image,
                           void* unsorted_scores,
                           void* unsorted_bbox_indices,
                           void* sorted_scores,
                           void* sorted_bbox_indices,
                           void* workspace,
                           int score_bits) {
  void* d_offsets = workspace;
  void* cubWorkspace = GetNextWorkspacePtr(reinterpret_cast<int8_t*>(d_offsets),
                                           (num_images + 1) * sizeof(int));

  SetUniformOffsets(stream,
                    num_images,
                    num_items_per_image,
                    reinterpret_cast<int*>(d_offsets));

  const int array_len = num_images * num_items_per_image;
  size_t temp_storage_bytes =
      CalcCubSortPairsWorkspaceSize<T_SCORE, int>(array_len, num_images);
  size_t begin_bit = 0;
  size_t end_bit = sizeof(T_SCORE) * 8;
  if (sizeof(T_SCORE) == 2 && score_bits > 0 && score_bits <= 10) {
    end_bit = 10;
    begin_bit = end_bit - score_bits;
  }
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      cubWorkspace,
      temp_storage_bytes,
      reinterpret_cast<T_SCORE*>(unsorted_scores),
      reinterpret_cast<T_SCORE*>(sorted_scores),
      reinterpret_cast<int*>(unsorted_bbox_indices),
      reinterpret_cast<int*>(sorted_bbox_indices),
      array_len,
      num_images,
      reinterpret_cast<int*>(d_offsets),
      reinterpret_cast<int*>(d_offsets) + 1,
      begin_bit,
      end_bit,
      stream);
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipGetLastError());
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaGetLastError());
#endif
}

template <typename T>
void InferNMS(gpuStream_t stream,
              const int N,
              const int per_batch_boxes_size,
              const int per_batch_scores_size,
              const bool share_location,
              const int background_label_id,
              const int num_preds_per_class,
              const int num_classes,
              const int top_k,
              const int keep_top_k,
              const float score_threshold,
              const float iou_threshold,
              const void* loc_data,
              const void* conf_data,
              void* keep_count,
              void* nmsed_boxes,
              void* nmsed_scores,
              void* nmsed_classes,
              void* nmsed_indices,
              void* nmsed_valid_mask,
              void* workspace,
              bool is_normalized,
              bool conf_sigmoid,
              bool clip_boxes,
              int score_bits,
              bool caffe_semantics) {
  PADDLE_ENFORCE_EQ(
      share_location,
      true,
      common::errors::Unimplemented("share_location=false is not supported."));

  // Prepare workspaces
  size_t bbox_data_size =
      CalcDetectionForwardBBoxDataSize<T>(N, per_batch_boxes_size);
  void* bbox_data_raw = workspace;
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpyAsync(bbox_data_raw,
                                            loc_data,
                                            bbox_data_size,
                                            hipMemcpyDeviceToDevice,
                                            stream));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(bbox_data_raw,
                                             loc_data,
                                             bbox_data_size,
                                             cudaMemcpyDeviceToDevice,
                                             stream));
#endif
  void* bbox_data = bbox_data_raw;

  const int num_scores = N * per_batch_scores_size;
  size_t total_scores_size =
      CalcDetectionForwardPreNMSSize<T>(N, per_batch_scores_size);
  void* scores =
      GetNextWorkspacePtr(reinterpret_cast<int8_t*>(bbox_data), bbox_data_size);
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpyAsync(
      scores, conf_data, total_scores_size, hipMemcpyDeviceToDevice, stream));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
      scores, conf_data, total_scores_size, cudaMemcpyDeviceToDevice, stream));
#endif

  size_t indices_size =
      CalcDetectionForwardPreNMSSize<int>(N, per_batch_scores_size);
  void* indices =
      GetNextWorkspacePtr(reinterpret_cast<int8_t*>(scores), total_scores_size);

  size_t post_nms_scores_size =
      CalcDetectionForwardPostNMSSize<T>(N, num_classes, top_k);
  size_t post_nms_indices_size = CalcDetectionForwardPostNMSSize<int>(
      N, num_classes, top_k);  // indices are full int32
  void* post_nms_scores =
      GetNextWorkspacePtr(reinterpret_cast<int8_t*>(indices), indices_size);
  void* post_nms_indices = GetNextWorkspacePtr(
      reinterpret_cast<int8_t*>(post_nms_scores), post_nms_scores_size);

  void* sorting_workspace = GetNextWorkspacePtr(
      reinterpret_cast<int8_t*>(post_nms_indices), post_nms_indices_size);
  // Sort the scores so that the following NMS could be applied.
  float score_shift = 0.f;
  SortScoresPerClassGPU<T>(stream,
                           N,
                           num_classes,
                           num_preds_per_class,
                           background_label_id,
                           score_threshold,
                           scores,
                           indices,
                           sorting_workspace,
                           score_bits,
                           score_shift);

  // This is set to true as the input bounding boxes are of the format [ymin,
  // xmin, ymax, xmax]. The default implementation assumes [xmin, ymin, xmax,
  // ymax]
  bool flip_xy = true;
  // NMS
  AllClassNMSGPU<T, T>(stream,
                       N,
                       num_classes,
                       num_preds_per_class,
                       top_k,
                       iou_threshold,
                       share_location,
                       is_normalized,
                       bbox_data,
                       scores,
                       indices,
                       post_nms_scores,
                       post_nms_indices,
                       flip_xy,
                       score_shift,
                       caffe_semantics);

  // Sort the bounding boxes after NMS using scores
  SortScoresPerImageGPU<T>(stream,
                           N,
                           num_classes * top_k,
                           post_nms_scores,
                           post_nms_indices,
                           scores,
                           indices,
                           sorting_workspace,
                           score_bits);

  // Gather data from the sorted bounding boxes after NMS
  GatherNMSOutputsGPU<T, T>(stream,
                            share_location,
                            N,
                            num_preds_per_class,
                            num_classes,
                            top_k,
                            keep_top_k,
                            indices,
                            scores,
                            bbox_data,
                            keep_count,
                            nmsed_boxes,
                            nmsed_scores,
                            nmsed_classes,
                            nmsed_indices,
                            nmsed_valid_mask,
                            clip_boxes,
                            score_shift);
}

template <typename T, typename Context>
void MultiClassNMSGPUKernel(const Context& ctx,
                            const DenseTensor& bboxes,
                            const DenseTensor& scores,
                            const paddle::optional<DenseTensor>& rois_num,
                            float score_threshold,
                            int nms_top_k,
                            int keep_top_k,
                            float nms_threshold,
                            bool normalized,
                            float nms_eta,
                            int background_label,
                            DenseTensor* out,
                            DenseTensor* index,
                            DenseTensor* nms_rois_num) {
  bool return_index = index != nullptr;
  bool has_roisnum = rois_num.get_ptr() != nullptr;
  auto score_dims = scores.dims();
  auto score_size = score_dims.size();

  bool is_supported = (score_size == 3) && (nms_top_k >= 0) &&
                      (nms_top_k <= 4096) && (keep_top_k >= 0) &&
                      (nms_eta == 1.0) && !has_roisnum;
  if (!is_supported) {
    VLOG(6)
        << "This configuration is not supported by GPU kernel. Falling back to "
           "CPU kernel. "
           "Expect (score_size == 3) && (nms_top_k >= 0) && (nms_top_k <= 4096)"
           "(keep_top_k >= 0) && (nms_eta == 1.0) && !has_roisnum, "
           "got score_size="
        << score_size << ", nms_top_k=" << nms_top_k
        << ", keep_top_k=" << keep_top_k << ", nms_eta=" << nms_eta
        << ", has_roisnum=" << has_roisnum;

    DenseTensor bboxes_cpu, scores_cpu, rois_num_cpu_tenor;
    DenseTensor out_cpu, index_cpu, nms_rois_num_cpu;
    paddle::optional<DenseTensor> rois_num_cpu(paddle::none);
    auto cpu_place = phi::CPUPlace();
    auto gpu_place = ctx.GetPlace();

    // copy from GPU to CPU
    phi::Copy(ctx, bboxes, cpu_place, false, &bboxes_cpu);
    phi::Copy(ctx, scores, cpu_place, false, &scores_cpu);
    if (has_roisnum) {
      phi::Copy(
          ctx, *rois_num.get_ptr(), cpu_place, false, &rois_num_cpu_tenor);
      rois_num_cpu = paddle::optional<DenseTensor>(rois_num_cpu_tenor);
    }
    ctx.Wait();
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto* cpu_ctx = static_cast<phi::CPUContext*>(pool.Get(cpu_place));
    MultiClassNMSKernel<T, phi::CPUContext>(*cpu_ctx,
                                            bboxes_cpu,
                                            scores_cpu,
                                            rois_num_cpu,
                                            score_threshold,
                                            nms_top_k,
                                            keep_top_k,
                                            nms_threshold,
                                            normalized,
                                            nms_eta,
                                            background_label,
                                            &out_cpu,
                                            &index_cpu,
                                            &nms_rois_num_cpu);
    // copy back
    phi::Copy(ctx, out_cpu, gpu_place, false, out);
    phi::Copy(ctx, index_cpu, gpu_place, false, index);
    phi::Copy(ctx, nms_rois_num_cpu, gpu_place, false, nms_rois_num);
    return;
  }

  // Calculate input shapes
  int64_t batch_size = score_dims[0];
  const int64_t per_batch_boxes_size =
      bboxes.dims()[1] * bboxes.dims()[2];  // M * 4
  const int64_t per_batch_scores_size =
      scores.dims()[1] * scores.dims()[2];       // C * M
  const int64_t num_priors = bboxes.dims()[1];   // M
  const int64_t num_classes = scores.dims()[1];  // C
  const bool share_location = true;
  auto stream = reinterpret_cast<const Context&>(ctx).stream();
  // Sanity check
  PADDLE_ENFORCE_LE(
      nms_top_k,
      num_priors,
      common::errors::InvalidArgument("Expect nms_top_k (%d)"
                                      " <= num of boxes per batch (%d).",
                                      nms_top_k,
                                      num_priors));
  PADDLE_ENFORCE_LE(keep_top_k,
                    nms_top_k,
                    common::errors::InvalidArgument("Expect keep_top_k (%d)"
                                                    " <= nms_top_k (%d).",
                                                    keep_top_k,
                                                    nms_top_k));

  // Transform the layout of bboxes and scores
  // bboxes: [N,M,4] -> [N,1,M,4]
  DenseTensor transformed_bboxes(bboxes.type());
  transformed_bboxes.ShareDataWith(bboxes).Resize(
      {bboxes.dims()[0], 1, bboxes.dims()[1], bboxes.dims()[2]});
  // scores: [N, C, M] => [N, C, M, 1]
  DenseTensor transformed_scores(scores.type());
  transformed_scores.ShareDataWith(scores).Resize(
      {scores.dims()[0], scores.dims()[1], scores.dims()[2], 1});

  // Prepare intermediate outputs for NMS kernels
  DenseTensor keep_count(DataType::INT32);
  keep_count.Resize({batch_size});
  if (nms_rois_num != nullptr) {
    nms_rois_num->Resize({batch_size});
    ctx.template Alloc<int>(nms_rois_num);
    keep_count.ShareDataWith(*nms_rois_num);
  } else {
    ctx.template Alloc<int>(&keep_count);
  }

  DenseTensor nmsed_indices(DataType::INT32);
  nmsed_indices.Resize({batch_size * keep_top_k, 1});
  ctx.template Alloc<int>(&nmsed_indices);

  DenseTensor nmsed_valid_mask(DataType::INT32);
  nmsed_valid_mask.Resize({batch_size * keep_top_k});
  ctx.template Alloc<int>(&nmsed_valid_mask);

  DenseTensor nmsed_boxes(bboxes.dtype());
  DenseTensor nmsed_scores(scores.dtype());
  DenseTensor nmsed_classes(scores.dtype());
  nmsed_boxes.Resize({batch_size * keep_top_k, 4});
  nmsed_scores.Resize({batch_size * keep_top_k, 1});
  nmsed_classes.Resize({batch_size * keep_top_k, 1});
  ctx.template Alloc<T>(&nmsed_boxes);
  ctx.template Alloc<T>(&nmsed_scores);
  ctx.template Alloc<T>(&nmsed_classes);

  auto workspace_size =
      CalcDetectionInferenceWorkspaceSize<T>(share_location,
                                             batch_size,
                                             per_batch_boxes_size,
                                             per_batch_scores_size,
                                             num_classes,
                                             num_priors,
                                             nms_top_k);

  DenseTensor workspace = DenseTensor();
  workspace.Resize({static_cast<int64_t>(workspace_size)});
  T* workspace_ptr = ctx.template Alloc<T>(&workspace);

  // Launch the NMS kernel
  InferNMS<T>(stream,
              batch_size,
              per_batch_boxes_size,
              per_batch_scores_size,
              share_location,
              background_label,
              num_priors,
              num_classes,
              nms_top_k,
              keep_top_k,
              score_threshold,
              nms_threshold,
              transformed_bboxes.data<T>(),
              transformed_scores.data<T>(),
              keep_count.data<int>(),
              nmsed_boxes.data<T>(),
              nmsed_scores.data<T>(),
              nmsed_classes.data<T>(),
              nmsed_indices.data<int>(),
              nmsed_valid_mask.data<int>(),
              workspace_ptr,
              normalized,
              false,
              false,
              0,
              true);

  // Post-processing to get the final outputs
  // Concat the individual class, score and boxes outputs
  // into a [N * M, 6] tensor.
  DenseTensor raw_out;
  raw_out.Resize({batch_size * keep_top_k, 6});
  ctx.template Alloc<T>(&raw_out);
  phi::funcs::ConcatFunctor<Context, T> concat;
  concat(ctx, {nmsed_classes, nmsed_scores, nmsed_boxes}, 1, &raw_out);

  // Output of NMS kernel may include invalid entries, which is
  // marked by nmsed_valid_mask. Eliminate the invalid entries
  // by gathering the valid ones.

  // 1. Get valid indices
  DenseTensor valid_indices;
  NonZeroKernel<int, Context>(ctx, nmsed_valid_mask, &valid_indices);
  // 2. Perform gathering
  const int64_t valid_samples = valid_indices.dims()[0];
  out->Resize({valid_samples, 6});
  ctx.template Alloc<T>(out);
  phi::funcs::GPUGatherNd<T, int64_t>(ctx, raw_out, valid_indices, out);
  index->Resize({valid_samples, 1});
  ctx.template Alloc<int>(index);
  phi::funcs::GPUGatherNd<int, int64_t>(
      ctx, nmsed_indices, valid_indices, index);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    multiclass_nms3, GPU, ALL_LAYOUT, phi::MultiClassNMSGPUKernel, float) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}
