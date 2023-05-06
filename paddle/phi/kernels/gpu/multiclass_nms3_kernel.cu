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

#ifndef PADDLE_WITH_HIP

#include "paddle/phi/kernels/multiclass_nms3_kernel.h"
#include "paddle/phi/kernels/impl/multiclass_nms3_kernel_impl.h"

#include <cub/cub.cuh>
#include "cuda.h"  // NOLINT

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
size_t cubSortPairsWorkspaceSize(int num_items, int num_segments) {
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
size_t detectionForwardBBoxDataSize(int N, int C1) {
  return N * C1 * sizeof(T);
}

template <typename T>
size_t detectionForwardBBoxPermuteSize(bool shareLocation, int N, int C1) {
  return shareLocation ? 0 : N * C1 * sizeof(T);
}

template <typename T>
size_t detectionForwardPreNMSSize(int N, int C2) {
  return N * C2 * sizeof(T);
}

template <typename T>
size_t detectionForwardPostNMSSize(int N, int numClasses, int topK) {
  return N * numClasses * topK * sizeof(T);
}

size_t calculateTotalWorkspaceSize(size_t* workspaces, int count) {
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
size_t sortScoresPerClassWorkspaceSize(const int num,
                                       const int num_classes,
                                       const int num_preds_per_class) {
  size_t wss[4];
  const int arrayLen = num * num_classes * num_preds_per_class;
  wss[0] = arrayLen * sizeof(T);                   // temp scores
  wss[1] = arrayLen * sizeof(int);                 // temp indices
  wss[2] = (num * num_classes + 1) * sizeof(int);  // offsets
  wss[3] = cubSortPairsWorkspaceSize<T, int>(
      arrayLen, num * num_classes);  // cub workspace

  return calculateTotalWorkspaceSize(wss, 4);
}

template <typename T>
size_t sortScoresPerImageWorkspaceSize(const int num_images,
                                       const int num_items_per_image) {
  const int arrayLen = num_images * num_items_per_image;
  size_t wss[2];
  wss[0] = (num_images + 1) * sizeof(int);  // offsets
  wss[1] =
      cubSortPairsWorkspaceSize<T, int>(arrayLen, num_images);  // cub workspace

  return calculateTotalWorkspaceSize(wss, 2);
}

template <typename T>
size_t detectionInferenceWorkspaceSize(bool shareLocation,
                                       int N,
                                       int C1,
                                       int C2,
                                       int numClasses,
                                       int numPredsPerClass,
                                       int topK) {
  size_t wss[6];
  wss[0] = detectionForwardBBoxDataSize<T>(N, C1);
  wss[1] = detectionForwardPreNMSSize<T>(N, C2);
  wss[2] = detectionForwardPreNMSSize<int>(N, C2);
  wss[3] = detectionForwardPostNMSSize<T>(N, numClasses, topK);
  wss[4] = detectionForwardPostNMSSize<int>(N, numClasses, topK);
  wss[5] = std::max(
      sortScoresPerClassWorkspaceSize<T>(N, numClasses, numPredsPerClass),
      sortScoresPerImageWorkspaceSize<T>(N, numClasses * topK));
  return calculateTotalWorkspaceSize(wss, 6);
}

// ALIGNPTR
int8_t* alignPtr(int8_t* ptr, uintptr_t to) {
  uintptr_t addr = (uintptr_t)ptr;
  if (addr % to) {
    addr += to - addr % to;
  }
  return reinterpret_cast<int8_t*>(addr);
}

// NEXTWORKSPACEPTR
int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize) {
  uintptr_t addr = (uintptr_t)ptr;
  addr += previousWorkspaceSize;
  return alignPtr(reinterpret_cast<int8_t*>(addr), CUDA_MEM_ALIGN);
}

/* ==================
 * sortScoresPerClass
 * ================== */
template <typename T_SCORE, unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta) __global__
    void prepareSortData(const int num,
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
  const int numPredsPerBatch = num_classes * num_preds_per_class;
  T_SCORE clip_val =
      T_SCORE(static_cast<float>(score_shift) + 1.f - 1.f / 1024.f);
  if (cur_idx < numPredsPerBatch) {
    const int class_idx = cur_idx / num_preds_per_class;
    for (int i = 0; i < num; i++) {
      const int targetIdx = i * numPredsPerBatch + cur_idx;
      const T_SCORE score = conf_scores_gpu[targetIdx];

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
        temp_scores[targetIdx] = score_shift;
        temp_idx[targetIdx] = -1;
        conf_scores_gpu[targetIdx] = score_shift;
      } else {  // "Clear" scores lower than threshold
        if (static_cast<float>(score) > confidence_threshold) {
          // add shift of 1.0 to normalize the score values
          // to the range [1, 2).
          // add a constant shift to scores will not change the sort
          // result, but will help reduce the computation because
          // we only need to sort the mantissa part of the floating-point
          // numbers
          temp_scores[targetIdx] = score + score_shift;
          if (static_cast<float>(score_shift) > 0.f &&
              (temp_scores[targetIdx] >= clip_val))
            temp_scores[targetIdx] = clip_val;
          temp_idx[targetIdx] = cur_idx + i * numPredsPerBatch;
        } else {
          // Set scores to 0
          // Set label = -1
          // add shift of 1.0 to normalize the score values
          // to the range [1, 2).
          // add a constant shift to scores will not change the sort
          // result, but will help reduce the computation because
          // we only need to sort the mantissa part of the floating-point
          // numbers
          temp_scores[targetIdx] = score_shift;
          temp_idx[targetIdx] = -1;
          conf_scores_gpu[targetIdx] = score_shift;
          // TODO(tizheng): HERE writing memory too many times
        }
      }

      if ((cur_idx % num_preds_per_class) == 0) {
        const int offset_ct = i * num_classes + cur_idx / num_preds_per_class;
        d_offsets[offset_ct] = offset_ct * num_preds_per_class;
        // set the last element in d_offset
        if (blockIdx.x == 0 && threadIdx.x == 0)
          d_offsets[num * num_classes] = num * numPredsPerBatch;
      }
    }
  }
}

template <typename T_SCORE>
void sortScoresPerClass_gpu(cudaStream_t stream,
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
  const int arrayLen = num * num_classes * num_preds_per_class;
  void* temp_idx = nextWorkspacePtr(reinterpret_cast<int8_t*>(temp_scores),
                                    arrayLen * sizeof(T_SCORE));
  void* d_offsets = nextWorkspacePtr(reinterpret_cast<int8_t*>(temp_idx),
                                     arrayLen * sizeof(int));
  size_t cubOffsetSize = (num_segments + 1) * sizeof(int);
  void* cubWorkspace =
      nextWorkspacePtr(reinterpret_cast<int8_t*>(d_offsets), cubOffsetSize);

  const int BS = 512;
  const int GS = (num_classes * num_preds_per_class + BS - 1) / BS;
  // prepare the score, index, and offsets for CUB radix sort
  // also normalize the scores to the range [1, 2)
  // so we only need to sort the mantissa of floating-point numbers
  // since their sign bit and exponential bits are identical
  // we will subtract the 1.0 shift in gatherTopDetections()
  prepareSortData<T_SCORE, BS>
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
      cubSortPairsWorkspaceSize<T_SCORE, int>(arrayLen, num_segments);
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
      arrayLen,
      num_segments,
      reinterpret_cast<int*>(d_offsets),
      reinterpret_cast<int*>(d_offsets) + 1,
      begin_bit,
      end_bit,
      stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaGetLastError());
}

/* ===========
 * allClassNMS
 * =========== */
template <typename T_BBOX>
__device__ float bboxSize(const Bbox<T_BBOX>& bbox, const bool normalized) {
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
__device__ void intersectBbox(const Bbox<T_BBOX>& bbox1,
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
__device__ Bbox<T_BBOX> getDiagonalMinMaxSortedBox(const Bbox<T_BBOX>& bbox1) {
  Bbox<T_BBOX> result;
  result.xmin = min(bbox1.xmin, bbox1.xmax);
  result.xmax = max(bbox1.xmin, bbox1.xmax);

  result.ymin = min(bbox1.ymin, bbox1.ymax);
  result.ymax = max(bbox1.ymin, bbox1.ymax);
  return result;
}

template <typename T_BBOX>
__device__ float jaccardOverlap(const Bbox<T_BBOX>& bbox1,
                                const Bbox<T_BBOX>& bbox2,
                                const bool normalized,
                                const bool caffeSemantics) {
  Bbox<T_BBOX> intersect_bbox;

  Bbox<T_BBOX> localbbox1 = getDiagonalMinMaxSortedBox(bbox1);
  Bbox<T_BBOX> localbbox2 = getDiagonalMinMaxSortedBox(bbox2);

  intersectBbox(localbbox1, localbbox2, &intersect_bbox);

  float intersect_width, intersect_height;
  // Only when using Caffe semantics, IOU calculation adds "1" to width and
  // height if bbox is not normalized.
  // https://github.com/weiliu89/caffe/blob/ssd/src/caffe/util/bbox_util.cpp#L92-L97
  if (normalized || !caffeSemantics) {
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
    float bbox1_size = bboxSize(localbbox1, normalized);
    float bbox2_size = bboxSize(localbbox2, normalized);
    return intersect_size / (bbox1_size + bbox2_size - intersect_size);
  } else {
    return 0.;
  }
}

template <typename T_SCORE, typename T_BBOX, int TSIZE>
__global__ void allClassNMS_kernel(
    const int num,
    const int num_classes,
    const int num_preds_per_class,
    const int top_k,
    const float nms_threshold,
    const bool share_location,
    const bool isNormalized,
    T_BBOX* bbox_data,  // bbox_data should be float to preserve location
                        // information
    T_SCORE* beforeNMS_scores,
    int* beforeNMS_index_array,
    T_SCORE* afterNMS_scores,
    int* afterNMS_index_array,
    bool flipXY,
    const float score_shift,
    bool caffeSemantics) {
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
          loc_bboxIndex[t] = beforeNMS_index_array[item_idx];
        } else {
          loc_bboxIndex[t] = -1;
        }

        if (loc_bboxIndex[t] != -1) {
          const int bbox_data_idx =
              share_location
                  ? (loc_bboxIndex[t] % num_preds_per_class + bbox_idx_offset)
                  : loc_bboxIndex[t];

          loc_bbox[t].xmin = flipXY ? bbox_data[bbox_data_idx * 4 + 1]
                                    : bbox_data[bbox_data_idx * 4 + 0];
          loc_bbox[t].ymin = flipXY ? bbox_data[bbox_data_idx * 4 + 0]
                                    : bbox_data[bbox_data_idx * 4 + 1];
          loc_bbox[t].xmax = flipXY ? bbox_data[bbox_data_idx * 4 + 3]
                                    : bbox_data[bbox_data_idx * 4 + 2];
          loc_bbox[t].ymax = flipXY ? bbox_data[bbox_data_idx * 4 + 2]
                                    : bbox_data[bbox_data_idx * 4 + 3];
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
              ? (beforeNMS_index_array[ref_item_idx] % num_preds_per_class +
                 bbox_idx_offset)
              : beforeNMS_index_array[ref_item_idx];
    }
    while ((ref_bbox_idx != -1) && ref_item_idx < max_read_idx) {
      Bbox<T_BBOX> ref_bbox;
      ref_bbox.xmin = flipXY ? bbox_data[ref_bbox_idx * 4 + 1]
                             : bbox_data[ref_bbox_idx * 4 + 0];
      ref_bbox.ymin = flipXY ? bbox_data[ref_bbox_idx * 4 + 0]
                             : bbox_data[ref_bbox_idx * 4 + 1];
      ref_bbox.xmax = flipXY ? bbox_data[ref_bbox_idx * 4 + 3]
                             : bbox_data[ref_bbox_idx * 4 + 2];
      ref_bbox.ymax = flipXY ? bbox_data[ref_bbox_idx * 4 + 2]
                             : bbox_data[ref_bbox_idx * 4 + 3];

      // Eliminate shared memory RAW hazard
      __syncthreads();

      for (int t = 0; t < TSIZE; t++) {
        const int cur_idx = threadIdx.x + blockDim.x * t;
        const int item_idx = offset + cur_idx;

        if ((kept_bboxinfo_flag[cur_idx]) && (item_idx > ref_item_idx)) {
          if (jaccardOverlap(
                  ref_bbox, loc_bbox[t], isNormalized, caffeSemantics) >
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
                ? (beforeNMS_index_array[ref_item_idx] % num_preds_per_class +
                   bbox_idx_offset)
                : beforeNMS_index_array[ref_item_idx];
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
        afterNMS_scores[write_item_idx] =
            kept_bboxinfo_flag[cur_idx]
                ? T_SCORE(beforeNMS_scores[read_item_idx])
                : T_SCORE(score_shift);
        afterNMS_index_array[write_item_idx] =
            kept_bboxinfo_flag[cur_idx] ? loc_bboxIndex[t] : -1;
      }
    }
  }
}

template <typename T_SCORE, typename T_BBOX>
void allClassNMS_gpu(cudaStream_t stream,
                     const int num,
                     const int num_classes,
                     const int num_preds_per_class,
                     const int top_k,
                     const float nms_threshold,
                     const bool share_location,
                     const bool isNormalized,
                     void* bbox_data,
                     void* beforeNMS_scores,
                     void* beforeNMS_index_array,
                     void* afterNMS_scores,
                     void* afterNMS_index_array,
                     bool flipXY,
                     const float score_shift,
                     bool caffeSemantics) {
#define P(tsize) allClassNMS_kernel<T_SCORE, T_BBOX, (tsize)>

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
      isNormalized,
      reinterpret_cast<T_BBOX*>(bbox_data),
      reinterpret_cast<T_SCORE*>(beforeNMS_scores),
      reinterpret_cast<int*>(beforeNMS_index_array),
      reinterpret_cast<T_SCORE*>(afterNMS_scores),
      reinterpret_cast<int*>(afterNMS_index_array),
      flipXY,
      score_shift,
      caffeSemantics);

  PADDLE_ENFORCE_GPU_SUCCESS(cudaGetLastError());
}

/* ==================
 * sortScoresPerImage
 * ================== */
template <unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta) __global__
    void setUniformOffsets_kernel(const int num_segments,
                                  const int offset,
                                  int* d_offsets) {
  const int idx = blockIdx.x * nthds_per_cta + threadIdx.x;
  if (idx <= num_segments) d_offsets[idx] = idx * offset;
}

void setUniformOffsets(cudaStream_t stream,
                       const int num_segments,
                       const int offset,
                       int* d_offsets) {
  const int BS = 32;
  const int GS = (num_segments + 1 + BS - 1) / BS;
  setUniformOffsets_kernel<BS>
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
    void gatherNMSOutputs_kernel(const bool shareLocation,
                                 const int numImages,
                                 const int numPredsPerClass,
                                 const int numClasses,
                                 const int topK,
                                 const int keepTopK,
                                 const int* indices,
                                 const T_SCORE* scores,
                                 const T_BBOX* bboxData,
                                 int* numDetections,
                                 T_BBOX* nmsedBoxes,
                                 T_BBOX* nmsedScores,
                                 T_BBOX* nmsedClasses,
                                 int* nmsedIndices,
                                 int* nmsedValidMask,
                                 bool clipBoxes,
                                 const T_SCORE scoreShift) {
  if (keepTopK > topK) return;
  for (int i = blockIdx.x * nthds_per_cta + threadIdx.x;
       i < numImages * keepTopK;
       i += gridDim.x * nthds_per_cta) {
    const int imgId = i / keepTopK;
    const int detId = i % keepTopK;
    const int offset = imgId * numClasses * topK;
    const int index = indices[offset + detId];
    const T_SCORE score = scores[offset + detId];
    if (index == -1) {
      nmsedClasses[i] = -1;
      nmsedScores[i] = 0;
      nmsedBoxes[i * 4] = 0;
      nmsedBoxes[i * 4 + 1] = 0;
      nmsedBoxes[i * 4 + 2] = 0;
      nmsedBoxes[i * 4 + 3] = 0;
      nmsedIndices[i] = -1;
      nmsedValidMask[i] = 0;
    } else {
      const int bboxOffset =
          imgId *
          (shareLocation ? numPredsPerClass : (numClasses * numPredsPerClass));
      const int bboxId =
          ((shareLocation ? (index % numPredsPerClass)
                          : index % (numClasses * numPredsPerClass)) +
           bboxOffset) *
          4;
      nmsedClasses[i] = (index % (numClasses * numPredsPerClass)) /
                        numPredsPerClass;  // label
      nmsedScores[i] = score;              // confidence score
      nmsedScores[i] = nmsedScores[i] - scoreShift;
      const T_BBOX xMin = bboxData[bboxId];
      const T_BBOX yMin = bboxData[bboxId + 1];
      const T_BBOX xMax = bboxData[bboxId + 2];
      const T_BBOX yMax = bboxData[bboxId + 3];
      // clipped bbox xmin
      nmsedBoxes[i * 4] = clipBoxes ? saturate(xMin) : xMin;
      // clipped bbox ymin
      nmsedBoxes[i * 4 + 1] = clipBoxes ? saturate(yMin) : yMin;
      // clipped bbox xmax
      nmsedBoxes[i * 4 + 2] = clipBoxes ? saturate(xMax) : xMax;
      // clipped bbox ymax
      nmsedBoxes[i * 4 + 3] = clipBoxes ? saturate(yMax) : yMax;
      nmsedIndices[i] = bboxId >> 2;
      nmsedValidMask[i] = 1;
      atomicAdd(&numDetections[i / keepTopK], 1);
    }
  }
}

template <typename T_BBOX, typename T_SCORE>
void gatherNMSOutputs_gpu(cudaStream_t stream,
                          const bool shareLocation,
                          const int numImages,
                          const int numPredsPerClass,
                          const int numClasses,
                          const int topK,
                          const int keepTopK,
                          const void* indices,
                          const void* scores,
                          const void* bboxData,
                          void* numDetections,
                          void* nmsedBoxes,
                          void* nmsedScores,
                          void* nmsedClasses,
                          void* nmsedIndices,
                          void* nmsedValidMask,
                          bool clipBoxes,
                          const float scoreShift) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(numDetections, 0, numImages * sizeof(int), stream));
  const int BS = 32;
  const int GS = 32;
  gatherNMSOutputs_kernel<T_BBOX, T_SCORE, BS>
      <<<GS, BS, 0, stream>>>(shareLocation,
                              numImages,
                              numPredsPerClass,
                              numClasses,
                              topK,
                              keepTopK,
                              reinterpret_cast<const int*>(indices),
                              reinterpret_cast<const T_SCORE*>(scores),
                              reinterpret_cast<const T_BBOX*>(bboxData),
                              reinterpret_cast<int*>(numDetections),
                              reinterpret_cast<T_BBOX*>(nmsedBoxes),
                              reinterpret_cast<T_BBOX*>(nmsedScores),
                              reinterpret_cast<T_BBOX*>(nmsedClasses),
                              reinterpret_cast<int*>(nmsedIndices),
                              reinterpret_cast<int*>(nmsedValidMask),
                              clipBoxes,
                              T_SCORE(scoreShift));

  PADDLE_ENFORCE_GPU_SUCCESS(cudaGetLastError());
}

template <typename T_SCORE>
void sortScoresPerImage_gpu(cudaStream_t stream,
                            const int num_images,
                            const int num_items_per_image,
                            void* unsorted_scores,
                            void* unsorted_bbox_indices,
                            void* sorted_scores,
                            void* sorted_bbox_indices,
                            void* workspace,
                            int score_bits) {
  void* d_offsets = workspace;
  void* cubWorkspace = nextWorkspacePtr(reinterpret_cast<int8_t*>(d_offsets),
                                        (num_images + 1) * sizeof(int));

  setUniformOffsets(stream,
                    num_images,
                    num_items_per_image,
                    reinterpret_cast<int*>(d_offsets));

  const int arrayLen = num_images * num_items_per_image;
  size_t temp_storage_bytes =
      cubSortPairsWorkspaceSize<T_SCORE, int>(arrayLen, num_images);
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
      arrayLen,
      num_images,
      reinterpret_cast<int*>(d_offsets),
      reinterpret_cast<int*>(d_offsets) + 1,
      begin_bit,
      end_bit,
      stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaGetLastError());
}

template <typename T>
void nmsInference(cudaStream_t stream,
                  const int N,
                  const int perBatchBoxesSize,
                  const int perBatchScoresSize,
                  const bool shareLocation,
                  const int backgroundLabelId,
                  const int numPredsPerClass,
                  const int numClasses,
                  const int topK,
                  const int keepTopK,
                  const float scoreThreshold,
                  const float iouThreshold,
                  const void* locData,
                  const void* confData,
                  void* keepCount,
                  void* nmsedBoxes,
                  void* nmsedScores,
                  void* nmsedClasses,
                  void* nmsedIndices,
                  void* nmsedValidMask,
                  void* workspace,
                  bool isNormalized,
                  bool confSigmoid,
                  bool clipBoxes,
                  int scoreBits,
                  bool caffeSemantics) {
  PADDLE_ENFORCE_EQ(
      shareLocation,
      true,
      phi::errors::Fatal("shareLocation=false is not supported."));
  size_t bboxDataSize = detectionForwardBBoxDataSize<T>(N, perBatchBoxesSize);
  void* bboxDataRaw = workspace;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
      bboxDataRaw, locData, bboxDataSize, cudaMemcpyDeviceToDevice, stream));
  void* bboxData = bboxDataRaw;

  const int numScores = N * perBatchScoresSize;
  size_t totalScoresSize = detectionForwardPreNMSSize<T>(N, perBatchScoresSize);
  void* scores =
      nextWorkspacePtr(reinterpret_cast<int8_t*>(bboxData), bboxDataSize);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
      scores, confData, totalScoresSize, cudaMemcpyDeviceToDevice, stream));

  size_t indicesSize = detectionForwardPreNMSSize<int>(N, perBatchScoresSize);
  void* indices =
      nextWorkspacePtr(reinterpret_cast<int8_t*>(scores), totalScoresSize);

  size_t postNMSScoresSize =
      detectionForwardPostNMSSize<T>(N, numClasses, topK);
  size_t postNMSIndicesSize = detectionForwardPostNMSSize<int>(
      N, numClasses, topK);  // indices are full int32
  void* postNMSScores =
      nextWorkspacePtr(reinterpret_cast<int8_t*>(indices), indicesSize);
  void* postNMSIndices = nextWorkspacePtr(
      reinterpret_cast<int8_t*>(postNMSScores), postNMSScoresSize);

  void* sortingWorkspace = nextWorkspacePtr(
      reinterpret_cast<int8_t*>(postNMSIndices), postNMSIndicesSize);
  // Sort the scores so that the following NMS could be applied.
  float scoreShift = 0.f;
  sortScoresPerClass_gpu<T>(stream,
                            N,
                            numClasses,
                            numPredsPerClass,
                            backgroundLabelId,
                            scoreThreshold,
                            scores,
                            indices,
                            sortingWorkspace,
                            scoreBits,
                            scoreShift);

  // This is set to true as the input bounding boxes are of the format [ymin,
  // xmin, ymax, xmax]. The default implementation assumes [xmin, ymin, xmax,
  // ymax]
  bool flipXY = true;
  // NMS
  allClassNMS_gpu<T, T>(stream,
                        N,
                        numClasses,
                        numPredsPerClass,
                        topK,
                        iouThreshold,
                        shareLocation,
                        isNormalized,
                        bboxData,
                        scores,
                        indices,
                        postNMSScores,
                        postNMSIndices,
                        flipXY,
                        scoreShift,
                        caffeSemantics);

  // Sort the bounding boxes after NMS using scores
  sortScoresPerImage_gpu<T>(stream,
                            N,
                            numClasses * topK,
                            postNMSScores,
                            postNMSIndices,
                            scores,
                            indices,
                            sortingWorkspace,
                            scoreBits);

  // Gather data from the sorted bounding boxes after NMS
  gatherNMSOutputs_gpu<T, T>(stream,
                             shareLocation,
                             N,
                             numPredsPerClass,
                             numClasses,
                             topK,
                             keepTopK,
                             indices,
                             scores,
                             bboxData,
                             keepCount,
                             nmsedBoxes,
                             nmsedScores,
                             nmsedClasses,
                             nmsedIndices,
                             nmsedValidMask,
                             clipBoxes,
                             scoreShift);
}

template <typename T, typename Context>
void MultiClassNMSKernel(const Context& ctx,
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
    VLOG(3)
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
    MultiClassNMSCPUKernel<T, phi::CPUContext>(*cpu_ctx,
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

  int64_t batch_size = score_dims[0];
  int64_t box_dim = bboxes.dims()[2];
  int64_t out_dim = box_dim + 2;

  const int64_t per_batch_boxes_size =
      bboxes.dims()[1] * bboxes.dims()[2];  // M * 4
  const int64_t per_batch_scores_size =
      scores.dims()[1] * scores.dims()[2];       // C * M
  const int64_t num_priors = bboxes.dims()[1];   // M
  const int64_t num_classes = scores.dims()[1];  // C
  const bool share_location = true;
  auto stream = reinterpret_cast<const Context&>(ctx).stream();

  PADDLE_ENFORCE_LE(
      nms_top_k,
      num_priors,
      phi::errors::InvalidArgument("Expect nms_top_k (%d)"
                                   " <= num of boxes per batch (%d).",
                                   nms_top_k,
                                   num_priors));
  PADDLE_ENFORCE_LE(keep_top_k,
                    nms_top_k,
                    phi::errors::InvalidArgument("Expect keep_top_k (%d)"
                                                 " <= nms_top_k (%d).",
                                                 keep_top_k,
                                                 nms_top_k));

  // bboxes: [N,M,4] -> [N,1,M,4]
  DenseTensor transformed_bboxes(bboxes.type());
  transformed_bboxes.ShareDataWith(bboxes).Resize(
      {bboxes.dims()[0], 1, bboxes.dims()[1], bboxes.dims()[2]});
  // scores: [N, C, M] => [N, C, M, 1]
  DenseTensor transformed_scores(scores.type());
  transformed_scores.ShareDataWith(scores).Resize(
      {scores.dims()[0], scores.dims()[1], scores.dims()[2], 1});

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
      detectionInferenceWorkspaceSize<T>(share_location,
                                         batch_size,
                                         per_batch_boxes_size,
                                         per_batch_scores_size,
                                         num_classes,
                                         num_priors,
                                         nms_top_k);

  DenseTensor workspace = DenseTensor();
  workspace.Resize({static_cast<int64_t>(workspace_size)});
  T* workspace_ptr = ctx.template Alloc<T>(&workspace);

  nmsInference<T>(stream,
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

  // concat
  // out [N * M, 6]
  DenseTensor raw_out;
  raw_out.Resize({batch_size * keep_top_k, 6});
  ctx.template Alloc<T>(&raw_out);
  phi::funcs::ConcatFunctor<Context, T> concat;
  concat(ctx, {nmsed_classes, nmsed_scores, nmsed_boxes}, 1, &raw_out);

  // get valid indices
  DenseTensor valid_indices;
  NonZeroKernel<int, Context>(ctx, nmsed_valid_mask, &valid_indices);

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

PD_REGISTER_KERNEL(multiclass_nms3,  // cuda_only
                   GPU,
                   ALL_LAYOUT,
                   phi::MultiClassNMSKernel,
                   float) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}

#endif
