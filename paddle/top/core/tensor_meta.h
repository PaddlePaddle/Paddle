/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <vector>

#ifdef PADDLE_WITH_MKLDNN
#include "mkldnn.hpp"
#endif

#include "paddle/top/core/backend.h"
#include "paddle/top/core/dtype.h"
#include "paddle/top/core/layout.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/ddim.h"
// Note: mixed_vector include many header now, LoD will be
// used on CUDA device? Can we use small_vector here?
// #include "paddle/fluid/framework/mixed_vector.h"

namespace pt {

// template <typename T>
// using Vector = paddle::framework::Vector<T>;

/*
 * LoD is short for Level of Details.
 *
 * - in a level, each element indicates relative offset of the lower level
 * - the first element should be 0 and that indicates that this sequence start
 * from 0
 * - each sequence's begin and end(no-inclusive) is level[id, id+1]
 *
 * For example:
 *    3-level LoD stores
 *
 *    0 2 3
 *    0 2 4 7
 *    0 2 5 7 10 12 15 20
 */
// using LoD = std::vector<paddle::framework::Vector<size_t>>;
using LoD = std::vector<std::vector<size_t>>;

/**
 * The Meta data member of DenseTensor.
 *
 * Here the `meta` represents information describing the basic features and
 * data features of Tensor, and does not include the status information of
 * Tensor
 *
 * Note: TensorMeta is a struct, the members are named like
 * ordinary nonmember variables, such as `type` instead of `type_`.
 * And we direct access its members, in addition to constructor, destructor
 * and functions for setting data members, can not provide other functions.
 */
struct TensorMeta {
  TensorMeta() = delete;
  TensorMeta(const TensorMeta&) = delete;
  TensorMeta& operator=(const TensorMeta&) = delete;
  // TensorMeta(TensorMeta&&) = delete;
  TensorMeta& operator=(TensorMeta&&) = delete;

  TensorMeta(TensorMeta&& meta)
      : dims(meta.dims),
        backend(meta.backend),
        type(meta.type),
        layout(meta.layout),
        numel(meta.numel),
        offset(meta.offset),
        lod(meta.lod) {}

  // Bad constructor, may introduce bug
  // explicit TensorMeta(DDim dims) : dims(dims) {}

  // Compatible Contructor
  TensorMeta(const DDim& dims,
             Backend backend,
             DataType type,
             DataLayout layout = DataLayout::kNCHW,
             size_t offset = 0UL,
             const LoD& lod = {})
      : dims(dims),
        backend(backend),
        type(type),
        layout(layout),
        offset(offset),
        lod(lod) {
    int64_t init_numel = paddle::framework::product(dims);
    if (init_numel > 0) {
      numel = init_numel;
    }
  }

  virtual ~TensorMeta() = default;

  DDim dims;

  Backend backend{Backend::kCPU};
  DataType type{DataType::kFLOAT32};
  DataLayout layout{DataLayout::kNCHW};

  /**
   * [ Why not calculate numel based on dims? ]
   *
   * Tensor may be 0-dimensional, but 0-dimensional Tensor may have values.
   * For example:
   *
   *   import paddle
   *
   *   a = paddle.to_tensor([1, 2, 3])
   *   print(a[0].shape) # expected: []
   *   print(a[0].numel()) # expected: 1
   *
   * Now Paddle can not get expected result above, because the old Tensor's
   * numel is calculated based on dims.
   */
  int64_t numel{1};

  size_t offset{0};

  /**
   * [ Why basic TensorMeta hold LoD? ]
   *
   * LoDTensor is still the main Tensor concept in Paddle.
   * Although only a small number of ops need to use LoD information,
   * LoD may need to be passed between Op's input and output, which is
   * difficult to remove in a short time.
   *
   * But we don't want to add a Tensor type because of LoD, which makes
   * the concept complicated, so LoD is a member held by Tensor by default.
   */
  LoD lod;
};

#ifdef PADDLE_WITH_MKLDNN
struct MKLDNNTensorMeta : public TensorMeta {
  MKLDNNTensorMeta(
      const DDim& dims,
      Backend backend,
      DataType type,
      DataLayout layout,
      size_t offset = 0UL,
      const LoD& lod = {},
      mkldnn::memory::format_tag format = mkldnn::memory::format_tag::undef)
      : TensorMeta(dims, backend, type, layout, offset, lod), format(format) {}

  ~MKLDNNTensorMeta() override {}

  /**
   * @brief the detail format of memory block which have layout as kMKLDNN
   *
   * @note MKLDNN lib support various memory format like nchw, nhwc, nChw8C,
   *       nChw16c, etc. For a MKLDNN memory block, layout will be set as
   *       DataLayout::kMKLDNN meanwhile detail memory format will be kept in
   *       this field.
   */
  mkldnn::memory::format_tag format = mkldnn::memory::format_tag::undef;
};
#endif

}  // namespace pt
