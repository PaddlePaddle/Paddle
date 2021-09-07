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

#ifdef PADDLE_WITH_MKLDNN

#include "mkldnn.hpp"

#include "paddle/tcmpt/core/dense_tensor.h"

namespace pt {

class MKLDNNDenseTensor : public DenseTensor {
 public:
  // Not allowed to initialize a tensor without descriptive metadata
  MKLDNNDenseTensor() = delete;

  MKLDNNDenseTensor(const MKLDNNDenseTensor&) = delete;
  MKLDNNDenseTensor& operator=(const MKLDNNDenseTensor&) = delete;
  MKLDNNDenseTensor(MKLDNNDenseTensor&&) = delete;
  MKLDNNDenseTensor& operator=(MKLDNNDenseTensor&&) = delete;

  MKLDNNDenseTensor(const TensorMeta& meta, const TensorStatus& status)
      : DenseTensor(meta, status) {}

  mkldnn::memory::format_tag format() const { return format_; }

  void set_format(const mkldnn::memory::format_tag format) { format_ = format; }

 private:
  /**
   * @brief the detail format of memory block which have layout as kMKLDNN
   *
   * @note MKLDNN lib support various memory format like nchw, nhwc, nChw8C,
   *       nChw16c, etc. For a MKLDNN memory block, layout will be set as
   *       DataLayout::kMKLDNN meanwhile detail memory format will be kept in
   *       this field.
   */
  mkldnn::memory::format_tag format_ = mkldnn::memory::format_tag::undef;
};

}  // namespace pt

#endif
