/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "dnnl.hpp"  // NOLINT
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

/// \brief The onednn tensor is used when run onednn related kernel
class OneDNNTensor : public DenseTensor {
 public:
  /// \brief construct a onednn tensor by DenseTensor
  explict OneDNNTensor(const DenseTensor& dense_tensor)
      : DenseTensor(dense_tensor) {}

  OneDNNTensor(const OneDNNTensor& other) : DenseTensor(other) {
    format_ = other.format_;
    mem_desc_ = other.mem_desc_;
  }

  OneDNNTensor() : DenseTensor() {}

  DenseTensor& operator=(const OneDNNTensor& other) {
    DenseTensor::operator=(other) format_ = other.format_;
    mem_desc_ = other.mem_desc_;
    return *this;
  }

  DenseTensor& operator=(DenseTensor&& other) {
    DenseTensor::operator=(other) format_ = other.format_;
    mem_desc_ = other.mem_desc_;
    return *this;
  }

  /// \brief Destroy the tensor object and release exclusive resources.
  virtual ~OneDNNTensor() = default;

  dnnl::memory::desc mem_desc() const {
    return mem_desc_
               ? mem_desc_
               : dnnl::memory::desc(phi::vectorize(meta_.dims),
                                    phi::TransToMKLDNNDataType(meta_.dtype),
                                    format_);
  }

  dnnl::memory::format_tag format() const {
    return mem_desc_ ? paddle::platform::GetMKLDNNFormat(mem_desc_) : format_;
  }

  inline void set_mem_desc(const dnnl::memory::desc& mem_desc) {
    mem_desc_ = mem_desc;
    meta_.layout = DataLayout::kMKLDNN;
  }

  inline void set_format(const dnnl::memory::format_tag format) {
    format_ = format;
  }

 private:
  /**
   * @brief the detail format of memory block which have layout as kMKLDNN
   *
   * @note MKLDNN lib support various memory format like nchw, nhwc, nChw8C,
   *       nChw16c, etc. For a MKLDNN memory block, layout will be set as
   *       DataLayout::kMKLDNN meanwhile detail memory format will be kept in
   *       this field.
   */
  dnnl::memory::format_tag format_ = dnnl::memory::format_tag::undef;

  /// \brief memory descriptor of tensor which have layout set as kMKLDNN
  dnnl::memory::desc mem_desc_;
};

}  // namespace phi

#endif
