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

#pragma once

#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "dtu/hlir/builder/hlir_builder.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace gcu {
static DataType PrimitiveTypeToDataType(const builder::PrimitiveType& dtype) {
  if (dtype == builder::PrimitiveType::PRED()) {
    return DataType::BOOL;
  } else if (dtype == builder::PrimitiveType::S8()) {
    return DataType::INT8;
  } else if (dtype == builder::PrimitiveType::U8()) {
    return DataType::UINT8;
  } else if (dtype == builder::PrimitiveType::F16()) {
    return DataType::FLOAT16;
  } else if (dtype == builder::PrimitiveType::BF16()) {
    return DataType::BFLOAT16;
  } else if (dtype == builder::PrimitiveType::S16()) {
    return DataType::INT16;
  } else if (dtype == builder::PrimitiveType::U16()) {
    return DataType::UINT16;
  } else if (dtype == builder::PrimitiveType::F32()) {
    return DataType::FLOAT32;
  } else if (dtype == builder::PrimitiveType::S32()) {
    return DataType::INT32;
  } else if (dtype == builder::PrimitiveType::U32()) {
    return DataType::UINT32;
  } else if (dtype == builder::PrimitiveType::S64()) {
    return DataType::INT64;
  } else if (dtype == builder::PrimitiveType::U64()) {
    return DataType::UINT64;
  } else if (dtype == builder::PrimitiveType::F64()) {
    return DataType::FLOAT64;
  }
  return DataType::UNDEFINED;
}

class GcuTensor;
// for layout transfer
enum class Layout : int { NCHW, NHWC, HWCN, NCDHW, NDHWC, DHWCN };
std::vector<int64_t> GetPermByFormat(const std::string& src_format,
                                     const std::string& dst_format);
// only for pd origin tensor valid
Layout GetFormatByPdShape(const std::vector<int64_t>& pdshape);

std::vector<int64_t> TransShapeByFormat(const std::vector<int64_t>& shape,
                                        const std::string& src_format,
                                        const std::string& dst_format);

size_t GetElementSize(const builder::PrimitiveType& dtype);

std::string DtypeToString(const builder::PrimitiveType& dtype);

Layout StringToLayout(const std::string& layout);

std::string LayoutToString(const Layout& layout);

void GcuTranspose(const GcuTensor& in,
                  GcuTensor& out,  // NOLINT
                  const std::vector<int64_t>& perm);

class GcuTensor {
 public:
  GcuTensor(const std::vector<int64_t>& shape,
            Layout format,
            builder::PrimitiveType dtype);
  GcuTensor(const GcuTensor& gcu_tensor) = default;

  std::vector<int64_t> GetShape() const { return shape_; }

  std::string GetShapeStr() const {
    std::ostringstream ss;
    ss << "[";
    for (const int64_t& s : shape_) {
      ss << s << ",";
    }
    ss << "]";
    return ss.str();
  }

  void SetShape(const std::vector<int64_t>& shape);

  builder::PrimitiveType GetDataType() const { return dtype_; }

  void SetDataType(const builder::PrimitiveType& dtype);

  Layout GetFormat() const { return format_; }

  std::string GetFormatStr() const { return LayoutToString(format_); }

  void SetFormat(const Layout& format) { format_ = format; }

  void* Data() const { return data_; }

  void SetData(void* data) { data_ = data; }

  size_t Size() { return size_; }

 private:
  std::vector<int64_t> shape_;
  Layout format_;
  builder::PrimitiveType dtype_;
  int64_t numel_ = 0;
  size_t size_ = 0;
  void* data_ = nullptr;
};

class GcuTensorTable {
  using GcuTensorPtr = std::shared_ptr<GcuTensor>;

 public:
  void BuffTensor(const void* pdtensor,
                  const GcuTensorPtr& gcu_tensor,
                  bool is_global = false) {
    // std::lock_guard<std::mutex> lock(mu_);
    if (!is_global) {
      tmp_tensors_.insert(pdtensor);
      tmp_tensors_layout_[pdtensor] = gcu_tensor;
    } else {
      global_tensors_.insert(pdtensor);
      global_tensors_layout_[pdtensor] = gcu_tensor;
    }
  }

  bool IsIn(const void* pdtensor) {
    // std::lock_guard<std::mutex> lock(mu_);
    if (IsInGlobal(pdtensor)) {
      return true;
    } else {
      return IsInTmp(pdtensor);
    }
  }

  bool IsInGlobal(const void* pdtensor) {
    // std::lock_guard<std::mutex> lock(mu_);
    return global_tensors_.count(pdtensor) != 0;
  }

  bool IsInTmp(const void* pdtensor) {
    // std::lock_guard<std::mutex> lock(mu_);
    return tmp_tensors_.count(pdtensor) != 0;
  }

  GcuTensorPtr GetTransedGcuTensor(const void* pdtensor) {
    // std::lock_guard<std::mutex> lock(mu_);
    if (GcuTensorTable::GetInstance()->IsInGlobal(pdtensor)) {
      return global_tensors_layout_[pdtensor];
    } else if (GcuTensorTable::GetInstance()->IsInTmp(pdtensor)) {
      return tmp_tensors_layout_[pdtensor];
    } else {
      // PADDLE_THROW(platform::errors::InvalidArgument(
      //     "pdtensor %p is not buffered in gcu tensor table global area" ,
      //     pdtensor));
      return nullptr;
    }
  }

  void ClearBuffedTensors(bool is_global = false) {
    // std::lock_guard<std::mutex> lock(mu_);
    if (!is_global) {
      tmp_tensors_.clear();
      tmp_tensors_layout_.clear();
    } else {
      global_tensors_.clear();
      global_tensors_layout_.clear();
    }
  }

  static GcuTensorTable* GetInstance() {
    static GcuTensorTable manager;
    return &manager;
  }

 private:
  GcuTensorTable() = default;
  ~GcuTensorTable() {
    std::lock_guard<std::mutex> lock(mu_);
    ClearBuffedTensors(true);
    ClearBuffedTensors(false);
  }

 private:
  std::mutex mu_;
  // buffered tensor struct to avoid memory reuse
  std::unordered_set<const void*> global_tensors_;
  std::unordered_set<const void*> tmp_tensors_;
  std::map<const void*, GcuTensorPtr> global_tensors_layout_;
  std::map<const void*, GcuTensorPtr> tmp_tensors_layout_;
};
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
