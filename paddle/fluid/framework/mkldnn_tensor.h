/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <mkldnn.h>
#include <vector>
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace framework {

class MKLDNNTensorData;

namespace detail {
inline std::shared_ptr<const MKLDNNTensorData> ExtractMKLDNNDataFromTensor(
    const Tensor& tensor);
inline std::shared_ptr<MKLDNNTensorData> ExtractMKLDNNDataFromTensor(
    Tensor* tensor);
inline void SetMKLDNNData(Tensor* tensor, const mkldnn::engine& engine);
}

inline mkldnn::memory::format to_mkldnn_format(DataLayout layout) {
  switch (layout) {
    case DataLayout::kNHWC:
      // TODO(pzelazko) currently Tensor has nhwc layout by default, but
      // eventually nchw is used.
      // return mkldnn::memory::format::nhwc;
      return mkldnn::memory::format::nchw;
    case DataLayout::kNCHW:
      return mkldnn::memory::format::nchw;
    case DataLayout::kAnyLayout:
      return mkldnn::memory::format::nchw;
    default:
      return mkldnn::memory::format::format_undef;
  }
}

inline DataLayout to_paddle_layout(mkldnn::memory::format format) {
  switch (format) {
    case mkldnn::memory::format::nhwc:
      return DataLayout::kNHWC;
    case mkldnn::memory::format::nchw:
      return DataLayout::kNHWC;  // currently NHWC is default layout despite
                                 // it's treated as NCHW
    case mkldnn::memory::format::oihw:  // from paddle perspective oihw is nchw
      return DataLayout::kNCHW;
    default:
      return DataLayout::kMKLDNN;
  }
}

class MKLDNNTensorData : public Tensor::ExtendedData {
 public:
  explicit MKLDNNTensorData(const mkldnn::engine& engine) : engine_(engine) {}
  inline mkldnn::memory::format GetFormat() const { return format_; }

  inline void SetFormat(mkldnn::memory::format format) { format_ = format; }

  inline const mkldnn::engine& GetEngine() const { return engine_; }

  inline mkldnn::memory::data_type GetDataType() const { return data_type_; }

 private:
  const mkldnn::memory::data_type data_type_{mkldnn::memory::data_type::f32};
  mkldnn::memory::format format_{mkldnn::memory::format::format_undef};
  const mkldnn::engine engine_;
};

// Decorator class for unmutable Tensor
class MKLDNNTensor {
 public:
  explicit MKLDNNTensor(const Tensor& tensor) : tensor_(tensor) {
    PADDLE_ENFORCE(tensor.layout() == DataLayout::kMKLDNN,
                   "Tensor should has MKLDNN data layout set");

    data_ = detail::ExtractMKLDNNDataFromTensor(tensor);
    PADDLE_ENFORCE(data_ != nullptr,
                   "Tensor should be initalized with MKLDNN "
                   "data when passed to "
                   "MKLDNNTensor class");
  }

  virtual ~MKLDNNTensor() {}

  inline static bool IsInitialized(const Tensor& tensor);

  mkldnn::memory::format GetFormat() const { return data_->GetFormat(); }
  const mkldnn::engine& GetEngine() const { return data_->GetEngine(); }
  mkldnn::memory::data_type GetDataType() const { return data_->GetDataType(); }

  inline mkldnn::memory GetMemory() const {
    return mkldnn::memory({GetMemoryDesc(), GetEngine()},
                          const_cast<void*>(tensor_.data<void>()));
  }

  inline mkldnn::memory::desc GetMemoryDesc() const {
    return GetMemoryDesc(GetFormat());
  }

  inline mkldnn::memory::desc GetMemoryDesc(
      mkldnn::memory::format format) const {
    std::vector<int> dimensions = vectorize2int(tensor_.dims());
    return mkldnn::memory::desc({dimensions}, GetDataType(), format);
  }

  inline void Reorder(Tensor* out, DataLayout dst_layout,
                      platform::Place place) const;

 private:
  const Tensor& tensor_;
  std::shared_ptr<const MKLDNNTensorData> data_;
};

// Decorator class for Tensor
class MKLDNNTensorMutable : public MKLDNNTensor {
 public:
  inline static MKLDNNTensorMutable Create(Tensor* tensor,
                                           const mkldnn::engine& engine);

  inline mkldnn::memory GetMemory() {
    return mkldnn::memory({GetMemoryDesc(), data_->GetEngine()},
                          tensor_->data<void>());
  }

  inline mkldnn::memory GetMutableMemory(const platform::Place& place) {
    return mkldnn::memory(
        {GetMemoryDesc(), data_->GetEngine()},
        static_cast<void*>(tensor_->mutable_data<float>(place)));
  }

  inline void SetFormat(mkldnn::memory::format format) {
    data_->SetFormat(format);
    // TODO(pzelazko): change dims here
  }

  inline void Reorder(mkldnn::memory::format dst_format);

 private:
  MKLDNNTensorMutable(Tensor* tensor, const mkldnn::engine& engine)
      : MKLDNNTensor(*tensor), tensor_(tensor) {
    data_ = detail::ExtractMKLDNNDataFromTensor(tensor);
  }

  Tensor* tensor_;
  std::shared_ptr<MKLDNNTensorData> data_;
};

inline MKLDNNTensorMutable MKLDNNTensorMutable::Create(
    Tensor* tensor, const mkldnn::engine& engine) {
  if (!tensor->get_extended_data()) {
    detail::SetMKLDNNData(tensor, engine);
  }

  if (!MKLDNNTensor::IsInitialized(*tensor)) {
    tensor->set_layout(DataLayout::kMKLDNN);
  }

  return MKLDNNTensorMutable(tensor, engine);
}

inline bool MKLDNNTensor::IsInitialized(const Tensor& tensor) {
  return tensor.layout() == DataLayout::kMKLDNN;
}

inline void MKLDNNTensorMutable::Reorder(mkldnn::memory::format dst_format) {
  PADDLE_ENFORCE(tensor_->type().hash_code() == typeid(float).hash_code(),
                 "MKLDNN tensor should be created from float type Tensor");

  mkldnn::memory::format src_format{data_->GetFormat()};
  if (src_format == mkldnn::memory::format::nchw &&
      (src_format == mkldnn::memory::format::Ohwi8o ||
       src_format == mkldnn::memory::format::oihw)) {
    src_format = mkldnn::memory::format::oihw;
  }

  if (src_format == dst_format) {
    return;
  }

  VLOG(3) << "reordering " << reinterpret_cast<void*>(tensor_) << " to "
          << dst_format << std::endl;

  mkldnn::memory src_memory{{GetMemoryDesc(src_format), data_->GetEngine()},
                            const_cast<void*>(tensor_->data<void>())};
  data_->SetFormat(dst_format);
  // TODO(pzelazko) change axis here
  mkldnn::memory dst_memory{{GetMemoryDesc(dst_format), data_->GetEngine()},
                            const_cast<void*>(tensor_->data<void>())};

  auto reorder = mkldnn::reorder(src_memory, dst_memory);
  std::vector<mkldnn::primitive> pipeline{reorder};
  mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
}

inline void MKLDNNTensor::Reorder(Tensor* out, DataLayout dst_layout,
                                  platform::Place place) const {
  PADDLE_ENFORCE(dst_layout != DataLayout::kAnyLayout);
  if (dst_layout == DataLayout::kMKLDNN) {
    return;
  }

  mkldnn::memory::format src_format{GetFormat()};
  mkldnn::memory::format dst_format{to_mkldnn_format(dst_layout)};
  VLOG(3) << "reordering from " << src_format << " to " << dst_format
          << std::endl;
  if (src_format == dst_format) {
    out->ShareDataWith(tensor_);
    out->set_extended_data(nullptr);
    out->set_layout(dst_layout);
    return;
  }

  mkldnn::memory src_memory{{GetMemoryDesc(src_format), GetEngine()},
                            const_cast<void*>(tensor_.data<void>())};
  // TODO(pzelazko) change axis here

  std::vector<int> dimensions = vectorize2int(tensor_.dims());

  out->Resize(make_ddim(dimensions));
  out->mutable_data<float>(place);

  mkldnn::memory::desc dst_memory_desc(
      {dimensions}, mkldnn::memory::data_type::f32, dst_format);
  mkldnn::memory dst_memory{{dst_memory_desc, GetEngine()},
                            const_cast<void*>(out->data<void>())};

  out->set_layout(dst_layout);

  auto reorder = mkldnn::reorder(src_memory, dst_memory);
  std::vector<mkldnn::primitive> pipeline{reorder};
  mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
}

inline mkldnn::memory::format GetMKLDNNFormat(const Tensor& tensor) {
  DataLayout layout = tensor.layout();
  if (layout == DataLayout::kMKLDNN) {
    return MKLDNNTensor(tensor).GetFormat();
  }
  return to_mkldnn_format(layout);
}

namespace detail {
inline std::shared_ptr<const MKLDNNTensorData> ExtractMKLDNNDataFromTensor(
    const Tensor& tensor) {
  std::shared_ptr<const framework::Tensor::ExtendedData> data =
      tensor.get_extended_data();
  if (data) {
    return std::dynamic_pointer_cast<const MKLDNNTensorData>(data);
  }
  return nullptr;
}

inline std::shared_ptr<MKLDNNTensorData> ExtractMKLDNNDataFromTensor(
    Tensor* tensor) {
  std::shared_ptr<framework::Tensor::ExtendedData> data =
      tensor->get_extended_data();
  if (data) {
    return std::dynamic_pointer_cast<MKLDNNTensorData>(data);
  }
  return nullptr;
}

inline void SetMKLDNNData(Tensor* tensor, const mkldnn::engine& engine) {
  std::shared_ptr<MKLDNNTensorData> data = ExtractMKLDNNDataFromTensor(tensor);
  if (!data) {
    data = std::make_shared<MKLDNNTensorData>(engine);
    tensor->set_extended_data(data);
  }
  data->SetFormat(to_mkldnn_format(tensor->layout()));
}
}  // namespace detail

}  // namespace framework
}  // namespace paddle
