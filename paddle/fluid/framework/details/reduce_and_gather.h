//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <map>
#include <vector>
#include "paddle/fluid/framework/details/reduce_and_gather.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows.h"
namespace paddle {
namespace framework {
namespace details {

struct ReduceLoDTensor {
  const std::vector<const LoDTensor *> &src_tensors_;
  LoDTensor &dst_tensor_;

  ReduceLoDTensor(const std::vector<const LoDTensor *> &src, LoDTensor *dst)
      : src_tensors_(src), dst_tensor_(*dst) {}

  template <typename T>
  void apply() const {
    PADDLE_ENFORCE(!src_tensors_.empty());
    auto &t0 = *src_tensors_[0];
    PADDLE_ENFORCE_NE(t0.numel(), 0);

    dst_tensor_.Resize(t0.dims());
    T *dst = dst_tensor_.mutable_data<T>(platform::CPUPlace());

    for (size_t i = 0; i < src_tensors_.size(); ++i) {
      auto &t = *src_tensors_[i];
      if (dst == t.data<T>()) {
        continue;
      }

      PADDLE_ENFORCE_EQ(t.dims(), t0.dims());
      PADDLE_ENFORCE_EQ(t.type(), t0.type());
      std::transform(t.data<T>(), t.data<T>() + t.numel(), dst, dst,
                     [](T a, T b) -> T { return a + b; });
    }
  }
};

struct ReduceBufferData {
  const std::vector<const void *> &src_data_;
  void *dst_data_;
  int64_t numel_;

  ReduceBufferData(const std::vector<const void *> &src, void *dst,
                   int64_t numel)
      : src_data_(src), dst_data_(dst), numel_(numel) {}

  template <typename T>
  void apply() const {
    T *dst_data = reinterpret_cast<T *>(dst_data_);
    for (size_t i = 0; i < src_data_.size(); ++i) {
      auto srd_data = reinterpret_cast<const T *>(src_data_[i]);
      VLOG(10) << "dst: " << dst_data_ << ", " << srd_data;
      if (srd_data == dst_data_) {
        continue;
      }

      std::transform(srd_data, srd_data + numel_, dst_data, dst_data,
                     [](T a, T b) -> T { return a + b; });
    }
  }
};

inline void GatherLocalSelectedRows(
    const std::vector<const SelectedRows *> &src_selecte_rows_,
    const std::vector<platform::Place> &in_places,
    const std::map<platform::Place, platform::DeviceContext *> &dev_ctxes,
    const platform::Place &out_place, SelectedRows *dst_selecte_rows) {
  PADDLE_ENFORCE(!src_selecte_rows_.empty());

  std::vector<Tensor> in_tensors;
  std::vector<int64_t> out_rows;

  for (auto in_sr_ptr : src_selecte_rows_) {
    auto &in_sr = *in_sr_ptr;
    in_tensors.emplace_back(in_sr.value());
    out_rows.insert(out_rows.end(), in_sr.rows().begin(), in_sr.rows().end());
  }

  auto &pre_in = src_selecte_rows_[0];

  auto &dst_tensor = *dst_selecte_rows;
  dst_tensor.set_height(pre_in->height());
  dst_tensor.set_rows(out_rows);
  size_t rows = out_rows.size();
  DDim out_dim = pre_in->GetCompleteDims();
  out_dim[0] = static_cast<int64_t>(rows);
  dst_tensor.mutable_value()->Resize(out_dim);
  dst_tensor.mutable_value()->mutable_data(out_place, pre_in->value().type());
  Tensor *out_tensor = dst_tensor.mutable_value();

  // copy
  int s = 0, e = 0;
  for (size_t j = 0; j < in_tensors.size(); ++j) {
    e += in_tensors[j].dims()[0];
    auto sub_out = out_tensor->Slice(s, e);
    paddle::framework::TensorCopy(in_tensors[j], out_place,
                                  *(dev_ctxes.at(in_places[j])), &sub_out);
    s = e;
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
