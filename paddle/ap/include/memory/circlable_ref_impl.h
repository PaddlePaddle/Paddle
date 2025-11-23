// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/memory/circlable_ref_impl_base.h"
#include "paddle/ap/include/memory/circlable_ref_list.h"

namespace ap::memory {

template <typename T>
class CirclableRefImpl : public CirclableRefImplBase {
 public:
  explicit CirclableRefImpl(const std::shared_ptr<T>& data) : data_(data) {}
  ~CirclableRefImpl() override { EraseIterFromList(); }

  void ClearRef() override { data_.reset(); }

  void EraseIterFromList() {
    if (circlable_ref_list_weak_ptr_.has_value() &&
        weak_ref_iter_.has_value()) {
      if (auto ptr = circlable_ref_list_weak_ptr_.value().lock()) {
        const auto& ret = ptr->EraseWeakRef(this);
        (void)ret;
      }
    }
  }

  explicit operator bool() const { return static_cast<bool>(data_); }

  adt::Result<const T*> Get() const {
    const auto* ptr = data_.get();
    ADT_CHECK(ptr != nullptr) << adt::errors::TypeError{
        "ptr is deleted. please check CirclableRefList is alive."};
    return ptr;
  }

  adt::Result<T*> Mut() {
    auto* ptr = data_.get();
    ADT_CHECK(ptr != nullptr) << adt::errors::TypeError{
        "ptr is deleted. please check CirclableRefList is alive."};
    return ptr;
  }

  const std::shared_ptr<T>& shared_ptr() const { return data_; }

 private:
  std::shared_ptr<T> data_;
};

}  // namespace ap::memory
