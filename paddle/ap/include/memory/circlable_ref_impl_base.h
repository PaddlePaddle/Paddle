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

#include <list>
#include <memory>
#include <optional>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/memory/circlable_ref_list_base.h"

namespace ap::memory {

class CirclableRefImplBase
    : public std::enable_shared_from_this<CirclableRefImplBase> {
 public:
  CirclableRefImplBase(const CirclableRefImplBase&) = delete;
  CirclableRefImplBase(CirclableRefImplBase&&) = delete;
  virtual ~CirclableRefImplBase() {}

  virtual void ClearRef() = 0;

  using WeakRefList = std::list<std::weak_ptr<CirclableRefImplBase>>;
  using CirclableRefListPtr = std::weak_ptr<CirclableRefListBase>;

  const std::optional<WeakRefList::iterator>& weak_ref_iter() const {
    return weak_ref_iter_;
  }

  const std::optional<CirclableRefListPtr>& circlable_ref_list_ptr() const {
    return circlable_ref_list_weak_ptr_;
  }

  adt::Result<adt::Ok> InitWeakRefIterAndList(WeakRefList::iterator iter,
                                              const CirclableRefListPtr& list) {
    ADT_CHECK(!weak_ref_iter_.has_value());
    ADT_CHECK(!circlable_ref_list_weak_ptr_.has_value());
    weak_ref_iter_ = iter;
    circlable_ref_list_weak_ptr_ = list;
    return adt::Ok{};
  }

  void ClearIterAndRef() {
    weak_ref_iter_ = std::nullopt;
    circlable_ref_list_weak_ptr_ = std::nullopt;
    ClearRef();
  }

 protected:
  CirclableRefImplBase() = default;

  std::optional<WeakRefList::iterator> weak_ref_iter_;
  std::optional<CirclableRefListPtr> circlable_ref_list_weak_ptr_;
};

}  // namespace ap::memory
