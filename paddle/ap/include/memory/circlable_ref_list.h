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
#include "paddle/ap/include/memory/circlable_ref_list_base.h"

namespace ap::memory {

class CirclableRefList : public CirclableRefListBase {
 public:
  CirclableRefList(const CirclableRefList&) = delete;
  CirclableRefList(CirclableRefList&&) = delete;
  ~CirclableRefList() override {
    const auto& ok = EraseAllWeakRef();
    (void)ok;
  }

  CirclableRefList() {}

  const WeakRefList& weak_refs() const { return weak_refs_; }

  WeakRefList::iterator AddWeakRef(
      const std::shared_ptr<CirclableRefImplBase>& ref) override {
    std::weak_ptr<CirclableRefImplBase> weak_ref{ref};
    WeakRefList::iterator weak_iter =
        weak_refs_.insert(weak_refs_.end(), weak_ref);
    return weak_iter;
  }

  adt::Result<WeakRefList::iterator> EraseWeakRef(
      CirclableRefImplBase* ref) override {
    ADT_CHECK(ref->weak_ref_iter().has_value());
    ADT_CHECK(ref->circlable_ref_list_ptr().has_value());
    const auto& weak_ptr = ref->circlable_ref_list_ptr().value();
    if (!weak_ptr.expired()) {
      ADT_CHECK(weak_ptr.lock().get() == this);
    } else {
      // called by my own destructor.
      ADT_CHECK(weak_ptr.lock().get() == nullptr);
    }
    auto iter = weak_refs_.erase(ref->weak_ref_iter().value());
    ref->ClearIterAndRef();
    return iter;
  }

 private:
  adt::Result<adt::Ok> EraseAllWeakRef() {
    for (auto iter = weak_refs_.begin(); iter != weak_refs_.end();) {
      if (auto ref = iter->lock()) {
        ADT_CHECK(ref->weak_ref_iter().has_value());
        ADT_CHECK(iter == ref->weak_ref_iter().value());
        ADT_LET_CONST_REF(next_iter, this->EraseWeakRef(ref.get()));
        iter = next_iter;
      } else {
        ++iter;
      }
    }
    return adt::Ok{};
  }
  WeakRefList weak_refs_;
};

}  // namespace ap::memory
