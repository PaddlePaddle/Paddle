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

#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/memory/circlable_ref_impl.h"
#include "paddle/ap/include/memory/circlable_ref_list.h"

namespace ap::memory {

template <typename Derived, typename T>
class CirclableRef {
 public:
  CirclableRef(const CirclableRef&) = default;
  CirclableRef(CirclableRef&&) = default;
  explicit CirclableRef(const std::shared_ptr<CirclableRefImpl<T>>& impl)
      : impl_(impl) {}
  CirclableRef& operator=(const CirclableRef&) = default;
  CirclableRef& operator=(CirclableRef&&) = default;

  explicit operator bool() const { return impl_->operator bool(); }

  adt::Result<const T*> Get() const { return impl_->Get(); }

  adt::Result<T*> Mut() const { return impl_->Mut(); }

  adt::Result<std::shared_ptr<T>> shared_ptr() const {
    ADT_CHECK(this->operator bool());
    return impl_->shared_ptr();
  }

  bool operator==(const CirclableRef& other) const {
    return this->impl_ == other.impl_;
  }

  static Derived Make(const std::shared_ptr<CirclableRefListBase>& ref_list,
                      const std::shared_ptr<T>& obj) {
    auto impl = std::make_shared<CirclableRefImpl<T>>(obj);
    auto iter = ref_list->AddWeakRef(impl);
    const auto& ok = impl->InitWeakRefIterAndList(iter, ref_list);
    (void)ok;
    return Derived(impl);
  }

 protected:
  T* raw_ptr() const { return impl_->shared_ptr().get(); }

  std::shared_ptr<CirclableRefImpl<T>> impl_;
};

}  // namespace ap::memory
