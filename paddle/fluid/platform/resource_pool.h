// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace platform {

template <typename T>
class ResourcePool : public std::enable_shared_from_this<ResourcePool<T>> {
 private:
  struct ResourceDeleter {
   public:
    explicit ResourceDeleter(ResourcePool<T> *pool)
        : instance_(pool->shared_from_this()) {}

    void operator()(T *ptr) const { instance_->Restore(ptr); }

   private:
    std::shared_ptr<ResourcePool<T>> instance_;
  };

 public:
  static std::shared_ptr<ResourcePool<T>> Create(
      const std::function<T *()> &creator,
      const std::function<void(T *)> &deleter) {
    return std::shared_ptr<ResourcePool<T>>(
        new ResourcePool<T>(creator, deleter));
  }

  ~ResourcePool() {
    for (auto *ptr : instances_) {
      deleter_(ptr);
    }
  }

  std::shared_ptr<T> New() {
    std::lock_guard<std::mutex> guard(mtx_);
    T *obj = nullptr;
    if (instances_.empty()) {
      obj = creator_();
      PADDLE_ENFORCE_NOT_NULL(obj,
                              platform::errors::PermissionDenied(
                                  "The creator should not return nullptr."));
      VLOG(10) << "Create new instance " << TypePtrName();
    } else {
      obj = instances_.back();
      instances_.pop_back();
      VLOG(10) << "Pop new instance " << TypePtrName()
               << " from pool, size=" << instances_.size();
    }
    return std::shared_ptr<T>(obj, ResourceDeleter(this));
  }

 private:
  static std::string TypePtrName() {
    return platform::demangle(typeid(T *).name());  // NOLINT
  }

 private:
  ResourcePool(const std::function<T *()> &creator,
               const std::function<void(T *)> &deleter)
      : creator_(creator), deleter_(deleter) {}

  void Restore(T *ptr) {
    std::lock_guard<std::mutex> guard(mtx_);
    instances_.emplace_back(ptr);
    VLOG(10) << "Restore " << TypePtrName()
             << " into pool, size=" << instances_.size();
  }

 private:
  std::vector<T *> instances_;
  const std::function<T *()> creator_;
  const std::function<void(T *)> deleter_;

  std::mutex mtx_;
};

}  // namespace platform
}  // namespace paddle
