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

#include <functional>
#include <map>
#include <memory>
#include <mutex>  // NOLINT [build/c++11]
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dtu/hlir/builder/hlir_builder.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device/gcu/register/register.h"
#include "paddle/fluid/platform/device/gcu/utils/layout.h"
#include "paddle/fluid/platform/device/gcu/utils/types.h"
#include "paddle/fluid/platform/device/gcu/utils/utils.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace platform {
namespace gcu {
class TensorItem {
 public:
  TensorItem() = default;
  ~TensorItem() = default;
  explicit TensorItem(const std::string& name) { name_ = name; }
  TensorItem(const TensorItem& tensor_table) = default;
  TensorItem(const std::string& name, const GcuTransInfo& trans_info)
      : name_(name), trans_info_(trans_info) {}
  std::string Name() const { return name_; }
  void SetName(const std::string& tensor_name) { name_ = tensor_name; }
  bool IsShapeTransed() const { return is_shape_transed_; }
  void SetShapeTransed() { is_shape_transed_ = true; }
  void SetTransInfo(const GcuTransInfo& trans_info) {
    trans_info_ = trans_info_;
  }
  GcuTransInfo GetTransInfo() const { return trans_info_; }
  bool IsValid() { return !name_.empty(); }

  bool operator<(const TensorItem& item) const { return name_ < item.Name(); }

 private:
  std::string name_;
  GcuTransInfo trans_info_;
  bool is_shape_transed_ = false;
};

class TensorTable {
  using TensorItemsPtr = std::shared_ptr<std::set<TensorItem>>;

 public:
  void Insert(const int64_t& program_id, const TensorItem& item) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (program_to_items_.count(program_id) == 0) {
      auto items = std::make_shared<std::set<TensorItem>>();
      items->insert(item);
      program_to_items_[program_id] = items;
    } else {
      auto it = program_to_items_.find(program_id);
      (it->second)->insert(item);
    }
  }

  bool IsInHostTrans(const std::string& name) {
    auto item = GetTensorItem(name);
    return item.IsValid();
  }

  bool IsInHostNoTrans(const std::string& name) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    return var_to_host_no_trans_item_.count(name) != 0;
  }

  // if same name , return the first match
  TensorItem GetHostNoTransTensorItem(const std::string& name) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    return var_to_host_no_trans_item_[name];
  }

  void InsertHostNoTransTensorItem(const TensorItem& item) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    var_to_host_no_trans_item_[item.Name()] = item;
  }

  void UpdateItem(const int64_t& program_id, const TensorItem& item) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    PADDLE_ENFORCE_NE(
        program_to_items_.count(program_id),
        0,
        platform::errors::NotFound(
            "Update Item failed.Not found any item according to program id %ld",
            program_id));
    auto iter = program_to_items_.find(program_id);
    PADDLE_ENFORCE_NE(
        iter->second->count(item),
        0,
        platform::errors::NotFound("Update Item failed.Not found any item "
                                   "according to in program id %ld items.",
                                   program_id));
    iter->second->emplace(item);
  }

  void UpdateItem(const TensorItem& item) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    bool is_hit = false;
    for (auto& p : program_to_items_) {
      if (p.second->count(item) == 0) {
        continue;
      }
      p.second->emplace(item);
      is_hit = true;
    }
    PADDLE_ENFORCE_NE(is_hit,
                      false,
                      platform::errors::NotFound(
                          "Update Item failed.Not found any item by name %s",
                          item.Name().c_str()));
  }

  std::set<TensorItem> GetItemsByProgramId(const int64_t& program_id) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    auto it = program_to_items_.find(program_id);
    return *(it->second);
  }

  TensorItem GetTensorItem(const int64_t& program_id, const std::string& name) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    auto program_it = program_to_items_.find(program_id);
    if (program_it == program_to_items_.end()) {
      VLOG(1) << "[Warning] can not find tensor tables according to programid! "
                 "pid is:"
              << program_id;
      return TensorItem();
    }
    if (program_it->second->find(TensorItem(name)) ==
        program_it->second->end()) {
      return TensorItem();
    }
    return *(program_it->second->find(TensorItem(name)));
  }
  // if same name , return the first match
  TensorItem GetTensorItem(const std::string& name) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    for (const auto& items : program_to_items_) {
      for (const auto& item : *(items.second)) {
        if (item.Name() == name) {
          return item;
        }
      }
    }
    VLOG(1) << "can not lookup item by name " << name
            << ". create an empty item as return!";
    return TensorItem();
  }

  static TensorTable* GetInstance() {
    static TensorTable manager;
    return &manager;
  }

 private:
  std::recursive_mutex mu_;
  std::map<int64_t, TensorItemsPtr> program_to_items_;

 public:
  std::map<std::string, TensorItem> var_to_host_no_trans_item_;
};
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
