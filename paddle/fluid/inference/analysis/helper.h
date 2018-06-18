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

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace analysis {

template <typename Vec>
int AccuDims(Vec &&vec, int size) {
  int res = 1;
  for (int i = 0; i < size; i++) {
    res *= std::forward<Vec>(vec)[i];
  }
  return res;
}

#define SET_TYPE(type__) dic_[typeid(type__).hash_code()] = #type__;
/*
 * Map typeid to representation.
 */
struct DataTypeNamer {
  static const DataTypeNamer &Global() {
    static auto *x = new DataTypeNamer();
    return *x;
  }

  template <typename T>
  const std::string &repr() const {
    auto x = typeid(T).hash_code();
    PADDLE_ENFORCE(dic_.count(x), "unknown type for representation");
    return dic_.at(x);
  }

  const std::string &repr(size_t &hash) const {  // NOLINT
    PADDLE_ENFORCE(dic_.count(hash), "unknown type for representation");
    return dic_.at(hash);
  }

 private:
  DataTypeNamer() {
    SET_TYPE(int);
    SET_TYPE(bool);
    SET_TYPE(float);
    SET_TYPE(void *);
  }

  std::unordered_map<decltype(typeid(int).hash_code()),  // NOLINT
                     std::string>
      dic_;
};
#undef SET_TYPE

template <typename IteratorT>
class iterator_range {
  IteratorT begin_, end_;

 public:
  template <typename Container>
  explicit iterator_range(Container &&c) : begin_(c.begin()), end_(c.end()) {}

  iterator_range(const IteratorT &begin, const IteratorT &end)
      : begin_(begin), end_(end) {}

  const IteratorT &begin() const { return begin_; }
  const IteratorT &end() const { return end_; }
};

/*
 * An registry helper class, with its records keeps the order they registers.
 */
template <typename T>
class OrderedRegistry {
 public:
  T *Register(const std::string &name, T *x) {
    PADDLE_ENFORCE(!dic_.count(name));
    dic_[name] = data_.size();
    data_.emplace_back(std::unique_ptr<T>(x));
    return data_.back().get();
  }

  T *Lookup(const std::string &name) {
    auto it = dic_.find(name);
    if (it == dic_.end()) return nullptr;
    return data_[it->second].get();
  }

 protected:
  std::unordered_map<std::string, int> dic_;
  std::vector<std::unique_ptr<T>> data_;
};

template <typename T>
T &GetFromScope(const framework::Scope &scope, const std::string &name) {
  framework::Variable *var = scope.FindVar(name);
  PADDLE_ENFORCE(var != nullptr);
  return *var->GetMutable<T>();
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

#define PADDLE_DISALLOW_COPY_AND_ASSIGN(type__) \
  type__(const type__ &) = delete;              \
  void operator=(const type__ &) = delete;
