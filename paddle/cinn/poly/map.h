// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <glog/logging.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/poly/dim.h"
#include "paddle/cinn/poly/domain.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace poly {

struct Iterator {
  std::string id;

  Iterator() = default;
  explicit Iterator(const std::string& id) : id(id) {}
  Iterator(const Iterator& x) : id(x.id) {}  // NOLINT
  explicit Iterator(Iterator&& x) : id(std::move(x.id)) {}

  Iterator& operator=(const Iterator& other);
  friend bool operator==(const Iterator& a, const Iterator& b) {
    return a.id == b.id;
  }
  friend bool operator!=(const Iterator& a, const Iterator& b) {
    return !(a.id == b.id);
  }

  friend std::ostream& operator<<(std::ostream& os, const Iterator& x);
};

struct Condition {
  std::string cond;

  explicit Condition(std::string cond) : cond(std::move(cond)) {}

  friend std::ostream& operator<<(std::ostream& os, const Condition& x) {
    os << x.__str__();
    return os;
  }

  std::string __str__() const {
    return utils::StringFormat("%s", cond.c_str());
  }
};

/**
 * A wrapper on isl::map.
 */
class Map {
 public:
  Map(isl::ctx ctx,
      std::string id,
      std::vector<Iterator> domain_iterators,
      std::vector<Iterator> range_iterators,
      std::vector<Condition> conds,
      std::string range_id = "");

  //! Get the corresponding ISL map.
  isl::map to_isl() const;

  //! Get the ISL style map representation, such as '{ S[i,j] -> [i,j]: }'.
  std::string __str__() const;

 protected:
  isl::ctx ctx_;
  std::string id_;
  std::vector<Iterator> domain_iterators_;
  std::vector<Iterator> range_iterators_;
  std::vector<Condition> conds_;
  std::string range_id_;
};

class Aff : public Map {
 public:
  Aff(isl::ctx ctx,
      std::string id,
      std::vector<Iterator> domain_iterators,
      std::vector<Iterator> range_iterators,
      std::vector<Condition> conds)
      : Map(std::move(ctx),
            std::move(id),
            std::move(domain_iterators),
            std::move(range_iterators),
            std::move(conds),
            "") {}

  isl::aff to_isl() const { return isl::aff(ctx_, __str__()); }
};

std::ostream& operator<<(std::ostream& os, const Map& x);
std::ostream& operator<<(std::ostream& os, const Aff& x);
static bool operator<(const Iterator& a, const Iterator& b) {
  return a.id < b.id;
}

}  // namespace poly
}  // namespace cinn
