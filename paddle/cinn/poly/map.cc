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

#include "paddle/cinn/poly/map.h"

#include "paddle/cinn/poly/isl_utils.h"
#include "paddle/cinn/utils/functional.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace poly {

std::string Map::__str__() const {
  PADDLE_ENFORCE_EQ(
      !domain_iterators_.empty(),
      true,
      ::common::errors::InvalidArgument(
          "The domain iterators are empty. Please provide valid iterators."));

  auto get_ids_repr = [](const std::vector<Iterator>& ids) {
    std::vector<std::string> fields;
    std::transform(ids.begin(),
                   ids.end(),
                   std::back_inserter(fields),
                   [](const Iterator& x) { return x.id; });
    return utils::Join(fields, ", ");
  };

  auto domain_iterators_repr = get_ids_repr(domain_iterators_);
  auto range_iterators_repr = get_ids_repr(range_iterators_);

  std::vector<std::string> conds_fields;
  std::transform(conds_.begin(),
                 conds_.end(),
                 std::back_inserter(conds_fields),
                 [](const Condition& x) { return x.__str__(); });
  auto conds_repr = utils::Join(conds_fields, " and ");

  if (!conds_.empty()) {
    return utils::StringFormat("{ %s[%s] -> %s[%s]: %s }",
                               id_.c_str(),
                               domain_iterators_repr.c_str(),
                               range_id_.c_str(),
                               range_iterators_repr.c_str(),
                               conds_repr.c_str());
  }

  return utils::StringFormat("{ %s[%s] -> %s[%s] }",
                             id_.c_str(),
                             domain_iterators_repr.c_str(),
                             range_id_.c_str(),
                             range_iterators_repr.c_str());
}

Map::Map(isl::ctx ctx,
         std::string id,
         std::vector<Iterator> domain_iterators,
         std::vector<Iterator> range_iterators,
         std::vector<Condition> conds,
         std::string range_id)
    : ctx_(ctx),
      id_(std::move(id)),
      domain_iterators_(std::move(domain_iterators)),
      range_iterators_(std::move(range_iterators)),
      conds_(std::move(conds)),
      range_id_(std::move(range_id)) {}

isl::map Map::to_isl() const {
  auto map = isl::map(ctx_, __str__());
  // set dimension names
  auto handler = [](const Iterator& x) { return x.id; };
  auto domain_dim_names = utils::Map<std::vector<Iterator>, std::string>(
      domain_iterators_, handler);
  auto range_dim_names =
      utils::Map<std::vector<Iterator>, std::string>(range_iterators_, handler);
  isl_set_dim_names(&map, isl_dim_in, domain_dim_names);
  isl_set_dim_names(&map, isl_dim_out, range_dim_names);
  return map;
}

std::ostream& operator<<(std::ostream& os, const Iterator& x) {
  os << utils::StringFormat("<Iterator: %s>", x.id.c_str());
  return os;
}

std::ostream& operator<<(std::ostream& os, const Map& x) {
  os << x.__str__();
  return os;
}

std::ostream& operator<<(std::ostream& os, const Aff& x) {
  os << x.__str__();
  return os;
}

Iterator& Iterator::operator=(const Iterator& other) {
  id = other.id;
  return *this;
}

}  // namespace poly
}  // namespace cinn
