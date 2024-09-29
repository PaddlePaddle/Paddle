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

#include "paddle/cinn/poly/isl_utils.h"

#include <glog/logging.h>
#include <isl/cpp.h>

#include <algorithm>
#include <set>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace poly {
using utils::Join;
using utils::StringFormat;

std::vector<std::string> isl_get_dim_names(const isl::set &x) {
  std::vector<std::string> res;
  for (int i = 0; i < isl_set_dim(x.get(), isl_dim_set); i++) {
    res.push_back(isl_set_get_dim_name(x.get(), isl_dim_set, i));
  }
  return res;
}

std::vector<std::string> isl_get_dim_names(const isl::map &x,
                                           isl_dim_type dim_type) {
  std::vector<std::string> res;
  for (int i = 0; i < isl_map_dim(x.get(), dim_type); i++) {
    res.push_back(isl_map_get_dim_name(x.get(), dim_type, i));
  }
  return res;
}

std::vector<std::string> isl_get_dim_names(isl_set *set) {
  std::vector<std::string> res;
  for (int i = 0; i < isl_set_dim(set, isl_dim_set); i++) {
    res.push_back(isl_set_get_dim_name(set, isl_dim_set, i));
  }
  return res;
}

void isl_set_dim_names(isl::map *map,
                       isl_dim_type dim_type,
                       const std::vector<std::string> &names) {
  const int dim = isl_map_dim(map->get(), dim_type);
  PADDLE_ENFORCE_EQ(
      dim,
      names.size(),
      ::common::errors::InvalidArgument(
          "The size of names should be equal to the dimension of the map."));

  for (int i = 0; i < dim; i++) {
    *map = isl::manage(
        isl_map_set_dim_name(map->release(), dim_type, i, names[i].c_str()));
  }
}

void isl_set_dim_names(isl::set *set, const std::vector<std::string> &names) {
  int dim = isl_set_dim(set->get(), isl_dim_set);
  PADDLE_ENFORCE_EQ(
      dim,
      names.size(),
      ::common::errors::InvalidArgument(
          "The size of names should be equal to the dimension of the set."));

  for (int i = 0; i < dim; i++) {
    *set = isl::manage(
        isl_set_set_dim_name(set->release(), isl_dim_set, i, names[i].c_str()));
  }
}

isl::union_map isl_maps_to_union_map(const std::vector<isl::map> &maps) {
  PADDLE_ENFORCE_EQ(!maps.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The input vector of isl::map is empty. "
                        "Please ensure the vector is not empty."));
  isl::union_map umap =
      isl::manage(isl_union_map_from_map(maps.front().copy()));
  for (int i = 1; i < maps.size(); i++) {
    umap = isl::manage(isl_union_map_add_map(umap.release(), maps[i].copy()));
  }
  return umap;
}

isl::union_set isl_sets_to_union_set(const std::vector<isl::set> &sets) {
  PADDLE_ENFORCE_EQ(!sets.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The input vector of isl::set is empty. "
                        "Please ensure the vector is not empty."));
  isl::union_set uset =
      isl::manage(isl_union_set_from_set(sets.front().copy()));
  for (int i = 1; i < sets.size(); i++) {
    uset = isl::manage(isl_union_set_add_set(uset.release(), sets[i].copy()));
  }
  return uset;
}

std::string isl_map_get_statement_repr(__isl_keep isl_map *map,
                                       isl_dim_type type) {
  PADDLE_ENFORCE_NOT_NULL(
      map,
      ::common::errors::InvalidArgument(
          "The input isl_map is null. "
          "Please ensure the map is properly initialized."));
  auto tuple_name = isl_map_get_tuple_name(map, type);
  std::vector<std::string> dims;

  for (int i = 0; i < isl_map_dim(map, type); i++) {
    dims.push_back(isl_map_get_dim_name(map, type, i));
  }
  return StringFormat("%s[%s]", tuple_name, Join(dims, ", ").c_str());
}

std::vector<std::string> isl_get_dim_names(isl_map *map,
                                           isl_dim_type dim_type) {
  std::vector<std::string> res;
  int n = isl_map_dim(map, dim_type);
  for (int i = 0; i < n; i++) {
    res.push_back(isl_map_get_dim_name(map, dim_type, i));
  }
  return res;
}

isl::set SetGetDims(isl::set set, const std::vector<int> &dims) {
  std::string tuple_name = isl_set_get_tuple_name(set.get());
  auto dim_names = isl_get_dim_names(set);
  std::vector<std::string> selected_dim_names;
  for (int v : dims) {
    PADDLE_ENFORCE_LT(
        v,
        dim_names.size(),
        ::common::errors::InvalidArgument(
            "The dim index should be less than the size of dim names."));
    selected_dim_names.push_back(dim_names[v]);
  }

  std::string transform_repr =
      StringFormat("{ %s[%s] -> %s[%s] }",
                   tuple_name.c_str(),             //
                   Join(dim_names, ", ").c_str(),  //
                   tuple_name.c_str(),             //
                   Join(selected_dim_names, ", ").c_str());
  isl::map transform(set.ctx(), transform_repr);
  return set.apply(transform);
}

isl_set *isl_get_preceding_axis(isl_set *set, int level, bool with_tuple_name) {
  int n = isl_set_dim(set, isl_dim_set);
  PADDLE_ENFORCE_LT(
      level,
      n,
      ::common::errors::InvalidArgument(
          "The level should be less than the dimension of the set."));

  std::vector<std::string> domain_iterators;
  std::vector<std::string> range_iterators;

  for (int i = 0; i < n; i++) {
    domain_iterators.push_back(cinn::UniqName("i" + std::to_string(i)));
  }

  for (int i = 0; i < level; i++) {
    range_iterators.push_back(cinn::UniqName("i" + std::to_string(i)));
  }

  const char *statement = isl_set_get_tuple_name(set);

  std::string repr =
      utils::StringFormat("{ %s[%s] -> %s[%s] }",
                          statement,
                          utils::Join(domain_iterators, ", ").c_str(),
                          statement,
                          utils::Join(range_iterators, ", ").c_str());
  auto transform =
      isl::manage(isl_map_read_from_str(isl_set_get_ctx(set), repr.c_str()));

  return isl_set_apply(set, transform.release());
}

int isl_get_original_axes_from_optimized_level(isl_set __isl_keep *a,
                                               int level) {
  int original_level = -1;
  std::vector<std::tuple<int, int>> iden_dim_offsets;
  for (int i = 0; i <= level;) {
    original_level++;
    if (isl_set_axis_has_noparam_constant_bound(a, original_level)) {
      auto range = isl_set_get_axis_range(a, original_level);
      auto &minv = std::get<0>(range);
      auto &maxv = std::get<1>(range);

      int min_iv = minv.get_num_si();
      int max_iv = maxv.get_num_si();
      if (max_iv == min_iv) {
        i--;
      }
    }
    i++;
  }
  return original_level;
}

int isl_get_preceding_removed_axes_counts(isl_set __isl_keep *a, int level) {
  int removed_axes_counts = 0;
  std::vector<std::tuple<int, int>> iden_dim_offsets;
  for (int i = 0; i <= level; i++) {
    if (isl_set_axis_has_noparam_constant_bound(a, i)) {
      auto range = isl_set_get_axis_range(a, i);
      auto &minv = std::get<0>(range);
      auto &maxv = std::get<1>(range);

      int min_iv = minv.get_num_si();
      int max_iv = maxv.get_num_si();
      if (max_iv == min_iv) {
        removed_axes_counts++;
      }
    }
  }
  return removed_axes_counts;
}

bool isl_is_removed_axis(isl_set __isl_keep *a, int level) {
  std::vector<std::tuple<int, int>> iden_dim_offsets;
  if (isl_set_axis_has_noparam_constant_bound(a, level)) {
    auto range = isl_set_get_axis_range(a, level);
    auto &minv = std::get<0>(range);
    auto &maxv = std::get<1>(range);

    int min_iv = minv.get_num_si();
    int max_iv = maxv.get_num_si();
    if (max_iv == min_iv) {
      return true;
    }
  }
  return false;
}

int isl_max_level_compatible(isl_set *a, isl_set *b) {
  int an = isl_set_dim(a, isl_dim_set);
  int bn = isl_set_dim(b, isl_dim_set);
  PADDLE_ENFORCE_GE(
      an,
      0,
      ::common::errors::InvalidArgument(
          "The dimension of the set should be greater than or equal to 0."));
  PADDLE_ENFORCE_GE(
      bn,
      0,
      ::common::errors::InvalidArgument(
          "The dimension of the set should be greater than or equal to 0."));

  int compatible_level = -1;
  for (int i = 0; i < std::min(an, bn); i++) {
    isl::set a_prefix =
        isl::manage(isl_get_preceding_axis(isl_set_copy(a), i, false));
    isl::set b_prefix =
        isl::manage(isl_get_preceding_axis(isl_set_copy(b), i, false));

    a_prefix = isl::manage(isl_set_set_tuple_name(a_prefix.release(), "s"));
    b_prefix = isl::manage(isl_set_set_tuple_name(b_prefix.release(), "s"));
    if (isl_set_is_equal(a_prefix.get(), b_prefix.get()))
      compatible_level = i;
    else
      break;
  }

  return compatible_level;
}

isl_set *isl_remove_axis_by_name(isl_set *set, const char *axis_name) {
  std::string tuple_name = isl_set_get_tuple_name(set);
  int offset = isl_set_find_dim_by_name(set, isl_dim_set, axis_name);
  set = isl_set_remove_dims(set, isl_dim_set, offset, 1);
  set = isl_set_set_tuple_name(set, tuple_name.c_str());
  return set;
}

isl_map *isl_remove_axis_by_name(isl_map *map,
                                 isl_dim_type dim_type,
                                 const char *axis_name) {
  int offset = isl_map_find_dim_by_name(map, dim_type, axis_name);
  std::string tuple_name = isl_map_get_tuple_name(map, dim_type);
  map = isl_map_remove_dims(map, dim_type, offset, 1);
  map = isl_map_set_tuple_name(map, dim_type, tuple_name.c_str());
  return map;
}
isl_set *isl_rename_axis(isl_set *set, int offset, const char *name) {
  return isl_set_set_dim_name(set, isl_dim_set, offset, name);
}
isl_map *isl_rename_axis(isl_map *map,
                         isl_dim_type dim_type,
                         int offset,
                         const char *name) {
  return isl_map_set_dim_name(map, dim_type, offset, name);
}

isl_set *isl_simplify(isl_set __isl_take *set) {
  set = isl_set_coalesce(set);
  set = isl_set_remove_redundancies(set);
  return set;
}

isl::union_set isl_union_set_from_sets(llvm::ArrayRef<isl::set> sets) {
  PADDLE_ENFORCE_EQ(!sets.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The input ArrayRef of isl::set is empty. "
                        "Please ensure the ArrayRef is not empty."));
  isl::union_set res = isl::manage(isl_union_set_from_set(sets.front().copy()));
  for (int i = 1; i < sets.size(); i++) {
    res = isl::manage(isl_union_set_add_set(res.release(), sets[i].copy()));
  }
  return res;
}

std::tuple<isl::val, isl::val> isl_set_get_axis_range_by_name(
    isl_set *set, std::string axis_name) {
  std::vector<std::string> from_iters;
  for (int i = 0; i < isl_set_dim(set, isl_dim_set); i++) {
    auto *name = isl_set_get_dim_name(set, isl_dim_set, i);
    if (name) {
      from_iters.push_back(name);
    } else {
      from_iters.push_back("__emp__" + std::to_string(i));
    }
  }

  isl::aff aff(isl_set_get_ctx(set),
               utils::StringFormat("{ %s[%s] -> [%s] }",
                                   isl_set_get_tuple_name(set),
                                   utils::Join(from_iters, ",").c_str(),
                                   axis_name.c_str()));

  isl::val max_val = isl::manage(isl_set_max_val(set, aff.get()));
  isl::val min_val = isl::manage(isl_set_min_val(set, aff.get()));

  return std::make_tuple(min_val, max_val);
}

std::tuple<isl::val, isl::val> isl_set_get_axis_range(isl_set *set, int pos) {
  PADDLE_ENFORCE_EQ(
      isl_set_dim_is_bounded(set, isl_dim_set, pos),
      true,
      ::common::errors::InvalidArgument(
          "The dimension at position %d of the isl_set is not bounded. "
          "An unbounded dimension cannot get range. Please check the set: %s",
          pos,
          isl_set_to_str(set)));

  std::vector<std::string> from_iters;
  std::string target_axis_name;
  for (int i = 0; i < isl_set_dim(set, isl_dim_set); i++) {
    auto *name = isl_set_get_dim_name(set, isl_dim_set, i);
    if (name) {
      from_iters.push_back(name);
    } else {
      from_iters.push_back("__emp__" + std::to_string(i));
    }
    if (pos == i) target_axis_name = from_iters.back();
  }

  isl::aff aff(isl_set_get_ctx(set),
               utils::StringFormat("{ %s[%s] -> [%s] }",
                                   isl_set_get_tuple_name(set),
                                   utils::Join(from_iters, ",").c_str(),
                                   target_axis_name.c_str()));

  isl::val max_val = isl::manage(isl_set_max_val(set, aff.get()));
  isl::val min_val = isl::manage(isl_set_min_val(set, aff.get()));

  return std::make_tuple(min_val, max_val);
}

bool isl_set_axis_has_noparam_constant_bound(isl_set __isl_keep *set, int pos) {
  if (!isl_set_dim_is_bounded(set, isl_dim_set, pos)) return false;
  set = isl_simplify(isl_set_copy(set));
  set = isl_set_drop_unused_params(set);

  isl_pw_aff *min_val = isl_set_dim_min(isl_set_copy(set), pos);
  isl_pw_aff *max_val = isl_set_dim_max(isl_set_copy(set), pos);
  VLOG(3) << "set: " << isl_set_to_str(set);
  VLOG(3) << "min_val: " << isl_pw_aff_to_str(min_val);
  VLOG(3) << "max_val: " << isl_pw_aff_to_str(max_val);

  isl::set context(isl_set_get_ctx(set), "{:}");
  auto is_dim_a_constant = [&](isl_pw_aff *__isl_give val) {
    val = isl_pw_aff_drop_unused_params(val);
    val = isl_pw_aff_align_params(val, isl_space_copy(context.space().get()));

    bool is_param_involved = false;
    isl_pw_aff_foreach_piece(
        val,
        [](isl_set *__isl_give set,
           isl_aff *__isl_give aff,
           void *user) -> isl_stat {
          // Ignore the set piece, e.g. [_cp_C_0, _cp_C_1] -> { cache[0, 0] :
          // _cp_C_0 = 0 and _cp_C_1 = 0 } will get a set [_cp_C_0, _cp_C_1] ->
          // {  : _cp_C_0 = 0 and _cp_C_1 = 0 }
          if (set) {
            // ignore
          }

          PADDLE_ENFORCE_NOT_NULL(
              aff,
              ::common::errors::InvalidArgument(
                  "The isl_aff is null. "
                  "Please ensure the isl_aff is properly initialized."));
          auto &is_param_involved = *reinterpret_cast<bool *>(user);
          if (is_param_involved) return isl_stat_ok;

          // drop unused params, so the Aff [n]->{ [(0)] } will be []->{ [(0)] }
          auto *pw_aff = isl_pw_aff_from_aff(aff);
          pw_aff = isl_pw_aff_drop_unused_params(pw_aff);

          // check if some params is involved.
          isl::set params = isl::manage(isl_pw_aff_params(pw_aff));
          is_param_involved = isl_set_dim(params.get(), isl_dim_param) > 0;

          isl_set_free(set);
          return isl_stat_ok;
        },
        reinterpret_cast<void *>(&is_param_involved));

    return !is_param_involved;
  };

  return is_dim_a_constant(max_val) && is_dim_a_constant(min_val);
}

isl::map isl_set_dim_name_if_null(
    isl_map *map, std::function<std::string(isl_dim_type, int)> namer) {
  int in_dims = isl_map_dim(map, isl_dim_in);
  int out_dims = isl_map_dim(map, isl_dim_out);
  auto set_name = [&](isl_dim_type dim_type) {
    for (int i = 0; i < isl_map_dim(map, dim_type); i++) {
      if (!isl_map_get_dim_name(map, dim_type, i)) {
        map =
            isl_map_set_dim_name(map, dim_type, i, namer(dim_type, i).c_str());
      }
    }
  };

  set_name(isl_dim_in);
  set_name(isl_dim_out);

  return isl::manage(map);
}

isl::set isl_set_dim_name_if_null(
    isl_set *set, std::function<std::string(isl_dim_type, int)> namer) {
  for (int i = 0; i < isl_set_dim(set, isl_dim_set); i++) {
    if (!isl_set_get_dim_name(set, isl_dim_set, i)) {
      set = isl_set_set_dim_name(
          set, isl_dim_set, i, namer(isl_dim_set, i).c_str());
    }
  }
  return isl::manage(set);
}

isl::map RemoveAxesByInputNames(const isl::map &x,
                                const isl::set &origin_domain,
                                const std::vector<std::string> &dim_in_names) {
  std::string map_str = isl_map_to_str(x.get());
  isl::ctx this_ctx = x.ctx();
  isl::map temp_transform(this_ctx, map_str);
  auto related_output_names =
      GetRelatedOutputAxes(x, origin_domain, dim_in_names);
  if (dim_in_names.empty()) return temp_transform;
  for (auto &i : dim_in_names) {
    temp_transform = isl::manage(isl_remove_axis_by_name(
        temp_transform.release(), isl_dim_in, i.c_str()));
  }
  for (auto &i : related_output_names) {
    temp_transform = isl::manage(isl_remove_axis_by_name(
        temp_transform.release(), isl_dim_out, i.c_str()));
  }
  return temp_transform;
}

isl::map RemoveAxesByOutputNames(
    const isl::map &x,
    const isl::set &origin_domain,
    const std::vector<std::string> &dim_out_names) {
  std::string map_str = isl_map_to_str(x.get());
  isl::ctx this_ctx = x.ctx();
  isl::map temp_transform(this_ctx, map_str);
  auto related_input_names =
      GetRelatedInputAxes(x, origin_domain, dim_out_names);
  if (dim_out_names.empty()) return temp_transform;
  for (auto &i : dim_out_names) {
    temp_transform = isl::manage(isl_remove_axis_by_name(
        temp_transform.release(), isl_dim_out, i.c_str()));
  }
  for (auto &i : related_input_names) {
    temp_transform = isl::manage(isl_remove_axis_by_name(
        temp_transform.release(), isl_dim_in, i.c_str()));
  }
  return temp_transform;
}

std::vector<std::string> GetRelatedOutputAxes(
    const isl::map &x,
    const isl::set &origin_domain,
    const std::vector<std::string> &dim_in_names) {
  std::string map_str = isl_map_to_str(x.get());
  VLOG(1) << "GetRelatedOutputAxes map_str is : " << map_str;
  isl::ctx this_ctx = x.ctx();
  isl::map temp_transform(this_ctx, map_str);
  auto dim_out_names = isl_get_dim_names(temp_transform, isl_dim_out);
  std::set<std::string> dim_in_set;
  for (auto &i : dim_in_names) {
    VLOG(1) << "GetRelatedOutputAxes dim_in_names is : " << i;
    dim_in_set.insert(i);
  }
  std::set<std::string> res_set;
  for (auto &i : dim_out_names) {
    auto related_in_dim =
        GetRelatedInputAxes(temp_transform, origin_domain, {i});
    for (auto &j : related_in_dim) {
      if (dim_in_set.count(j) > 0) {
        res_set.insert(i);
      }
    }
  }
  std::vector<std::string> res;
  for (auto &i : res_set) {
    VLOG(1) << "GetRelatedOutputAxes res is : " << i;
    res.push_back(i);
  }
  return res;
}

std::vector<std::string> GetRelatedInputAxes(
    const isl::map &x,
    const isl::set &origin_domain,
    const std::vector<std::string> &dim_out_names,
    bool strict) {
  std::string map_str = isl_map_to_str(x.get());
  VLOG(1) << "GetRelatedInputAxes map_str is : " << map_str;
  isl::ctx this_ctx = x.ctx();
  isl::map temp_transform(this_ctx, map_str);
  auto dim_in_names = isl_get_dim_names(temp_transform, isl_dim_in);
  for (auto &i : dim_out_names) {
    VLOG(1) << "GetRelatedInputAxes dim_out_names is : " << i;
    temp_transform = isl::manage(isl_remove_axis_by_name(
        temp_transform.release(), isl_dim_out, i.c_str()));
  }
  std::string deleted_map = isl_map_to_str(temp_transform.get());
  std::vector<std::string> res;
  std::set<std::string> out_set;
  std::set<std::string> out_set_without_suffix;
  std::string set_str = isl_set_to_str(origin_domain.get());
  isl::ctx set_ctx = origin_domain.ctx();
  isl::set temp_set(this_ctx, set_str);
  auto transformed_domain = temp_set.apply(x);
  for (auto &i : dim_out_names) {
    out_set.insert(i);
    if (utils::EndsWith(i, "_inner") || utils::EndsWith(i, "_outer")) {
      out_set_without_suffix.insert(utils::RemoveSuffix(i));
    }
  }
  for (auto &i : dim_in_names) {
    if (utils::Count(&map_str, i) != utils::Count(&deleted_map, i)) {
      VLOG(1) << "GetRelatedInputAxes res is : " << i;
      res.push_back(i);
    } else if (out_set_without_suffix.count(i) > 0 && !strict) {
      VLOG(1) << "GetRelatedInputAxes res is : " << i;
      res.push_back(i);
    } else if (out_set.count(i) > 0) {
      auto range1 = isl_set_get_axis_range_by_name(origin_domain.get(), i);
      auto &minv1 = std::get<0>(range1);
      auto &maxv1 = std::get<1>(range1);
      auto range2 = isl_set_get_axis_range_by_name(transformed_domain.get(), i);
      auto &minv2 = std::get<0>(range2);
      auto &maxv2 = std::get<1>(range2);
      int min_iv1 = minv1.get_num_si();
      int max_iv1 = maxv1.get_num_si();
      int min_iv2 = minv2.get_num_si();
      int max_iv2 = maxv2.get_num_si();
      if (min_iv1 == max_iv1 && min_iv2 == max_iv2) {
        res.push_back(i);
      }
    }
  }
  return res;
}
}  // namespace poly
}  // namespace cinn
