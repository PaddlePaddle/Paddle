/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace paddle {
namespace distributed {

struct CommContext {
  CommContext() = default;

  CommContext(const std::string &name,
              const std::vector<std::string> &names,
              const std::vector<std::string> &emap,
              const std::vector<int64_t> &sections,
              const std::vector<std::string> &origin_names,
              int trainer_id,
              bool merge_add = true,
              bool is_sparse = true,
              bool is_distributed = false,
              int table_id = -1,
              bool is_tensor_table = false,
              bool is_datanorm_table = false,
              int64_t program_id = -1,
              const std::vector<int32_t> &remote_sparse_ids = {})
      : var_name(name),
        splited_varnames(names),
        epmap(emap),
        height_sections(sections),
        origin_varnames(origin_names),
        trainer_id(trainer_id),
        merge_add(merge_add),
        is_sparse(is_sparse),
        is_distributed(is_distributed),
        table_id(table_id),
        program_id(program_id),
        is_tensor_table(is_tensor_table),
        is_datanorm_table(is_datanorm_table),
        remote_sparse_ids(remote_sparse_ids) {}

  CommContext(const CommContext &ctx) {
    var_name = ctx.var_name;
    splited_varnames = ctx.splited_varnames;
    epmap = ctx.epmap;
    height_sections = ctx.height_sections;
    trainer_id = ctx.trainer_id;
    merge_add = ctx.merge_add;
    is_sparse = ctx.is_sparse;
    origin_varnames = ctx.origin_varnames;
    is_distributed = ctx.is_distributed;
    table_id = ctx.table_id;
    program_id = ctx.program_id;
    is_tensor_table = ctx.is_tensor_table;
    is_datanorm_table = ctx.is_datanorm_table;
    remote_sparse_ids = ctx.remote_sparse_ids;
  }

  std::string print() const {
    std::stringstream ss;

    ss << "varname: " << var_name << " trainer_id: " << trainer_id << " ";
    ss << " table_id: " << table_id;

    std::for_each(
        remote_sparse_ids.begin(), remote_sparse_ids.end(), [&](const int &i) {
          ss << "remote_sparse_id: " << i << " ";
        });

    for (size_t i = 0; i < splited_varnames.size(); i++) {
      ss << "slice varname: " << splited_varnames[i] << " ep: " << epmap[i]
         << " section: " << height_sections[i] << " ";
    }

    ss << "origin varnames: ";
    for (size_t i = 0; i < origin_varnames.size(); i++) {
      ss << origin_varnames[i] << " ";
    }

    ss << " aggregation->add: " << merge_add;
    ss << " is_sparse: " << is_sparse;
    ss << " is_distributed: " << is_distributed << "\n";
    ss << " table_id: " << table_id << "\n";
    ss << " program_id: " << program_id << "\n";
    ss << " is_tensor_table: " << is_tensor_table << "\n";
    ss << " is_datanorm_table: " << is_datanorm_table << "\n";

    return ss.str();
  }

  std::string var_name;
  std::vector<std::string> splited_varnames;
  std::vector<std::string> epmap;
  std::vector<int64_t> height_sections;
  std::vector<std::string> origin_varnames;
  int trainer_id;
  bool merge_add;
  bool is_sparse;
  bool is_distributed;
  int table_id;
  int64_t program_id;
  bool is_tensor_table;
  bool is_datanorm_table;
  std::vector<int32_t> remote_sparse_ids;
};

}  // namespace distributed
}  // namespace paddle
