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

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"

namespace paddle {
namespace operators {

inline bool NeedSend(const framework::Scope &scope,
                     const std::string &varname) {
  // dummy variable is only used in parallel executor to represent
  // some dependency relationship, we don't need to send/recv it.
  // TODO(paddle-dev): Why would parallel executor logic leaked into here?
  if (varname.find(framework::ir::Node::kControlDepVarName) !=
      std::string::npos)
    return false;
  auto *var = scope.FindVar(varname);
  PADDLE_ENFORCE_NOT_NULL(var, "Can not find variable '%s' in the send side.",
                          varname);
  if (var->IsType<framework::LoDTensor>()) {
    return var->Get<framework::LoDTensor>().IsInitialized();
  } else if (var->IsType<framework::SelectedRows>()) {
    return var->Get<framework::SelectedRows>().rows().size() > 0UL;
  } else {
    PADDLE_THROW(
        "Variable type in send side should be in "
        "[LodTensor, SelectedRows]");
  }
  return false;
}

inline std::string GetTensorDetails(const framework::Scope &scope,
                                    const std::string &varname) {
  auto *var = scope.FindVar(varname);

  std::stringstream ss;
  ss << "------------  " << varname << "  ---------------\n";

  if (var.IsType<framework::LoDTensor>()) {
    auto &var_t = var.Get<framework::LoDTensor>();
    const auto *data = var_t.data<float>();

    for (int i = 0; i < var_t.numel(); i++) {
      ss << data[i] << " ";
    }
  } else {
    auto &var_t = var.Get<framework::SelectedRows>();
    auto &rows = var_t.rows();

    std::vector<int64_t> rs(rows.begin(), rows.end());
    std::sort(rs.begin(), rs.end());

    auto &values = var_t.value();

    ss << "ROWS: \n";
    for (auto &id : rows) {
      ss << id << " ";
    }

    ss << "\n ROWS SORT:\n";
    for (auto &id : rs) {
      ss << id << " ";
    }

    ss << "\n ROWS: " << rs.size() << " VALUES: " << values.numel() << "\n";

    ss << "\nVALUES: \n";

    const auto *data = values.data<float>();
    const auto dim = values.numel() / rs.size();

    std::vector<int64_t> print_r{570, 342789, 499868, 999497};

    for (int64_t i = 0; i < rows.size(); i++) {
      if (std::find(print_r.begin(), print_r.end(), rows[i]) == print_r.end()) {
        continue;
      }

      ss << "row: " << rows[i] << " val: ";
      for (int x = 0; x < dim; x++) {
        ss << data[i * dim + x] << " ";
      }
      ss << "\n";
    }
  }

  ss << "\n------------------------------------------------\n";

  return ss.str();
}

inline std::string GetTensorDetails(const framework::Scope &scope,
                                    const std::string &var_name) {
  if (var_name != "SparseFeatFactors@GRAD" && var_name != "fc_3.w_0@GRAD") {
    return "";
  }

  auto *var = scope.FindVar(var_name);
  return GetTensorDetails(*var, var_name);
}

inline std::vector<int64_t> ToAbsoluteSection(
    const std::vector<int64_t> &height_sections) {
  std::vector<int64_t> abs_sections;
  abs_sections.resize(height_sections.size());
  abs_sections[0] = 0;
  for (size_t i = 1; i < height_sections.size(); ++i) {
    abs_sections[i] = height_sections[i - 1] + abs_sections[i - 1];
  }
  return abs_sections;
}

inline size_t GetSectionIndex(int64_t id,
                              const std::vector<int64_t> &abs_sections) {
  for (size_t i = 1; i < abs_sections.size(); ++i) {
    if (id < abs_sections[i]) {
      return i - 1;
    }
  }
  return abs_sections.size() - 1;
}

}  // namespace operators
}  // namespace paddle
