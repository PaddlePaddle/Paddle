/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <iostream>
#include <sstream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/phi/core/distributed/type_defs.h"
#include "paddle/phi/infermeta/spmd_rules/replicated.h"
#include "paddle/phi/infermeta/spmd_rules/rules.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

using phi::distributed::ProcessMesh;
using phi::distributed::TensorDistAttr;

const std::vector<int64_t>& get_dims_mapping(
    const phi::distributed::ArgDistAttr& dist_attr);

bool is_partial(const phi::distributed::ArgDistAttr& dist_attr);

const std::set<int64_t> get_partial_dims(
    const phi::distributed::ArgDistAttr& dist_attr);
void check_dim_mapping(const phi::distributed::ArgDistAttr& dist_attr,
                       const std::vector<int64_t>& dim_mapping,
                       const std::string& line = "");

void check_partial_dims(const phi::distributed::ArgDistAttr& dist_attr,
                        const std::set<int64_t>& dims,
                        const std::string& line = "");

void clean_partial_status(phi::distributed::ArgDistAttr* dist_attr);

void clean_partial_dims(phi::distributed::ArgDistAttr* dist_attr,
                        std::vector<int64_t> dims);

void set_partial_status(phi::distributed::ArgDistAttr* dist_attr,
                        std::vector<int64_t> dims);

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
