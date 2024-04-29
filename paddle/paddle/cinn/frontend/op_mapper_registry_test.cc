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

#include "paddle/cinn/frontend/op_mapper_registry.h"

#include <gtest/gtest.h>

#include <typeinfo>

#include "paddle/cinn/frontend/op_mappers/use_op_mappers.h"
#include "paddle/cinn/utils/registry.h"

namespace cinn {
namespace frontend {

TEST(OpMapperRegistryTest, list_all_opmappers) {
  auto all_opmappers_names = OpMapperRegistry::Global()->ListAllNames();
  LOG(INFO) << "Total has " << all_opmappers_names.size()
            << " registered OpMappers:\n"
            << cinn::utils::Join(all_opmappers_names, ", ");
  ASSERT_FALSE(all_opmappers_names.empty());
}

TEST(OpMapperRegistryTest, basic) {
  auto kernel = OpMapperRegistry::Global()->Find("sigmoid");
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(typeid(*kernel), typeid(OpMapper));
  ASSERT_EQ(kernel->name, "sigmoid");
}

}  // namespace frontend
}  // namespace cinn
