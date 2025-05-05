// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/paddle/pass/ap_registry_helper.h"
#include "paddle/ap/include/registry/registry_mgr.h"

namespace cinn::dialect::ir {

namespace {

using ap::registry::Registry;
using ap::registry::RegistryMgr;
using ap::registry::RegistrySingleton;

}  // namespace

ap::adt::Result<Registry> ApRegistryHelper::SingletonRegistry() {
  ADT_RETURN_IF_ERR(RegistryMgr::Singleton()->LoadAllOnce());
  ADT_LET_CONST_REF(registry, RegistrySingleton::Singleton());
  return registry;
}

}  // namespace cinn::dialect::ir
