// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ir/core/builtin_attribute.h"

namespace ir {
std::string StrAttribute::data() const { return storage()->GetAsKey(); }

uint32_t StrAttribute::size() const { return storage()->GetAsKey().size(); }

bool BoolAttribute::data() const { return storage()->GetAsKey(); }

float FloatAttribute::data() const { return storage()->GetAsKey(); }

double DoubleAttribute::data() const { return storage()->GetAsKey(); }

int32_t Int32_tAttribute::data() const { return storage()->GetAsKey(); }

int64_t Int64_tAttribute::data() const { return storage()->GetAsKey(); }

std::vector<Attribute> ArrayAttribute::data() const {
  return storage()->GetAsKey();
}

void* PointerAttribute::data() const { return storage()->GetAsKey(); }

}  // namespace ir
