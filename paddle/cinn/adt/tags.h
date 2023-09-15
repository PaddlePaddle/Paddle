// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/adt/adt.h"

namespace cinn::adt {

DEFINE_ADT_TAG(tIn);
DEFINE_ADT_TAG(tOut);

DEFINE_ADT_TAG(tVar);
DEFINE_ADT_TAG(tSSAShadow);
DEFINE_ADT_TAG(tAnchor);
DEFINE_ADT_TAG(tStride);

DEFINE_ADT_TAG(tInMsgBox);
DEFINE_ADT_TAG(tOutMsgBox);

DEFINE_ADT_TAG(tBreak);

DEFINE_ADT_TAG(tHasNoConflictValue);

}  // namespace cinn::adt
