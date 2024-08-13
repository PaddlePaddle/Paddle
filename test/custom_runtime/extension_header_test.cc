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

#include "glog/logging.h"
#include "gtest/gtest.h"

// Note(qili93): ensure compile with one header file 'extension.h' only,
// !!! do not fix this ut by adding other header files (PR#60842) !!!
#include "paddle/phi/extension.h"

TEST(CustomDevice, extension_header) {
  VLOG(1) << "check extension header support compile only";
}
