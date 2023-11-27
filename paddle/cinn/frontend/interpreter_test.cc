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

#include "paddle/cinn/frontend/interpreter.h"

#include <gtest/gtest.h>

#include "paddle/cinn/runtime/use_extern_funcs.h"

PD_DEFINE_string(model_dir, "", "");

namespace cinn::frontend {

TEST(Interpreter, basic) {
  Interpreter executor({"A"}, {{1, 30}});
  executor.LoadPaddleModel(FLAGS_model_dir, common::DefaultTarget(), true);
  executor.Run();
  // fc_0.tmp_2 is eliminated by OpFusion, so here
  // change to get tenor of the out variable
  executor.GetTensor("save_infer_model/scale_0.tmp_0");
}

}  // namespace cinn::frontend
