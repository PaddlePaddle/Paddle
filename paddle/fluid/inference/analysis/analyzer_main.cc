// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

/*
 * This file implements analysizer -- an executation help to analyze and
 * optimize trained model.
 */
#include "paddle/fluid/inference/analysis/analyzer.h"
#include <gflags/gflags.h>
#include <glog/logging.h>

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  using paddle::inference::analysis::Analyzer;
  using paddle::inference::analysis::Argument;

  Argument argument;
  Analyzer analyzer;
  analyzer.Run(&argument);

  return 0;
}
