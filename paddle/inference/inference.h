/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "paddle/framework/block_desc.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/program_desc.h"

namespace paddle {

class InferenceEngine {
public:
  InferenceEngine() : program_(nullptr), load_program_(nullptr) {}
  ~InferenceEngine() {
    delete program_;
    delete load_program_;
  }

  void LoadInferenceModel(const std::string& dirname,
                          const std::vector<std::string>& feed_var_names,
                          const std::vector<std::string>& fetch_var_names);
  void Execute(const std::vector<framework::LoDTensor>& feeds,
               std::vector<framework::LoDTensor>& fetchs);

private:
  bool IsParameter(const framework::VarDesc* var);
  void GenerateLoadProgram(const std::string& dirname);
  void PrependFeedOp();
  void AppendFetchOp();

private:
  framework::ProgramDesc* program_;
  framework::ProgramDesc* load_program_;
  std::vector<std::string> feed_var_names_;
  std::vector<std::string> fetch_var_names_;
};

}  // namespace paddle
