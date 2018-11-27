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

#include "paddle/fluid/inference/api/paddle_pass_builder.h"

#include <glog/logging.h>

namespace paddle {

void PaddlePassBuilder::AppendPass(const std::string &pass_type) {
  passes_.push_back(pass_type);
}

void PaddlePassBuilder::TurnOnDebug() {
  std::vector<std::string> passes;
  auto it = std::begin(passes_);
  while (it != std::end(passes_)) {
    if (*it != "graph_viz_pass") {
      it = passes_.insert(it + 1, "graph_viz_pass");
    } else {
      ++it;
    }
  }
}

std::string PaddlePassBuilder::DebugString() {
  std::stringstream ss;
  ss << "Passes to apply:\n";
  for (auto &pass : passes_) {
    ss << "  - " << pass << '\n';
  }
  return ss.str();
}

void PaddlePassBuilder::DeletePass(const std::string &pass_type) {
  auto it = std::begin(passes_);
  while (it != std::end(passes_)) {
    if (*it == pass_type) {
      it = passes_.erase(it);
    } else {
      ++it;
    }
  }
}

void PaddlePassBuilder::InsertPass(size_t idx, const std::string &pass_type) {
  passes_.insert(std::begin(passes_) + idx, pass_type);
}

void PaddlePassBuilder::DeletePass(size_t idx) {
  passes_.erase(std::begin(passes_) + idx);
}

void GpuPassStrategy::EnableMKLDNN() {
  LOG(ERROR) << "GPU not support MKLDNN yet";
}

void PaddlePassBuilder::AppendAnalysisPass(const std::string &pass) {
  analysis_passes_.push_back(pass);
}

}  // namespace paddle
