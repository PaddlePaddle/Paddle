/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <ostream>
#include <string>

#include "paddle/fluid/inference/api/paddle_inference_api.h"

namespace paddle {
namespace inference {

thread_local int num_spaces = 0;

static std::string GenSpaces(int num_spaces) {
  std::ostringstream os;
  for (int i = 0; i < num_spaces; ++i) {
    os << "  ";
  }
  return os.str();
}

std::ostream &operator<<(std::ostream &os,
                         const PaddlePredictor::Config &config) {
  os << GenSpaces(num_spaces) << "PaddlePredictor::Config {\n";
  num_spaces++;
  os << GenSpaces(num_spaces) << "model_dir: " << config.model_dir << "\n";
  num_spaces--;
  os << GenSpaces(num_spaces) << "}\n";
  return os;
}

std::ostream &operator<<(std::ostream &os, const NativeConfig &config) {
  os << GenSpaces(num_spaces) << "NativeConfig {\n";
  num_spaces++;
  os << *reinterpret_cast<const PaddlePredictor::Config *>(&config);
  os << GenSpaces(num_spaces) << "use_gpu: " << config.use_gpu << "\n";
  os << GenSpaces(num_spaces) << "device: " << config.device << "\n";
  os << GenSpaces(num_spaces)
     << "fraction_of_gpu_memory: " << config.fraction_of_gpu_memory << "\n";
  os << GenSpaces(num_spaces)
     << "specify_input_name: " << config.specify_input_name << "\n";
  num_spaces--;
  os << GenSpaces(num_spaces) << "}\n";
  return os;
}

std::ostream &operator<<(std::ostream &os, const AnalysisConfig &config) {
  os << GenSpaces(num_spaces) << "AnalysisConfig {\n";
  num_spaces++;
  os << config.ToNativeConfig();
  if (!config.model_from_memory()) {
    os << GenSpaces(num_spaces) << "prog_file: " << config.prog_file() << "\n";
    os << GenSpaces(num_spaces) << "param_file: " << config.params_file()
       << "\n";
  } else {
    os << GenSpaces(num_spaces)
       << "prog_file and param_file: load from memory \n";
  }
  os << GenSpaces(num_spaces) << "enable_ir_optim: " << config.ir_optim()
     << "\n";
  os << GenSpaces(num_spaces)
     << "cpu_num_threads: " << config.cpu_math_library_num_threads() << "\n";
  os << GenSpaces(num_spaces)
     << "use_tensorrt: " << config.tensorrt_engine_enabled() << "\n";
  os << GenSpaces(num_spaces) << "use_mkldnn: " << config.mkldnn_enabled()
     << "\n";
  num_spaces--;
  os << GenSpaces(num_spaces) << "}\n";
  return os;
}

}  // namespace inference
}  // namespace paddle
