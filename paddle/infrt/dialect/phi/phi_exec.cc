// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/host_context/paddle_mlir.h"

void print_usage() {
  std::cout << "Error inputs format, two kinds of inputs are supported:\n";
  std::cout << "    [1] ./paddle-mlir-convert $path_to_model_file "
               "$path_to_params_file\n";
  std::cout << "    [2] ./paddle-mlir-convert $path_to_model_dir(__model__ + "
               "params)\n";
}

bool parse_inputs(int argc,
                  char** argv,
                  std::string* model_file_name,
                  std::string* params_file_name) {
  switch (argc) {
    case 1: {
      print_usage();
      return false;
    }
    case 2: {
      *model_file_name = std::string(argv[1]) + std::string("/__model__");
      *params_file_name = std::string(argv[1]) + std::string("/params");
      return true;
    }
    case 3: {
      *model_file_name = argv[1];
      *params_file_name = argv[2];
      return true;
    }
    default: { return false; }
  }
}

int main(int argc, char** argv) {
  std::string model_file_name;
  std::string params_file_name;
  if (parse_inputs(argc, argv, &model_file_name, &params_file_name)) {
    MLIRModelGenImpl myGen;
    auto module_ = myGen.ImportPaddleModel(model_file_name, params_file_name);
    module_.dump();
  }
}
