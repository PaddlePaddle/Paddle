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

#include <time.h>
#include <iostream>
#include "paddle/inference/inference.h"

int main(int argc, char* argv[]) {
  std::string dirname =
      "/home/work/liuyiqun/PaddlePaddle/Paddle/paddle/inference/"
      "recognize_digits_mlp.inference.model";
  std::vector<std::string> feed_var_names = {"x"};
  std::vector<std::string> fetch_var_names = {"fc_2.tmp_2"};
  paddle::InferenceEngine* desc = new paddle::InferenceEngine();
  desc->LoadInferenceModel(dirname, feed_var_names, fetch_var_names);

  paddle::framework::LoDTensor input;
  srand(time(0));
  float* input_ptr =
      input.mutable_data<float>({1, 784}, paddle::platform::CPUPlace());
  for (int i = 0; i < 784; ++i) {
    input_ptr[i] = rand() / (static_cast<float>(RAND_MAX));
  }

  std::vector<paddle::framework::LoDTensor> feeds;
  feeds.push_back(input);
  std::vector<paddle::framework::LoDTensor> fetchs;
  desc->Execute(feeds, fetchs);

  for (size_t i = 0; i < fetchs.size(); ++i) {
    auto dims_i = fetchs[i].dims();
    std::cout << "dims_i:";
    for (int j = 0; j < dims_i.size(); ++j) {
      std::cout << " " << dims_i[j];
    }
    std::cout << std::endl;
    std::cout << "result:";
    float* output_ptr = fetchs[i].data<float>();
    for (int j = 0; j < paddle::framework::product(dims_i); ++j) {
      std::cout << " " << output_ptr[j];
    }
    std::cout << std::endl;
  }
  return 0;
}
