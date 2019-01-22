/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <mkldnn.hpp>
#include <fstream>
#include "paddle/fluid/platform/port.h"
#include <omp.h>

DEFINE_string(input, "", "input data path");
DEFINE_string(weights, "", "weights data path");
DEFINE_string(bias, "", "bias data path");
DEFINE_string(output, "", "output data path");
DEFINE_string(inference_model_dir, "", "placeholder");

static void* LoadData(const char* filename, size_t size) {
  std::ifstream input_stream(filename, std::ios::binary);
  if(input_stream.fail()) {
    std::cerr << filename << " doesn't exist" << std::endl;
    ADD_FAILURE();
  }

  float* buffer = (float*)malloc(size * sizeof(float));
  float data_point;
  for(size_t i = 0; i < size; i++) {
    input_stream.read((char*)&data_point, sizeof(float));
    buffer[i] = data_point;
  }
  return (void*)buffer;
}

TEST(conv_accuracy_test, test) {

  omp_set_num_threads(1);

  auto strides = {1, 1};
  auto paddings = {0, 0};

  std::vector<int> src_dims = {1, 1024, 20, 20};
  std::vector<int> w_dims = {256, 1024, 1, 1};
  std::vector<int> bias_dims = {256};
  std::vector<int> dst_dims = {1, 256, 20, 20};

  auto input_data = LoadData(FLAGS_input.c_str(), 409600);
  auto w_data = LoadData(FLAGS_weights.c_str(), 262144);
  auto bias_data = LoadData(FLAGS_bias.c_str(), 256);
  auto correct_data = LoadData(FLAGS_output.c_str(), 102400);
  auto output_data = (float*)calloc(102400, sizeof(float));

  mkldnn::engine engine(mkldnn::engine::kind::cpu, 0);
  using mkldnn::memory;

  memory::desc src_desc = {src_dims, memory::data_type::f32, memory::format::nChw16c};
  memory::desc w_desc = {w_dims, memory::data_type::f32, memory::format::OIhw16i16o};
  memory::desc bias_desc = {bias_dims, memory::data_type::f32, memory::format::x};
  memory::desc out_desc = {dst_dims, memory::data_type::f32, memory::format::nChw16c};

  auto src_memory = mkldnn::memory({src_desc, engine}, input_data);
  auto w_memory = mkldnn::memory({w_desc, engine}, w_data);
  auto bias_memory = mkldnn::memory({bias_desc, engine}, bias_data);
  auto dst_memory = mkldnn::memory({out_desc, engine}, (void*)output_data);

  auto conv_desc =
          mkldnn::convolution_forward::desc(mkldnn::prop_kind::forward_inference, mkldnn::convolution_direct,
           src_desc, w_desc, bias_desc, out_desc, strides, paddings,
           paddings, mkldnn::padding_kind::zero);
  mkldnn::primitive_attr conv_attr;
  mkldnn::post_ops post_operations;
  constexpr float scale = 1.0f;
  constexpr float negative_slope = 0.0f;
  constexpr float placeholder = 0.0f;
  post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_relu,
                                 negative_slope, placeholder);
  conv_attr.set_post_ops(post_operations);

  auto conv_prim_desc = mkldnn::convolution_forward::primitive_desc(conv_desc, conv_attr, engine);
  auto conv_p = mkldnn::convolution_forward(conv_prim_desc, src_memory, w_memory, bias_memory, dst_memory);

  using mkldnn::stream;
  std::vector<mkldnn::primitive> pipeline;
  pipeline.push_back(conv_p);
  stream(stream::kind::eager).submit(pipeline).wait();

  float* correct_data_f = (float*) correct_data;
  for(int i = 0; i < 102400; i++) {
    ASSERT_NEAR(output_data[i], correct_data_f[i], 0.0000001);
  }

  free(input_data);
  free(w_data);
  free(bias_data);
  free(correct_data);
  free(output_data);
}

