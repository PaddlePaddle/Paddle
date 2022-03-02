// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <string>

// #include "gperftools/profiler.h"

#include "paddle/fluid/framework/new_executor/standalone_executor.h"

USE_OP_ITSELF(fill_constant);
USE_OP(uniform_random);
USE_OP(lookup_table);
USE_OP(transpose2);
USE_OP_ITSELF(reshape2);
USE_OP(split);
USE_OP(slice);
USE_OP(concat);
USE_OP(matmul);
USE_OP_ITSELF(elementwise_add);
USE_OP(sigmoid);
USE_OP(tanh);
USE_OP(elementwise_mul);
USE_OP(softmax_with_cross_entropy);
USE_OP_ITSELF(reduce_mean);
USE_OP_ITSELF(reduce_sum);
USE_OP(reduce_sum_grad);
USE_OP(reduce_mean_grad);
USE_OP_ITSELF(reshape2_grad);
USE_OP(softmax_with_cross_entropy_grad);
USE_OP_ITSELF(elementwise_add_grad);
USE_OP(matmul_grad);
USE_OP(square);
USE_OP(transpose2_grad);
USE_OP(concat_grad);
USE_OP(elementwise_mul_grad);
USE_OP(sigmoid_grad);
USE_OP(tanh_grad);
USE_OP(sum);
USE_OP(slice_grad);
USE_OP(lookup_table_grad);
USE_OP(sqrt);
USE_OP(elementwise_max);
USE_OP(elementwise_div);
USE_OP(sgd);
USE_OP(squared_l2_norm);
USE_OP(memcpy_h2d);
USE_OP(memcpy_d2h);
DECLARE_double(eager_delete_tensor_gb);

namespace paddle {
namespace framework {

ProgramDesc load_from_file(const std::string& file_name) {
  std::ifstream fin(file_name, std::ios::in | std::ios::binary);
  fin.seekg(0, std::ios::end);
  std::string buffer(fin.tellg(), ' ');
  fin.seekg(0, std::ios::beg);
  fin.read(&buffer[0], buffer.size());
  fin.close();
  ProgramDesc program_desc(buffer);
  return program_desc;
}

TEST(StandaloneExecutor, run) {
  FLAGS_eager_delete_tensor_gb = 0.1;
  int64_t batch_size = 20;

  auto place = platform::CUDAPlace(0);
  auto test_prog = load_from_file("lm_startup_program");
  auto main_prog = load_from_file("lm_main_program");

  auto& global_block = main_prog.Block(0);

  auto& op1 = global_block.AllOps()[1];
  auto shape1 = BOOST_GET_CONST(std::vector<int64_t>, op1->GetAttr("shape"));
  shape1[0] = batch_size * 20;
  op1->SetAttr("shape", shape1);

  auto& op2 = global_block.AllOps()[2];
  auto shape2 = BOOST_GET_CONST(std::vector<int64_t>, op2->GetAttr("shape"));
  shape2[0] = batch_size;
  op2->SetAttr("shape", shape2);

  auto& op3 = global_block.AllOps()[3];
  auto shape3 = BOOST_GET_CONST(std::vector<int64_t>, op3->GetAttr("shape"));
  shape3[0] = batch_size;
  op3->SetAttr("shape", shape3);

  Scope scope;
  StandaloneExecutor exec(place, test_prog, main_prog, &scope);
  exec.Run({}, {}, {});
  auto start = std::chrono::steady_clock::now();

  // ProfilerStart("new_executor.prof");

  for (size_t i = 0; i < 2320; ++i) {
    if (i % 200 == 0) {
      std::cout << i << std::endl;
    }

    exec.Run({}, {}, {});
  }

  // ProfilerStop();

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff = end - start;

  std::cout << "time cost " << diff.count() << std::endl;
  // ASSERT_LT(diff.count(), 30);
}

}  // namespace framework
}  // namespace paddle
