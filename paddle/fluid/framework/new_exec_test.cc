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

#include <iostream>
#include <string>

#include <chrono>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device_context.h"

#include "paddle/fluid/pybind/pybind.h"

#include "gperftools/profiler.h"
#include "paddle/fluid/framework/new_exec.h"
#include "paddle/fluid/platform/init.h"

int main() {
  paddle::framework::InitDevices();
  paddle::framework::VariableScope global_scope;
  auto place = paddle::platform::CUDAPlace(0);
  auto test_prog = paddle::framework::load_from_file("lm_startup_program");
  {
    paddle::framework::build_variable_scope(test_prog, &global_scope);

    std::vector<paddle::framework::OpFuncNode> vec_func_list;
    std::vector<paddle::framework::OperatorBase*> op_list;
    paddle::framework::build_op_func_list(test_prog, op_list, vec_func_list,
                                          &global_scope, place);

    // paddle::framework::exec_op_func_list( vec_func_list, op_list,
    // global_scope, place );
  }

  cerr << "run main" << endl;
  auto main_prog = paddle::framework::load_from_file("lm_main_program");

  paddle::framework::build_variable_scope(main_prog, &global_scope);

  std::vector<paddle::framework::OpFuncNode> vec_main_func_list;
  std::vector<paddle::framework::OperatorBase*> op_main_list;
  paddle::framework::build_op_func_list(
      main_prog, op_main_list, vec_main_func_list, &global_scope, place);
  paddle::framework::Scope scope;
  paddle::framework::InterpreterCore interp_core(place, main_prog, test_prog,
                                                 &scope);
  auto start = std::chrono::steady_clock::now();
  ProfilerStart("new_executor.prof");
  for (size_t i = 0; i < 2320; ++i) {
    if (i % 200 == 0) {
      cerr << i << endl;
    }
    // paddle::framework::exec_op_func_list( vec_main_func_list, op_main_list,
    // global_scope, place );
    std::vector<paddle::framework::Tensor> vec_out;
    interp_core.run({}, {}, {}, vec_out);
  }
  ProfilerStop();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff = end - start;

  cerr << "time cost " << diff.count() << endl;

  return 1;
}
