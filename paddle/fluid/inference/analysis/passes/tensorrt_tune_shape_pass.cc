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

#include "paddle/fluid/inference/analysis/passes/tensorrt_tune_shape_pass.h"

#include <string>
#include <utility>

#include "glog/logging.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/inference/api/paddle_api.h"
#include "paddle/fluid/inference/utils/io_utils.h"
#include "paddle/fluid/inference/io.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
class Graph;
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace analysis {

using framework::ir::Graph;
using framework::ir::Node;
using framework::ir::TopologyVarientSort;
using framework::NaiveExecutor;
using framework::Variable;
typedef std::vector<std::map<std::string,std::tuple<std::vector<int32_t>,void*,int>>>  tune_input_data;

std::string TensorrtTuneShapePass::repr() const { return "tensorrt tune shape pass"; }

void TensorrtTuneShapePass::PrepareScope(Argument *argument, framework::Scope *scope) {
  platform::Place place;
  place = platform::CPUPlace();

  if (argument->model_dir_valid()) {
    auto program =
        LoadModel(argument->model_dir(), scope, place);
  } else if (argument->model_program_path_valid() &&
             argument->model_params_path_valid()) {
    auto program = LoadModel(
        argument->model_program_path(), argument->model_params_path(),
        scope, place,
        argument->model_from_memory_valid() && argument->model_from_memory());
  }
}

std::unique_ptr<framework::ProgramDesc> TensorrtTuneShapePass::LoadModel(
    const std::string &path, framework::Scope *scope,
    const platform::Place &place) {
  framework::Executor exe(place);
  return Load(&exe, scope, path);
}

std::unique_ptr<framework::ProgramDesc> TensorrtTuneShapePass::LoadModel(
    const std::string &program_path, const std::string &params_path,
    framework::Scope *scope, const platform::Place &place,
    bool model_from_memory) {
  framework::Executor exe(place);
  if (!model_from_memory) {
    return Load(&exe, scope, program_path, params_path);
  } else {
    return LoadFromMemory(&exe, scope, program_path, params_path);
  }
}

void TensorrtTuneShapePass::StatisticShapeRangeInfo(std::map<std::string, std::vector<std::vector<int32_t>>> shape_info) {
  std::map<std::string, std::vector<int32_t>> min_shapes;
  std::map<std::string, std::vector<int32_t>> max_shapes;
  std::map<std::string, std::vector<int32_t>> opt_shapes;
  for (auto it : shape_info) {
    auto name = it.first;
    auto shapes = it.second;

    std::vector<int32_t> min_shape(shapes[0].begin(), shapes[0].end());
    std::vector<int32_t> max_shape(shapes[0].begin(), shapes[0].end());
    std::vector<int32_t> opt_shape(shapes[0].begin(), shapes[0].end());

    auto ShapeMaxFreq = [](const std::map<int32_t, int32_t> &m) -> int32_t {
      std::vector<std::pair<int32_t, int32_t>> counter;
      for (auto &it : m) counter.push_back(it);
      std::sort(
          counter.begin(), counter.end(),
          [](std::pair<int32_t, int32_t> &a, std::pair<int32_t, int32_t> &b) {
            return a.second > b.second;
          });
      return counter[0].first;
    };

    for (size_t d = 0; d < shapes[0].size(); ++d) {
      std::map<int32_t, int32_t> counter;
      for (size_t i = 0; i < shapes.size(); ++i) {
        counter[shapes[i][d]] += 1;
        if (shapes[i][d] < min_shape[d]) min_shape[d] = shapes[i][d];
        if (shapes[i][d] > max_shape[d]) max_shape[d] = shapes[i][d];
      }
      opt_shape[d] = ShapeMaxFreq(counter);
    }

    min_shapes[name] = min_shape;
    max_shapes[name] = max_shape;
    opt_shapes[name] = opt_shape;
  }

  inference::SerializeShapeRangeInfo("jzz_test.pbtxt",
                                     min_shapes, max_shapes, opt_shapes);
}

void TensorrtTuneShapePass::RunImpl(Argument* argument) {
  // executor
  std::unique_ptr<NaiveExecutor> executor;
  executor.reset(new paddle::framework::NaiveExecutor(paddle::platform::CUDAPlace(argument->gpu_device_id())));

  // program after ir_graph_build
  auto program = argument->main_program_ptr();
  
  // scope has persistable variables
  //auto main_scope = argument->scope_ptr();
  framework::Scope *main_scope = new framework::Scope();
  PrepareScope(argument, main_scope);
  auto& scope = main_scope->NewScope();

  // create  vars
  executor->CreateVariables(*program, 0, false, &scope);
  
  // executor prepare
  executor->Prepare(&scope, *program, 0, false);
  
  // CreateFeedFetchVar
  auto *var = scope.Var("feed");
  var->GetMutable<framework::FeedList>();
  var = scope.Var("fetch");
  var->GetMutable<framework::FetchList>();

  // set input tensor
  auto input_data = argument->tensorrt_tune_input();

  for (auto input_map:input_data){
    for(auto iter=input_map.begin(); iter!=input_map.end(); ++iter){
      auto *var_name = scope.FindVar(iter->first);
      auto *tensor = var_name->GetMutable<paddle::framework::LoDTensor>();
      auto *lod_tensor = static_cast<paddle::framework::LoDTensor *>(tensor);
      lod_tensor->Resize(phi::make_ddim(std::get<0>(iter->second)));
      size_t ele_size = tensor->numel() * sizeof(float);
      paddle::platform::DeviceContextPool &pool = paddle::platform::DeviceContextPool::Instance();
      paddle::platform::CUDAPlace gpu_place(argument->gpu_device_id());
      auto *t_data = tensor->mutable_data<float>(gpu_place);
      auto *dev_ctx = static_cast<const paddle::platform::CUDADeviceContext *>(pool.Get(gpu_place));
      paddle::memory::Copy(gpu_place, static_cast<void *>(t_data), paddle::platform::CPUPlace(), static_cast<float*>(std::get<1>(iter->second)), ele_size, dev_ctx->stream());
    }
    executor->Run();
  }

  // auto *var_name = scope.FindVar("inputs");
  // auto *tensor = var_name->GetMutable<paddle::framework::LoDTensor>();
  // auto *lod_tensor = static_cast<paddle::framework::LoDTensor *>(tensor);
  // lod_tensor->Resize(phi::make_ddim({1, 3, 224, 224}));

  // size_t ele_size = tensor->numel() * sizeof(float);
  // std::vector<float> data(1 * 3 * 224 * 224, 1.0);

  // paddle::platform::DeviceContextPool &pool = paddle::platform::DeviceContextPool::Instance();
  // paddle::platform::CUDAPlace gpu_place(argument->gpu_device_id());
  // auto *t_data = tensor->mutable_data<float>(gpu_place);
  // auto *dev_ctx = static_cast<const paddle::platform::CUDADeviceContext *>(pool.Get(gpu_place));
  // paddle::memory::Copy(gpu_place, static_cast<void *>(t_data), paddle::platform::CPUPlace(), data.data(), ele_size, dev_ctx->stream());

  // // executor run
  // executor->Run();

  // collect_shape_range_info
  std::vector<std::string> var_names = scope.LocalVarNames();
  std::map<std::string, std::vector<std::vector<int32_t>>> shape_info;
  for (const auto &name : var_names) {
    auto *tmp_var = scope.GetVar(name);
    if (!tmp_var->IsType<framework::LoDTensor>()) {
      continue;
    }
    framework::DDim dim = tmp_var->Get<framework::LoDTensor>().dims();
    std::vector<int32_t> shape(dim.size());
    for (size_t i = 0; i < shape.size(); ++i){
      shape[i] = dim[i];
    }
    shape_info[name].emplace_back(shape);
  }

  StatisticShapeRangeInfo(shape_info);

  std::cout << "Run Tune Pass Done." << std::endl;

  return;
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle