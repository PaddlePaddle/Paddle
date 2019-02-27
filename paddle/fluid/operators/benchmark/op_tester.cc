/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/benchmark/op_tester.h"
#include <fstream>
#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/fluid/pybind/pybind.h"

namespace paddle {
namespace operators {
namespace benchmark {

DEFINE_string(op_config_list, "", "Path of op config file.");
DEFINE_int32(specified_config_id, -1, "Test the specified op config.");

void OpTester::Init(const std::string &filename) {
  Init(OpTesterConfig(filename));
}

void OpTester::Init(const OpTesterConfig &config) {
  config_ = config;

  auto &op_desc_info = framework::OpInfoMap::Instance();
  // Initialize the OpDesc
  if (op_desc_info.Has(config_.op_type)) {
    type_ = config_.op_type;
    op_desc_.SetType(config_.op_type);

    CreateInputVarDesc();
    CreateOutputVarDesc();
  } else {
    LOG(FATAL) << "Op \"" << config_.op_type << "\" is not registered.";
  }

  if (config_.device_id >= 0) {
    place_ = paddle::platform::CUDAPlace(config_.device_id);
  } else {
    place_ = paddle::platform::CPUPlace();
  }

  framework::InitDevices(false);
  scope_.reset(new paddle::framework::Scope());

  op_ = framework::OpRegistry::CreateOp(op_desc_);
  CreateVariables(scope_.get());
}

void OpTester::Run() {
  if (config_.print_debug_string) {
    LOG(INFO) << DebugString();
  }

  // Warm up
  RunImpl();

  platform::Timer timer;
  if (config_.profile) {
    if (platform::is_cpu_place(place_)) {
      platform::EnableProfiler(platform::ProfilerState::kCPU);
    } else {
#ifdef PADDLE_WITH_CUDA
      platform::EnableProfiler(platform::ProfilerState::kAll);
      platform::SetDeviceId(config_.device_id);
#else
      PADDLE_THROW("'CUDAPlace' is not supported in CPU only device.");
#endif
    }

    timer.Start();
    for (int i = config_.repeat; i > 0; --i) {
      RunImpl();
    }
    timer.Pause();
    platform::DisableProfiler(platform::EventSortingKey::kDefault,
                              "op_tester_profiler");
  } else {
    timer.Start();
    for (int i = config_.repeat; i > 0; --i) {
      RunImpl();
    }
    timer.Pause();
  }
  config_.runtime = timer.ElapsedMS() / config_.repeat;
  LOG(INFO) << "=== Run " << config_.repeat
            << " times, latency: " << config_.runtime << " ms ===";
}

void OpTester::RunImpl() {
  op_->Run(*scope_, place_);
  platform::DeviceContextPool::Instance().Get(place_)->Wait();
  scope_->DropKids();
}

std::vector<std::string> OpTester::GetOpProtoInputNames() {
  std::vector<std::string> input_names;
  const framework::proto::OpProto &proto =
      framework::OpInfoMap::Instance().Get(type_).Proto();
  for (int i = 0; i != proto.inputs_size(); ++i) {
    const auto &input = proto.inputs(i);
    input_names.push_back(input.name());
  }
  return input_names;
}

std::vector<std::string> OpTester::GetOpProtoOutputNames() {
  std::vector<std::string> output_names;
  const framework::proto::OpProto &proto =
      framework::OpInfoMap::Instance().Get(type_).Proto();
  for (int i = 0; i != proto.outputs_size(); ++i) {
    const auto &output = proto.outputs(i);
    output_names.push_back(output.name());
  }
  return output_names;
}

void OpTester::CreateInputVarDesc() {
  std::vector<std::string> input_names = GetOpProtoInputNames();
  for (auto &name : input_names) {
    const OpInputConfig *input = config_.GetInput(name);
    if (input == nullptr) {
      LOG(FATAL) << "The input " << name << " of op " << config_.op_type
                 << " is not correctlly provided.";
    }

    std::string var_name = config_.op_type + "." + name;
    framework::VarDesc *var = Var(var_name);
    // Need to support more type
    var->SetType(framework::proto::VarType::LOD_TENSOR);
    var->SetPersistable(false);
    var->SetDataType(framework::proto::VarType::FP32);
    var->SetShape(input->dims);

    op_desc_.SetInput(name, {var_name});
    input_lods_[var_name] = input->lod;
  }
}

void OpTester::CreateOutputVarDesc() {
  std::vector<std::string> output_names = GetOpProtoOutputNames();
  for (auto &name : output_names) {
    std::string var_name = config_.op_type + "." + name;
    framework::VarDesc *var = Var(var_name);
    // Need to support more type
    var->SetType(framework::proto::VarType::LOD_TENSOR);
    var->SetPersistable(false);
    var->SetDataType(framework::proto::VarType::FP32);

    op_desc_.SetOutput(name, {var_name});
  }
}

framework::VarDesc *OpTester::Var(const std::string &name) {
  auto it = vars_.find(name);
  if (it != vars_.end()) {
    return it->second.get();
  }
  auto *var = new framework::VarDesc(name);
  vars_[name].reset(var);
  return var;
}

template <typename T>
void OpTester::SetupTensor(framework::LoDTensor *tensor,
                           const std::vector<int64_t> &shape, T lower,
                           T upper) {
  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);

  T *ptr = tensor->mutable_data<T>(framework::make_ddim(shape), place_);
  if (platform::is_cpu_place(place_)) {
    for (int i = 0; i < tensor->numel(); ++i) {
      ptr[i] = static_cast<T>(uniform_dist(rng) * (upper - lower) + lower);
    }
  } else {
    framework::LoDTensor cpu_tensor;
    T *cpu_ptr = cpu_tensor.mutable_data<T>(framework::make_ddim(shape),
                                            platform::CPUPlace());
    for (int i = 0; i < cpu_tensor.numel(); ++i) {
      cpu_ptr[i] = static_cast<T>(uniform_dist(rng) * (upper - lower) + lower);
    }
    TensorCopySync(cpu_tensor, place_, tensor);
  }
}

void OpTester::CreateVariables(framework::Scope *scope) {
  for (auto &item : vars_) {
    auto &var = item.second;
    if (var->Name() == framework::kEmptyVarName) {
      continue;
    }

    auto *ptr = scope->Var(var->Name());
    framework::InitializeVariable(ptr, var->GetType());
    if (var->Persistable()) {
      VLOG(3) << "Create Variable " << var->Name()
              << " global, which pointer is " << ptr;
    } else {
      VLOG(3) << "Create Variable " << var->Name()
              << " locally, which pointer is " << ptr;
    }
  }

  for (auto &item : input_lods_) {
    // Allocate memory for input tensor
    auto &var_name = item.first;
    VLOG(3) << "Allocate memory for tensor " << var_name;

    auto &var_desc = vars_[var_name];
    std::vector<int64_t> shape = var_desc->GetShape();

    auto *var = scope->Var(var_name);
    auto *tensor = var->GetMutable<framework::LoDTensor>();
    SetupTensor<float>(tensor, shape, static_cast<float>(0.0),
                       static_cast<float>(1.0));

    VLOG(3) << "Set lod for tensor " << var_name;
    std::vector<std::vector<size_t>> &lod_vec = item.second;
    framework::LoD lod;
    for (size_t i = 0; i < lod_vec.size(); ++i) {
      lod.push_back(lod_vec[i]);
    }
    tensor->set_lod(lod);
  }
}

static std::string GenSpaces(int count) {
  std::stringstream ss;
  for (int i = 0; i < count; ++i) {
    ss << "  ";
  }
  return ss.str();
}

std::string OpTester::DebugString() {
  std::stringstream ss;
  int count = 0;
  for (auto &item : vars_) {
    auto &var = item.second;
    ss << GenSpaces(count++) << "vars {\n";
    ss << GenSpaces(count) << "name: \"" << var->Name() << "\"\n";
    ss << GenSpaces(count++) << "type: {\n";
    ss << GenSpaces(count) << "type: LOD_TENSOR\n";
    ss << GenSpaces(count++) << "lod_tensor {\n";
    ss << GenSpaces(count++) << "tensor {\n";
    ss << GenSpaces(count) << "data_type: FP32\n";
    std::vector<int64_t> shape = var->GetShape();
    for (auto d : shape) {
      ss << GenSpaces(count) << "dims: " << d << "\n";
    }
    ss << GenSpaces(--count) << "}\n";
    ss << GenSpaces(--count) << "}\n";
    ss << GenSpaces(--count) << "}\n";
    ss << GenSpaces(count) << "persistable: " << var->Persistable() << "\n";
    ss << GenSpaces(--count) << "}\n";
  }
  ss << GenSpaces(count++) << "ops {\n";
  for (auto &name : op_desc_.InputNames()) {
    ss << GenSpaces(count++) << "inputs {\n";
    ss << GenSpaces(count) << "parameters: \"" << name << "\"\n";
    ss << GenSpaces(count) << "arguments: \"" << op_desc_.Input(name)[0]
       << "\"\n";
    ss << GenSpaces(--count) << "}\n";
  }
  for (auto &name : op_desc_.OutputNames()) {
    ss << GenSpaces(count++) << "outputs {\n";
    ss << GenSpaces(count) << "parameters: \"" << name << "\"\n";
    ss << GenSpaces(count) << "arguments: \"" << op_desc_.Output(name)[0]
       << "\"\n";
    ss << GenSpaces(--count) << "}\n";
  }
  ss << GenSpaces(count) << "type: " << op_desc_.Type() << "\n";
  ss << GenSpaces(--count) << "}\n";
  return ss.str();
}

TEST(op_tester, base) {
  if (!FLAGS_op_config_list.empty()) {
    std::ifstream fin(FLAGS_op_config_list, std::ios::in | std::ios::binary);
    PADDLE_ENFORCE(static_cast<bool>(fin), "Cannot open file %s",
                   FLAGS_op_config_list.c_str());
    std::vector<OpTesterConfig> op_configs;
    while (!fin.eof()) {
      OpTesterConfig config;
      bool result = config.Init(fin);
      if (result) {
        op_configs.push_back(config);
      }
    }
    if (FLAGS_specified_config_id >= 0 &&
        FLAGS_specified_config_id < static_cast<int>(op_configs.size())) {
      OpTester tester;
      tester.Init(op_configs[FLAGS_specified_config_id]);
      tester.Run();
    } else {
      for (size_t i = 0; i < op_configs.size(); ++i) {
        OpTester tester;
        tester.Init(op_configs[i]);
        tester.Run();
      }
    }
  } else {
    OpTester tester;
    OpTesterConfig config;
    config.op_type = "elementwise_add";
    config.inputs.resize(2);
    config.inputs[0].name = "X";
    config.inputs[0].dims = {64, 64};
    config.inputs[1].name = "Y";
    config.inputs[1].dims = {64, 1};
    tester.Init(config);
    tester.Run();
  }
}

}  // namespace benchmark
}  // namespace operators
}  // namespace paddle
