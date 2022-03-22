// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

// #include <glog/logging.h>
#include "paddle/fluid/framework/details/dump_tensor.h"
#include "paddle/fluid/operators/tensor_formatter.h"

namespace paddle {
namespace framework {
namespace details {

// This function only work when setting these two environment variables
// export FLAGS_dump_tensor=1
void DumpTensor2File(const framework::LoDTensor& tensor,
                     const std::string& folder_path,
                     const std::string& var_name) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);

  std::string filename = var_name;
  std::size_t pos = filename.find("/");
  while (pos != std::string::npos) {
    filename.replace(pos, 1, ".");
    pos = filename.find("/");
  }

  std::string mkdir_cmd = "mkdir -p " + folder_path;
  PADDLE_ENFORCE_EQ(
      system(mkdir_cmd.c_str()), 0,
      platform::errors::NotFound("Cannot create folder %s", folder_path));

  std::string file_path = folder_path + filename + ".txt";
  std::ofstream fout(file_path);
  PADDLE_ENFORCE_EQ(
      static_cast<bool>(fout), true,
      platform::errors::NotFound("Cannot open %s to write", filename));

  operators::TensorFormatter formatter;
  fout << formatter.Format(tensor, filename, "");

  fout.close();
  VLOG(4) << "Save tensor to text file " << file_path;
}

inline bool CheckWhiteBlackListFormEnv(const std::string& op_type) {
  bool dump_op_flag = false;

  // WhiteList: export PADDLE_DUMP_OP_LIST="unsqueeze2,cast"
  const char* dump_op_list = std::getenv("PADDLE_DUMP_OP_LIST");
  if (dump_op_list != NULL) {
    std::stringstream ss(dump_op_list);
    std::string dump_op_type;
    while (std::getline(ss, dump_op_type, ',')) {
      VLOG(4) << "Check PADDLE_DUMP_OP_LIST <" << dump_op_type << "> with OP ("
              << op_type << ")";
      if (dump_op_type == op_type) {
        dump_op_flag = true;
        break;
      }
    }
  } else {
    dump_op_flag = true;
  }

  // BlackList: export PADDLE_DUMP_OP_SKIP="unsqueeze2,cast"
  const char* dump_op_skip = std::getenv("PADDLE_DUMP_OP_SKIP");
  if (dump_op_skip != NULL) {
    std::stringstream ss(dump_op_skip);
    std::string skip_op_type;
    while (std::getline(ss, skip_op_type, ',')) {
      VLOG(4) << "Check PADDLE_DUMP_OP_SKIP <" << skip_op_type << "> with OP ("
              << op_type << ")";
      if (skip_op_type == op_type) {
        dump_op_flag = false;
        break;
      }
    }
  }
  return dump_op_flag;
}

void DumpTensor(const std::string& op_type, const std::string& var_name,
                const std::string& value_type, const framework::Variable* var,
                const platform::Place& place) {
  VLOG(4) << "Saving: " << op_type << ", VarName: " << var_name;

  if (!CheckWhiteBlackListFormEnv(op_type)) {
    VLOG(4) << "Op(" << op_type << ") CheckWhiteBlackListFormEnv Failed.";
    return;
  }

  if (!var->IsType<framework::LoDTensor>()) {
    VLOG(4) << var_name << " var_name is not LoDTensor, NOT need to check";
    return;
  }

  framework::LoDTensor tensor = var->Get<framework::LoDTensor>();
  if (tensor.memory_size() == 0) {
    VLOG(4) << var_name << " var_name need not to check, but memory_size == 0";
    return;
  }
  VLOG(4) << "begin check OP(" << op_type << "), var_name:" << var_name
          << ", place:" << tensor.place() << ", numel:" << tensor.numel();
  std::string folder_path = "dump_tensor/" + op_type + "/" + value_type + "/";

  if (platform::is_gpu_place(tensor.place())) {
    framework::LoDTensor cpu_tensor;
    cpu_tensor.Resize(tensor.dims());
    framework::TensorCopySync(tensor, platform::CPUPlace(), &cpu_tensor);
    DumpTensor2File(cpu_tensor, folder_path, var_name);
    return;
  }

  DumpTensor2File(tensor, folder_path, var_name);
}

void DumpTensor(const framework::OperatorBase& op,
                const framework::Scope& exec_scope,
                const platform::Place& place) {
  auto inputs = op.Inputs();
  auto outputs = op.Outputs();
  for (auto it = inputs.begin(); it != inputs.end(); ++it) {
    auto& input = *it;
    auto input_name = input.first;
    for (size_t i = 0; i < input.second.size(); ++i) {
      auto var_name = input.second[i];
      Variable* var = exec_scope.FindVar(var_name);
      if (var == nullptr) continue;
      DumpTensor(op.Type(), var_name, "InputVars", var, place);
    }
  }

  for (auto it = outputs.begin(); it != outputs.end(); ++it) {
    auto& output = *it;
    auto output_name = output.first;
    for (size_t i = 0; i < output.second.size(); ++i) {
      auto var_name = output.second[i];
      Variable* var = exec_scope.FindVar(var_name);
      if (var == nullptr) continue;
      DumpTensor(op.Type(), var_name, "OutputVars", var, place);
    }
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
