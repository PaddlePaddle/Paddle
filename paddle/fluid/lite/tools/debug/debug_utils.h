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

#pragma once
#include <gflags/gflags.h>
#include <algorithm>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/api/cxx_api.h"
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/utils/string.h"

DEFINE_string(model_dir, "", "Model dir path");
DEFINE_string(input_file, "", "Input datas file path");
DEFINE_string(topo_output_file, "", "Runtime topology order output file path");
DEFINE_bool(output_topo, true, "Dump runtime topology or not");
DEFINE_string(tensor_output_file, "", "Tensor output file path");
DEFINE_bool(output_vars, true, "Dump vars or not");
DEFINE_bool(output_weights, true, "Dump weight tensors or not");
DEFINE_string(
    tensor_names, "",
    "If tensor_names is not empty, then only this tensors will be dump");
DEFINE_int32(tensor_output_length, -1,
             "Output tensor data length, dims size will be used if "
             "output_tensor_length < 0");
DEFINE_int32(arm_thread_num, 1, "Arm thread nums, 1 as default");
DEFINE_string(separator, ",", "Deafult separator, use in string split");

namespace paddle {
namespace lite {
namespace tools {
namespace debug {

struct DebugConfig {
  // arguments
  std::string model_dir;
  std::string topo_output_file;
  std::string tensor_output_file;
  std::string input_file;
  std::vector<std::string> tensor_names;
  bool output_weights;
  bool output_topo;
  bool output_vars;
  int tensor_output_length;
  int arm_thread_num;

  std::unordered_map<std::string, lite::VarDesc> var_descs;
  std::vector<std::vector<std::string>> input_values;
};

template <typename T>
std::vector<T> Split2Vector(const std::string& input,
                            const std::string& separator) {
  std::vector<T> tgt;
  std::vector<std::string> inputs = Split(input, separator);
  tgt.resize(inputs.size());
  std::stringstream ss;
  for (int i = 0; i < inputs.size(); ++i) {
    ss << inputs[i] << " ";
  }
  for (int i = 0; i < inputs.size(); ++i) {
    ss >> tgt[i];
  }
  return tgt;
}

void CollectFeedVarsInfo(std::unordered_map<int, std::string>* feed_vars_info,
                         const framework::proto::ProgramDesc& desc) {
  CHECK(feed_vars_info);
  for (const auto& proto_op_desc : desc.blocks(0).ops()) {
    lite::OpDesc op_desc(proto_op_desc);
    auto op_type = op_desc.Type();
    if (op_type == "feed") {
      (*feed_vars_info)
          .emplace(op_desc.GetAttr<int>("col"), op_desc.Output("Out").front());
    }
  }
}
template <typename T>
void FillTensorData(lite::Tensor* tensor, const DebugConfig& conf, int col) {
  CHECK(tensor);
  auto dim_size = tensor->dims().production();
  auto* data = tensor->mutable_data<T>();
  if (conf.input_values.size() > 0) {
    CHECK(col < conf.input_values[0].size())
        << "Input data fields out of index. field_len: "
        << conf.input_values[0].size() << " col: " << col;
    std::vector<T> input_data(
        std::move(Split2Vector<T>(conf.input_values[0][col], " ")));
    CHECK(input_data.size() == dim_size)
        << "Input data field[" << col
        << "] mismatch TensorDim: " << input_data.size() << " vs " << dim_size;
    for (int i = 0; i < dim_size; i++) {
      data[i] = input_data[i];
    }
  } else {
    LOG(INFO) << "------------> Use all-ones input";
    for (int i = 0; i < dim_size; i++) {
      data[i] = 1;
    }
  }
}

void CheckDim(std::vector<DDim::value_type>* dim) {
  CHECK(dim);
  for (int i = 0; i < dim->size(); ++i) {
    if ((*dim)[i] < 0) (*dim)[i] = -(*dim)[i];
  }
}

void PrepareModelInputTensor(const DebugConfig& conf, lite::Scope* scope,
                             const framework::proto::ProgramDesc& desc) {
  CHECK(scope);

  std::unordered_map<int, std::string> feed_vars_info;
  CollectFeedVarsInfo(&feed_vars_info, desc);
  auto* feed_var =
      scope->FindVar("feed")->GetMutable<std::vector<lite::Tensor>>();
  feed_var->resize(feed_vars_info.size());

  for (auto& item : feed_vars_info) {
    auto& var_desc = conf.var_descs.at(item.second);
    auto val_type = var_desc.GetDataType();
    auto dim = var_desc.GetShape();
    CheckDim(&dim);
    auto* input_tensor = &feed_var->at(item.first);
    input_tensor->Resize(DDim(dim));
    switch (val_type) {
#define FILL_TENSOR_BY_TYPE_ONCE(pb_type__, type__)         \
  case framework::proto::VarType::pb_type__:                \
    FillTensorData<type__>(input_tensor, conf, item.first); \
    break

      FILL_TENSOR_BY_TYPE_ONCE(UINT8, uint8_t);
      FILL_TENSOR_BY_TYPE_ONCE(INT8, int8_t);
      FILL_TENSOR_BY_TYPE_ONCE(INT16, int16_t);
      FILL_TENSOR_BY_TYPE_ONCE(INT32, int32_t);
      FILL_TENSOR_BY_TYPE_ONCE(INT64, int64_t);
      FILL_TENSOR_BY_TYPE_ONCE(FP32, float);
      FILL_TENSOR_BY_TYPE_ONCE(FP64, double);

      default:
        LOG(FATAL) << "Unsupported data type: " << static_cast<int>(val_type);
#undef FILL_TENSOR_BY_TYPE_ONCE
    }
  }
}

void ParseInputFile(DebugConfig* conf) {
  CHECK(conf);
  if (conf->input_file.empty()) return;
  auto& inputs = conf->input_values;
  std::ifstream fd(conf->input_file);
  CHECK(fd.is_open()) << "Open input file: " << conf->input_file << " failed!";
  std::string line;
  while (std::getline(fd, line)) {
    inputs.emplace_back(std::move(Split(line, FLAGS_separator)));
  }
  LOG(INFO) << "Load data:" << inputs.size() << " items";
}

void ParseConfig(DebugConfig* conf) {
  CHECK(conf);
#define CHECK_NON_EMPTY(name__) \
  CHECK(!FLAGS_##name__.empty()) << "Option " << #name__ << " can't be empty."
  CHECK_NON_EMPTY(model_dir);
  if (FLAGS_output_topo) {
    CHECK_NON_EMPTY(topo_output_file);
  }
  if (FLAGS_output_vars || FLAGS_output_weights) {
    CHECK_NON_EMPTY(tensor_output_file);
  }
#undef CHECK_NON_EMPTY
  conf->model_dir = FLAGS_model_dir;
  conf->topo_output_file = FLAGS_topo_output_file;
  conf->tensor_output_file = FLAGS_tensor_output_file;
  conf->input_file = FLAGS_input_file;
  conf->output_weights = FLAGS_output_weights;
  conf->output_vars = FLAGS_output_vars;
  conf->output_topo = FLAGS_output_topo;
  conf->tensor_output_length = FLAGS_tensor_output_length;
  conf->arm_thread_num = FLAGS_arm_thread_num;

  if (!FLAGS_tensor_names.empty()) {
    conf->tensor_names = Split(FLAGS_tensor_names, FLAGS_separator);
  }

  ParseInputFile(conf);
}

void CollectAndDumpTopoInfo(const std::vector<Instruction>& instructions,
                            const DebugConfig& conf) {
  if (!conf.output_topo) return;
  LOG(INFO) << "----------------- dump topo file";
  std::ofstream os(conf.topo_output_file);
  CHECK(os.is_open());
  for (auto& inst : instructions) {
    auto* op_info = inst.op()->op_info();
    CHECK(op_info);
    os << op_info->Type() << "\t";
    os << "(";
#define DUMP_TOPO_INFO_ONCE(name__)                   \
  {                                                   \
    auto argnames = op_info->name__##ArgumentNames(); \
    for (int i = 0; i < argnames.size(); ++i) {       \
      os << argnames[i] << ":";                       \
      auto vars = op_info->name__(argnames[i]);       \
      for (int j = 0; j < vars.size(); ++j) {         \
        os << vars[j];                                \
        if (j != vars.size() - 1) os << "#";          \
      }                                               \
      if (i != argnames.size() - 1) os << " ";        \
    }                                                 \
  }
    DUMP_TOPO_INFO_ONCE(Input);
    os << ")\t(";
    DUMP_TOPO_INFO_ONCE(Output);
    os << ")\n";
#undef DUMP_TOPO_INFO_ONCE
  }
  os.close();
}

void CollectVarDescs(std::unordered_map<std::string, lite::VarDesc>* var_descs,
                     const framework::proto::ProgramDesc& desc) {
  CHECK(var_descs);
  CHECK(!desc.blocks().empty());
  std::unordered_set<std::string> weights;
  for (auto proto_var_desc : desc.blocks(0).vars()) {
    lite::VarDesc var_desc(proto_var_desc);
    (*var_descs).emplace(var_desc.Name(), std::move(var_desc));
  }
}

std::unordered_set<std::string> CollectUnusedVars(
    const std::vector<Instruction>& instructions) {
  std::unordered_set<std::string> unused;
  std::unordered_set<std::string> all_inputs;
  for (auto& inst : instructions) {
    for (const auto& name : inst.op()->op_info()->input_names()) {
      all_inputs.insert(name);
    }
  }

  for (auto& inst : instructions) {
    for (const auto& name : inst.op()->op_info()->output_names()) {
      if (all_inputs.count(name) == 0) unused.insert(name);
    }
  }

  return unused;
}

std::string GetTensorRepr(const lite::Tensor& tensor, int out_data_len) {
  std::stringstream ss;
  auto size = tensor.dims().production();
  if (out_data_len >= 0) {
    size = std::min(size, static_cast<DDim::value_type>(out_data_len));
  }
  for (int i = 0; i < size; i++) {
    ss << tensor.template data<float>()[i];
    if (i != size - 1) ss << " ";
  }
  return ss.str();
}

void CollectAndDumpTensorInfo(const std::vector<Instruction>& instructions,
                              const framework::proto::ProgramDesc& desc,
                              const DebugConfig& conf) {
  CHECK(instructions.size() > 0) << "No instruction found";
  const auto* scope = const_cast<lite::OpLite*>(instructions[0].op())->scope();
  std::ofstream os(conf.tensor_output_file);
  CHECK(os.is_open());

  std::unordered_set<std::string> dump_vars;
#define DUMP_TENSOR_ONCE(name__)                                  \
  LOG(INFO) << "----------------- dump tensor: " << name__;       \
  auto& tensor = scope->FindVar(name__)->Get<lite::Tensor>();     \
  os << name__ << "\t" << tensor.dims() << "\t"                   \
     << GetTensorRepr(tensor, conf.tensor_output_length) << "\n"; \
  dump_vars.insert(name__)

#define DUMP_OP_TENSOR_ONCE(name__, skip__)                              \
  for (const auto& name : inst.op()->op_info()->name__##_names()) {      \
    bool is_weight = conf.var_descs.at(name).Persistable();              \
    if (unused.count(name) != 0 || name == #skip__ ||                    \
        (!conf.output_weights && is_weight) ||                           \
        (!conf.output_vars && !is_weight) || dump_vars.count(name) != 0) \
      continue;                                                          \
    DUMP_TENSOR_ONCE(name);                                              \
  }

  if (conf.tensor_names.size() == 0) {
    std::unordered_set<std::string> unused(
        std::move(CollectUnusedVars(instructions)));

    for (auto& inst : instructions) {
      DUMP_OP_TENSOR_ONCE(input, feed);
      DUMP_OP_TENSOR_ONCE(output, fetch);
    }
  } else {
    for (const auto& name : conf.tensor_names) {
      DUMP_TENSOR_ONCE(name);
    }
  }
#undef DUMP_OP_TENSOR_ONCE
#undef DUMP_TENSOR_ONCE
  os.close();
}

}  // namespace debug
}  // namespace tools
}  // namespace lite
}  // namespace paddle
