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

#include "paddle/fluid/inference/api/onnxruntime_predictor.h"

#include <glog/logging.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid//platform/device/gpu/gpu_types.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/framework/version.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/passes/memory_optimize_pass.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/utils/io_utils.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/pten/api/ext/op_meta_info.h"

namespace paddle {

framework::proto::VarType::Type ConvertONNXType(
    ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return framework::proto::VarType::FP32;
    // case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    //   return DataType::FP16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return framework::proto::VarType::INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return framework::proto::VarType::INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return framework::proto::VarType::INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return framework::proto::VarType::UINT8;
    default:
      LOG(ERROR) << "unsupported ONNX Tensor Type: " << static_cast<int>(type);
      return framework::proto::VarType::FP32;
  }
}

bool ConvertToONNX(const AnalysisConfig &config) { return true; }

/*
bool ConvertToONNX(const AnalysisConfig& config) {
  if (!config.model_dir().empty()) {
    // TODO(heliqi jiangjiajun): Paddle2ONNX not support
    // config.model_dir() + "/__model__"
    // config.model_dir() + var_name
    return false;
  } else if (config.prog_file().empty() || config.params_file().empty()) {
    LOG(ERROR) << string::Sprintf(
        "not valid model path '%s' or program path '%s' or params path '%s'.",
config.model_dir(),
        config.prog_file(), config.params_file());
    return false;
  }
  auto parser = paddle2onnx::PaddleParser();
  if(!parser.Init(config.prog_file(), config.params_file()) {
    VLOG(3) << "paddle2onnx parser init error";
    return false;
  }
  paddle2onnx::ModelExporter me;
  std::set<std::string> unsupported_ops;
  if (!me.CheckIfOpSupported(parser, &unsupported_ops)) {
    VLOG(3) << "there are some operators not supported by Paddle2ONNX ";
    return false;
  }
  if(me.GetMinOpset(parser, false) < 0) {
     VLOG(3) << "there are some operators' version not supported by Paddle2ONNX
";
    return false;
  }
  return true;
}
*/

bool ONNXRuntimePredictor::Init() {
  VLOG(3) << "ONNXRuntime Predictor::init()";

  // Now ONNXRuntime only suuport CPU
  if (config_.use_gpu()) {
    place_ = paddle::platform::CUDAPlace(config_.gpu_device_id());
  } else {
    place_ = paddle::platform::CPUPlace();
  }
  scope_.reset(new paddle::framework::Scope());
  sub_scope_ = &scope_->NewScope();

  /*
  auto parser = paddle2onnx::PaddleParser();
  if (!config_.prog_file().empty() && !config_.params_file().empty()) {
    parser.Init(config_.prog_file(), config_.params_file())
  } else {
    return false;
  }
  paddle2onnx::ModelExporter me;
  std::string onnx_proto = me.Run(parser, 9, true, false);
  */

  Ort::SessionOptions session_options;
  // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  // session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
  if (config_.profile_enabled()) {
    LOG(WARNING) << "ONNXRuntime Profiler is activated, which might affect the "
                    "performance";
    session_options.EnableProfiling("ONNX");
  } else {
    VLOG(2) << "ONNXRuntime Profiler is deactivated, and no profiling report "
               "will be "
               "generated.";
  }
  session_options.SetInterOpNumThreads(config_.cpu_math_library_num_threads());
  session_options.SetIntraOpNumThreads(config_.cpu_math_library_num_threads());
  VLOG(2) << "ONNXRuntime threads " << config_.cpu_math_library_num_threads();
  session_ = {env_, config_.prog_file().c_str(), session_options};
  // session_ = {env_, onnx_proto.data(), onnx_proto.size(), session_options};

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Allocator allocator(session_, memory_info);

  framework::proto::VarType::Type proto_type =
      framework::proto::VarType::LOD_TENSOR;
  size_t n_inputs = session_.GetInputCount();
  for (size_t i = 0; i < n_inputs; ++i) {
    auto input_name = session_.GetInputName(i, allocator);
    auto type_info = session_.GetInputTypeInfo(i);
    std::vector<int64_t> shape =
        type_info.GetTensorTypeAndShapeInfo().GetShape();
    ONNXTensorElementDataType data_type =
        type_info.GetTensorTypeAndShapeInfo().GetElementType();
    input_desc_.emplace_back(
        ONNXDesc{.name = input_name, .shape = shape, .dtype = data_type});
    auto *ptr = scope_->Var(input_name);
    // framework::proto::VarType::Type proto_type =
    // framework::proto::VarType::FEED_MINIBATCH;
    framework::InitializeVariable(ptr, proto_type);
    allocator.Free(input_name);
  }

  size_t n_outputs = session_.GetOutputCount();
  for (size_t i = 0; i < n_outputs; ++i) {
    auto output_name = session_.GetOutputName(i, allocator);
    auto type_info = session_.GetOutputTypeInfo(i);
    std::vector<int64_t> shape =
        type_info.GetTensorTypeAndShapeInfo().GetShape();
    ONNXTensorElementDataType data_type =
        type_info.GetTensorTypeAndShapeInfo().GetElementType();
    output_desc_.emplace_back(
        ONNXDesc{.name = output_name, .shape = shape, .dtype = data_type});
    auto *ptr = scope_->Var(output_name);
    // framework::proto::VarType::Type proto_type =
    // framework::proto::VarType::FETCH_LIST;
    framework::InitializeVariable(ptr, proto_type);
    allocator.Free(output_name);
  }

  return true;
}

/*
{
  Ort::AllocatorWithDefaultOptions allocator;

  //输出模型输入节点的数量
  size_t num_input_nodes = session.GetInputCount();
  size_t num_output_nodes = session.GetOutputCount();
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<const char*> output_node_names(num_output_nodes);
  std::vector<int64_t> input_node_dims;

  bool dynamic_flag = false;

  //迭代所有的输入节点
  for (int i = 0; i < num_input_nodes; i++) {
        //输出输入节点的名称
      char* input_name = session.GetInputName(i, allocator);
     std::cout << "Input " << i << ": name=" << input_name << std::endl;
      input_node_names[i] = input_name;

      // 输出输入节点的类型
      Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

      ONNXTensorElementDataType type = tensor_info.GetElementType();
      printf("Input %d : type=%d\n", i, type);

      input_node_dims = tensor_info.GetShape();
      //输入节点的打印维度
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
      //打印各个维度的大小
      for (int j = 0; j < input_node_dims.size(); j++)
      {
          printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
          if (input_node_dims[j] < 1)
          {
              dynamic_flag  = true;
          }
      }

      input_node_dims[0] = 1;
  }
  //打印输出节点信息，方法类似
  for (int i = 0; i < num_output_nodes; i++)
  {
      char* output_name = session.GetOutputName(i, allocator);
      printf("Output: %d name=%s\n", i, output_name);
      output_node_names[i] = output_name;
      Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      ONNXTensorElementDataType type = tensor_info.GetElementType();
      printf("Output %d : type=%d\n", i, type);
      auto output_node_dims = tensor_info.GetShape();
      printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
      for (int j = 0; j < input_node_dims.size(); j++)
          printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
  }

  size_t input_tensor_size = 2*3*3*3;
  std::vector<float> input_tensor_values(input_tensor_size, 1.0);
  // 为输入数据创建一个Tensor对象
  try
  {
      OrtValue * input_val = nullptr;
      std::vector<int64_t> input_shape = {2, 3, 3, 3};
      auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
OrtMemTypeDefault);
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
input_tensor_values.data(), input_tensor_size, input_shape.data(), 4);

      // // 推理得到结果
      // auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);

      // // Get pointer to output tensor float values
      // float* floatarr = output_tensors.front().GetTensorMutableData<float>();
      // for (auto j =0; j < 10; ++j)
      //   for (auto i = 0; i < 4; ++i)
      //     std::cout << "floatarr i " << j + i << "  " << floatarr[i] <<
std::endl;

      // 另一种形式
      Ort::IoBinding io_binding{session};
      io_binding.BindInput(input_node_names[0], input_tensor);
      Ort::MemoryInfo output_mem_info =
Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

      io_binding.BindOutput(output_node_names[0], output_mem_info);
      session.Run(Ort::RunOptions{ nullptr }, io_binding);
      auto output_tensors2 = io_binding.GetOutputValues();
      float* floatarr2 = output_tensors2.front().GetTensorMutableData<float>();
      for (auto j =0; j < 10; ++j)
      for (auto i = 0; i < 4; ++i)
        std::cout << "floatarr22 i " << j*4 + i << "  " << floatarr2[i+j*4] <<
std::endl;

      // output_tensors.size(),
      printf("Number of outputs = %d  %d\n",  output_tensors2.size());

  }
  catch (Ort::Exception& e)
  {
      printf(e.what());
  }
  printf("Done!\n");

}
*/

template <>
std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kONNXRuntime>(
    const AnalysisConfig &config) {
  if (config.glog_info_disabled()) {
    FLAGS_logtostderr = 1;
    FLAGS_minloglevel = 2;  // GLOG_ERROR
  }

  PADDLE_ENFORCE_EQ(
      config.is_valid(), true,
      platform::errors::InvalidArgument(
          "Note: Each config can only be used for one predictor."));

  VLOG(3) << "create ONNXRuntimePredictor";

  std::unique_ptr<PaddlePredictor> predictor(new ONNXRuntimePredictor(config));
  // Each config can only be used for one predictor.
  config.SetInValid();
  auto predictor_p = dynamic_cast<ONNXRuntimePredictor *>(predictor.get());

  if (!predictor_p->Init()) {
    return nullptr;
  }

  return predictor;
}

std::vector<std::string> ONNXRuntimePredictor::GetInputNames() {
  std::vector<std::string> input_names;
  for (auto input_desc : input_desc_) {
    input_names.push_back(input_desc.name);
  }
  return input_names;
}

std::map<std::string, std::vector<int64_t>>
ONNXRuntimePredictor::GetInputTensorShape() {
  std::map<std::string, std::vector<int64_t>> input_shapes;
  for (auto input_desc : input_desc_) {
    input_shapes[input_desc.name] = input_desc.shape;
  }
  return input_shapes;
}

std::vector<std::string> ONNXRuntimePredictor::GetOutputNames() {
  std::vector<std::string> output_names;
  for (auto output_desc : output_desc_) {
    output_names.push_back(output_desc.name);
  }
  return output_names;
}

std::unique_ptr<ZeroCopyTensor> ONNXRuntimePredictor::GetInputTensor(
    const std::string &name) {
  PADDLE_ENFORCE_NOT_NULL(scope_->FindVar(name),
                          platform::errors::PreconditionNotMet(
                              "The in variable named %s is not found in the "
                              "scope of the ONNXPredictor.",
                              name));
  std::unique_ptr<ZeroCopyTensor> res(
      new ZeroCopyTensor(static_cast<void *>(scope_.get())));
  res->input_or_output_ = true;
  res->SetName(name);
  if (platform::is_cpu_place(place_)) {
    res->SetPlace(PaddlePlace::kCPU);
  } else {
    auto gpu_place = place_;
    res->SetPlace(PaddlePlace::kGPU, gpu_place.GetDeviceId());
  }
  return res;
}

std::unique_ptr<ZeroCopyTensor> ONNXRuntimePredictor::GetOutputTensor(
    const std::string &name) {
  PADDLE_ENFORCE_NOT_NULL(scope_->FindVar(name),
                          platform::errors::PreconditionNotMet(
                              "The out variable named %s is not found in the "
                              "scope of the ONNXPredictor.",
                              name));
  std::unique_ptr<ZeroCopyTensor> res(
      new ZeroCopyTensor(static_cast<void *>(scope_.get())));
  res->input_or_output_ = false;
  res->SetName(name);
  if (platform::is_cpu_place(place_)) {
    res->SetPlace(PaddlePlace::kCPU);
  } else {
    auto gpu_place = place_;
    res->SetPlace(PaddlePlace::kGPU, gpu_place.GetDeviceId());
  }
  return res;
}

Ort::Value ONNXRuntimePredictor::GetOrtValue(const ONNXDesc &desc,
                                             const char *device_name) {
  Ort::MemoryInfo memory_info(device_name, OrtDeviceAllocator,
                              place_.GetDeviceId(), OrtMemTypeDefault);
  auto *var = scope_->FindVar(desc.name);
  auto *tensor = var->GetMutable<framework::LoDTensor>();
  size_t size = tensor->numel() * framework::SizeOfType(tensor->type());
  std::vector<int64_t> shape = framework::vectorize<int64_t>(tensor->dims());
  return Ort::Value::CreateTensor(memory_info,
                                  static_cast<void *>(tensor->data()), size,
                                  shape.data(), shape.size(), desc.dtype);
}

void ONNXRuntimePredictor::AsTensor(const Ort::Value &value,
                                    const ONNXDesc &desc) {
  auto info = value.GetTensorTypeAndShapeInfo();

  auto *var = scope_->FindVar(desc.name);
  auto *tensor = var->GetMutable<framework::LoDTensor>();
  tensor->Resize(framework::make_ddim(info.GetShape()));
  auto dtype = ConvertONNXType(info.GetElementType());
  auto *ptr = tensor->mutable_data(place_, dtype);

  if (platform::is_cpu_place(place_)) {
    std::memcpy(ptr, const_cast<void *>(value.GetTensorData<void>()),
                tensor->numel() * framework::SizeOfType(dtype));
  } else {
    auto src_place = place_;
    auto dst_place = place_;
    memory::Copy(dst_place, ptr, src_place,
                 const_cast<void *>(value.GetTensorData<void>()),
                 tensor->numel() * framework::SizeOfType(dtype));
  }
}

bool ONNXRuntimePredictor::Run(const std::vector<PaddleTensor> &inputs,
                               std::vector<PaddleTensor> *output_data,
                               int batch_size) {
  LOG(ERROR) << "Not support Run";
  return false;
}

bool ONNXRuntimePredictor::ZeroCopyRun() {
  try {
    Ort::IoBinding binding(session_);
    std::vector<Ort::Value> inputs;
    std::vector<Ort::Value> outputs;
    Ort::RunOptions options;

    inputs.reserve(input_desc_.size());
    const char *device_name = config_.use_gpu() ? "Cuda" : "Cpu";
    for (auto desc : input_desc_) {
      inputs.push_back(GetOrtValue(desc, device_name));
      binding.BindInput(desc.name.c_str(), inputs.back());
    }

    // TODO(heliqi): Optimizatio —— move to  Init()
    for (auto desc : output_desc_) {
      Ort::MemoryInfo memory_info(device_name, OrtDeviceAllocator,
                                  place_.GetDeviceId(), OrtMemTypeDefault);
      binding.BindOutput(desc.name.c_str(), memory_info);
    }

    session_.Run({}, binding);

    outputs = binding.GetOutputValues();
    for (size_t i = 0; i < output_desc_.size(); ++i) {
      AsTensor(outputs[i], output_desc_[i]);
    }
  } catch (const std::exception &e) {
    LOG(ERROR) << e.what();
    return false;
  }

  return true;
}

/*
bool ONNXRuntimePredictor::LoadProgramDesc() {
  // Initialize the inference program
  std::string filename;
  if (!config_.model_dir().empty()) {
    filename = config_.model_dir() + "/__model__";
  } else if (!config_.prog_file().empty()) {
    // All parameters are saved in a single file.
    // The file names should be consistent with that used
    // in Python API `fluid.io.save_inference_model`.
    filename = config_.prog_file();
  } else {
    if (config_.model_dir().empty() && config_.prog_file().empty()) {
      LOG(ERROR)
          << "Either model_dir or (prog_file, param_file) should be set.";
      return false;
    }
    LOG(ERROR) << string::Sprintf(
        "not valid model path '%s' or program path '%s'.", config_.model_dir(),
        config_.params_file());
    return false;
  }
  // Create ProgramDesc
  framework::proto::ProgramDesc proto;
  if (!config_.model_from_memory()) {
    std::string pb_content;
    // Read binary
    std::ifstream fin(filename, std::ios::in | std::ios::binary);
    PADDLE_ENFORCE_EQ(
        static_cast<bool>(fin.is_open()), true,
        platform::errors::NotFound(
            "Cannot open file %s, please confirm whether the file is normal.",
            filename));
    fin.seekg(0, std::ios::end);
    pb_content.resize(fin.tellg());
    fin.seekg(0, std::ios::beg);
    fin.read(&(pb_content.at(0)), pb_content.size());
    fin.close();

    proto.ParseFromString(pb_content);
  } else {
    proto.ParseFromString(config_.prog_file());
  }
  // inference_program_.reset(new framework::ProgramDesc(proto));
  return true;
}
*/

std::unique_ptr<PaddlePredictor> ONNXRuntimePredictor::Clone() {
  LOG(ERROR) << "Not support Clone()";
  return nullptr;
}

uint64_t ONNXRuntimePredictor::TryShrinkMemory() {
  // ClearIntermediateTensor();
  return paddle::memory::Release(place_);
}

void ONNXRuntimePredictor::ClearIntermediateTensor() {
  // PADDLE_ENFORCE_NOT_NULL(inference_program_.get(),
  //                         platform::errors::PreconditionNotMet(
  //                             "The inference program should be loaded
  //                             first."));
  // const auto &global_block = inference_program_->MutableBlock(0);
  // for (auto *var : global_block->AllVars()) {
  //   if (!IsPersistable(var)) {
  //     const std::string name = var->Name();
  //     auto *variable = executor_->scope()->FindVar(name);
  //     if (variable != nullptr && variable->IsType<framework::LoDTensor>() &&
  //         name != "feed" && name != "fetch") {
  //       VLOG(3) << "Clear Intermediate Tensor: " << name;
  //       auto *t = variable->GetMutable<framework::LoDTensor>();
  //       t->clear();
  //     }
  //   }
  // }
}

ONNXRuntimePredictor::~ONNXRuntimePredictor() {
  if (sub_scope_) {
    scope_->DeleteScope(sub_scope_);
  }
  memory::Release(place_);
}

}  // namespace paddle
