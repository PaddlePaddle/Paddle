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

#include "paddle/fluid/pybind/inference_api.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/inference/api/analysis_predictor.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_pass_builder.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {
using paddle::AnalysisPredictor;
using paddle::NativeConfig;
using paddle::NativePaddlePredictor;
using paddle::PaddleBuf;
using paddle::PaddleDType;
using paddle::PaddlePassBuilder;
using paddle::PaddlePlace;
using paddle::PaddlePredictor;
using paddle::PaddleTensor;
using paddle::PassStrategy;
using paddle::ZeroCopyTensor;

namespace {
void BindPaddleDType(py::module *m);
void BindPaddleBuf(py::module *m);
void BindPaddleTensor(py::module *m);
void BindPaddlePlace(py::module *m);
void BindPaddlePredictor(py::module *m);
void BindNativeConfig(py::module *m);
void BindNativePredictor(py::module *m);
void BindAnalysisConfig(py::module *m);
void BindAnalysisPredictor(py::module *m);
void BindZeroCopyTensor(py::module *m);
void BindPaddlePassBuilder(py::module *m);

#ifdef PADDLE_WITH_MKLDNN
void BindMkldnnQuantizerConfig(py::module *m);
#endif

template <typename T>
PaddleBuf PaddleBufCreate(py::array_t<T> data) {
  PaddleBuf buf(data.size() * sizeof(T));
  std::copy_n(static_cast<const T *>(data.data()), data.size(),
              static_cast<T *>(buf.data()));
  return buf;
}

template <typename T>
void PaddleBufReset(PaddleBuf &buf, py::array_t<T> data) {  // NOLINT
  buf.Resize(data.size() * sizeof(T));
  std::copy_n(static_cast<const T *>(data.data()), data.size(),
              static_cast<T *>(buf.data()));
}

template <typename T>
constexpr PaddleDType PaddleTensorGetDType();

template <>
constexpr PaddleDType PaddleTensorGetDType<int32_t>() {
  return PaddleDType::INT32;
}

template <>
constexpr PaddleDType PaddleTensorGetDType<int64_t>() {
  return PaddleDType::INT64;
}

template <>
constexpr PaddleDType PaddleTensorGetDType<float>() {
  return PaddleDType::FLOAT32;
}

template <typename T>
PaddleTensor PaddleTensorCreate(
    py::array_t<T> data, const std::string name = "",
    const std::vector<std::vector<size_t>> &lod = {}, bool copy = true) {
  PaddleTensor tensor;

  if (copy) {
    PaddleBuf buf(data.size() * sizeof(T));
    std::copy_n(static_cast<const T *>(data.data()), data.size(),
                static_cast<T *>(buf.data()));
    tensor.data = std::move(buf);
  } else {
    tensor.data = PaddleBuf(data.mutable_data(), data.size() * sizeof(T));
  }

  tensor.dtype = PaddleTensorGetDType<T>();
  tensor.name = name;
  tensor.lod = lod;
  tensor.shape.resize(data.ndim());
  std::copy_n(data.shape(), data.ndim(), tensor.shape.begin());

  return tensor;
}

py::dtype PaddleDTypeToNumpyDType(PaddleDType dtype) {
  py::dtype dt;
  switch (dtype) {
    case PaddleDType::INT32:
      dt = py::dtype::of<int32_t>();
      break;
    case PaddleDType::INT64:
      dt = py::dtype::of<int64_t>();
      break;
    case PaddleDType::FLOAT32:
      dt = py::dtype::of<float>();
      break;
    default:
      LOG(FATAL) << "unsupported dtype";
  }

  return dt;
}

py::array PaddleTensorGetData(PaddleTensor &tensor) {  // NOLINT
  py::dtype dt = PaddleDTypeToNumpyDType(tensor.dtype);
  return py::array(std::move(dt), {tensor.shape}, tensor.data.data());
}

template <typename T>
void ZeroCopyTensorCreate(ZeroCopyTensor &tensor,  // NOLINT
                          py::array_t<T> data) {
  std::vector<int> shape;
  std::copy_n(data.shape(), data.ndim(), std::back_inserter(shape));
  tensor.Reshape(std::move(shape));
  tensor.copy_from_cpu(static_cast<const T *>(data.data()));
}

size_t PaddleGetDTypeSize(PaddleDType dt) {
  size_t size{0};
  switch (dt) {
    case PaddleDType::INT32:
      size = sizeof(int32_t);
      break;
    case PaddleDType::INT64:
      size = sizeof(int64_t);
      break;
    case PaddleDType::FLOAT32:
      size = sizeof(float);
      break;
    default:
      LOG(FATAL) << "unsupported dtype";
  }
  return size;
}

py::array ZeroCopyTensorToNumpy(ZeroCopyTensor &tensor) {  // NOLINT
  py::dtype dt = PaddleDTypeToNumpyDType(tensor.type());
  auto tensor_shape = tensor.shape();
  py::array::ShapeContainer shape(tensor_shape.begin(), tensor_shape.end());
  py::array array(dt, std::move(shape));

  switch (tensor.type()) {
    case PaddleDType::INT32:
      tensor.copy_to_cpu(static_cast<int32_t *>(array.mutable_data()));
      break;
    case PaddleDType::INT64:
      tensor.copy_to_cpu(static_cast<int64_t *>(array.mutable_data()));
      break;
    case PaddleDType::FLOAT32:
      tensor.copy_to_cpu<float>(static_cast<float *>(array.mutable_data()));
      break;
    default:
      LOG(FATAL) << "unsupported dtype";
  }
  return array;
}
}  // namespace

void BindInferenceApi(py::module *m) {
  BindPaddleDType(m);
  BindPaddleBuf(m);
  BindPaddleTensor(m);
  BindPaddlePlace(m);
  BindPaddlePredictor(m);
  BindNativeConfig(m);
  BindNativePredictor(m);
  BindAnalysisConfig(m);
  BindAnalysisPredictor(m);
  BindZeroCopyTensor(m);
  BindPaddlePassBuilder(m);
#ifdef PADDLE_WITH_MKLDNN
  BindMkldnnQuantizerConfig(m);
#endif
  m->def("create_paddle_predictor",
         &paddle::CreatePaddlePredictor<AnalysisConfig>);
  m->def("create_paddle_predictor",
         &paddle::CreatePaddlePredictor<NativeConfig>);
  m->def("paddle_dtype_size", &paddle::PaddleDtypeSize);
}

namespace {
void BindPaddleDType(py::module *m) {
  py::enum_<PaddleDType>(*m, "PaddleDType")
      .value("FLOAT32", PaddleDType::FLOAT32)
      .value("INT64", PaddleDType::INT64)
      .value("INT32", PaddleDType::INT32);
}

void BindPaddleBuf(py::module *m) {
  py::class_<PaddleBuf>(*m, "PaddleBuf")
      .def(py::init<size_t>())
      .def(py::init([](std::vector<float> &data) {
        auto buf = PaddleBuf(data.size() * sizeof(float));
        std::memcpy(buf.data(), static_cast<void *>(data.data()), buf.length());
        return buf;
      }))
      .def(py::init(&PaddleBufCreate<int32_t>))
      .def(py::init(&PaddleBufCreate<int64_t>))
      .def(py::init(&PaddleBufCreate<float>))
      .def("resize", &PaddleBuf::Resize)
      .def("reset",
           [](PaddleBuf &self, std::vector<float> &data) {
             self.Resize(data.size() * sizeof(float));
             std::memcpy(self.data(), data.data(), self.length());
           })
      .def("reset", &PaddleBufReset<int32_t>)
      .def("reset", &PaddleBufReset<int64_t>)
      .def("reset", &PaddleBufReset<float>)
      .def("empty", &PaddleBuf::empty)
      .def("tolist",
           [](PaddleBuf &self, const std::string &dtype) -> py::list {
             py::list l;
             if (dtype == "int32") {
               auto *data = static_cast<int32_t *>(self.data());
               auto size = self.length() / sizeof(int32_t);
               l = py::cast(std::vector<int32_t>(data, data + size));
             } else if (dtype == "int64") {
               auto *data = static_cast<int64_t *>(self.data());
               auto size = self.length() / sizeof(int64_t);
               l = py::cast(std::vector<int64_t>(data, data + size));
             } else if (dtype == "float32") {
               auto *data = static_cast<float *>(self.data());
               auto size = self.length() / sizeof(float);
               l = py::cast(std::vector<float>(data, data + size));
             } else {
               LOG(FATAL) << "unsupported dtype";
             }
             return l;
           })
      .def("float_data",
           [](PaddleBuf &self) -> std::vector<float> {
             auto *data = static_cast<float *>(self.data());
             return {data, data + self.length() / sizeof(*data)};
           })
      .def("int64_data",
           [](PaddleBuf &self) -> std::vector<int64_t> {
             int64_t *data = static_cast<int64_t *>(self.data());
             return {data, data + self.length() / sizeof(*data)};
           })
      .def("int32_data",
           [](PaddleBuf &self) -> std::vector<int32_t> {
             int32_t *data = static_cast<int32_t *>(self.data());
             return {data, data + self.length() / sizeof(*data)};
           })
      .def("length", &PaddleBuf::length);
}

void BindPaddleTensor(py::module *m) {
  py::class_<PaddleTensor>(*m, "PaddleTensor")
      .def(py::init<>())
      .def(py::init(&PaddleTensorCreate<int32_t>), py::arg("data"),
           py::arg("name") = "",
           py::arg("lod") = std::vector<std::vector<size_t>>(),
           py::arg("copy") = true)
      .def(py::init(&PaddleTensorCreate<int64_t>), py::arg("data"),
           py::arg("name") = "",
           py::arg("lod") = std::vector<std::vector<size_t>>(),
           py::arg("copy") = true)
      .def(py::init(&PaddleTensorCreate<float>), py::arg("data"),
           py::arg("name") = "",
           py::arg("lod") = std::vector<std::vector<size_t>>(),
           py::arg("copy") = true)
      .def("as_ndarray", &PaddleTensorGetData)
      .def_readwrite("name", &PaddleTensor::name)
      .def_readwrite("shape", &PaddleTensor::shape)
      .def_readwrite("data", &PaddleTensor::data)
      .def_readwrite("dtype", &PaddleTensor::dtype)
      .def_readwrite("lod", &PaddleTensor::lod);
}

void BindPaddlePlace(py::module *m) {
  py::enum_<PaddlePlace>(*m, "PaddlePlace")
      .value("UNK", PaddlePlace::kUNK)
      .value("CPU", PaddlePlace::kCPU)
      .value("GPU", PaddlePlace::kGPU);
}

void BindPaddlePredictor(py::module *m) {
  auto paddle_predictor = py::class_<PaddlePredictor>(*m, "PaddlePredictor");
  paddle_predictor
      .def("run",
           [](PaddlePredictor &self, const std::vector<PaddleTensor> &inputs) {
             std::vector<PaddleTensor> outputs;
             self.Run(inputs, &outputs);
             return outputs;
           })
      .def("get_input_tensor", &PaddlePredictor::GetInputTensor)
      .def("get_output_tensor", &PaddlePredictor::GetOutputTensor)
      .def("get_input_names", &PaddlePredictor::GetInputNames)
      .def("get_output_names", &PaddlePredictor::GetOutputNames)
      .def("zero_copy_run", &PaddlePredictor::ZeroCopyRun)
      .def("clone", &PaddlePredictor::Clone)
      .def("get_serialized_program", &PaddlePredictor::GetSerializedProgram);

  auto config = py::class_<PaddlePredictor::Config>(paddle_predictor, "Config");
  config.def(py::init<>())
      .def_readwrite("model_dir", &PaddlePredictor::Config::model_dir);
}

void BindNativeConfig(py::module *m) {
  py::class_<NativeConfig, PaddlePredictor::Config>(*m, "NativeConfig")
      .def(py::init<>())
      .def_readwrite("use_gpu", &NativeConfig::use_gpu)
      .def_readwrite("device", &NativeConfig::device)
      .def_readwrite("fraction_of_gpu_memory",
                     &NativeConfig::fraction_of_gpu_memory)
      .def_readwrite("prog_file", &NativeConfig::prog_file)
      .def_readwrite("param_file", &NativeConfig::param_file)
      .def_readwrite("specify_input_name", &NativeConfig::specify_input_name)
      .def("set_cpu_math_library_num_threads",
           &NativeConfig::SetCpuMathLibraryNumThreads)
      .def("cpu_math_library_num_threads",
           &NativeConfig::cpu_math_library_num_threads);
}

void BindNativePredictor(py::module *m) {
  py::class_<NativePaddlePredictor, PaddlePredictor>(*m,
                                                     "NativePaddlePredictor")
      .def(py::init<const NativeConfig &>())
      .def("init", &NativePaddlePredictor::Init)
      .def("run",
           [](NativePaddlePredictor &self,
              const std::vector<PaddleTensor> &inputs) {
             std::vector<PaddleTensor> outputs;
             self.Run(inputs, &outputs);
             return outputs;
           })
      .def("get_input_tensor", &NativePaddlePredictor::GetInputTensor)
      .def("get_output_tensor", &NativePaddlePredictor::GetOutputTensor)
      .def("zero_copy_run", &NativePaddlePredictor::ZeroCopyRun)
      .def("clone", &NativePaddlePredictor::Clone)
      .def("scope", &NativePaddlePredictor::scope,
           py::return_value_policy::reference);
}

void BindAnalysisConfig(py::module *m) {
  py::class_<AnalysisConfig> analysis_config(*m, "AnalysisConfig");

  py::enum_<AnalysisConfig::Precision>(analysis_config, "Precision")
      .value("Float32", AnalysisConfig::Precision::kFloat32)
      .value("Int8", AnalysisConfig::Precision::kInt8)
      .value("Half", AnalysisConfig::Precision::kHalf)
      .export_values();

  analysis_config.def(py::init<const AnalysisConfig &>())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, const std::string &>())
      .def("set_model", (void (AnalysisConfig::*)(const std::string &)) &
                            AnalysisConfig::SetModel)
      .def("set_model", (void (AnalysisConfig::*)(const std::string &,
                                                  const std::string &)) &
                            AnalysisConfig::SetModel)
      .def("set_prog_file", &AnalysisConfig::SetProgFile)
      .def("set_params_file", &AnalysisConfig::SetParamsFile)
      .def("model_dir", &AnalysisConfig::model_dir)
      .def("prog_file", &AnalysisConfig::prog_file)
      .def("params_file", &AnalysisConfig::params_file)
      .def("enable_use_gpu", &AnalysisConfig::EnableUseGpu,
           py::arg("memory_pool_init_size_mb"), py::arg("device_id") = 0)
      .def("disable_gpu", &AnalysisConfig::DisableGpu)
      .def("use_gpu", &AnalysisConfig::use_gpu)
      .def("gpu_device_id", &AnalysisConfig::gpu_device_id)
      .def("memory_pool_init_size_mb",
           &AnalysisConfig::memory_pool_init_size_mb)
      .def("fraction_of_gpu_memory_for_pool",
           &AnalysisConfig::fraction_of_gpu_memory_for_pool)
      .def("switch_ir_optim", &AnalysisConfig::SwitchIrOptim,
           py::arg("x") = true)
      .def("ir_optim", &AnalysisConfig::ir_optim)
      .def("enable_memory_optim", &AnalysisConfig::EnableMemoryOptim)
      .def("enable_profile", &AnalysisConfig::EnableProfile)
      .def("disable_glog_info", &AnalysisConfig::DisableGlogInfo)
      .def("glog_info_disabled", &AnalysisConfig::glog_info_disabled)
      .def("set_optim_cache_dir", &AnalysisConfig::SetOptimCacheDir)
      .def("switch_use_feed_fetch_ops", &AnalysisConfig::SwitchUseFeedFetchOps,
           py::arg("x") = true)
      .def("use_feed_fetch_ops_enabled",
           &AnalysisConfig::use_feed_fetch_ops_enabled)
      .def("switch_specify_input_names",
           &AnalysisConfig::SwitchSpecifyInputNames, py::arg("x") = true)
      .def("specify_input_name", &AnalysisConfig::specify_input_name)
      .def("enable_tensorrt_engine", &AnalysisConfig::EnableTensorRtEngine,
           py::arg("workspace_size") = 1 << 20, py::arg("max_batch_size") = 1,
           py::arg("min_subgraph_size") = 3,
           py::arg("precision_mode") = AnalysisConfig::Precision::kFloat32,
           py::arg("use_static") = false, py::arg("use_calib_mode") = true)
      .def("tensorrt_engine_enabled", &AnalysisConfig::tensorrt_engine_enabled)
      .def("switch_ir_debug", &AnalysisConfig::SwitchIrDebug,
           py::arg("x") = true)
      .def("enable_mkldnn", &AnalysisConfig::EnableMKLDNN)
      .def("mkldnn_enabled", &AnalysisConfig::mkldnn_enabled)
      .def("set_cpu_math_library_num_threads",
           &AnalysisConfig::SetCpuMathLibraryNumThreads)
      .def("cpu_math_library_num_threads",
           &AnalysisConfig::cpu_math_library_num_threads)
      .def("to_native_config", &AnalysisConfig::ToNativeConfig)
      .def("enable_quantizer", &AnalysisConfig::EnableMkldnnQuantizer)
#ifdef PADDLE_WITH_MKLDNN
      .def("quantizer_config", &AnalysisConfig::mkldnn_quantizer_config,
           py::return_value_policy::reference)
#endif
      .def("set_mkldnn_op", &AnalysisConfig::SetMKLDNNOp)
      .def("set_model_buffer", &AnalysisConfig::SetModelBuffer)
      .def("model_from_memory", &AnalysisConfig::model_from_memory)
      .def("delete_pass",
           [](AnalysisConfig &self, const std::string &pass) {
             self.pass_builder()->DeletePass(pass);
           })
      .def("pass_builder", &AnalysisConfig::pass_builder,
           py::return_value_policy::reference);
}

#ifdef PADDLE_WITH_MKLDNN
void BindMkldnnQuantizerConfig(py::module *m) {
  py::class_<MkldnnQuantizerConfig> quantizer_config(*m,
                                                     "MkldnnQuantizerConfig");
  quantizer_config.def(py::init<const MkldnnQuantizerConfig &>())
      .def(py::init<>())
      .def("set_quant_data",
           [](MkldnnQuantizerConfig &self,
              const std::vector<PaddleTensor> &data) {
             auto warmup_data =
                 std::make_shared<std::vector<PaddleTensor>>(data);
             self.SetWarmupData(warmup_data);
             return;
           })
      .def("set_quant_batch_size", &MkldnnQuantizerConfig::SetWarmupBatchSize)
      .def(
          "set_enabled_op_types",
          (void (MkldnnQuantizerConfig::*)(std::unordered_set<std::string> &)) &
              MkldnnQuantizerConfig::SetEnabledOpTypes);
}
#endif

void BindAnalysisPredictor(py::module *m) {
  py::class_<AnalysisPredictor, PaddlePredictor>(*m, "AnalysisPredictor")
      .def(py::init<const AnalysisConfig &>())
      .def("init", &AnalysisPredictor::Init)
      .def(
          "run",
          [](AnalysisPredictor &self, const std::vector<PaddleTensor> &inputs) {
            std::vector<PaddleTensor> outputs;
            self.Run(inputs, &outputs);
            return outputs;
          })
      .def("get_input_tensor", &AnalysisPredictor::GetInputTensor)
      .def("get_output_tensor", &AnalysisPredictor::GetOutputTensor)
      .def("get_input_names", &AnalysisPredictor::GetInputNames)
      .def("get_output_names", &AnalysisPredictor::GetOutputNames)
      .def("get_input_tensor_shape", &AnalysisPredictor::GetInputTensorShape)
      .def("zero_copy_run", &AnalysisPredictor::ZeroCopyRun)
      .def("create_feed_fetch_var", &AnalysisPredictor::CreateFeedFetchVar)
      .def("prepare_feed_fetch", &AnalysisPredictor::PrepareFeedFetch)
      .def("prepare_argument", &AnalysisPredictor::PrepareArgument)
      .def("optimize_inference_program",
           &AnalysisPredictor::OptimizeInferenceProgram)
      .def("analysis_argument", &AnalysisPredictor::analysis_argument,
           py::return_value_policy::reference)
      .def("clone", &AnalysisPredictor::Clone)
      .def("scope", &AnalysisPredictor::scope,
           py::return_value_policy::reference)
      .def("program", &AnalysisPredictor::program,
           py::return_value_policy::reference)
      .def("get_serialized_program", &AnalysisPredictor::GetSerializedProgram)
      .def("mkldnn_quantize", &AnalysisPredictor::MkldnnQuantize)
      .def("SaveOptimModel", &AnalysisPredictor::SaveOptimModel,
           py::arg("dir"));
}

void BindZeroCopyTensor(py::module *m) {
  py::class_<ZeroCopyTensor>(*m, "ZeroCopyTensor")
      .def("reshape", &ZeroCopyTensor::Reshape)
      .def("copy_from_cpu", &ZeroCopyTensorCreate<int32_t>)
      .def("copy_from_cpu", &ZeroCopyTensorCreate<int64_t>)
      .def("copy_from_cpu", &ZeroCopyTensorCreate<float>)
      .def("copy_to_cpu", &ZeroCopyTensorToNumpy)
      .def("shape", &ZeroCopyTensor::shape)
      .def("set_lod", &ZeroCopyTensor::SetLoD)
      .def("lod", &ZeroCopyTensor::lod)
      .def("type", &ZeroCopyTensor::type);
}

void BindPaddlePassBuilder(py::module *m) {
  py::class_<PaddlePassBuilder>(*m, "PaddlePassBuilder")
      .def(py::init<const std::vector<std::string> &>())
      .def("set_passes",
           [](PaddlePassBuilder &self, const std::vector<std::string> &passes) {
             self.ClearPasses();
             for (auto pass : passes) {
               self.AppendPass(std::move(pass));
             }
           })
      .def("append_pass", &PaddlePassBuilder::AppendPass)
      .def("insert_pass", &PaddlePassBuilder::InsertPass)
      .def("delete_pass",
           [](PaddlePassBuilder &self, const std::string &pass_type) {
             self.DeletePass(pass_type);
           })
      .def("append_analysis_pass", &PaddlePassBuilder::AppendAnalysisPass)
      .def("turn_on_debug", &PaddlePassBuilder::TurnOnDebug)
      .def("debug_string", &PaddlePassBuilder::DebugString)
      .def("all_passes", &PaddlePassBuilder::AllPasses,
           py::return_value_policy::reference)
      .def("analysis_passes", &PaddlePassBuilder::AnalysisPasses);

  py::class_<PassStrategy, PaddlePassBuilder>(*m, "PassStrategy")
      .def(py::init<const std::vector<std::string> &>())
      .def("enable_cudnn", &PassStrategy::EnableCUDNN)
      .def("enable_mkldnn", &PassStrategy::EnableMKLDNN)
      .def("enable_ngraph", &PassStrategy::EnableNgraph)
      .def("enable_mkldnn_quantizer", &PassStrategy::EnableMkldnnQuantizer)
      .def("use_gpu", &PassStrategy::use_gpu);

  py::class_<CpuPassStrategy, PassStrategy>(*m, "CpuPassStrategy")
      .def(py::init<>())
      .def(py::init<const CpuPassStrategy &>())
      .def("enable_cudnn", &CpuPassStrategy::EnableCUDNN)
      .def("enable_mkldnn", &CpuPassStrategy::EnableMKLDNN)
      .def("enable_ngraph", &CpuPassStrategy::EnableNgraph)
      .def("enable_mkldnn_quantizer", &CpuPassStrategy::EnableMkldnnQuantizer);

  py::class_<GpuPassStrategy, PassStrategy>(*m, "GpuPassStrategy")
      .def(py::init<>())
      .def(py::init<const GpuPassStrategy &>())
      .def("enable_cudnn", &GpuPassStrategy::EnableCUDNN)
      .def("enable_mkldnn", &GpuPassStrategy::EnableMKLDNN)
      .def("enable_ngraph", &GpuPassStrategy::EnableNgraph)
      .def("enable_mkldnn_quantizer", &GpuPassStrategy::EnableMkldnnQuantizer);
}
}  // namespace
}  // namespace pybind
}  // namespace paddle
