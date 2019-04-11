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
#include <pybind11/stl.h>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "paddle/fluid/inference/api/analysis_predictor.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {
using paddle::PaddleDType;
using paddle::PaddleBuf;
using paddle::PaddleTensor;
using paddle::PaddlePlace;
using paddle::PaddlePredictor;
using paddle::NativeConfig;
using paddle::NativePaddlePredictor;
using paddle::AnalysisPredictor;

static void BindPaddleDType(py::module *m);
static void BindPaddleBuf(py::module *m);
static void BindPaddleTensor(py::module *m);
static void BindPaddlePlace(py::module *m);
static void BindPaddlePredictor(py::module *m);
static void BindNativeConfig(py::module *m);
static void BindNativePredictor(py::module *m);
static void BindAnalysisConfig(py::module *m);
static void BindAnalysisPredictor(py::module *m);

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

  m->def("create_paddle_predictor",
         &paddle::CreatePaddlePredictor<AnalysisConfig>);
  m->def("create_paddle_predictor",
         &paddle::CreatePaddlePredictor<NativeConfig>);
  m->def("paddle_dtype_size", &paddle::PaddleDtypeSize);
}

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
      .def(py::init([](std::vector<int64_t> &data) {
        auto buf = PaddleBuf(data.size() * sizeof(int64_t));
        std::memcpy(buf.data(), static_cast<void *>(data.data()), buf.length());
        return buf;
      }))
      .def("resize", &PaddleBuf::Resize)
      .def("reset",
           [](PaddleBuf &self, std::vector<float> &data) {
             self.Resize(data.size() * sizeof(float));
             std::memcpy(self.data(), data.data(), self.length());
           })
      .def("reset",
           [](PaddleBuf &self, std::vector<int64_t> &data) {
             self.Resize(data.size() * sizeof(int64_t));
             std::memcpy(self.data(), data.data(), self.length());
           })
      .def("empty", &PaddleBuf::empty)
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
      .def("zero_copy_run", &PaddlePredictor::ZeroCopyRun)
      .def("clone", &PaddlePredictor::Clone);

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
           py::arg("use_static") = true)
      .def("enable_anakin_engine", &AnalysisConfig::EnableAnakinEngine,
           py::arg("max_batch_size") = 1,
           py::arg("max_input_shape") =
               std::map<std::string, std::vector<int>>(),
           py::arg("min_subgraph_size") = 6,
           py::arg("precision_mode") = AnalysisConfig::Precision::kFloat32,
           py::arg("passes_filter") = std::vector<std::string>(),
           py::arg("ops_filter") = std::vector<std::string>())
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
      .def("set_mkldnn_op", &AnalysisConfig::SetMKLDNNOp)
      .def("set_model_buffer", &AnalysisConfig::SetModelBuffer)
      .def("model_from_memory", &AnalysisConfig::model_from_memory)
      .def("pass_builder", &AnalysisConfig::pass_builder,
           py::return_value_policy::reference);
}

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
      .def("zero_copy_run", &AnalysisPredictor::ZeroCopyRun)
      .def("clone", &AnalysisPredictor::Clone)
      .def("scope", &AnalysisPredictor::scope,
           py::return_value_policy::reference);
}

}  // namespace pybind
}  // namespace paddle
