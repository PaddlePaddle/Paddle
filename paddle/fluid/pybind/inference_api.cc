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

#include <pybind11/functional.h>
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
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/api/paddle_infer_contrib.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_pass_builder.h"
#include "paddle/fluid/inference/utils/io_utils.h"
#include "paddle/phi/core/compat/convert_utils.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/core/cuda_stream.h"
#endif

#ifdef PADDLE_WITH_ONNXRUNTIME
#include "paddle/fluid/inference/api/onnxruntime_predictor.h"
#endif

namespace py = pybind11;

namespace pybind11 {
namespace detail {

// Note: use same enum number of float16 in numpy.
// import numpy as np
// print np.dtype(np.float16).num  # 23
constexpr int NPY_FLOAT16_ = 23;
constexpr int NPY_UINT16_ = 4;

// Note: Since float16 is not a builtin type in C++, we register
// paddle::platform::float16 as numpy.float16.
// Ref: https://github.com/pybind/pybind11/issues/1776
template <>
struct npy_format_descriptor<paddle_infer::float16> {
  static py::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16_);
    return reinterpret_borrow<py::dtype>(ptr);
  }
  static std::string format() {
    // Note: "e" represents float16.
    // Details at:
    // https://docs.python.org/3/library/struct.html#format-characters.
    return "e";
  }
  static constexpr auto name = _("float16");
};

}  // namespace detail
}  // namespace pybind11

namespace paddle {
namespace pybind {
using paddle::AnalysisPredictor;
using paddle::NativeConfig;
using paddle::NativePaddlePredictor;
using paddle::PaddleBuf;
using paddle::PaddleDataLayout;
using paddle::PaddleDType;
using paddle::PaddlePassBuilder;
using paddle::PaddlePlace;
using paddle::PaddlePredictor;
using paddle::PaddleTensor;
using paddle::PassStrategy;
using paddle::ZeroCopyTensor;

namespace {
void BindPaddleDType(py::module *m);
void BindPaddleDataLayout(py::module *m);
void BindPaddleBuf(py::module *m);
void BindPaddleTensor(py::module *m);
void BindPaddlePlace(py::module *m);
void BindPaddlePredictor(py::module *m);
void BindNativeConfig(py::module *m);
void BindNativePredictor(py::module *m);
void BindLiteNNAdapterConfig(py::module *m);
void BindAnalysisConfig(py::module *m);
void BindAnalysisPredictor(py::module *m);
void BindZeroCopyTensor(py::module *m);
void BindPaddlePassBuilder(py::module *m);
void BindPaddleInferPredictor(py::module *m);
void BindPaddleInferTensor(py::module *m);
void BindPredictorPool(py::module *m);

#ifdef PADDLE_WITH_MKLDNN
void BindMkldnnQuantizerConfig(py::module *m);
#endif

template <typename T>
PaddleBuf PaddleBufCreate(
    py::array_t<T, py::array::c_style | py::array::forcecast> data) {
  PaddleBuf buf(data.size() * sizeof(T));
  std::copy_n(static_cast<const T *>(data.data()),
              data.size(),
              static_cast<T *>(buf.data()));
  return buf;
}

template <typename T>
void PaddleBufReset(
    PaddleBuf &buf,                                                    // NOLINT
    py::array_t<T, py::array::c_style | py::array::forcecast> data) {  // NOLINT
  buf.Resize(data.size() * sizeof(T));
  std::copy_n(static_cast<const T *>(data.data()),
              data.size(),
              static_cast<T *>(buf.data()));
}

template <typename T>
PaddleTensor PaddleTensorCreate(
    py::array_t<T, py::array::c_style | py::array::forcecast> data,
    const std::string name = "",
    const std::vector<std::vector<size_t>> &lod = {},
    bool copy = true) {
  PaddleTensor tensor;

  if (copy) {
    PaddleBuf buf(data.size() * sizeof(T));
    std::copy_n(static_cast<const T *>(data.data()),
                data.size(),
                static_cast<T *>(buf.data()));
    tensor.data = std::move(buf);
  } else {
    tensor.data = PaddleBuf(data.mutable_data(), data.size() * sizeof(T));
  }

  tensor.dtype = inference::PaddleTensorGetDType<T>();
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
    case PaddleDType::UINT8:
      dt = py::dtype::of<uint8_t>();
      break;
    case PaddleDType::FLOAT16:
      dt = py::dtype::of<paddle_infer::float16>();
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported data type. Now only supports INT32, INT64, UINT8 and "
          "FLOAT32."));
  }

  return dt;
}

py::array PaddleTensorGetData(PaddleTensor &tensor) {  // NOLINT
  py::dtype dt = PaddleDTypeToNumpyDType(tensor.dtype);
  return py::array(std::move(dt), {tensor.shape}, tensor.data.data());
}

template <typename T>
void ZeroCopyTensorCreate(
    ZeroCopyTensor &tensor,  // NOLINT
    py::array_t<T, py::array::c_style | py::array::forcecast> data) {
  std::vector<int> shape;
  std::copy_n(data.shape(), data.ndim(), std::back_inserter(shape));
  tensor.Reshape(std::move(shape));
  tensor.copy_from_cpu(static_cast<const T *>(data.data()));
}

/// \brief Experimental interface.
/// Create the Strings tensor from data.
/// \param tensor The tensor will be created and
/// the tensor value is same as data.
/// \param data The input text.
void ZeroCopyStringTensorCreate(ZeroCopyTensor &tensor,  // NOLINT
                                const paddle_infer::Strings *data) {
  size_t shape = data->size();
  tensor.ReshapeStrings(shape);
  tensor.copy_strings_from_cpu(data);
}

template <typename T>
void PaddleInferTensorCreate(
    paddle_infer::Tensor &tensor,  // NOLINT
    py::array_t<T, py::array::c_style | py::array::forcecast> data) {
  std::vector<int> shape;
  std::copy_n(data.shape(), data.ndim(), std::back_inserter(shape));
  tensor.Reshape(std::move(shape));
  tensor.CopyFromCpu(static_cast<const T *>(data.data()));
}

paddle_infer::PlaceType ToPaddleInferPlace(
    phi::AllocationType allocation_type) {
  if (allocation_type == phi::AllocationType::CPU) {
    return paddle_infer::PlaceType::kCPU;
  } else if (allocation_type == phi::AllocationType::GPU) {
    return paddle_infer::PlaceType::kGPU;
  } else {
    return paddle_infer::PlaceType::kCPU;
  }
}

void PaddleInferShareExternalData(paddle_infer::Tensor &tensor,  // NOLINT
                                  phi::DenseTensor input_tensor) {
  std::vector<int> shape;
  for (int i = 0; i < input_tensor.dims().size(); ++i) {
    shape.push_back(input_tensor.dims()[i]);
  }
  if (input_tensor.dtype() == phi::DataType::FLOAT32) {
    tensor.ShareExternalData(
        static_cast<float *>(input_tensor.data()),
        shape,
        ToPaddleInferPlace(input_tensor.place().GetType()));
  } else if (input_tensor.dtype() == phi::DataType::FLOAT16) {
    tensor.ShareExternalData(
        static_cast<paddle::platform::float16 *>(input_tensor.data()),
        shape,
        ToPaddleInferPlace(input_tensor.place().GetType()));
  }
}

/// \brief Experimental interface.
/// Create the Strings tensor from data.
/// \param tensor The tensor will be created and
/// the tensor value is same as data.
/// \param data The input text.
void PaddleInferStringTensorCreate(paddle_infer::Tensor &tensor,  // NOLINT
                                   const paddle_infer::Strings *data) {
  VLOG(3) << "Create PaddleInferTensor, dtype = Strings ";
  size_t shape = data->size();
  tensor.ReshapeStrings(shape);
  tensor.CopyStringsFromCpu(data);
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
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported data type. Now only supports INT32, INT64 and "
          "FLOAT32."));
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
    case PaddleDType::FLOAT16:
      tensor.copy_to_cpu<paddle::platform::float16>(
          static_cast<paddle::platform::float16 *>(array.mutable_data()));
      break;
    case PaddleDType::UINT8:
      tensor.copy_to_cpu<uint8_t>(static_cast<uint8_t *>(array.mutable_data()));
      break;
    case PaddleDType::INT8:
      tensor.copy_to_cpu<int8_t>(static_cast<int8_t *>(array.mutable_data()));
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported data type. Now only supports INT32, INT64, UINT8 and "
          "FLOAT32."));
  }
  return array;
}

py::array PaddleInferTensorToNumpy(paddle_infer::Tensor &tensor) {  // NOLINT
  py::dtype dt = PaddleDTypeToNumpyDType(tensor.type());
  auto tensor_shape = tensor.shape();
  py::array::ShapeContainer shape(tensor_shape.begin(), tensor_shape.end());
  py::array array(dt, std::move(shape));

  switch (tensor.type()) {
    case PaddleDType::INT32:
      tensor.CopyToCpu(static_cast<int32_t *>(array.mutable_data()));
      break;
    case PaddleDType::INT64:
      tensor.CopyToCpu(static_cast<int64_t *>(array.mutable_data()));
      break;
    case PaddleDType::FLOAT32:
      tensor.CopyToCpu<float>(static_cast<float *>(array.mutable_data()));
      break;
    case PaddleDType::FLOAT16:
      tensor.CopyToCpu<paddle::platform::float16>(
          static_cast<paddle::platform::float16 *>(array.mutable_data()));
      break;
    case PaddleDType::UINT8:
      tensor.CopyToCpu(static_cast<uint8_t *>(array.mutable_data()));
      break;
    case PaddleDType::INT8:
      tensor.CopyToCpu(static_cast<int8_t *>(array.mutable_data()));
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported data type. Now only supports INT32, INT64 and "
          "FLOAT32."));
  }
  return array;
}

py::bytes SerializePDTensorToBytes(PaddleTensor &tensor) {  // NOLINT
  std::stringstream ss;
  paddle::inference::SerializePDTensorToStream(&ss, tensor);
  return static_cast<py::bytes>(ss.str());
}

void CopyPaddleInferTensor(paddle_infer::Tensor &dst,  // NOLINT
                           const paddle_infer::Tensor &src) {
  return paddle_infer::contrib::TensorUtils::CopyTensor(&dst, src);
}

}  // namespace

void BindInferenceApi(py::module *m) {
  BindPaddleDType(m);
  BindPaddleDataLayout(m);
  BindPaddleBuf(m);
  BindPaddleTensor(m);
  BindPaddlePlace(m);
  BindPaddlePredictor(m);
  BindNativeConfig(m);
  BindNativePredictor(m);
  BindLiteNNAdapterConfig(m);
  BindAnalysisConfig(m);
  BindAnalysisPredictor(m);
  BindPaddleInferPredictor(m);
  BindZeroCopyTensor(m);
  BindPaddleInferTensor(m);
  BindPaddlePassBuilder(m);
  BindPredictorPool(m);
#ifdef PADDLE_WITH_MKLDNN
  BindMkldnnQuantizerConfig(m);
#endif
  m->def("create_paddle_predictor",
         &paddle::CreatePaddlePredictor<AnalysisConfig>,
         py::arg("config"));
  m->def("create_paddle_predictor",
         &paddle::CreatePaddlePredictor<NativeConfig>,
         py::arg("config"));
  m->def("create_predictor",
         [](const paddle_infer::Config &config)
             -> std::unique_ptr<paddle_infer::Predictor> {
           auto pred = std::unique_ptr<paddle_infer::Predictor>(
               new paddle_infer::Predictor(config));
           return pred;
         });
  m->def(
      "_get_phi_kernel_name",
      [](const std::string &fluid_op_name) {
        return phi::TransToPhiKernelName(fluid_op_name);
      },
      py::return_value_policy::reference);
  m->def("copy_tensor", &CopyPaddleInferTensor);
  m->def("paddle_dtype_size", &paddle::PaddleDtypeSize);
  m->def("paddle_tensor_to_bytes", &SerializePDTensorToBytes);
  m->def("get_version", &paddle_infer::GetVersion);
  m->def("get_trt_compile_version", &paddle_infer::GetTrtCompileVersion);
  m->def("get_trt_runtime_version", &paddle_infer::GetTrtRuntimeVersion);
  m->def("get_num_bytes_of_data_type", &paddle_infer::GetNumBytesOfDataType);
  m->def("convert_to_mixed_precision_bind",
         &paddle_infer::ConvertToMixedPrecision,
         py::arg("model_file"),
         py::arg("params_file"),
         py::arg("mixed_model_file"),
         py::arg("mixed_params_file"),
         py::arg("mixed_precision"),
         py::arg("backend"),
         py::arg("keep_io_types") = true,
         py::arg("black_list") = std::unordered_set<std::string>());
}

namespace {
void BindPaddleDType(py::module *m) {
  py::enum_<PaddleDType>(*m, "PaddleDType")
      .value("FLOAT32", PaddleDType::FLOAT32)
      .value("INT64", PaddleDType::INT64)
      .value("INT32", PaddleDType::INT32);
}

void BindPaddleDataLayout(py::module *m) {
  py::enum_<PaddleDataLayout>(*m, "PaddleDataLayout")
      .value("UNK", PaddleDataLayout::kUNK)
      .value("Any", PaddleDataLayout::kAny)
      .value("NHWC", PaddleDataLayout::kNHWC)
      .value("NCHW", PaddleDataLayout::kNCHW);
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
               PADDLE_THROW(platform::errors::Unimplemented(
                   "Unsupported data type. Now only supports INT32, INT64 and "
                   "FLOAT32."));
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
      .def(py::init(&PaddleTensorCreate<int32_t>),
           py::arg("data"),
           py::arg("name") = "",
           py::arg("lod") = std::vector<std::vector<size_t>>(),
           py::arg("copy") = true)
      .def(py::init(&PaddleTensorCreate<int64_t>),
           py::arg("data"),
           py::arg("name") = "",
           py::arg("lod") = std::vector<std::vector<size_t>>(),
           py::arg("copy") = true)
      .def(py::init(&PaddleTensorCreate<float>),
           py::arg("data"),
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
      .value("GPU", PaddlePlace::kGPU)
      .value("XPU", PaddlePlace::kXPU)
      .value("NPU", PaddlePlace::kNPU);
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
      .def("clone", [](PaddlePredictor &self) { return self.Clone(nullptr); })
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      .def("clone",
           [](PaddlePredictor &self, phi::CUDAStream &stream) {
             return self.Clone(stream.raw_stream());
           })
#endif
      .def("get_serialized_program", &PaddlePredictor::GetSerializedProgram);

  auto config = py::class_<PaddlePredictor::Config>(paddle_predictor, "Config");
  config.def(py::init<>())
      .def_readwrite("model_dir", &PaddlePredictor::Config::model_dir);
}

void BindNativeConfig(py::module *m) {
  py::class_<NativeConfig, PaddlePredictor::Config>(*m, "NativeConfig")
      .def(py::init<>())
      .def_readwrite("use_gpu", &NativeConfig::use_gpu)
      .def_readwrite("use_xpu", &NativeConfig::use_xpu)
      .def_readwrite("use_npu", &NativeConfig::use_npu)
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
      .def("clone",
           [](NativePaddlePredictor &self) { return self.Clone(nullptr); })
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      .def("clone",
           [](NativePaddlePredictor &self, phi::CUDAStream &stream) {
             return self.Clone(stream.raw_stream());
           })
#endif
      .def("scope",
           &NativePaddlePredictor::scope,
           py::return_value_policy::reference);
}

void BindAnalysisConfig(py::module *m) {
  py::class_<AnalysisConfig> analysis_config(*m, "AnalysisConfig");

  py::enum_<AnalysisConfig::Precision>(analysis_config, "Precision")
      .value("Float32", AnalysisConfig::Precision::kFloat32)
      .value("Int8", AnalysisConfig::Precision::kInt8)
      .value("Half", AnalysisConfig::Precision::kHalf)
      .value("Bfloat16", AnalysisConfig::Precision::kBf16)
      .export_values();

  analysis_config.def(py::init<>())
      .def(py::init<const AnalysisConfig &>())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, const std::string &>())
      .def("summary", &AnalysisConfig::Summary)
      .def("set_model",
           (void(AnalysisConfig::*)(const std::string &)) &
               AnalysisConfig::SetModel)
      .def("set_model",
           (void(AnalysisConfig::*)(const std::string &, const std::string &)) &
               AnalysisConfig::SetModel)
      .def("set_prog_file", &AnalysisConfig::SetProgFile)
      .def("set_params_file", &AnalysisConfig::SetParamsFile)
      .def("model_dir", &AnalysisConfig::model_dir)
      .def("prog_file", &AnalysisConfig::prog_file)
      .def("params_file", &AnalysisConfig::params_file)
      .def("enable_use_gpu",
           &AnalysisConfig::EnableUseGpu,
           py::arg("memory_pool_init_size_mb"),
           py::arg("device_id") = 0,
           py::arg("precision_mode") = AnalysisConfig::Precision::kFloat32)
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      .def("set_exec_stream",
           [](AnalysisConfig &self, phi::CUDAStream &stream) {
             self.SetExecStream(stream.raw_stream());
           })
#endif
      .def("enable_xpu",
           &AnalysisConfig::EnableXpu,
           py::arg("l3_workspace_size") = 16 * 1024 * 1024,
           py::arg("locked") = false,
           py::arg("autotune") = true,
           py::arg("autotune_file") = "",
           py::arg("precision") = "int16",
           py::arg("adaptive_seqlen") = false,
           py::arg("enable_multi_stream") = false)
      .def("set_xpu_device_id",
           &AnalysisConfig::SetXpuDeviceId,
           py::arg("device_id") = 0)
      .def("enable_custom_device",
           &AnalysisConfig::EnableCustomDevice,
           py::arg("device_type"),
           py::arg("device_id") = 0)
      .def("enable_npu", &AnalysisConfig::EnableNpu, py::arg("device_id") = 0)
      .def("enable_ipu",
           &AnalysisConfig::EnableIpu,
           py::arg("ipu_device_num") = 1,
           py::arg("ipu_micro_batch_size") = 1,
           py::arg("ipu_enable_pipelining") = false,
           py::arg("ipu_batches_per_step") = 1)
      .def("set_ipu_config",
           &AnalysisConfig::SetIpuConfig,
           py::arg("ipu_enable_fp16") = false,
           py::arg("ipu_replica_num") = 1,
           py::arg("ipu_available_memory_proportion") = 1.0,
           py::arg("ipu_enable_half_partial") = false,
           py::arg("ipu_enable_model_runtime_executor") = false)
      .def("set_ipu_custom_info",
           &AnalysisConfig::SetIpuCustomInfo,
           py::arg("ipu_custom_ops_info") =
               std::vector<std::vector<std::string>>({}),
           py::arg("ipu_custom_patterns") = std::map<std::string, bool>({}))
      .def("load_ipu_config",
           &AnalysisConfig::LoadIpuConfig,
           py::arg("config_path"))
      .def("disable_gpu", &AnalysisConfig::DisableGpu)
      .def("enable_onnxruntime", &AnalysisConfig::EnableONNXRuntime)
      .def("disable_onnxruntime", &AnalysisConfig::DisableONNXRuntime)
      .def("onnxruntime_enabled", &AnalysisConfig::use_onnxruntime)
      .def("use_opencl", &AnalysisConfig::use_opencl)
      .def("enable_ort_optimization", &AnalysisConfig::EnableORTOptimization)
      .def("use_gpu", &AnalysisConfig::use_gpu)
      .def("use_xpu", &AnalysisConfig::use_xpu)
      .def("use_npu", &AnalysisConfig::use_npu)
      .def("gpu_device_id", &AnalysisConfig::gpu_device_id)
      .def("xpu_device_id", &AnalysisConfig::xpu_device_id)
      .def("npu_device_id", &AnalysisConfig::npu_device_id)
      .def("memory_pool_init_size_mb",
           &AnalysisConfig::memory_pool_init_size_mb)
      .def("fraction_of_gpu_memory_for_pool",
           &AnalysisConfig::fraction_of_gpu_memory_for_pool)
      .def("switch_ir_optim",
           &AnalysisConfig::SwitchIrOptim,
           py::arg("x") = true)
      .def("ir_optim", &AnalysisConfig::ir_optim)
      .def("enable_memory_optim",
           &AnalysisConfig::EnableMemoryOptim,
           py::arg("x") = true)
      .def("enable_profile", &AnalysisConfig::EnableProfile)
      .def("disable_glog_info", &AnalysisConfig::DisableGlogInfo)
      .def("glog_info_disabled", &AnalysisConfig::glog_info_disabled)
      .def("set_optim_cache_dir", &AnalysisConfig::SetOptimCacheDir)
      .def("switch_use_feed_fetch_ops",
           &AnalysisConfig::SwitchUseFeedFetchOps,
           py::arg("x") = true)
      .def("use_feed_fetch_ops_enabled",
           &AnalysisConfig::use_feed_fetch_ops_enabled)
      .def("switch_specify_input_names",
           &AnalysisConfig::SwitchSpecifyInputNames,
           py::arg("x") = true)
      .def("specify_input_name", &AnalysisConfig::specify_input_name)
      .def("enable_tensorrt_engine",
           &AnalysisConfig::EnableTensorRtEngine,
           py::arg("workspace_size") = 1 << 30,
           py::arg("max_batch_size") = 1,
           py::arg("min_subgraph_size") = 3,
           py::arg("precision_mode") = AnalysisConfig::Precision::kFloat32,
           py::arg("use_static") = false,
           py::arg("use_calib_mode") = true)
      .def("enable_tensorrt_memory_optim",
           &AnalysisConfig::EnableTensorRTMemoryOptim,
           py::arg("engine_memory_sharing") = true,
           py::arg("sharing_identifier") = 0)
      .def("tensorrt_precision_mode", &AnalysisConfig::tensorrt_precision_mode)
      .def("set_trt_dynamic_shape_info",
           &AnalysisConfig::SetTRTDynamicShapeInfo,
           py::arg("min_input_shape") =
               std::map<std::string, std::vector<int>>({}),
           py::arg("max_input_shape") =
               std::map<std::string, std::vector<int>>({}),
           py::arg("optim_input_shape") =
               std::map<std::string, std::vector<int>>({}),
           py::arg("disable_trt_plugin_fp16") = false)
      .def("tensorrt_dynamic_shape_enabled",
           &AnalysisConfig::tensorrt_dynamic_shape_enabled)
      .def("enable_tensorrt_varseqlen", &AnalysisConfig::EnableVarseqlen)
      .def("tensorrt_varseqlen_enabled",
           &AnalysisConfig::tensorrt_varseqlen_enabled)
      .def("collect_shape_range_info", &AnalysisConfig::CollectShapeRangeInfo)
      .def("shape_range_info_path", &AnalysisConfig::shape_range_info_path)
      .def("shape_range_info_collected",
           &AnalysisConfig::shape_range_info_collected)
      .def("enable_tuned_tensorrt_dynamic_shape",
           &AnalysisConfig::EnableTunedTensorRtDynamicShape)
      .def("tuned_tensorrt_dynamic_shape",
           &AnalysisConfig::tuned_tensorrt_dynamic_shape)
      .def("trt_allow_build_at_runtime",
           &AnalysisConfig::trt_allow_build_at_runtime)
      .def("exp_disable_tensorrt_ops", &AnalysisConfig::Exp_DisableTensorRtOPs)
      .def("enable_tensorrt_dla",
           &AnalysisConfig::EnableTensorRtDLA,
           py::arg("dla_core") = 0)
      .def("tensorrt_dla_enabled", &AnalysisConfig::tensorrt_dla_enabled)
      .def("enable_tensorrt_inspector",
           &AnalysisConfig::EnableTensorRtInspector)
      .def("tensorrt_inspector_enabled",
           &AnalysisConfig::tensorrt_inspector_enabled)
      .def("tensorrt_engine_enabled", &AnalysisConfig::tensorrt_engine_enabled)
      .def("enable_dlnne",
           &AnalysisConfig::EnableDlnne,
           py::arg("min_subgraph_size") = 3,
           py::arg("max_batch_size") = 1,
           py::arg("use_static_batch") = false,
           py::arg("weight_share_mode") = "0",
           py::arg("disable_nodes_by_outputs") =
               std::unordered_set<std::string>(),
           py::arg("input_shape_dict") =
               std::map<std::string, std::vector<int64_t>>(),
           py::arg("use_calib_mode") = false,
           py::arg("precision_mode") = AnalysisConfig::Precision::kFloat32)
      .def("enable_lite_engine",
           &AnalysisConfig::EnableLiteEngine,
           py::arg("precision_mode") = AnalysisConfig::Precision::kFloat32,
           py::arg("zero_copy") = false,
           py::arg("passes_filter") = std::vector<std::string>(),
           py::arg("ops_filter") = std::vector<std::string>())
      .def("enable_opencl", &AnalysisConfig::EnableOpenCL)
      .def("lite_engine_enabled", &AnalysisConfig::lite_engine_enabled)
      .def("switch_ir_debug",
           &AnalysisConfig::SwitchIrDebug,
           py::arg("x") = true)
      .def("enable_mkldnn", &AnalysisConfig::EnableMKLDNN)
      .def("mkldnn_enabled", &AnalysisConfig::mkldnn_enabled)
      .def("set_cpu_math_library_num_threads",
           &AnalysisConfig::SetCpuMathLibraryNumThreads)
      .def("cpu_math_library_num_threads",
           &AnalysisConfig::cpu_math_library_num_threads)
      .def("to_native_config", &AnalysisConfig::ToNativeConfig)
      .def("enable_quantizer", &AnalysisConfig::EnableMkldnnQuantizer)
      .def("enable_mkldnn_bfloat16", &AnalysisConfig::EnableMkldnnBfloat16)
#ifdef PADDLE_WITH_MKLDNN
      .def("quantizer_config",
           &AnalysisConfig::mkldnn_quantizer_config,
           py::return_value_policy::reference)
      .def("set_mkldnn_cache_capacity",
           &AnalysisConfig::SetMkldnnCacheCapacity,
           py::arg("capacity") = 0)
      .def("set_bfloat16_op", &AnalysisConfig::SetBfloat16Op)
      .def("enable_mkldnn_int8",
           &AnalysisConfig::EnableMkldnnInt8,
           py::arg("mkldnn_int8_enabled_op_types") =
               std::unordered_set<std::string>({}))
      .def("mkldnn_int8_enabled", &AnalysisConfig::mkldnn_int8_enabled)
      .def("disable_mkldnn_fc_passes",
           &AnalysisConfig::DisableMkldnnFcPasses,
           R"DOC(
           Disable Mkldnn FC
           Args:
                None.
           Returns:
                None.
           Examples:
               .. code-block:: python
                from paddle.inference import Config

                config = Config("")
                config.enable_mkldnn()
                config.disable_mkldnn_fc_passes()
           )DOC")
#endif
      .def("set_mkldnn_op", &AnalysisConfig::SetMKLDNNOp)
      .def("set_model_buffer", &AnalysisConfig::SetModelBuffer)
      .def("model_from_memory", &AnalysisConfig::model_from_memory)
      .def("delete_pass",
           [](AnalysisConfig &self, const std::string &pass) {
             self.pass_builder()->DeletePass(pass);
           })
      .def(
          "pass_builder",
          [](AnalysisConfig &self) {
            return dynamic_cast<PaddlePassBuilder *>(self.pass_builder());
          },
          py::return_value_policy::reference)
      .def("nnadapter", &AnalysisConfig::NNAdapter)
      .def("set_dist_config", &AnalysisConfig::SetDistConfig)
      .def("dist_config", &AnalysisConfig::dist_config);

  py::class_<DistConfig>(*m, "DistConfig")
      .def(py::init<>())
      .def("set_carrier_id", &DistConfig::SetCarrierId)
      .def("set_comm_init_config", &DistConfig::SetCommInitConfig)
      .def("set_endpoints", &DistConfig::SetEndpoints)
      .def("set_ranks", &DistConfig::SetRanks)
      .def("enable_dist_model", &DistConfig::EnableDistModel)
      .def("carrier_id", &DistConfig::carrier_id)
      .def("current_endpoint", &DistConfig::current_endpoint)
      .def("trainer_endpoints", &DistConfig::trainer_endpoints)
      .def("nranks", &DistConfig::nranks)
      .def("rank", &DistConfig::rank)
      .def("comm_init_config", &DistConfig::comm_init_config)
      .def("use_dist_model", &DistConfig::use_dist_model);
}

void BindLiteNNAdapterConfig(py::module *m) {
  py::class_<LiteNNAdapterConfig> lite_nnadapter_config(*m,
                                                        "LiteNNAdapterConfig");

  lite_nnadapter_config
      .def("set_device_names", &LiteNNAdapterConfig::SetDeviceNames)
      .def("set_context_properties", &LiteNNAdapterConfig::SetContextProperties)
      .def("set_model_cache_dir", &LiteNNAdapterConfig::SetModelCacheDir)
      .def("set_model_cache_buffers",
           &LiteNNAdapterConfig::SetModelCacheBuffers)
      .def("set_subgraph_partition_config_path",
           &LiteNNAdapterConfig::SetSubgraphPartitionConfigPath)
      .def("set_subgraph_partition_config_buffer",
           &LiteNNAdapterConfig::SetSubgraphPartitionConfigBuffer)
      .def("enable", &LiteNNAdapterConfig::Enable)
      .def("disable", &LiteNNAdapterConfig::Disable);
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
      .def("set_enabled_op_types", &MkldnnQuantizerConfig::SetEnabledOpTypes);
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
      .def("clear_intermediate_tensor",
           &AnalysisPredictor::ClearIntermediateTensor)
      .def("try_shrink_memory", &AnalysisPredictor::TryShrinkMemory)
      .def("create_feed_fetch_var", &AnalysisPredictor::CreateFeedFetchVar)
      .def("prepare_feed_fetch", &AnalysisPredictor::PrepareFeedFetch)
      .def("prepare_argument", &AnalysisPredictor::PrepareArgument)
      .def("optimize_inference_program",
           &AnalysisPredictor::OptimizeInferenceProgram)
      .def("analysis_argument",
           &AnalysisPredictor::analysis_argument,
           py::return_value_policy::reference)
      .def("clone", [](AnalysisPredictor &self) { return self.Clone(nullptr); })
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      .def("clone",
           [](AnalysisPredictor &self, phi::CUDAStream &stream) {
             return self.Clone(stream.raw_stream());
           })
#endif
      .def("scope",
           &AnalysisPredictor::scope,
           py::return_value_policy::reference)
      .def("program",
           &AnalysisPredictor::program,
           py::return_value_policy::reference)
      .def("get_serialized_program", &AnalysisPredictor::GetSerializedProgram)
      .def("mkldnn_quantize", &AnalysisPredictor::MkldnnQuantize)
      .def(
          "SaveOptimModel", &AnalysisPredictor::SaveOptimModel, py::arg("dir"));
}

void BindPaddleInferPredictor(py::module *m) {
  py::class_<paddle_infer::Predictor>(*m, "PaddleInferPredictor")
      .def(py::init<const paddle_infer::Config &>())
      .def("get_input_names", &paddle_infer::Predictor::GetInputNames)
      .def("get_output_names", &paddle_infer::Predictor::GetOutputNames)
      .def("get_input_handle", &paddle_infer::Predictor::GetInputHandle)
      .def("get_output_handle", &paddle_infer::Predictor::GetOutputHandle)
      .def("run",
           [](paddle_infer::Predictor &self) {
#ifdef PADDLE_WITH_ASCEND_CL
             pybind11::gil_scoped_release release;
#endif
             self.Run();
           })
      .def("clone",
           [](paddle_infer::Predictor &self) { return self.Clone(nullptr); })
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      .def("clone",
           [](paddle_infer::Predictor &self, phi::CUDAStream &stream) {
             return self.Clone(stream.raw_stream());
           })
#endif
      .def("try_shrink_memory", &paddle_infer::Predictor::TryShrinkMemory)
      .def("clear_intermediate_tensor",
           &paddle_infer::Predictor::ClearIntermediateTensor)
      .def("register_output_hook",
           &paddle_infer::Predictor::RegisterOutputHook);
}

void BindZeroCopyTensor(py::module *m) {
  py::class_<ZeroCopyTensor>(*m, "ZeroCopyTensor")
      .def(
          "reshape",
          py::overload_cast<const std::vector<int> &>(&ZeroCopyTensor::Reshape))
      .def("reshape",
           py::overload_cast<const std::size_t &>(
               &paddle_infer::Tensor::ReshapeStrings))
      .def("copy_from_cpu", &ZeroCopyTensorCreate<int32_t>)
      .def("copy_from_cpu", &ZeroCopyTensorCreate<int64_t>)
      .def("copy_from_cpu", &ZeroCopyTensorCreate<float>)
      .def("copy_from_cpu", &ZeroCopyTensorCreate<paddle_infer::float16>)
      .def("copy_from_cpu", &ZeroCopyStringTensorCreate)
      .def("copy_to_cpu", &ZeroCopyTensorToNumpy)
      .def("shape", &ZeroCopyTensor::shape)
      .def("set_lod", &ZeroCopyTensor::SetLoD)
      .def("lod", &ZeroCopyTensor::lod)
      .def("type", &ZeroCopyTensor::type);
}

void BindPaddleInferTensor(py::module *m) {
  py::class_<paddle_infer::Tensor>(*m, "PaddleInferTensor")
      .def("reshape",
           py::overload_cast<const std::vector<int> &>(
               &paddle_infer::Tensor::Reshape))
      .def("reshape",
           py::overload_cast<const std::size_t &>(
               &paddle_infer::Tensor::ReshapeStrings))
      .def("copy_from_cpu_bind", &PaddleInferTensorCreate<int32_t>)
      .def("copy_from_cpu_bind", &PaddleInferTensorCreate<int64_t>)
      .def("copy_from_cpu_bind", &PaddleInferTensorCreate<float>)
      .def("copy_from_cpu_bind",
           &PaddleInferTensorCreate<paddle_infer::float16>)
      .def("copy_from_cpu_bind", &PaddleInferStringTensorCreate)
      .def("share_external_data_bind", &PaddleInferShareExternalData)
      .def("copy_to_cpu", &PaddleInferTensorToNumpy)
      .def("shape", &paddle_infer::Tensor::shape)
      .def("set_lod", &paddle_infer::Tensor::SetLoD)
      .def("lod", &paddle_infer::Tensor::lod)
      .def("type", &paddle_infer::Tensor::type);
}

void BindPredictorPool(py::module *m) {
  py::class_<paddle_infer::services::PredictorPool>(*m, "PredictorPool")
      .def(py::init<const paddle_infer::Config &, size_t>())
      .def("retrive",
           &paddle_infer::services::PredictorPool::Retrive,
           py::return_value_policy::reference);
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
      .def("all_passes",
           &PaddlePassBuilder::AllPasses,
           py::return_value_policy::reference)
      .def("analysis_passes", &PaddlePassBuilder::AnalysisPasses);

  py::class_<PassStrategy, PaddlePassBuilder>(*m, "PassStrategy")
      .def(py::init<const std::vector<std::string> &>())
      .def("enable_cudnn", &PassStrategy::EnableCUDNN)
      .def("enable_mkldnn", &PassStrategy::EnableMKLDNN)
      .def("enable_mkldnn_quantizer", &PassStrategy::EnableMkldnnQuantizer)
      .def("enable_mkldnn_bfloat16", &PassStrategy::EnableMkldnnBfloat16)
      .def("use_gpu", &PassStrategy::use_gpu);

  py::class_<CpuPassStrategy, PassStrategy>(*m, "CpuPassStrategy")
      .def(py::init<>())
      .def(py::init<const CpuPassStrategy &>())
      .def("enable_cudnn", &CpuPassStrategy::EnableCUDNN)
      .def("enable_mkldnn", &CpuPassStrategy::EnableMKLDNN)
      .def("enable_mkldnn_quantizer", &CpuPassStrategy::EnableMkldnnQuantizer)
      .def("enable_mkldnn_bfloat16", &CpuPassStrategy::EnableMkldnnBfloat16);

  py::class_<GpuPassStrategy, PassStrategy>(*m, "GpuPassStrategy")
      .def(py::init<>())
      .def(py::init<const GpuPassStrategy &>())
      .def("enable_cudnn", &GpuPassStrategy::EnableCUDNN)
      .def("enable_mkldnn", &GpuPassStrategy::EnableMKLDNN)
      .def("enable_mkldnn_quantizer", &GpuPassStrategy::EnableMkldnnQuantizer)
      .def("enable_mkldnn_bfloat16", &GpuPassStrategy::EnableMkldnnBfloat16);
}
}  // namespace
}  // namespace pybind
}  // namespace paddle
