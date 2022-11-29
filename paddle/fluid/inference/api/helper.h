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

#pragma once

#include <glog/logging.h>

#include <fstream>
#if !defined(_WIN32)
#include <sys/time.h>
#endif
#include <algorithm>
#include <chrono>  // NOLINT
#include <functional>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/memory/stats.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/phi/backends/dynload/port.h"

extern std::string paddle::framework::DataTypeToString(
    const framework::proto::VarType::Type type);

namespace paddle {
namespace inference {

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

inline PaddleDType ConvertToPaddleDType(
    paddle::framework::proto::VarType::Type type) {
  if (type == paddle::framework::proto::VarType::FP32) {
    return PaddleDType::FLOAT32;
  } else if (type == paddle::framework::proto::VarType::INT64) {
    return PaddleDType::INT64;
  } else if (type == paddle::framework::proto::VarType::INT32) {
    return PaddleDType::INT32;
  } else if (type == paddle::framework::proto::VarType::UINT8) {
    return PaddleDType::UINT8;
  } else {
    PADDLE_THROW(paddle::platform::errors::Unimplemented(
        "The paddle dtype convert function only supports FLOAT32, INT64, INT32 "
        "and UINT8 now. But "
        "we get %d here.",
        static_cast<int>(type)));
    return PaddleDType::FLOAT32;
  }
}

using paddle::framework::DataTypeToString;

// Timer for timer
class Timer {
 public:
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point startu;

  void tic() { start = std::chrono::high_resolution_clock::now(); }
  double toc() {
    startu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(startu -
                                                                  start);
    double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
    return used_time_ms;
  }
};

static int GetUniqueId() {
  static int id = 0;
  return id++;
}

static void split(const std::string &str,
                  char sep,
                  std::vector<std::string> *pieces,
                  bool ignore_null = true) {
  pieces->clear();
  if (str.empty()) {
    if (!ignore_null) {
      pieces->push_back(str);
    }
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}

template <typename T>
static T convert(const std::string &item,
                 std::function<T(const std::string &item)> func) {
  T res;
  try {
    res = func(item);
  } catch (std::invalid_argument &e) {
    std::string message =
        "invalid_argument exception when try to convert : " + item;
    LOG(ERROR) << message;
    PADDLE_THROW(platform::errors::InvalidArgument(
        "invalid_argument exception when try to convert %s.", item));
  } catch (std::out_of_range &e) {
    std::string message =
        "out_of_range exception when try to convert : " + item;
    LOG(ERROR) << message;
    PADDLE_THROW(platform::errors::InvalidArgument(
        "out_of_range exception when try to convert %s.", item));
  } catch (...) {
    std::string message = "unexpected exception when try to convert " + item;
    LOG(ERROR) << message;
    PADDLE_THROW(platform::errors::InvalidArgument(
        "unexpected exception when try to convert %s.", item));
  }
  return res;
}

static void split_to_float(const std::string &str,
                           char sep,
                           std::vector<float> *fs) {
  std::vector<std::string> pieces;
  split(str, sep, &pieces);
  std::transform(pieces.begin(),
                 pieces.end(),
                 std::back_inserter(*fs),
                 [](const std::string &v) {
                   return convert<float>(v, [](const std::string &item) {
                     return std::stof(item);
                   });
                 });
}
static void split_to_int64(const std::string &str,
                           char sep,
                           std::vector<int64_t> *is) {
  std::vector<std::string> pieces;
  split(str, sep, &pieces);
  std::transform(pieces.begin(),
                 pieces.end(),
                 std::back_inserter(*is),
                 [](const std::string &v) {
                   return convert<int64_t>(v, [](const std::string &item) {
                     return std::stoll(item);
                   });
                 });
}
static void split_to_int(const std::string &str,
                         char sep,
                         std::vector<int> *is) {
  std::vector<std::string> pieces;
  split(str, sep, &pieces);
  std::transform(pieces.begin(),
                 pieces.end(),
                 std::back_inserter(*is),
                 [](const std::string &v) {
                   return convert<int>(v, [](const std::string &item) {
                     return std::stoi(item);
                   });
                 });
}
template <typename T>
std::string to_string(const std::vector<T> &vec) {
  std::stringstream ss;
  for (const auto &c : vec) {
    ss << c << " ";
  }
  return ss.str();
}
template <>
std::string to_string<std::vector<float>>(
    const std::vector<std::vector<float>> &vec);

template <>
std::string to_string<std::vector<std::vector<float>>>(
    const std::vector<std::vector<std::vector<float>>> &vec);

template <typename T>
int VecReduceToInt(const std::vector<T> &v) {
  return std::accumulate(v.begin(), v.end(), 1, [](T a, T b) { return a * b; });
}

template <typename T>
void CheckAssignedData(const std::vector<std::vector<T>> &data,
                       const int num_elems) {
  int num = 0;
  for (auto it = data.begin(); it != data.end(); ++it) {
    num += (*it).size();
  }
  PADDLE_ENFORCE_EQ(
      num,
      num_elems,
      platform::errors::OutOfRange(
          "The number of elements out of bounds. "
          "Expected number of elements = %d. But received %d. Suggested Fix: "
          "If the tensor is expected to assign %d elements, check the number "
          "of elements of your 'infer_data'.",
          num_elems,
          num,
          num_elems));
}

template <typename T>
static void TensorAssignData(PaddleTensor *tensor,
                             const std::vector<std::vector<T>> &data) {
  // Assign buffer
  int num_elems = VecReduceToInt(tensor->shape);
  CheckAssignedData(data, num_elems);
  tensor->data.Resize(sizeof(T) * num_elems);
  int c = 0;
  for (const auto &f : data) {
    for (T v : f) {
      static_cast<T *>(tensor->data.data())[c++] = v;
    }
  }
}

template <typename T>
static void TensorAssignData(PaddleTensor *tensor,
                             const std::vector<std::vector<T>> &data,
                             const std::vector<size_t> &lod) {
  int size = lod[lod.size() - 1];
  tensor->shape.assign({size, 1});
  tensor->lod.assign({lod});
  TensorAssignData(tensor, data);
}

template <typename T>
static void ZeroCopyTensorAssignData(ZeroCopyTensor *tensor,
                                     const std::vector<std::vector<T>> &data) {
  auto *ptr = tensor->mutable_data<T>(PaddlePlace::kCPU);
  int c = 0;
  for (const auto &f : data) {
    for (T v : f) {
      ptr[c++] = v;
    }
  }
}

template <typename T>
static void ZeroCopyTensorAssignData(ZeroCopyTensor *tensor,
                                     const PaddleBuf &data) {
  auto *ptr = tensor->mutable_data<T>(PaddlePlace::kCPU);
  for (size_t i = 0; i < data.length() / sizeof(T); i++) {
    ptr[i] = *(reinterpret_cast<T *>(data.data()) + i);
  }
}

static bool CompareTensor(const PaddleTensor &a, const PaddleTensor &b) {
  if (a.dtype != b.dtype) {
    LOG(ERROR) << "dtype not match";
    return false;
  }

  if (a.lod.size() != b.lod.size()) {
    LOG(ERROR) << "lod not match";
    return false;
  }
  for (size_t i = 0; i < a.lod.size(); i++) {
    if (a.lod[i].size() != b.lod[i].size()) {
      LOG(ERROR) << "lod not match";
      return false;
    }
    for (size_t j = 0; j < a.lod[i].size(); j++) {
      if (a.lod[i][j] != b.lod[i][j]) {
        LOG(ERROR) << "lod not match";
        return false;
      }
    }
  }

  if (a.shape.size() != b.shape.size()) {
    LOG(INFO) << "shape not match";
    return false;
  }
  for (size_t i = 0; i < a.shape.size(); i++) {
    if (a.shape[i] != b.shape[i]) {
      LOG(ERROR) << "shape not match";
      return false;
    }
  }

  auto *adata = static_cast<float *>(a.data.data());
  auto *bdata = static_cast<float *>(b.data.data());
  for (int i = 0; i < VecReduceToInt(a.shape); i++) {
    if (adata[i] != bdata[i]) {
      LOG(ERROR) << "data not match";
      return false;
    }
  }
  return true;
}

static std::string DescribeTensor(const PaddleTensor &tensor,
                                  int max_num_of_data = 15) {
  std::stringstream os;
  os << "Tensor [" << tensor.name << "]\n";
  os << " - type: ";
  switch (tensor.dtype) {
    case PaddleDType::FLOAT32:
      os << "float32";
      break;
    case PaddleDType::INT64:
      os << "int64";
      break;
    case PaddleDType::INT32:
      os << "int32";
      break;
    default:
      os << "unset";
  }
  os << '\n';

  os << " - shape: " << to_string(tensor.shape) << '\n';
  os << " - lod: ";
  for (auto &l : tensor.lod) {
    os << to_string(l) << "; ";
  }
  os << "\n";
  os << " - memory length: " << tensor.data.length();
  os << "\n";

  os << " - data: ";
  int dim = VecReduceToInt(tensor.shape);
  float *pdata = static_cast<float *>(tensor.data.data());
  for (int i = 0; i < dim; i++) {
    os << pdata[i] << " ";
  }
  os << '\n';
  return os.str();
}

static std::string DescribeZeroCopyTensor(const ZeroCopyTensor &tensor) {
  std::stringstream os;
  os << "Tensor [" << tensor.name() << "]\n";

  os << " - shape: " << to_string(tensor.shape()) << '\n';
  os << " - lod: ";
  for (auto &l : tensor.lod()) {
    os << to_string(l) << "; ";
  }
  os << "\n";
  PaddlePlace place;
  int size;
  const auto *data = tensor.data<float>(&place, &size);
  os << " - numel: " << size;
  os << "\n";
  os << " - data: ";
  for (int i = 0; i < size; i++) {
    os << data[i] << " ";
  }
  return os.str();
}

static void PrintTime(int batch_size,
                      int repeat,
                      int num_threads,
                      int tid,
                      double batch_latency,
                      int epoch = 1,
                      const framework::proto::VarType::Type data_type =
                          framework::proto::VarType::FP32) {
  PADDLE_ENFORCE_GT(
      batch_size,
      0,
      platform::errors::InvalidArgument("Non-positive batch size."));
  double sample_latency = batch_latency / batch_size;
  LOG(INFO) << "====== threads: " << num_threads << ", thread id: " << tid
            << " ======";
  LOG(INFO) << "====== batch size: " << batch_size << ", iterations: " << epoch
            << ", repetitions: " << repeat << " ======";
  LOG(INFO) << "====== batch latency: " << batch_latency
            << "ms, number of samples: " << batch_size * epoch
            << ", sample latency: " << sample_latency
            << "ms, fps: " << 1000.f / sample_latency
            << ", data type: " << DataTypeToString(data_type) << " ======";
}

static bool IsFileExists(const std::string &path) {
  std::ifstream file(path);
  bool exists = file.is_open();
  file.close();
  return exists;
}

void RegisterAllCustomOperator();

static inline double ToMegaBytes(size_t bytes) {
  return static_cast<double>(bytes) / (1 << 20);
}

static inline void DisplayMemoryInfo(platform::Place place,
                                     const std::string &hint) {
#ifdef PADDLE_WITH_CUDA
  // size_t free, total;
  // cudaSetDevice(place.GetDeviceId());
  // cudaMemGetInfo(&free, &total);
  // VLOG(1) << "[" << ToMegaBytes(total - free) << "MB/" << ToMegaBytes(total)
  // << "MB]";

  VLOG(1) << hint << " : [gpu current allocated memory: "
          << ToMegaBytes(paddle::memory::DeviceMemoryStatCurrentValue(
                 "Allocated", place.GetDeviceId()))
          << "MB], [gpu current reserved memory: "
          << ToMegaBytes(paddle::memory::DeviceMemoryStatCurrentValue(
                 "Reserved", place.GetDeviceId()))
          << "MB], [gpu peak allocated memory: "
          << ToMegaBytes(paddle::memory::DeviceMemoryStatPeakValue(
                 "Allocated", place.GetDeviceId()))
          << "MB], [gpu peak reserved memory: "
          << ToMegaBytes(paddle::memory::DeviceMemoryStatPeakValue(
                 "Reserved", place.GetDeviceId()))
          << "MB]";
#endif
  VLOG(1)
      << hint << " : [cpu current allocated memory: "
      << ToMegaBytes(paddle::memory::HostMemoryStatCurrentValue("Allocated", 0))
      << "MB], [cpu current reserved memory: "
      << ToMegaBytes(paddle::memory::HostMemoryStatCurrentValue("Reserved", 0))
      << "MB], [cpu peak allocated memory: "
      << ToMegaBytes(paddle::memory::HostMemoryStatPeakValue("Allocated", 0))
      << "MB], [cpu peak reserved memory: "
      << ToMegaBytes(paddle::memory::HostMemoryStatPeakValue("Reserved", 0))
      << "MB]";
}

}  // namespace inference
}  // namespace paddle
