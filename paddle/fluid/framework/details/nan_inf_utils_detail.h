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
#include <fstream>
#include <iostream>
#include <string>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#define MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define MKDIR(path) mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)
#endif

DECLARE_int32(check_nan_inf_level);
namespace paddle {
namespace framework {
namespace details {

void SetNanInfDebugPath(const std::string& nan_inf_path);

std::string GetNanPath();

template <typename T,
          typename MT,
          std::enable_if_t<std::is_same<T, float>::value, bool> = true>
HOSTDEVICE bool NeedPrint(MT max_value, MT min_value, int check_nan_inf_level) {
  if (check_nan_inf_level >= 3) {
    return true;
  } else if (check_nan_inf_level >= 2) {
    MT fp16_max =
        static_cast<MT>(std::numeric_limits<phi::dtype::float16>::max());
    return max_value > fp16_max || min_value < -fp16_max;
  }
  return false;
}

template <typename T,
          typename MT,
          std::enable_if_t<!std::is_same<T, float>::value, bool> = true>
HOSTDEVICE bool NeedPrint(MT max_value, MT min_value, int check_nan_inf_level) {
  if (check_nan_inf_level >= 3) {
    return true;
  }
  return false;
}

template <typename T, typename MT>
HOSTDEVICE void PrintForDifferentLevel(const char* debug_info,
                                       int64_t numel,
                                       int64_t num_nan,
                                       int64_t num_inf,
                                       int64_t num_zero,
                                       MT max_value,
                                       MT min_value,
                                       MT mean_value,
                                       int check_nan_inf_level) {
  if (num_nan > 0 || num_inf > 0) {
    printf(
        "[PRECISION] [ERROR] in %s, numel=%lld, num_nan=%lld, "
        "num_inf=%lld, num_zero=%lld, max=%e, min=%e, mean=%e\n",
        debug_info,
        static_cast<long long>(numel),     // NOLINT
        static_cast<long long>(num_nan),   // NOLINT
        static_cast<long long>(num_inf),   // NOLINT
        static_cast<long long>(num_zero),  // NOLINT
        static_cast<float>(max_value),
        static_cast<float>(min_value),
        static_cast<float>(mean_value));
    if (check_nan_inf_level == 0) {
#if !(defined(__NVCC__) || defined(__HIPCC__))
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "There are NAN or INF (num_nan=%lld, num_inf=%lld, num_zero=%lld) in "
          "%s.",
          static_cast<long long>(num_nan),   // NOLINT
          static_cast<long long>(num_inf),   // NOLINT
          static_cast<long long>(num_zero),  // NOLINT
          debug_info));
#endif
    }
  } else if (NeedPrint<T, MT>(max_value, min_value, check_nan_inf_level)) {
    printf(
        "[PRECISION] in %s, numel=%lld, num_zero=%lld, max=%e, min=%e, "
        "mean=%e\n",
        debug_info,
        static_cast<long long>(numel),     // NOLINT
        static_cast<long long>(num_zero),  // NOLINT
        static_cast<float>(max_value),
        static_cast<float>(min_value),
        static_cast<float>(mean_value));
  }
}

template <typename T, typename MT>
void PrintForDifferentLevelFile(const char* debug_info,
                                int64_t numel,
                                int64_t num_nan,
                                int64_t num_inf,
                                int64_t num_zero,
                                MT max_value,
                                MT min_value,
                                MT mean_value,
                                int check_nan_inf_level,
                                const std::string& log_name) {
  int dev_id = 0;
#ifdef PADDLE_WITH_HIP
  hipGetDevice(&dev_id);
#elif PADDLE_WITH_CUDA
  cudaGetDevice(&dev_id);
#endif
  auto file_path = GetNanPath();
  MKDIR(file_path.c_str());
  std::string file_name = "worker_" + log_name + "." + std::to_string(dev_id);
  std::string path = file_path + file_name;
  std::ofstream outfile(path, std::ios::app);
  if (!outfile.is_open()) {
    return;
  }

  if (num_nan > 0 || num_inf > 0) {
    outfile << "[PRECISION] [ERROR] in " << debug_info
            << ", numel=" << static_cast<long long>(numel)        // NOLINT
            << ", num_nan=" << static_cast<long long>(num_nan)    // NOLINT
            << ", num_inf=" << static_cast<long long>(num_inf)    // NOLINT
            << ", num_zero=" << static_cast<long long>(num_zero)  // NOLINT
            << ", max=" << static_cast<float>(max_value)
            << ", min=" << static_cast<float>(min_value)
            << ", mean=" << static_cast<float>(mean_value) << std::endl;
  } else if (NeedPrint<T, MT>(max_value, min_value, check_nan_inf_level)) {
    outfile << "[PRECISION] in " << debug_info
            << ", numel=" << static_cast<long long>(numel)        // NOLINT
            << ", num_zero=" << static_cast<long long>(num_zero)  // NOLINT
            << ", max=" << static_cast<float>(max_value)
            << ", min=" << static_cast<float>(min_value)
            << ", mean=" << static_cast<float>(mean_value) << std::endl;
  }
  outfile.close();
}

template <typename T>
inline std::string GetCpuHintString(const std::string& op_type,
                                    const std::string& var_name,
                                    const phi::Place& place,
                                    int device_id = -1) {
  std::string dtype_str = DataTypeToString(DataTypeTrait<T>::DataType());
  if (dtype_str == "float") {
    dtype_str = "fp32";
  } else if (dtype_str == "double") {
    dtype_str = "fp64";
  } else if (dtype_str == "::paddle::platform::float16") {
    dtype_str = "fp16";
  } else if (dtype_str == "::paddle::platform::bfloat16") {
    dtype_str = "bf16";
  }

  std::stringstream ss;
  if (platform::is_gpu_place(place)) {
    ss << "[device=gpu:" << device_id << ", ";
  } else {
    ss << "[device=cpu, ";
  }
  ss << "op=" << op_type << ", tensor=" << var_name << ", dtype=" << dtype_str
     << "]";
  return ss.str();
}

template <
    typename T,
    std::enable_if_t<!std::is_same<T, phi::dtype::complex<float>>::value &&
                         !std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
static void CheckNanInfCpuImpl(const T* value_ptr,
                               const int64_t numel,
                               const std::string& cpu_hint_str,
                               const std::string log_name = "cpu") {
  using MT = typename phi::dtype::template MPTypeTrait<T>::Type;

#ifdef _OPENMP
  // Use maximum 4 threads to collect the nan and inf information.
  int num_threads = std::max(omp_get_num_threads(), 1);
  num_threads = std::min(num_threads, 4);
#else
  int num_threads = 1;
#endif

  std::vector<int64_t> thread_num_nan(num_threads, 0);
  std::vector<int64_t> thread_num_inf(num_threads, 0);
  std::vector<int64_t> thread_num_zero(num_threads, 0);
  std::vector<MT> thread_min_value(num_threads, static_cast<MT>(value_ptr[0]));
  std::vector<MT> thread_max_value(num_threads, static_cast<MT>(value_ptr[0]));
  std::vector<MT> thread_mean_value(num_threads, static_cast<MT>(0));

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
  {
#ifdef _OPENMP
    int64_t tid = omp_get_thread_num();
    int64_t chunk_size = (numel + num_threads - 1) / num_threads;
    int64_t begin = tid * chunk_size;
    int64_t end = chunk_size + begin > numel ? numel : chunk_size + begin;
#else
    int64_t tid = 0;
    int64_t begin = 0;
    int64_t end = numel;
#endif
    for (int64_t i = begin; i < end; ++i) {
      MT value = static_cast<MT>(value_ptr[i]);

      thread_min_value[tid] = std::min(thread_min_value[tid], value);
      thread_max_value[tid] = std::max(thread_max_value[tid], value);
      thread_mean_value[tid] += value / static_cast<MT>(numel);

      if (std::isnan(value)) {
        thread_num_nan[tid] += 1;
      } else if (std::isinf(value)) {
        thread_num_inf[tid] += 1;
      }
      if (value == 0) {
        thread_num_zero[tid] += 1;
      }
    }
  }

  int64_t num_nan = 0;
  int64_t num_inf = 0;
  int64_t num_zero = 0;
  MT min_value = thread_min_value[0];
  MT max_value = thread_max_value[0];
  MT mean_value = static_cast<MT>(0);
  for (int i = 0; i < num_threads; ++i) {
    num_nan += thread_num_nan[i];
    num_inf += thread_num_inf[i];
    num_zero += thread_num_zero[i];
    min_value = std::min(thread_min_value[i], min_value);
    max_value = std::max(thread_max_value[i], max_value);
    mean_value += thread_mean_value[i];
  }
  auto file_path = GetNanPath();
  // Write log to file
  if (file_path.size() > 0) {
    VLOG(4) << "[FLAGS_check_nan_inf_level=" << FLAGS_check_nan_inf_level
            << "]. Write log to " << file_path;
    PrintForDifferentLevelFile<T, MT>(cpu_hint_str.c_str(),
                                      numel,
                                      num_nan,
                                      num_inf,
                                      num_zero,
                                      max_value,
                                      min_value,
                                      mean_value,
                                      FLAGS_check_nan_inf_level,
                                      log_name);
    return;
  }

  PrintForDifferentLevel<T, MT>(cpu_hint_str.c_str(),
                                numel,
                                num_nan,
                                num_inf,
                                num_zero,
                                max_value,
                                min_value,
                                mean_value,
                                FLAGS_check_nan_inf_level);
}

template <
    typename T,
    std::enable_if_t<std::is_same<T, phi::dtype::complex<float>>::value ||
                         std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
void CheckNanInfCpuImpl(const T* value_ptr,
                        const int64_t numel,
                        const std::string& cpu_hint_str,
                        const std::string log_name = "cpu") {
  using RealType = typename T::value_type;

  RealType real_sum = 0.0f, imag_sum = 0.0f;

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : real_sum) reduction(+ : imag_sum)
#endif
  for (int64_t i = 0; i < numel; ++i) {
    T value = value_ptr[i];
    real_sum += (value.real - value.real);
    imag_sum += (value.imag - value.imag);
  }

  if (std::isnan(real_sum) || std::isinf(real_sum) || std::isnan(imag_sum) ||
      std::isinf(imag_sum)) {
    // hot fix for compile failed in gcc4.8
    // here also need print detail info of nan or inf later
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "There are NAN or INF in %s.", cpu_hint_str));
  }
}

template <typename DeviceContext>
struct TensorCheckerVisitor {
  TensorCheckerVisitor(const std::string& o,
                       const std::string& v,
                       const phi::DenseTensor& t,
                       const platform::Place& p)
      : op_type(o), var_name(v), tensor(t), place(p) {}

  template <typename T>
  void apply(
      typename std::enable_if<std::is_integral<T>::value>::type* = 0) const {
    VLOG(10) << var_name << " need not to check, it's type is not float point";
  }

  template <typename T>
  void apply(
      typename std::enable_if<
          std::is_floating_point<T>::value ||
          std::is_same<T, ::paddle::platform::complex<float>>::value ||
          std::is_same<T, ::paddle::platform::complex<double>>::value>::type* =
          0) const;

  std::string op_type;
  std::string var_name;
  const phi::DenseTensor& tensor;
  const platform::Place& place;
};

template <typename DeviceContext>
void tensor_check(const std::string& op_type,
                  const std::string& var_name,
                  const phi::DenseTensor& tensor,
                  const platform::Place& place);

}  // namespace details
}  // namespace framework
}  // namespace paddle
