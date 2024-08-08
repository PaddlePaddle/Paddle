// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_MKLML
#include <omp.h>
#endif
#include <fstream>

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#define MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define MKDIR(path) mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)
#endif

namespace phi {
namespace funcs {

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
HOSTDEVICE bool NeedPrint(MT max_value UNUSED,
                          MT min_value UNUSED,
                          int check_nan_inf_level) {
  if (check_nan_inf_level >= 3) {
    return true;
  }
  return false;
}

template <typename T>
HOSTDEVICE static void SaveStatsAndValues(int64_t num_nan,
                                          int64_t num_inf,
                                          int64_t num_zero,
                                          T max_value,
                                          T min_value,
                                          T mean_value,
                                          int64_t* stats_ptr,
                                          float* values_ptr) {
  if (stats_ptr) {
    stats_ptr[0] = num_nan;
    stats_ptr[1] = num_inf;
    stats_ptr[2] = num_zero;
  }
  if (values_ptr) {
    values_ptr[0] = static_cast<float>(max_value);
    values_ptr[1] = static_cast<float>(min_value);
    values_ptr[2] = static_cast<float>(mean_value);
  }
}

HOSTDEVICE static void PrintAndThrowError(const char* debug_info,
                                          int64_t num_nan,
                                          int64_t num_inf,
                                          int64_t num_zero) {
#if !defined(__HIPCC__) && !defined(__CUDA_ARCH__)
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "There are NAN or INF (num_nan=%lld, num_inf=%lld, num_zero=%lld) in "
      "%s.",
      static_cast<long long>(num_nan),   // NOLINT
      static_cast<long long>(num_inf),   // NOLINT
      static_cast<long long>(num_zero),  // NOLINT
      debug_info));
#endif
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
      PrintAndThrowError(debug_info, num_nan, num_inf, num_zero);
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
void WriteToFileForDifferentLevel(const char* debug_info,
                                  int64_t numel,
                                  int64_t num_nan,
                                  int64_t num_inf,
                                  int64_t num_zero,
                                  MT max_value,
                                  MT min_value,
                                  MT mean_value,
                                  int check_nan_inf_level,
                                  const std::string& log_name,
                                  const std::string output_dir) {
  MKDIR(output_dir.c_str());
  std::string filename = output_dir + "worker_" + log_name;
  std::ofstream outfile(filename, std::ios::app);
  PADDLE_ENFORCE_EQ(outfile.is_open(),
                    true,
                    common::errors::Unavailable(
                        "Fail to open output file %s, please check the "
                        "specified output_dir (%s).",
                        filename,
                        output_dir));

  if (num_nan > 0 || num_inf > 0) {
    outfile << "[PRECISION] [ERROR] in " << debug_info
            << ", numel=" << static_cast<long long>(numel)        // NOLINT
            << ", num_nan=" << static_cast<long long>(num_nan)    // NOLINT
            << ", num_inf=" << static_cast<long long>(num_inf)    // NOLINT
            << ", num_zero=" << static_cast<long long>(num_zero)  // NOLINT
            << std::scientific << std::setprecision(6)
            << ", max=" << static_cast<float>(max_value)
            << ", min=" << static_cast<float>(min_value)
            << ", mean=" << static_cast<float>(mean_value) << std::endl;
  } else if (phi::funcs::NeedPrint<T, MT>(
                 max_value, min_value, check_nan_inf_level)) {
    outfile << "[PRECISION] in " << debug_info
            << ", numel=" << static_cast<long long>(numel)        // NOLINT
            << ", num_zero=" << static_cast<long long>(num_zero)  // NOLINT
            << std::scientific << std::setprecision(6)
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
  std::string dtype_str;
  phi::DataType dtype = phi::CppTypeToDataType<T>::Type();
  if (dtype == DataType::FLOAT32) {
    dtype_str = "fp32";
  } else if (dtype == DataType::FLOAT64) {
    dtype_str = "fp64";
  } else if (dtype == DataType::FLOAT16) {
    dtype_str = "fp16";
  } else if (dtype == DataType::BFLOAT16) {
    dtype_str = "bf16";
  }

  std::stringstream ss;
  if (place.GetType() == phi::AllocationType::GPU) {
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
static void CheckNumericsCpuImpl(const T* value_ptr,
                                 const int64_t numel,
                                 const std::string& cpu_hint_str,
                                 const int check_nan_inf_level,
                                 const std::string log_name,
                                 const std::string output_dir,
                                 int64_t* stats_ptr,
                                 float* values_ptr) {
  using MT = typename phi::dtype::template MPTypeTrait<T>::Type;

#ifdef PADDLE_WITH_MKLML
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

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel num_threads(num_threads)
#endif
  {
#ifdef PADDLE_WITH_MKLML
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
      if (value == static_cast<MT>(0)) {
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

  SaveStatsAndValues<MT>(num_nan,
                         num_inf,
                         num_zero,
                         max_value,
                         min_value,
                         mean_value,
                         stats_ptr,
                         values_ptr);

  // Write log to file
  if (output_dir.size() > 0) {
    WriteToFileForDifferentLevel<T, MT>(cpu_hint_str.c_str(),
                                        numel,
                                        num_nan,
                                        num_inf,
                                        num_zero,
                                        max_value,
                                        min_value,
                                        mean_value,
                                        check_nan_inf_level,
                                        log_name,
                                        output_dir);
  } else {
    PrintForDifferentLevel<T, MT>(cpu_hint_str.c_str(),
                                  numel,
                                  num_nan,
                                  num_inf,
                                  num_zero,
                                  max_value,
                                  min_value,
                                  mean_value,
                                  check_nan_inf_level);
  }
}

template <
    typename T,
    std::enable_if_t<std::is_same<T, phi::dtype::complex<float>>::value ||
                         std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
void CheckNumericsCpuImpl(const T* value_ptr,
                          const int64_t numel,
                          const std::string& cpu_hint_str,
                          const int check_nan_inf_level,
                          const std::string log_name,
                          const std::string output_dir,
                          int64_t* stats_ptr,
                          float* values_ptr) {
  using RealType = typename T::value_type;

  RealType real_sum = 0.0f, imag_sum = 0.0f;

#ifdef PADDLE_WITH_MKLML
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
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "There are NAN or INF in %s.", cpu_hint_str));
  }
}

}  // namespace funcs
}  // namespace phi
