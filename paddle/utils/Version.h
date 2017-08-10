/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <stddef.h>
#include <iostream>
#include "Common.h"

namespace paddle {

/**
 * namespace paddle::version
 * Some constexpr to detect paddle version.
 *    use paddle_trainer --version to print version information.
 *
 * Possible output as follow:
 * paddle version:
 *    withGpu: false
 *    withAvx: false
 *    withPyDataProvider: true
 *    withTimer: false
 *    withFpga: false
 *    real byte size: 4
 */

namespace version {

/**
 * @brief print paddle version and exit when --version flag setted. Otherwise,
 * do nothing.
 */
void printVersion();

void printVersion(std::ostream& os);
/**
 * @brief isWithGpu
 * @return return true if paddle compiled with GPU
 */
constexpr bool isWithGpu() {
#ifdef PADDLE_ONLY_CPU
  return false;
#else
  return true;
#endif
}

/**
 * @brief isWithPyDataProvider
 * @return return true if paddle compiled with PyDataProvider
 *
 * @note: A complete python interpreter is embeded into paddle binary if paddle
 * is compiled with PyDataProvider. Then the config parser just invoke python
 * method. Otherwise, ConfigParser just serializes config into protobuf, and
 * pass to C++ by using stdio.
 */
constexpr bool isWithPyDataProvider() {
#ifdef PADDLE_NO_PYTHON
  return false;
#else
  return true;
#endif
}

/**
 * @brief isWithTimer
 * @return true if paddle compiled with timer.
 */
constexpr bool isWithTimer() {
#ifdef PADDLE_DISABLE_TIMER
  return false;
#else
  return true;
#endif
}

/**
 * @brief isWithAvx
 * @return true if paddle compiled with AVX instructs.
 */
constexpr bool isWithAvx() {
#ifdef __AVX__
  return true;
#else
  return false;
#endif
}

/**
 * @brief isWithFpga
 * @return true if paddle compiled with FPGA for prediction.
 */
constexpr bool isWithFpga() {
#ifdef PADDLE_USE_FPGA
  return true;
#else
  return false;
#endif
}

/**
 * @brief sizeofReal
 * @return return the byte size of real
 */
constexpr size_t sizeofReal() { return sizeof(real); }

/**
 * @brief isPaddleUseDouble
 * @return true if paddle compiled with double precision.
 */
constexpr bool isPaddleUseDouble() { return sizeofReal() == sizeof(double); }

/**
 * @brief isPaddleUseFloat
 * @return true if paddle compiled with float precision
 */
constexpr bool isPaddleUseFloat() { return sizeofReal() == sizeof(float); }

}  //  namespace version

}  //  namespace paddle
