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

#include "Version.h"

#include <iomanip>
#include <numeric>
#include "Flags.h"
#include "Util.h"

DECLARE_bool(version);

namespace paddle {
namespace version {

void printVersion(std::ostream& os) {
#ifndef PADDLE_VERSION
#define PADDLE_VERSION "unknown"
#endif
// converts macro to string
// https://gcc.gnu.org/onlinedocs/cpp/Stringification.html
#define xstr(s) str(s)
#define str(s) #s

  os << "paddle version: " << xstr(PADDLE_VERSION) << std::endl
     << std::boolalpha << "\t"
     << "withGpu: " << version::isWithGpu() << std::endl
     << "\t"
     << "withAvx: " << version::isWithAvx() << std::endl
     << "\t"
     << "withPyDataProvider: " << version::isWithPyDataProvider() << std::endl
     << "\t"
     << "withTimer: " << version::isWithTimer() << std::endl
     << "\t"
     << "withFpga: " << version::isWithFpga() << std::endl
     << "\t"
     << "real byte size: " << version::sizeofReal() << std::endl
     << std::endl;
}

void printVersion() {
  if (FLAGS_version) {
    printVersion(std::cout);
    exit(0);
  }
}

}  //  namespace version
}  //  namespace paddle
