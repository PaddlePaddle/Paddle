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

#include "paddle/utils/CustomStackTrace.h"
#include "paddle/utils/StringUtil.h"
#include "paddle/utils/Util.h"

int main(int argc, char** argv) {
  paddle::initMain(argc, argv);

  for (size_t i = 0; i < 1000; ++i) {
    paddle::gLayerStackTrace.push("layer_" + paddle::str::to_string(i));
    if (i == 998) {
      throw "Unhandle exception";
    }
  }

  return 0;
}
