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

#include <string>

namespace paddle {
namespace framework {
namespace op_helpers {

/*
 * Generate the gradient variable's name of a forward varialbe.
 *
 * If a variable's name has a certain suffix, it means that the
 * variable is the gradient of another varibale.
 * e.g. Variable "x@GRAD" is the gradient of varibale "x".
 */
inline std::string GenGradName(const std::string& var) {
  static const std::string suffix{"@GRAD"};
  return var + suffix;
}

}  // namespace op_helpers
}  // namespace framework
}  // namespace paddle
