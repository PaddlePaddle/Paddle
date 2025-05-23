// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#include <sstream>
#include <stdexcept>
#include "paddle/ap/include/axpr/abstract_list.h"
#include "paddle/ap/include/axpr/bool_helper.h"
#include "paddle/ap/include/axpr/bool_int_double_helper.h"
#include "paddle/ap/include/axpr/builtin_frame_util.h"
#include "paddle/ap/include/axpr/builtin_high_order_func_type.h"
#include "paddle/ap/include/axpr/callable_helper.h"
#include "paddle/ap/include/axpr/data_value_util.h"
#include "paddle/ap/include/axpr/exception_method_class.h"
#include "paddle/ap/include/axpr/interpreter.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/string_util.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/axpr/value_method_class.h"
#include "paddle/ap/include/fs/builtin_functions.h"
#include "paddle/ap/include/memory/guard.h"

namespace ap::axpr {

adt::Result<axpr::Value> DirName(const axpr::Value&,
                                 const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1)
      << adt::errors::TypeError{"dirname() takes 1 argument, but " +
                                std::to_string(args.size()) + "were given."};
  ADT_LET_CONST_REF(filepath, args.at(0).template CastTo<std::string>());
  std::size_t pos = filepath.find_last_of("/\\");
  if (pos == std::string::npos) return std::string{};
  return filepath.substr(0, pos);
}

adt::Result<axpr::Value> BaseName(const axpr::Value&,
                                  const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1)
      << adt::errors::TypeError{"basename() takes 1 argument, but " +
                                std::to_string(args.size()) + "were given."};
  ADT_LET_CONST_REF(filepath, args.at(0).template CastTo<std::string>());
  std::size_t pos = filepath.find_last_of("/\\");
  if (pos == std::string::npos) return filepath;
  return filepath.substr(pos + 1);
}

void ForceLink() {}

}  // namespace ap::axpr
