// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <tuple>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/eager/type_defs.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/utils/optional.h"
#include "paddle/utils/small_vector.h"

namespace egr {

using Tensor = paddle::Tensor;
using TupleOfTwoTensors = std::tuple<Tensor, Tensor>;
using TupleOfThreeTensors = std::tuple<Tensor, Tensor, Tensor>;
using TupleOfFourTensors = std::tuple<Tensor, Tensor, Tensor, Tensor>;
using TupleOfFiveTensors = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>;
using TupleOfSixTensors =
    std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;
using TupleOfTensorAndVector =
    std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>>;

void DumpTensorToFile(const std::string& api_unique,
                      const std::string& api_name,
                      const std::string& arg_type,
                      const std::string& arg_name,
                      const Tensor& tensor);

void DumpTensorToFile(const std::string& api_unique,
                      const std::string& api_name,
                      const std::string& arg_type,
                      const std::string& arg_name,
                      const paddle::optional<Tensor>& tensor);

void DumpTensorToFile(const std::string& api_unique,
                      const std::string& api_name,
                      const std::string& arg_type,
                      const std::string& arg_name,
                      const std::vector<Tensor>& tensors);

void DumpTensorToFile(const std::string& api_unique,
                      const std::string& api_name,
                      const std::string& arg_type,
                      const std::string& arg_name,
                      const paddle::optional<std::vector<Tensor>>& tensors);

struct TensorDumpVisitor {
  TensorDumpVisitor(const std::string& api_unique,
                    const std::string& api_name,
                    const std::string& arg_type,
                    const std::string& arg_name,
                    const std::string& adr_name,
                    const phi::DenseTensor& t)
      : api_unique(api_unique),
        api_name(api_name),
        arg_type(arg_type),
        arg_name(arg_name),
        adr_name(adr_name),
        tensor(t) {}

  template <typename T>
  void apply(
      typename std::enable_if<std::is_integral<T>::value>::type* = 0) const {
    VLOG(10) << adr_name << " need not to check, it's type is not float point";
  }

  template <typename T>
  void apply(
      typename std::enable_if<
          std::is_floating_point<T>::value ||
          std::is_same<T, ::paddle::platform::complex<float>>::value ||
          std::is_same<T, ::paddle::platform::complex<double>>::value>::type* =
          0) const;

  std::string api_unique;
  std::string api_name;
  std::string arg_type;
  std::string arg_name;
  std::string adr_name;
  const phi::DenseTensor& tensor;
};

void tensor_dump(const std::string& api_unique,
                 const std::string& api_name,
                 const std::string& arg_type,
                 const std::string& arg_name,
                 const std::string& adr_name,
                 const phi::DenseTensor& tensor);

}  // namespace egr
