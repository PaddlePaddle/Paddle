/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/var_type.h"

namespace paddle {
namespace framework {
class LoDTensor;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {

class TensorFormatter {
 public:
  TensorFormatter() {}

  std::string Format(const framework::LoDTensor& print_tensor,
                     const std::string& tensor_name = "",
                     const std::string& message = "");

  template <typename T>
  void FormatData(const framework::LoDTensor& print_tensor,
                  std::stringstream& log_stream);

  void Print(const framework::LoDTensor& print_tensor,
             const std::string& tensor_name = "",
             const std::string& message = "");

  void SetPrintTensorType(bool print_tensor_type);
  void SetPrintTensorShape(bool print_tensor_shape);
  void SetPrintTensorLod(bool print_tensor_lod);
  void SetPrintTensorLayout(bool print_tensor_layout);
  void SetSummarize(int64_t summarize);

 private:
  int64_t summarize_ = -1;
  bool print_tensor_type_ = true;
  bool print_tensor_shape_ = true;
  bool print_tensor_lod_ = true;
  bool print_tensor_layout_ = true;
};

}  // namespace operators
}  // namespace paddle
