//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

template <typename T>
class AssignValueKernel : public framework::OpKernel<T> {
 public:
  //  virtual void Compute(const framework::ExecutionContext& ctx) const {
  //    auto shape = ctx.Attr<std::vector<int>>("shape");
  //    auto* out = ctx.Output<framework::Tensor>("Out");
  //    int dtype = ctx.Attr<int>("dtype");
  //    std::vector<T> values;
  //
  //    const char* value_name = nullptr;
  //    switch (dtype) {
  //      case framework::proto::VarType::BOOL:
  //        VLOG(4) << " -------- bool -------" << std::endl;
  //        value_name = "bool_values";
  //        values = ctx.Attr<std::vector<T>>(value_name);
  //        break;
  //      case framework::proto::VarType::INT32:
  //        VLOG(4) << " -------- int32 -------" << std::endl;
  //        value_name = "int32_values";
  //        values = ctx.Attr<std::vector<T>>(value_name);
  //        break;
  //      case framework::proto::VarType::INT64:
  //        VLOG(4) << " -------- int64 -------" << std::endl;
  //        value_name = "int64_values";
  //        values = ctx.Attr<std::vector<T>>(value_name);
  //
  //        break;
  //      case framework::proto::VarType::FP32:
  //        VLOG(4) << " -------- float32 -------" << std::endl;
  //        value_name = "fp32_values";
  //        values = ctx.Attr<std::vector<T>>(value_name);
  //
  //        break;
  ////      case framework::proto::VarType::FP64:
  ////        VLOG(4) << " -------- float64 -------" << std::endl;
  ////        value_name = "fp64_values";
  ////        values = ctx.Attr<std::vector<float>>("fp64_values");
  ////        //            values = static_cast<std::vector<T>>(values,
  ////        //            (std::vector<T>*)nullptr);
  ////        //            values =
  ////        //
  /// static_cast<std::vector<T>>(ctx.Attr<std::vector<float>>(value_name));
  ////        //            std::vector<int> sections =
  /// static_cast<std::vector<int>>(
  ////        // ctx->Attrs().Get<std::vector<int>>("sections"));
  ////
  ////        break;
  //      //        case framework::proto::VarType::BOOL:
  //      //            value_name = "bool_values";
  //      //            break;
  //      default:
  //        PADDLE_THROW("Unsupported dtype for assign_value_op: %d", dtype);
  //        break;
  //    }
  //
  //    VLOG(4) << " -------- 1 -------" << std::endl;
  //    //    auto values = ctx.Attr<std::vector<T>>(value_name);
  //    VLOG(4) << " -------- 2 -------" << std::endl;
  //    framework::TensorFromVector(values, ctx.device_context(), out);
  //    VLOG(4) << " -------- 3 -------" << std::endl;
  //    out->Resize(framework::make_ddim(shape));
  //    VLOG(4) << " -------- end -------" << std::endl;
  //  }
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto shape = ctx.Attr<std::vector<int>>("shape");
    auto* out = ctx.Output<framework::Tensor>("Out");
    int dtype = ctx.Attr<int>("dtype");

    const char* value_name = nullptr;
    switch (dtype) {
      case framework::proto::VarType::BOOL:
        VLOG(4) << " -------- bool -------" << std::endl;
        value_name = "bool_values";

        break;
      case framework::proto::VarType::INT32:
        VLOG(4) << " -------- int32 -------" << std::endl;
        value_name = "int32_values";
        break;
      case framework::proto::VarType::INT64:
        VLOG(4) << " -------- int64 -------" << std::endl;
        value_name = "int64_values";

        break;
      case framework::proto::VarType::FP32:
        VLOG(4) << " -------- float32 -------" << std::endl;
        value_name = "fp32_values";

        break;
      default:
        PADDLE_THROW("Unsupported dtype for assign_value_op: %d", dtype);
        break;
    }

    auto values = ctx.Attr<std::vector<T>>(value_name);
    framework::TensorFromVector(values, ctx.device_context(), out);
    out->Resize(framework::make_ddim(shape));
  }
};

}  // namespace operators
}  // namespace paddle
