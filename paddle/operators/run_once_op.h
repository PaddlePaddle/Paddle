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
#include "paddle/framework/operator.h"

namespace paddle {
namespace operators {

class RunOnceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {
    bool run_once = Attr<bool>("run_once");
    if (run_once && has_run_) {
      return;
    }
    framework::OperatorWithKernel::Run(scope, dev_ctx);
    has_run_ = true;
  }

 private:
  mutable bool has_run_{false};
};

class RunOnceOpInfoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  RunOnceOpInfoMaker(framework::OpProto* proto,
                     framework::OpAttrChecker* op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddAttr<bool>("run_once",
                  "whether that operator run only once or run multiple times")
        .SetDefault(false);
  }
};

}  // namespace operators
}  // namespace paddle
