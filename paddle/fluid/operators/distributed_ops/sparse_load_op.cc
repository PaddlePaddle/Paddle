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

#include <future>  // NOLINT
#include <ostream>

#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/distributed/communicator.h"
#include "paddle/fluid/operators/distributed/communicator_common.h"
#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/parameter_send.h"
#include "paddle/fluid/operators/distributed_ops/send_recv_util.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SparseLoadKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    auto filename = ctx.Attr<std::string>("file_path");
    std::ifstream fin(filename, std::ios::binary);
    PADDLE_ENFORCE_EQ(static_cast<bool>(fin), true,
                      platform::errors::Unavailable(
                          "Load operator fail to open file %s, please check "
                          "whether the model file is complete or damaged.",
                          filename));
    auto name = ctx.OutputNames("Out")[0];
    VLOG(1) << "Sparse Load Var name: " << name;
    auto *out_var = ctx.OutputVar("Out");
    VLOG(1) << "Sparse Load get out_var";
    PADDLE_ENFORCE_NOT_NULL(
        out_var, platform::errors::InvalidArgument(
                     "The variable %s to be loaded cannot be found.", name));

    if (out_var->IsType<framework::LoDTensor>()) {
      VLOG(1) << "Sparse Load LoadLodTensor";
      LoadLodTensor(fin, place, out_var, ctx);
    } else if (out_var->IsType<framework::SelectedRows>()) {
      VLOG(1) << "Sparse Load LoadSelectedRows";
      LoadSelectedRows(fin, place, out_var, ctx);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Load operator only supports loading LoDTensor and SelectedRows "
          "variable, %s has wrong type",
          name));
    }
  }

  void LoadLodTensor(std::istream &fin, const platform::Place &place,
                     framework::Variable *var,
                     const framework::ExecutionContext &ctx) const {
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);
    auto *tensor = var->GetMutable<framework::LoDTensor>();

    auto node_index = ctx.Attr<int64_t>("node_index");
    auto node_num = ctx.Attr<int64_t>("node_num");
    auto shape = ctx.Attr<std::vector<int64_t>>("shape");
    VLOG(1) << "Sparse LoadLodTensor node_num" << node_num;
    VLOG(1) << "Sparse LoadLodTensor node_index" << node_index;
    VLOG(1) << "Sparse LoadLodTensor shape[0]" << shape[0];
    PADDLE_ENFORCE_GE(node_index, 0, platform::errors::InvalidArgument(
                                         "node_num great than or equal to 0"));
    PADDLE_ENFORCE_GE(node_num, 1, platform::errors::InvalidArgument(
                                       "node_num great than or equal to 1"));
    DeserializeFromStream(fin, tensor, dev_ctx, node_index, node_num, shape);
  }

  void LoadSelectedRows(std::istream &fin, const platform::Place &place,
                        framework::Variable *var,
                        const framework::ExecutionContext &ctx) const {
    auto *selectedRows = var->GetMutable<framework::SelectedRows>();
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);
    auto node_index = ctx.Attr<int64_t>("node_index");
    auto node_num = ctx.Attr<int64_t>("node_num");
    auto shape = ctx.Attr<std::vector<int64_t>>("shape");
    PADDLE_ENFORCE_GE(node_index, 0, platform::errors::InvalidArgument(
                                         "node_num great than or equal to 0"));
    PADDLE_ENFORCE_GE(node_num, 1, platform::errors::InvalidArgument(
                                       "node_num great than or equal to 1"));
    VLOG(1) << "Sparse LoadSelectedRows node_num" << node_num;
    VLOG(1) << "Sparse LoadSelectedRows node_index" << node_index;
    VLOG(1) << "Sparse LoadSelectedRows shape[0]" << shape[0];
    DeserializeFromStream(fin, selectedRows, dev_ctx, node_index, node_num,
                          shape);
  }
};

class SparseLoadOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::OpKernelType kt = framework::OpKernelType(
        framework::proto::VarType::FP32, ctx.GetPlace());
    return kt;
  }
};

class SparseLoadOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddOutput("Out", "The LoDTensor / SelectedRows need to be loaded");
    AddAttr<std::string>("file_path",
                         R"(Variable will be loaded from "file_path")")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
    AddAttr<int64_t>("node_index", "role id from 0 ~ node_num.").SetDefault(0);
    AddAttr<int64_t>("node_num", "role nums which need load current varibale.")
        .SetDefault(0);
    AddAttr<std::vector<int64_t>>("shape",
                                  "(vector<int64_t>) The shape of the output")
        .SetDefault({});
    AddComment(R"DOC(
    SparseLoad OP, Load embedding on parameter server
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(sparse_load, ops::SparseLoadOp, ops::SparseLoadOpMaker);

REGISTER_OP_CPU_KERNEL(
    sparse_load,
    ops::SparseLoadKernel<paddle::platform::CPUDeviceContext, float>)
