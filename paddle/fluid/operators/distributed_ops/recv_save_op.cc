/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <stdint.h>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/version.h"
#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/parameter_recv.h"
#include "paddle/fluid/operators/distributed/rpc_common.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace operators {
class RecvSaveOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::proto::VarType::Type(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }
};

class RecvSaveOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(
Recv Save operator

This operator will serialize and write LoDTensor variable to file on disk.
)DOC");
    AddAttr<int>("dtype",
                 "(int, default 5 (FP32)) "
                 "Output data type")
        .SetDefault(framework::proto::VarType::FP32);

    AddAttr<bool>("overwrite",
                  "(boolean, default true)"
                  "Overwrite the output file if exist")
        .SetDefault(true);

    AddAttr<std::string>("file_path",
                         "(string)"
                         "The \"file_path\" where the variable will be saved.")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });

    AddAttr<std::vector<int64_t>>("shape",
                                  "(vector<int64_t>) The shape of the output")
        .SetDefault({});

    AddAttr<std::vector<std::string>>(
        "slice_varnames",
        "(string vector, default {}) "
        "sometimes we need to put received var in another name "
        "for example: we need var named 'moment_1@127.0.0.1:1001', "
        "and it real name on parameter server is 'moment_1'. ")
        .SetDefault({});

    AddAttr<std::vector<std::string>>(
        "remote_varnames",
        "(string vector, default {}) "
        "sometimes we need to put received var in another name "
        "for example: we need var named 'moment_1@127.0.0.1:1001', "
        "and it real name on parameter server is 'moment_1'. ")
        .SetDefault({});

    AddAttr<std::vector<std::string>>("slice_shapes",
                                      "(vector<int>) "
                                      "the length of each output along the "
                                      "specified axis.")
        .SetDefault({});

    AddAttr<std::vector<std::string>>("endpoints",
                                      "(string vector, default 127.0.0.1:6164)"
                                      "Server endpoints in the order of input "
                                      "variables for mapping")
        .SetDefault({});

    AddAttr<int>("trainer_id", "trainer id from 0 ~ worker_num.").SetDefault(0);
  }
};

template <typename DeviceContext, typename T>
class RecvSaveOpKernel : public framework::OpKernel<T> {
 private:
  void SerializeVersionToStream(std::ostream &os) const {
    {  // the 1st field, uint32_t version for LoDTensor
      os.write(reinterpret_cast<const char *>(&framework::kCurTensorVersion),
               sizeof(framework::kCurTensorVersion));
    }
    // the 2st field, LoD information
    // in this scene, skip LoD information.
    uint64_t size = 0;
    os.write(reinterpret_cast<const char *>(&size), sizeof(size));
  }

  void SerializeTensorHeaderToStream(
      std::ostream &os, const framework::proto::VarType::Type &type,
      const framework::DDim &dims) const {
    {  // the 1st field, uint32_t version
      constexpr uint32_t version = 0;
      os.write(reinterpret_cast<const char *>(&version), sizeof(version));
    }
    {  // the 2nd field, tensor description
      // int32_t  size
      // void*    protobuf message
      framework::proto::VarType::TensorDesc desc;
      desc.set_data_type(type);
      auto tensor_dims = framework::vectorize(dims);
      auto *pb_dims = desc.mutable_dims();
      pb_dims->Resize(static_cast<int>(tensor_dims.size()), 0);
      std::copy(tensor_dims.begin(), tensor_dims.end(), pb_dims->begin());
      int32_t size = desc.ByteSize();
      os.write(reinterpret_cast<const char *>(&size), sizeof(size));
      auto out = desc.SerializeAsString();
      os.write(out.data(), size);
    }
  }

  void SerializeTensorAppendToStream(std::ostream &os,
                                     const framework::Tensor &tensor) const {
    uint64_t size = tensor.numel() * framework::SizeOfType(tensor.type());
    auto *data_ptr = tensor.data<void>();

    PADDLE_ENFORCE_LT(size, std::numeric_limits<std::streamsize>::max(),
                      platform::errors::ResourceExhausted(
                          "tensor size %d overflow when writing tensor", size));
    os.write(static_cast<const char *>(data_ptr),
             static_cast<std::streamsize>(size));
  }

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();

    auto filename = ctx.Attr<std::string>("file_path");
    auto overwrite = ctx.Attr<bool>("overwrite");

    if (FileExists(filename) && !overwrite) {
      PADDLE_THROW(platform::errors::AlreadyExists(
          "%s is existed, cannot save to it when overwrite=false", filename));
    }

    MkDirRecursively(DirName(filename).c_str());

    auto origin_shape = ctx.Attr<std::vector<int64_t>>("shape");
    auto slice_shapes = ctx.Attr<std::vector<std::string>>("slice_shapes");
    auto slice_varnames = ctx.Attr<std::vector<std::string>>("slice_varnames");
    auto remote_varnames =
        ctx.Attr<std::vector<std::string>>("remote_varnames");
    auto endpoints = ctx.Attr<std::vector<std::string>>("endpoints");

    PADDLE_ENFORCE_EQ(slice_shapes.size(), slice_varnames.size(),
                      platform::errors::InvalidArgument(
                          "Expected attr len(slice_shapes) must be equal to "
                          "len(slice_varnames)"));

    PADDLE_ENFORCE_EQ(
        slice_shapes.size(), endpoints.size(),
        platform::errors::InvalidArgument(
            "Expected attr len(slice_shapes) must be equal to len(endpoints)"));

    auto data_type =
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));

    // it to save an output stream.
    std::ofstream fout(filename, std::ios::binary);
    PADDLE_ENFORCE_EQ(
        static_cast<bool>(fout), true,
        platform::errors::NotFound("Cannot open %s to write", filename));

    SerializeVersionToStream(fout);
    SerializeTensorHeaderToStream(fout, data_type,
                                  framework::make_ddim(origin_shape));

    framework::Scope &local_scope = ctx.scope().NewScope();

    auto trainer_id = ctx.Attr<int>("trainer_id");

    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &device_ctx = *pool.Get(place);

    distributed::RPCClient *rpc_client =
        distributed::RPCClient::GetInstance<RPCCLIENT_T>(trainer_id);

    for (size_t i = 0; i < slice_varnames.size(); i++) {
      auto &varname = slice_varnames[i];
      auto *var = local_scope.Var(varname);
      auto *tensor = var->GetMutable<framework::LoDTensor>();

      auto slice_string =
          string::split_string<std::string>(slice_shapes[i], ",");
      std::vector<int64_t> slice_shape;

      for (auto &dim : slice_string) {
        slice_shape.push_back(static_cast<int64_t>(std::stoull(dim)));
      }

      tensor->Resize(framework::make_ddim(slice_shape));

      distributed::VarHandlePtr ret;

      ret = rpc_client->AsyncGetVarNoBarrier(
          endpoints[i], device_ctx, local_scope, remote_varnames[i], varname);

      PADDLE_ENFORCE_NE(
          ret->Wait(), 0U,
          platform::errors::ExecutionTimeout(
              "rpc error when communication with %s", endpoints[i]));

      auto &c_tensor = var->Get<framework::LoDTensor>();

      SerializeTensorAppendToStream(fout, c_tensor);
      local_scope.EraseVars({varname});
    }

    fout.close();
    ctx.scope().DeleteScope(&local_scope);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(recv_save, ops::RecvSaveOp, ops::RecvSaveOpProtoMaker);

REGISTER_OP_CPU_KERNEL(
    recv_save, ops::RecvSaveOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::RecvSaveOpKernel<paddle::platform::CPUDeviceContext, double>,
    ops::RecvSaveOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::RecvSaveOpKernel<paddle::platform::CPUDeviceContext, int64_t>);
