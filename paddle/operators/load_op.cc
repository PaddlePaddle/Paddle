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

#include "paddle/framework/op_registry.h"

#include <fstream>

namespace paddle {
namespace operators {

class LoadOp : public framework::OperatorBase {
 public:
  LoadOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto filename = Attr<std::string>("file_path");
    std::ifstream fin(filename);
    PADDLE_ENFORCE(static_cast<bool>(fin), "Cannot open file %s for load op",
                   filename);

    auto out_var_name = Output("Out");
    auto *out_var = scope.FindVar(out_var_name);
    PADDLE_ENFORCE(out_var != nullptr, "Output variable %s cannot be found",
                   out_var_name);

    auto *tensor = out_var->GetMutable<framework::LoDTensor>();

    uint32_t version;
    fin.read(reinterpret_cast<char *>(&version), sizeof(version));
    PADDLE_ENFORCE_EQ(version, 0U, "Only version 0 is supported");
    framework::TensorDesc desc;
    {  // int32_t size
       // proto buffer
      int32_t size;
      fin.read(reinterpret_cast<char *>(&size), sizeof(size));
      std::unique_ptr<char[]> buf(new char[size]);
      fin.read(reinterpret_cast<char *>(buf.get()), size);
      PADDLE_ENFORCE(desc.ParseFromArray(buf.get(), size),
                     "Cannot parse tensor desc");
    }
    {  // read tensor
      std::vector<int64_t> dims;
      dims.reserve(static_cast<size_t>(desc.dims().size()));
      std::copy(desc.dims().begin(), desc.dims().end(),
                std::back_inserter(dims));
      tensor->Resize(framework::make_ddim(dims));

      void *buf;
      platform::Place cpu = platform::CPUPlace();
      switch (desc.data_type()) {
        case framework::FP32:
          buf = tensor->mutable_data<float>(cpu);
          break;
        case framework::FP64:
          buf = tensor->mutable_data<double>(cpu);
          break;
        case framework::INT32:
          buf = tensor->mutable_data<int>(cpu);
          break;
        case framework::INT64:
          buf = tensor->mutable_data<int64_t>(cpu);
          break;
        default:
          PADDLE_THROW("DataType %d not supported", desc.data_type());
      }
      fin.read(static_cast<char *>(buf), tensor->memory_size());
    }
    {  // read lod
      uint64_t lod_level;
      fin.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
      auto &lod = *tensor->mutable_lod();
      lod.resize(lod_level);
      for (uint64_t i = 0; i < lod_level; ++i) {
        uint64_t size;
        fin.read(reinterpret_cast<char *>(&size), sizeof(size));
        std::vector<size_t> tmp(size / sizeof(size_t));
        fin.read(reinterpret_cast<char *>(tmp.data()),
                 static_cast<std::streamsize>(size));
        lod[i] = tmp;
      }
    }

    auto place = dev_ctx.GetPlace();
    if (platform::is_gpu_place(place)) {
      // copy CPU to GPU
      framework::LoDTensor cpu_tensor;
      cpu_tensor.ShareDataWith(*tensor);
      cpu_tensor.set_lod(tensor->lod());

      // reset tensor
      out_var->Clear();
      tensor = out_var->GetMutable<framework::LoDTensor>();
      tensor->set_lod(cpu_tensor.lod());
      tensor->CopyFrom(cpu_tensor, place, dev_ctx);
    }
  }
};

class LoadOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LoadOpProtoMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddOutput("Out", "(Tensor) The tensor need to be loaded");
    AddAttr<std::string>("file_path",
                         "(string) "
                         "Variable will be loaded from \"file_path\".")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
    AddComment(R"DOC(
Load Operator.

Load operator will load a tensor variable from disk file.

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(load, ops::LoadOp, ops::LoadOpProtoMaker);
