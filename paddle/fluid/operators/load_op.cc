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
#include <fstream>

#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {

class LoadOp : public framework::OperatorBase {
 public:
  LoadOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    // FIXME(yuyang18): We save variable to local file now, but we should change
    // it to save an output stream.
    auto filename = Attr<std::string>("file_path");
    std::ifstream fin(filename, std::ios::binary);
    PADDLE_ENFORCE(static_cast<bool>(fin), "Cannot open file %s for load op",
                   filename);

    auto out_var_name = Output("Out");
    auto *out_var = scope.FindVar(out_var_name);
    PADDLE_ENFORCE(out_var != nullptr,
                   "Output variable %s cannot be found in scope %p",
                   out_var_name, &scope);

    if (out_var->IsType<framework::LoDTensor>()) {
      LoadLodTensor(fin, place, out_var);
    } else if (out_var->IsType<framework::SelectedRows>()) {
      LoadSelectedRows(fin, place, out_var);
    } else {
      PADDLE_ENFORCE(
          false,
          "Load only support LoDTensor and SelectedRows, %s has wrong type",
          out_var_name);
    }
  }

  void LoadLodTensor(std::istream &fin, const platform::Place &place,
                     framework::Variable *var) const {
    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);
    auto *tensor = var->GetMutable<framework::LoDTensor>();
    DeserializeFromStream(fin, tensor, dev_ctx);

    auto load_as_fp16 = Attr<bool>("load_as_fp16");
    auto in_dtype = tensor->type();
    auto out_dtype = load_as_fp16 ? framework::proto::VarType::FP16 : in_dtype;

    if (in_dtype != out_dtype) {
      // convert to float16 tensor
      auto in_kernel_type = framework::OpKernelType(in_dtype, place);
      auto out_kernel_type = framework::OpKernelType(out_dtype, place);
      framework::LoDTensor fp16_tensor;
      // copy LoD info to the new tensor
      fp16_tensor.set_lod(tensor->lod());
      framework::TransDataType(in_kernel_type, out_kernel_type, *tensor,
                               &fp16_tensor);

      // reset output tensor
      var->Clear();
      tensor = var->GetMutable<framework::LoDTensor>();
      tensor->set_lod(fp16_tensor.lod());
      tensor->ShareDataWith(fp16_tensor);
    }
  }

  void LoadSelectedRows(std::istream &fin, const platform::Place &place,
                        framework::Variable *var) const {
    auto *selectedRows = var->GetMutable<framework::SelectedRows>();
    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);
    framework::DeserializeFromStream(fin, selectedRows, dev_ctx);
    selectedRows->SyncIndex();
  }
};

class LoadOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "The LoDTensor / SelectedRows need to be loaded");
    AddAttr<bool>(
        "load_as_fp16",
        "If true, the tensor will be first loaded and then "
        "converted to float16 data type. Otherwise, the tensor will be "
        "directly loaded without data type conversion. Default is false.")
        .SetDefault(false);
    AddAttr<std::string>("file_path",
                         R"(Variable will be loaded from "file_path")")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
    AddComment(
        "Load operator will load a LoDTensor / SelectedRows variable from disk "
        "file.");
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(load, ops::LoadOp, ops::LoadOpProtoMaker);
