/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

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

#include "paddle/framework/op_registry.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace operators {

class LoadCombineOp : public framework::OperatorBase {
 public:
  LoadCombineOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::Place &place) const override {
    auto filename = Attr<std::string>("file_path");
    auto position_counter = Attr<int>("position_counter");

    std::ifstream fin(filename);
    PADDLE_ENFORCE(static_cast<bool>(fin),
                   "Cannot open file %s for load_combine op", filename);

    auto out_var_name = Output("Out");
    auto *out_var = scope.FindVar(out_var_name);
    PADDLE_ENFORCE(out_var != nullptr, "Output variable %s cannot be found",
                   out_var_name);

    auto *tensor = out_var->GetMutable<framework::LoDTensor>();

    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);

    uint64_t data_length;
    char *buffer = NULL;
    for (int i = 0; i <= position_counter; i++) {
      if (!buffer) delete[] buffer;

      // Error checking
      PADDLE_ENFORCE(static_cast<bool>(fin), "Cannot read more from file %s",
                     filename);

      // Read a fixed-width int, to get the number of bytes
      // for the serialized data.
      fin.read(reinterpret_cast<char *>(&data_length), sizeof(data_length));

      // Error checking
      PADDLE_ENFORCE(static_cast<bool>(fin), "Cannot read more from file %s",
                     filename);

      buffer = new char[data_length];

      // Read the serialized data into the buffer
      fin.read(buffer, data_length);
    }

    std::string current_serialized_data;
    current_serialized_data.assign(buffer, data_length);

    // Create an input string stream
    std::istringstream ist(current_serialized_data);
    DeserializeFromStream(ist, tensor, dev_ctx);

    if (!buffer) delete[] buffer;  // delete the last allocated memory

    if (platform::is_gpu_place(place)) {
      // copy CPU to GPU
      framework::LoDTensor cpu_tensor;
      cpu_tensor.ShareDataWith(*tensor);
      cpu_tensor.set_lod(tensor->lod());

      // reset tensor
      out_var->Clear();
      tensor = out_var->GetMutable<framework::LoDTensor>();
      tensor->set_lod(cpu_tensor.lod());
      Copy(cpu_tensor, place, dev_ctx, tensor);
    }
  }
};

class LoadCombineOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LoadCombineOpProtoMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddOutput("Out", "(Tensor) The tensor need to be load_combineed");
    AddAttr<int>("position_counter",
                 "(int) "
                 "It specifies the relative ordering of different parameters.")
        .AddCustomChecker([](const int &counter) { return counter >= 0; });
    AddAttr<std::string>("file_path",
                         "(string) "
                         "Variable will be load_combined from \"file_path\".")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
    AddComment(R"DOC(
LoadCombine Operator.

LoadCombine operator combines together various tensor variable into a file.

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(load_combine, ops::LoadCombineOp,
                  ops::LoadCombineOpProtoMaker);
