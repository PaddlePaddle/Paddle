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
#include "paddle/framework/data_type.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"

#include <boost/filesystem.hpp>
#include <numeric>

namespace paddle {
namespace operators {

class SaveOp : public framework::OperatorBase {
 public:
  SaveOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto filename = Attr<std::string>("file_path");
    auto overwrite = Attr<bool>("overwrite");

    boost::filesystem::path path(filename);
    if (boost::filesystem::exists(path) && !overwrite) {
      PADDLE_THROW("%s is existed, cannot save to it when overwrite=false",
                   filename, overwrite);
    }
    auto parent_dir = path.parent_path();
    if (!parent_dir.empty() && !boost::filesystem::exists(parent_dir)) {
      boost::system::error_code error;
      PADDLE_ENFORCE(boost::filesystem::create_directories(parent_dir, error),
                     "mkdir -p %s failed, %s", parent_dir.string(),
                     error.message());
    }

    // FIXME(yuyang18): We save variable to local file now, but we should change
    // it to save an output stream.
    std::ofstream fout(path.string());
    PADDLE_ENFORCE(static_cast<bool>(fout), "Cannot open %s to write",
                   path.string());

    auto iname = Input("X");
    auto *var = scope.FindVar(iname);
    PADDLE_ENFORCE(var != nullptr, "Cannot find variable %s for save_op",
                   iname);

    PADDLE_ENFORCE(var->IsType<framework::LoDTensor>(),
                   "SaveOp only support LoDTensor, %s has wrong type", iname);

    auto &tensor = var->Get<framework::LoDTensor>();

    {  // the 1st field, uint32_t version
      constexpr uint32_t version = 0;
      fout.write(reinterpret_cast<const char *>(&version), sizeof(version));
    }
    {  // the 2nd field, tensor description
       // int32_t  size
       // void*    protobuf message
      framework::TensorDesc desc;
      desc.set_data_type(framework::ToDataType(tensor.type()));
      auto dims = framework::vectorize(tensor.dims());
      auto *pb_dims = desc.mutable_dims();
      pb_dims->Reserve(static_cast<int>(dims.size()));
      std::copy(dims.begin(), dims.end(), std::back_inserter(*pb_dims));
      int32_t size = desc.ByteSize();
      fout.write(reinterpret_cast<const char *>(&size), sizeof(size));
      auto out = desc.SerializeAsString();
      fout.write(out.data(), size);
    }
    {  // the 3rd field, tensor data
      uint64_t size = tensor.memory_size();
      auto *data_ptr = tensor.data<void>();
      PADDLE_ENFORCE(size < std::numeric_limits<std::streamsize>::max(),
                     "Index overflow when writing tensor");
      fout.write(static_cast<const char *>(data_ptr),
                 static_cast<std::streamsize>(size));
    }
    {  // the 4th field, lod information
       // uint64_t lod_level
       // uint64_t lod_level_1 size in byte.
       // int*     lod_level_1 data
       // ...
      auto lod = tensor.lod();
      uint64_t size = lod.size();
      fout.write(reinterpret_cast<const char *>(&size), sizeof(size));

      for (auto &each : lod) {
        size = each.size() * sizeof(framework::LoD::value_type::value_type);
        fout.write(reinterpret_cast<const char *>(&size), sizeof(size));
        fout.write(reinterpret_cast<const char *>(each.data()),
                   static_cast<std::streamsize>(size));
      }
    }
  }
};

class SaveOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SaveOpProtoMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The tensor need to be saved");
    AddComment(R"DOC(Save operator
Save operator will serialize and write a tensor variable to disk file.
)DOC");
    AddAttr<bool>("overwrite", "Overwrite the output file if exist")
        .SetDefault(true);
    AddAttr<std::string>("file_path", "Variable will save to \"file_path\".")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(save, ops::SaveOp, ops::SaveOpProtoMaker);
