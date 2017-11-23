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
#include <sys/stat.h>
#include <fstream>
#include <numeric>

#include "paddle/framework/data_type.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

// TODO(yuyang18): If the functions below are needed by other files, move them
// to paddle::filesystem namespace.
constexpr char kSEP = '/';
static bool FileExists(const std::string &filepath) {
  struct stat buffer;
  return (stat(filepath.c_str(), &buffer) == 0);
}

static std::string DirName(const std::string &filepath) {
  auto pos = filepath.rfind(kSEP);
  if (pos == std::string::npos) {
    return "";
  }
  return filepath.substr(0, pos);
}

static void MkDir(const char *path) {
  if (mkdir(path, 0755)) {
    PADDLE_ENFORCE_EQ(errno, EEXIST, "%s mkdir failed!", path);
  }
}

static void MkDirRecursively(const char *fullpath) {
  if (*fullpath == '\0') return;  // empty string
  if (FileExists(fullpath)) return;

  MkDirRecursively(DirName(fullpath).c_str());
  MkDir(fullpath);
}

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

    if (FileExists(filename) && !overwrite) {
      PADDLE_THROW("%s is existed, cannot save to it when overwrite=false",
                   filename, overwrite);
    }

    MkDirRecursively(DirName(filename).c_str());

    // FIXME(yuyang18): We save variable to local file now, but we should change
    // it to save an output stream.
    std::ofstream fout(filename);
    PADDLE_ENFORCE(static_cast<bool>(fout), "Cannot open %s to write",
                   filename);

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
      pb_dims->Resize(static_cast<int>(dims.size()), 0);
      std::copy(dims.begin(), dims.end(), pb_dims->begin());
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
      if (platform::is_gpu_place(tensor.place())) {
#ifdef PADDLE_WITH_CUDA
        constexpr size_t kBufSize = 1024 * 1024 * 64;  // 64MB
        std::unique_ptr<char[]> buf(new char[kBufSize]);
        auto &gpu_dev_ctx =
            static_cast<const platform::CUDADeviceContext &>(dev_ctx);
        platform::CPUPlace cpu;
        uintptr_t data = reinterpret_cast<uintptr_t>(data_ptr);
        while (size != 0) {
          size_t size_to_write = std::min(kBufSize, static_cast<size_t>(size));
          memory::Copy(cpu, buf.get(),
                       boost::get<platform::GPUPlace>(tensor.place()),
                       reinterpret_cast<const void *>(data), size_to_write,
                       gpu_dev_ctx.stream());
          gpu_dev_ctx.Wait();
          fout.write(buf.get(), size_to_write);
          data += size_to_write;
          size -= size_to_write;
        }
#else
        PADDLE_THROW("Unexpected branch");
#endif
      } else {
        fout.write(static_cast<const char *>(data_ptr),
                   static_cast<std::streamsize>(size));
      }
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
    AddInput("X", "(Tensor ) Input tensor to be saved");
    AddComment(R"DOC(
Save operator

This operator will serialize and write a tensor variable to file on disk.
)DOC");
    AddAttr<bool>("overwrite",
                  "(boolean, default true)"
                  "Overwrite the output file if exist")
        .SetDefault(true);
    AddAttr<std::string>("file_path",
                         "(string)"
                         "The \"file_path\" where the variable will be saved.")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(save, ops::SaveOp, ops::SaveOpProtoMaker);
