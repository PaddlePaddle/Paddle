// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <fstream>
#include <string>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/jit/layer.h"
#include "paddle/phi/core/dense_tensor.h"

namespace paddle {
namespace jit {
using DenseTensor = phi::DenseTensor;

// Export Layer into local disk
class Serializer {
 public:
  void operator()(const Layer& layer, const std::string& file_dir);

  //  private:
  //   void WriteTensorData(const Layer& layer, const std::string& file_name)
  //   const;
  //   void WriteExtraInfo(const Layer& layer, const std::string& file_name)
  //   const;
  //   void WriteByteCode(const Layer& layer, const std::string& file_name)
  //   const;
};

class Deserializer {
 public:
  // TODO: use file prefix currently
  Layer operator()(const std::string& file_dir) {
    auto program = LoadProgram(file_dir + ".pdmodel");
    auto ivalues = ReadTensorData(file_dir + ".pdiparams");
    // TODO: we also need name of Params
    return Layer({program}, ivalues);
  }

 private:
  std::vector<IValue> ReadTensorData(const std::string& file_name) const {
    cout << "ReadTensorData " << file_name << endl;
    std::ifstream fin(file_name, std::ios::binary);
    // TODO: how to pass var_names;
    std::vector<std::string> var_names = {"linear_0.b_0", "linear_0.w_0",
                                          "linear_1.b_0", "linear_1.w_0"};
    std::vector<IValue> res;
    for (size_t i = 0; i < var_names.size(); ++i) {
      cout << "load Tensor: " << var_names[i] << endl;
      Tensor t(std::make_shared<DenseTensor>());
      LoadTensorFromBuffer(fin, dynamic_cast<DenseTensor*>(t.impl().get()));
      t.set_name(var_names[i]);
      res.emplace_back(t);
    }
    return res;
  };

  void LoadTensorFromBuffer(std::istream& buffer, DenseTensor* tensor) const {
    {
      // the 1st field, unit32_t version for LoDTensor
      uint32_t version;
      buffer.read(reinterpret_cast<char*>(&version), sizeof(version));
      cout << "version: " << version << endl;
    }
    {
      // the 2st field, LoD information
      uint64_t lod_level;
      buffer.read(reinterpret_cast<char*>(&lod_level), sizeof(lod_level));
      auto& lod = *tensor->mutable_lod();
      // TODO:support lod;
      cout << "lod_level: " << lod_level << endl;
      lod.resize(lod_level);
    }
    {
      // the 3st filed, Tensor
      {
        uint32_t version;
        buffer.read(reinterpret_cast<char*>(&version), sizeof(version));
        cout << "tensor version: " << version << endl;
      }
      framework::proto::VarType::TensorDesc desc;
      {
        int32_t size;
        buffer.read(reinterpret_cast<char*>(&size), sizeof(size));
        std::unique_ptr<char[]> buf(new char[size]);
        buffer.read(reinterpret_cast<char*>(buf.get()), size);
        desc.ParseFromArray(buf.get(), size);
      }
      {
        // read tensor
        std::vector<int64_t> dims;
        dims.reserve(desc.dims().size());
        std::copy(desc.dims().begin(), desc.dims().end(),
                  std::back_inserter(dims));
        cout << "dims: " << phi::make_ddim(dims) << endl;
        tensor->Resize(phi::make_ddim(dims));
        void* buf;
        size_t size = tensor->numel() * framework::SizeOfType(desc.data_type());
        buf = tensor->mutable_data(phi::CPUPlace());
        buffer.read(static_cast<char*>(buf), size);
      }
    }
  }

  // void ReadExtraInfo(const std::string& file_name) const;
  // void ReadByteCode(const std::string& file_name) const;

  framework::ProgramDesc LoadProgram(const std::string& file_name) {
    cout << "LoadProgram " << file_name << endl;
    std::ifstream fin(file_name, std::ios::in | std::ios::binary);
    fin.seekg(0, std::ios::end);
    std::string buffer(fin.tellg(), ' ');
    fin.seekg(0, std::ios::beg);
    fin.read(&buffer[0], buffer.size());
    fin.close();
    return framework::ProgramDesc(buffer);
  }
};

void Export(const Layer& layer, const std::string& file_path);

Layer Load(const std::string& file_path) {
  auto deserializer = Deserializer();
  return deserializer(file_path);
}

}  // namespace jit
}  // namespace paddle
