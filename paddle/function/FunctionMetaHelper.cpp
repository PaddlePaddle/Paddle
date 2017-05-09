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
#include "FunctionMetaHelper.h"
#include "paddle/topology/meta/Validator.h"
namespace paddle {
namespace function {

class MetaKernelType {
private:
  static KernelType getKernel(topology::Function& topo,
                              const std::string& metaName) {
    auto meta = topology::meta::FunctionMeta::get(topo.type);
    const KernelType* kernel = nullptr;
    auto err = meta->getMeta<KernelType>(metaName, &kernel);
    if (err.isOK()) return *kernel;
    const KernelTypeWithAttrs* kernelEx = nullptr;
    meta->getMeta<KernelTypeWithAttrs>(metaName, &kernelEx).check();
    return [&topo, kernelEx](const BufferArgs& in, const BufferArgs& out) {
      return (*kernelEx)(in, out, topo.attributes);
    };
  }

  Error setBufferType(topology::TensorPtr& tensor, BufferType type) {
    switch (type) {
      case TENSOR_NORMAL:
        tensor->setDataType(paddle::topology::DataType::DENSE)
            .setSequenceType(paddle::topology::SequenceType::NO_SEQUENCE);
        break;
      case TENSOR_SEQUENCE_ID:
        tensor->setDataType(paddle::topology::DataType::INTEGER)
            .setSequenceType(paddle::topology::SequenceType::SEQUENCE);
        break;
      case TENSOR_SEQUENCE_DATA:
        tensor->setDataType(paddle::topology::DataType::DENSE)
            .setSequenceType(paddle::topology::SequenceType::SEQUENCE);
        break;
      case TENSOR_SPARSE:
        tensor->setDataType(paddle::topology::DataType::SPARSE)
            .setSequenceType(paddle::topology::SequenceType::NO_SEQUENCE);
        break;
      default:
        return Error("Not supportted buffer type");
    }
    return Error();
  }

public:
  explicit MetaKernelType(const topology::Function& topo) : funcTopo_(topo) {
    kernel_ = getKernel(funcTopo_, topo.useGPU() ? "GPUKernel" : "CPUKernel");
  }

  Error operator()(const BufferArgs& in, const BufferArgs& out) {
    if (funcTopo_.inputs.size() == 0) {
      funcTopo_.inputs.reserve(in.size());
      for (size_t i = 0; i < in.size(); ++i) {
        funcTopo_.inputs.emplace_back(new topology::Tensor());
        auto err = setBufferType(funcTopo_.inputs.back(), in[i].bufferType());
        if (!err.isOK()) return err;
      }
      funcTopo_.outputs.reserve(out.size());
      for (size_t i = 0; i < out.size(); ++i) {
        funcTopo_.outputs.emplace_back(new topology::Tensor());
      }
    }

    for (size_t i = 0; i < funcTopo_.inputs.size(); ++i) {
      std::vector<int> shape;
      shape.resize(in[i].shape().ndims());
      for (size_t i = 0; i < shape.size(); ++i) {
        shape[i] = (int)in[i].shape()[i];
      }
      funcTopo_.inputs[i]->setShape(shape);
    }

    //! Only check when first invoke.
    auto err = topology::meta::validate(funcTopo_);
    if (!err.isOK()) return err;

    return kernel_(in, out);
  }

private:
  topology::Function funcTopo_;
  KernelType kernel_;
};

KernelType createKernel(const topology::Function& conf) {
  return MetaKernelType(conf);
}

}  // namespace function

}  // namespace paddle
