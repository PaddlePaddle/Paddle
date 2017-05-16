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
#include "Register.h"
#include <algorithm>
#include <iterator>
#include <sstream>
#include "paddle/topology/Validator.h"

namespace paddle {
namespace function {

class RegistedFunction {
private:
  static Function getKernel(topology::Function* topo,
                            const std::string& kernelMethodName) {
    auto meta = topology::meta::FunctionMeta::get(topo->type);
    const Function* kernel = nullptr;
    auto err = meta->metaAttributes_.get<Function>(kernelMethodName, &kernel);
    if (err.isOK()) return *kernel;
    const details::FunctionWithAttrs* kernelEx = nullptr;
    meta->metaAttributes_
        .get<details::FunctionWithAttrs>(kernelMethodName, &kernelEx)
        .check();
    return [topo, kernelEx](const BufferArgs& in, const BufferArgs& out) {
      return (*kernelEx)(in, out, topo->attributes);
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

  Error checkSame(topology::TensorPtr& tensor, const BufferArg& arg) {
    switch (arg.bufferType()) {
      case TENSOR_NORMAL:
        if (tensor->dataType() == paddle::topology::DataType::DENSE &&
            tensor->sequenceType() ==
                paddle::topology::SequenceType::NO_SEQUENCE)
          break;
      // fall through
      case TENSOR_SEQUENCE_ID:
        if (tensor->dataType() == paddle::topology::DataType::INTEGER &&
            tensor->sequenceType() == paddle::topology::SequenceType::SEQUENCE)
          break;
      case TENSOR_SEQUENCE_DATA:
        if (tensor->dataType() == paddle::topology::DataType::DENSE &&
            tensor->sequenceType() == paddle::topology::SequenceType::SEQUENCE)
          break;
      case TENSOR_SPARSE:
        if (tensor->dataType() == paddle::topology::DataType::SPARSE &&
            tensor->sequenceType() ==
                paddle::topology::SequenceType::NO_SEQUENCE)
          break;
      default:
        return Error("The Type mismatched");
    }
    // Check Shape
    auto& tensorShape = tensor->shape();
    auto& argShape = arg.shape();
    if (argShape.ndims() != tensorShape.size())
      return Error("Tensor dimension mismatch");
    for (size_t i = 0; i < tensorShape.size(); ++i) {
      if (tensorShape[i] !=
          std::remove_reference<decltype(tensorShape[i])>::type(argShape[i])) {
        std::ostringstream sout;
        sout << "Tensor shape mismatch: Tensor (";
        std::copy(tensorShape.begin(),
                  tensorShape.end(),
                  std::ostream_iterator<int>(sout, ","));
        sout << "), Arg (";
        std::copy(&argShape[0],
                  &argShape[0] + argShape.ndims(),
                  std::ostream_iterator<size_t>(sout, ","));
        sout << ")";

        return Error(sout.str().c_str());
      }
    }

    if (tensor->attributes.get<int>("arg_type") != arg.getArgType())
      return Error("Arg type mismatch");

    return Error();
  }

public:
  explicit RegistedFunction(const topology::Function& topo) : funcTopo_(topo) {
    kernel_ = getKernel(&funcTopo_, topo.useGPU() ? "GPUKernel" : "CPUKernel");
  }

  RegistedFunction(const RegistedFunction& o) = delete;  // DISABLE COPY
  RegistedFunction& operator=(const RegistedFunction& o) =
      delete;  // DISABLE COPY

  Error operator()(const BufferArgs& in, const BufferArgs& out) {
    if (funcTopo_.inputs.size() == 0) {
      funcTopo_.inputs.reserve(in.size());
      for (size_t i = 0; i < in.size(); ++i) {
        funcTopo_.inputs.emplace_back(new topology::Tensor());
        auto err = setBufferType(funcTopo_.inputs.back(), in[i].bufferType());
        if (!err.isOK()) return Error("InputTensor %d Error :%s", i, err.msg());
      }
      funcTopo_.outputs.reserve(out.size());
      for (size_t i = 0; i < out.size(); ++i) {
        funcTopo_.outputs.emplace_back(new topology::Tensor());
      }
    }

    for (size_t i = 0; i < funcTopo_.inputs.size(); ++i) {
      std::vector<size_t> tmp;
      tmp.resize(in[i].shape().ndims());
      for (size_t j = 0; j < tmp.size(); ++j) {
        tmp[j] = in[i].shape()[j];
      }
      funcTopo_.inputs[i]->setShape(tmp);
    }

    //! Only check when first invoke.
    auto err = topology::validateAndInferShape(funcTopo_);
    if (!err.isOK()) {
      return Error(
          "Function (%s) error: %s", funcTopo_.type.c_str(), err.msg());
    }

    for (size_t i = 0; i < out.size(); ++i) {
      err = checkSame(funcTopo_.outputs[i], out[i]);
      if (!err.isOK()) return Error("Output error %d: %s", i, err.msg());
    }

    return kernel_(in, out);
  }

private:
  topology::Function funcTopo_;
  Function kernel_;
};

Function createFunction(const topology::Function& conf) {
  auto ptr = std::make_shared<RegistedFunction>(conf);
  return [ptr](const BufferArgs& in, const BufferArgs& out) {
    return (*ptr)(in, out);
  };
}

}  // namespace function

}  // namespace paddle
