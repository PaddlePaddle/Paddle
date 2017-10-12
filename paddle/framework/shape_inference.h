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

#pragma once

#include "paddle/framework/ddim.h"

namespace paddle {
namespace framework {

// TODO(longfei): Once after both CompileTimeInferShapeContext and
// RuntimeInferShapeContext get merged, we can rename InferShapeContext into
// InferShapeContext so to replace the current InferShapeContext.
class InferShapeContext {
 public:
  virtual ~InferShapeContext() {}
  virtual bool HasInput(const std::string &name) const = 0;
  virtual bool HasOutput(const std::string &name) const = 0;

  virtual bool HasInputs(const std::string &name) const = 0;
  virtual bool HasOutputs(const std::string &name) const = 0;

  virtual framework::DDim GetInputDim(const std::string &name) const = 0;
  std::vector<framework::DDim> GetInputsDim(const std::string &name) const {
    const std::vector<std::string> &names = Inputs(name);
    return GetDims(names);
  }
  virtual void SetInputDim(const std::string &name,
                           const framework::DDim &dim) = 0;
  void SetInputsDim(const std::string &name,
                    const std::vector<framework::DDim> &dims) {
    auto &names = Inputs(name);
    SetDims(names, dims);
  }
  virtual framework::DDim GetOutputDim(const std::string &name) const = 0;
  std::vector<framework::DDim> GetOutputsDim(const std::string &name) const {
    const std::vector<std::string> &names = Outputs(name);
    return GetDims(names);
  }
  virtual void SetOutputDim(const std::string &name, const DDim &dim) = 0;
  void SetOutputsDim(const std::string &name,
                     const std::vector<framework::DDim> &dims) {
    auto &names = Outputs(name);
    SetDims(names, dims);
  }
  virtual AttrReader Attrs() const = 0;
  virtual const std::vector<std::string> &Inputs(
      const std::string &name) const = 0;
  virtual const std::vector<std::string> &Outputs(
      const std::string &name) const = 0;
  // TODO(qiao) implement this function
  void ShareLoD(const std::string &in, const std::string &out, size_t i = 0,
                size_t j = 0) const {}

 protected:
  virtual framework::DDim GetDim(const std::string &name) const = 0;
  virtual void SetDim(const std::string &name, const framework::DDim &dim) = 0;
  std::vector<framework::DDim> GetDims(
      const std::vector<std::string> &names) const {
    std::vector<framework::DDim> ret;
    ret.reserve(names.size());
    std::transform(
        names.begin(), names.end(), std::back_inserter(ret),
        [this](const std::string &name) { return this->GetDim(name); });
    return ret;
  }
  void SetDims(const std::vector<std::string> &names,
               const std::vector<framework::DDim> &dims) {
    size_t length = names.size();
    PADDLE_ENFORCE_EQ(length, dims.size());
    for (size_t i = 0; i < length; ++i) {
      SetDim(names[i], dims[i]);
    }
  }
};

}  // namespace framework
}  // namespace paddle
