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

#include "paddle/framework/attribute.h"
#include "paddle/framework/ddim.h"
#include "paddle/framework/framework.pb.h"

namespace paddle {
namespace framework {

class InferShapeContext {
 public:
  virtual ~InferShapeContext() = default;
  virtual bool HasInput(const std::string &name) const = 0;
  virtual bool HasOutput(const std::string &name) const = 0;

  std::vector<VarDesc::VarType> GetInputsVarType(const std::string &name) const;
  std::vector<VarDesc::VarType> GetOutputsVarType(
      const std::string &name) const;

  virtual bool HasInputs(const std::string &name) const = 0;
  virtual bool HasOutputs(const std::string &name) const = 0;

  virtual framework::DDim GetInputDim(const std::string &name) const = 0;

  std::vector<framework::DDim> GetInputsDim(const std::string &name) const;

  virtual void SetOutputDim(const std::string &name, const DDim &dim) = 0;
  void SetOutputsDim(const std::string &name,
                     const std::vector<framework::DDim> &dims);

  virtual AttrReader Attrs() const = 0;
  virtual const std::vector<std::string> &Inputs(
      const std::string &name) const = 0;
  virtual const std::vector<std::string> &Outputs(
      const std::string &name) const = 0;

  virtual void ShareLoD(const std::string &in, const std::string &out,
                        size_t i = 0, size_t j = 0) const = 0;

  virtual bool IsRuntime() const = 0;

 protected:
  virtual framework::DDim GetDim(const std::string &name) const = 0;
  virtual void SetDim(const std::string &name, const framework::DDim &dim) = 0;

  std::vector<framework::DDim> GetDims(
      const std::vector<std::string> &names) const;

  void SetDims(const std::vector<std::string> &names,
               const std::vector<framework::DDim> &dims);

  std::vector<VarDesc::VarType> GetVarTypes(
      const std::vector<std::string> &names) const;

  virtual VarDesc::VarType GetVarType(const std::string &name) const = 0;
};

}  // namespace framework
}  // namespace paddle
