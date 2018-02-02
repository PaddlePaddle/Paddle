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

  std::vector<proto::VarDesc::VarType> GetInputsVarType(
      const std::string &name) const;
  std::vector<proto::VarDesc::VarType> GetOutputsVarType(
      const std::string &name) const;

  virtual bool HasInputs(const std::string &name) const = 0;
  virtual bool HasOutputs(const std::string &name) const = 0;

  DDim GetInputDim(const std::string &name) const;

  std::vector<DDim> GetInputsDim(const std::string &name) const;
  DDim GetInputsElementDim(const std::string &name, int idx) const;

  void SetOutputDim(const std::string &name, const DDim &dim);
  void SetOutputsDim(const std::string &name, const std::vector<DDim> &dims);

  virtual AttrReader Attrs() const = 0;
  virtual const std::vector<std::string> &Inputs(
      const std::string &name) const = 0;
  virtual const std::vector<std::string> &Outputs(
      const std::string &name) const = 0;

  virtual void ShareLoD(const std::string &in, const std::string &out,
                        size_t i = 0, size_t j = 0) const = 0;

  virtual bool IsRuntime() const = 0;

  // Note: In while op, we need this to be public
  void SetDims(const std::vector<std::string> &names,
               const std::vector<DDim> &dims);

 protected:
  virtual DDim GetDim(const std::string &name) const = 0;
  virtual void SetDim(const std::string &name, const DDim &dim) = 0;

  std::vector<DDim> GetDims(const std::vector<std::string> &names) const;
  std::vector<proto::VarDesc::VarType> GetVarTypes(
      const std::vector<std::string> &names) const;

  virtual proto::VarDesc::VarType GetVarType(const std::string &name) const = 0;
};

}  // namespace framework
}  // namespace paddle
