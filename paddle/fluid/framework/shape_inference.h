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

#pragma once

#include <string>
#include <vector>
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace framework {

using InferShapeVarPtr = boost::variant<VarDesc *, Variable *>;

class InferShapeContext {
 public:
  virtual ~InferShapeContext() = default;
  virtual bool HasInput(const std::string &name) const = 0;
  virtual bool HasOutput(const std::string &name) const = 0;

  std::vector<proto::VarType::Type> GetInputsVarType(
      const std::string &name) const;
  std::vector<proto::VarType::Type> GetOutputsVarType(
      const std::string &name) const;

  virtual bool HasInputs(const std::string &name) const = 0;
  virtual bool HasOutputs(const std::string &name) const = 0;

  DDim GetInputDim(const std::string &name) const;
  std::vector<DDim> GetInputsDim(const std::string &name) const;
  std::vector<DDim> GetReaderDims(const std::string &name) const;
  DDim GetInputsElementDim(const std::string &name, int idx) const;

  void SetOutputDim(const std::string &name, const DDim &dim);
  void SetOutputsDim(const std::string &name, const std::vector<DDim> &dims);
  void SetReaderDims(const std::string &name, const std::vector<DDim> &dims);

  virtual AttrReader Attrs() const = 0;
  virtual const std::vector<std::string> &Inputs(
      const std::string &name) const = 0;
  virtual const std::vector<std::string> &Outputs(
      const std::string &name) const = 0;

  virtual void ShareLoD(const std::string &in, const std::string &out,
                        size_t i = 0, size_t j = 0) const = 0;

  virtual bool IsRuntime() const = 0;

  std::vector<InferShapeVarPtr> GetInputVarPtrs(const std::string &name);
  std::vector<InferShapeVarPtr> GetOutputVarPtrs(const std::string &name);
  virtual InferShapeVarPtr GetVarPtr(const std::string &name) = 0;

  // Note: In while op, we need this to be public
  void SetDims(const std::vector<std::string> &names,
               const std::vector<DDim> &dims);

 protected:
  virtual DDim GetDim(const std::string &name) const = 0;
  virtual void SetDim(const std::string &name, const DDim &dim) = 0;
  virtual std::vector<DDim> GetRepeatedDims(const std::string &name) const = 0;
  virtual void SetRepeatedDims(const std::string &name,
                               const std::vector<DDim> &dims) = 0;

  std::vector<DDim> GetDims(const std::vector<std::string> &names) const;

  std::vector<proto::VarType::Type> GetVarTypes(
      const std::vector<std::string> &names) const;

  virtual proto::VarType::Type GetVarType(const std::string &name) const = 0;
};

}  // namespace framework
}  // namespace paddle
