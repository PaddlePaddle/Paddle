//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/framework/reader.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class ReaderBase;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {
namespace reader {

static constexpr char kFileFormatSeparator[] = ".";

using FileReaderCreator =
    std::function<framework::ReaderBase*(const std::string&)>;

std::unordered_map<std::string, FileReaderCreator>& FileReaderRegistry();

template <typename Reader>
int RegisterFileReader(const std::string& filetype) {
  FileReaderRegistry()[filetype] = [](const std::string& fn) {
    return new Reader(fn);
  };
  return 0;
}

extern std::vector<phi::DDim> RestoreShapes(
    const std::vector<int>& shape_concat, const std::vector<int>& ranks);

class FileReaderMakerBase : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final;

 protected:
  virtual void Apply() = 0;
};

class FileReaderInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override;
};

class FileReaderInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override;
};

// general infershape for decorated reader
class DecoratedReaderInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override;
};

// general var type inference for decorated reader
class DecoratedReaderInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override;
};

class DecoratedReaderMakerBase : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final;

 protected:
  virtual void Apply() = 0;
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

#define REGISTER_FILE_READER_OPERATOR(op_name, ...)                    \
  REGISTER_OPERATOR(                                                   \
      op_name,                                                         \
      __VA_ARGS__,                                                     \
      paddle::operators::reader::FileReaderInferShape,                 \
      paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,  \
      paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>, \
      paddle::operators::reader::FileReaderInferVarType)

#define REGISTER_DECORATED_READER_OPERATOR(op_name, ...)               \
  REGISTER_OPERATOR(                                                   \
      op_name,                                                         \
      __VA_ARGS__,                                                     \
      paddle::operators::reader::DecoratedReaderInferShape,            \
      paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,  \
      paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>, \
      paddle::operators::reader::DecoratedReaderInferVarType)

#define REGISTER_FILE_READER(_filetype, _reader)            \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                           \
      _reg_file_reader_##_filetype,                         \
      "Must use REGISTER_FILE_READER in global namespace"); \
  int TouchFileReader##_filetype() { return 0; }            \
  int _reg_file_reader_entry_##filetype =                   \
      paddle::operators::reader::RegisterFileReader<_reader>(#_filetype)

#define USE_FILE_READER(filetype)         \
  extern int TouchFileReader##filetype(); \
  static int _use_##filetype = TouchFileReader##filetype()
