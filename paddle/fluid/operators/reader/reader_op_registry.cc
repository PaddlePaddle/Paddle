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

#include "paddle/fluid/operators/reader/reader_op_registry.h"
#include <string>
#include <vector>

namespace paddle {
namespace operators {
namespace reader {

std::vector<framework::DDim> RestoreShapes(const std::vector<int>& shape_concat,
                                           const std::vector<int>& ranks) {
  std::vector<framework::DDim> res;
  int offset = 0;
  for (int len : ranks) {
    auto start_it = shape_concat.begin() + offset;
    auto end_it = start_it + len;
    res.push_back(framework::make_ddim(std::vector<int>(start_it, end_it)));
    offset += len;
  }
  return res;
}

std::unordered_map<std::string, FileReaderCreator>& FileReaderRegistry() {
  static std::unordered_map<std::string, FileReaderCreator> regs;
  return regs;
}

std::unique_ptr<framework::ReaderBase> CreateReaderByFileName(
    const std::string& file_name) {
  size_t separator_pos = file_name.find_last_of(kFileFormatSeparator);
  PADDLE_ENFORCE_NE(separator_pos, std::string::npos,
                    "File name illegal! A legal file name should be like: "
                    "[file_name].[file_format] (e.g., 'data_file.recordio').");
  std::string filetype = file_name.substr(separator_pos + 1);

  auto itor = FileReaderRegistry().find(filetype);
  PADDLE_ENFORCE(itor != FileReaderRegistry().end(),
                 "No file reader registered for '%s' format.", filetype);
  framework::ReaderBase* reader = (itor->second)(file_name);
  return std::unique_ptr<framework::ReaderBase>(reader);
}

void FileReaderMakerBase::Make() {
  AddOutput("Out", "(ReaderHolder): The created random reader.").AsDuplicable();
  AddAttr<std::vector<int>>("shape_concat", "The concat of all data's shapes.");
  AddAttr<std::vector<int>>(
      "ranks",
      "The ranks of each data."
      "e.g."
      "shape_concat = [2,3,4,5,6]"
      "ranks = [3,2]"
      "It means the reader will generate two data each time,"
      "whose shapes are [2,3,4] and [5,6] respectively.");
  AddAttr<std::vector<int>>("lod_levels", "The LoD levels of each data.");
  AddAttr<bool>(
      "use_data_config",
      "Use the config of all datas like shape_concat/ranks/lod_levels")
      .SetDefault(true);
  Apply();
}

void FileReaderInferShape::operator()(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE(
      !ctx->IsRuntime(),
      "'FileReaderInferShape' should only be invoked during compile time.");

  PADDLE_ENFORCE(ctx->HasOutput("Out"),
                 "The output file reader should not be null.");
  bool use_data_config = ctx->Attrs().Get<bool>("use_data_config");
  if (use_data_config) {
    const auto shape_concat =
        ctx->Attrs().Get<std::vector<int>>("shape_concat");
    const auto ranks = ctx->Attrs().Get<std::vector<int>>("ranks");
    std::vector<framework::DDim> shapes = RestoreShapes(shape_concat, ranks);
    ctx->SetReaderDims("Out", shapes);

    const auto lod_levels = ctx->Attrs().Get<std::vector<int>>("lod_levels");
    PADDLE_ENFORCE_EQ(lod_levels.size(), shapes.size(),
                      "The number of 'lod_levels'(%d) doesn't match the number "
                      "of 'shapes'(%d).",
                      lod_levels.size(), shapes.size());
    framework::VarDesc* reader =
        boost::get<framework::VarDesc*>(ctx->GetOutputVarPtrs("Out")[0]);
    reader->SetLoDLevels(lod_levels);
  }
}

void FileReaderInferVarType::operator()(
    framework::InferVarTypeContext* ctx) const {
  std::string reader_name = ctx->Output("Out")[0];
  ctx->SetType(reader_name, framework::proto::VarType::READER);
}

void DecoratedReaderInferShape::operator()(
    framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE(!ctx->IsRuntime(),
                 "'DecoratedReaderInferShape' should only be invoked during "
                 "compile time.");

  PADDLE_ENFORCE(ctx->HasInput("UnderlyingReader"),
                 "Input(UnderlyingReader) should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("Out"),
                 "The output decorated reader should not be null.");
  ctx->SetReaderDims("Out", ctx->GetReaderDims("UnderlyingReader"));

  framework::VarDesc* in_reader = boost::get<framework::VarDesc*>(
      ctx->GetInputVarPtrs("UnderlyingReader")[0]);
  framework::VarDesc* out_reader =
      boost::get<framework::VarDesc*>(ctx->GetOutputVarPtrs("Out")[0]);
  out_reader->SetLoDLevels(in_reader->GetLoDLevels());
}

void DecoratedReaderInferVarType::operator()(
    framework::InferVarTypeContext* ctx) const {
  const std::string& in_reader_name = ctx->Input("UnderlyingReader")[0];
  const std::string& out_reader_name = ctx->Output("Out")[0];
  ctx->SetType(out_reader_name, framework::proto::VarType::READER);
  ctx->SetDataTypes(out_reader_name, ctx->GetDataTypes(in_reader_name));
}

void DecoratedReaderMakerBase::Make() {
  AddInput("UnderlyingReader",
           "(ReaderHolder) The underlying reader for creating a batch reader.");
  AddOutput("Out", "(ReaderHolder) The created batch reader.");
  Apply();
}

}  // namespace reader

}  // namespace operators
}  // namespace paddle
