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

namespace paddle::framework {
class VarDesc;
}  // namespace paddle::framework

namespace paddle::operators::reader {

std::vector<phi::DDim> RestoreShapes(const std::vector<int>& shape_concat,
                                     const std::vector<int>& ranks) {
  std::vector<phi::DDim> res;
  int offset = 0;
  for (int len : ranks) {
    auto start_it = shape_concat.begin() + offset;
    auto end_it = start_it + len;
    res.push_back(common::make_ddim(std::vector<int>(start_it, end_it)));
    offset += len;
  }
  return res;
}

std::unordered_map<std::string, FileReaderCreator>& FileReaderRegistry() {
  static std::unordered_map<std::string, FileReaderCreator> regs;
  return regs;
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
  AddAttr<std::vector<int>>("dtypes",
                            "The int value of enum dtypes of each data.");
  AddAttr<std::vector<int>>("need_check_feed",
                            "Whether to check shape and dtypes of input");
  AddAttr<bool>(
      "use_data_config",
      "Use the config of all datas like shape_concat/ranks/lod_levels")
      .SetDefault(true);
  Apply();
}

void FileReaderInferShape::operator()(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE_NE(
      ctx->IsRuntime(),
      true,
      common::errors::PreconditionNotMet("'FileReaderInferShape' should only "
                                         "be invoked during compile time."));

  PADDLE_ENFORCE_EQ(
      ctx->HasOutput("Out"),
      true,
      common::errors::NotFound("The output file reader should not be null."));
  bool use_data_config = ctx->Attrs().Get<bool>("use_data_config");
  if (use_data_config) {
    const auto shape_concat =
        ctx->Attrs().Get<std::vector<int>>("shape_concat");
    const auto ranks = ctx->Attrs().Get<std::vector<int>>("ranks");
    std::vector<phi::DDim> shapes = RestoreShapes(shape_concat, ranks);
    ctx->SetReaderDims("Out", shapes);

    const auto lod_levels = ctx->Attrs().Get<std::vector<int>>("lod_levels");
    PADDLE_ENFORCE_EQ(
        lod_levels.size(),
        shapes.size(),
        common::errors::InvalidArgument(
            "The number of 'lod_levels'(%d) doesn't match the number "
            "of 'shapes'(%d).",
            lod_levels.size(),
            shapes.size()));
    const auto dtypes = ctx->Attrs().Get<std::vector<int>>("dtypes");
    PADDLE_ENFORCE_EQ(
        dtypes.size(),
        shapes.size(),
        common::errors::InvalidArgument("The number of 'dtypes'(%d) doesn't "
                                        "match the number of 'shapes'(%d).",
                                        dtypes.size(),
                                        shapes.size()));
    const auto need_check_feed =
        ctx->Attrs().Get<std::vector<int>>("need_check_feed");
    PADDLE_ENFORCE_EQ(
        need_check_feed.size(),
        shapes.size(),
        common::errors::InvalidArgument(
            "The number of 'need_check_feed'(%d) doesn't match the "
            "number of 'shapes'(%d).",
            need_check_feed.size(),
            shapes.size()));
    framework::VarDesc* reader =
        PADDLE_GET(framework::VarDesc*, ctx->GetOutputVarPtrs("Out")[0]);
    reader->SetLoDLevels(lod_levels);
  }
}

void FileReaderInferVarType::operator()(
    framework::InferVarTypeContext* ctx) const {
  ctx->SetOutputType("Out", framework::proto::VarType::READER);
}

void DecoratedReaderInferShape::operator()(
    framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE_NE(
      ctx->IsRuntime(),
      true,
      common::errors::PreconditionNotMet(
          "'DecoratedReaderInferShape' should only be invoked during "
          "compile time."));

  PADDLE_ENFORCE_EQ(
      ctx->HasInput("UnderlyingReader"),
      true,
      common::errors::NotFound("Input(UnderlyingReader) should not be null."));
  PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"),
                    true,
                    common::errors::NotFound(
                        "The output decorated reader should not be null."));
  ctx->SetReaderDims("Out", ctx->GetReaderDims("UnderlyingReader"));

  framework::VarDesc* in_reader = PADDLE_GET(
      framework::VarDesc*, ctx->GetInputVarPtrs("UnderlyingReader")[0]);
  framework::VarDesc* out_reader =
      PADDLE_GET(framework::VarDesc*, ctx->GetOutputVarPtrs("Out")[0]);
  out_reader->SetLoDLevels(in_reader->GetLoDLevels());
}

void DecoratedReaderInferVarType::operator()(
    framework::InferVarTypeContext* ctx) const {
  ctx->SetOutputType("Out", framework::proto::VarType::READER);
  ctx->SetOutputDataTypes("Out", ctx->GetInputDataTypes("UnderlyingReader"));
}

void DecoratedReaderMakerBase::Make() {
  AddInput("UnderlyingReader",
           "(ReaderHolder) The underlying reader for creating a batch reader.");
  AddOutput("Out", "(ReaderHolder) The created batch reader.");
  Apply();
}

}  // namespace paddle::operators::reader
