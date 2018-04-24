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

namespace paddle {
namespace operators {
namespace reader {

template <typename T>
class RandomDataGenerator : public framework::ReaderBase {
 public:
  RandomDataGenerator(const std::vector<framework::DDim>& shapes, float min,
                      float max)
      : framework::ReaderBase(), min_(min), max_(max), shapes_(shapes) {
    PADDLE_ENFORCE_LE(
        min, max, "'min' shouldn't be greater than 'max'.(%f vs %f)", min, max);
    unsigned int seed = std::random_device()();
    engine_.seed(seed);
    dist_ = std::uniform_real_distribution<float>(min_, max_);
  }

  void ReadNext(std::vector<framework::LoDTensor>* out) override {
    out->clear();
    out->reserve(shapes_.size());
    for (const framework::DDim& shape : shapes_) {
      PADDLE_ENFORCE_GE(
          shape.size(), 2,
          "The rank of reader's output data should be 2 at least.(Now it's %d)",
          shape.size());
      framework::LoDTensor out_tensor;
      out_tensor.Resize(shape);
      T* data = out_tensor.mutable_data<T>(platform::CPUPlace());
      int64_t numel = framework::product(shape);
      for (int64_t i = 0; i < numel; ++i) {
        data[i] = dist_(engine_);
      }
      out->push_back(out_tensor);
    }
  }

  void ReInit() override { return; }

 private:
  float min_;
  float max_;
  std::minstd_rand engine_;
  std::uniform_real_distribution<float> dist_;
  std::vector<framework::DDim> shapes_;
};

template <typename T>
class CreateRandomDataGeneratorOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    const auto& shape_concat = Attr<std::vector<int>>("shape_concat");
    const auto& ranks = Attr<std::vector<int>>("ranks");
    PADDLE_ENFORCE(!shape_concat.empty() && !ranks.empty());
    PADDLE_ENFORCE_EQ(std::accumulate(ranks.begin(), ranks.end(), 0),
                      static_cast<int>(shape_concat.size()),
                      "The accumulate of all ranks should be equal to the "
                      "shape concat's length.");
    std::vector<framework::DDim> shapes = RestoreShapes(shape_concat, ranks);
    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();
    out->Reset(new RandomDataGenerator<T>(shapes, Attr<float>("min"),
                                          Attr<float>("max")));
  }
};

class CreateRandomDataGeneratorOpMaker : public FileReaderMakerBase {
 public:
  CreateRandomDataGeneratorOpMaker(OpProto* op_proto, OpAttrChecker* op_checker)
      : FileReaderMakerBase(op_proto, op_checker) {
    AddAttr<float>("min", "The lower bound of reader's uniform distribution.");
    AddAttr<float>("max", "The upper bound of reader's uniform distribution.");
    AddComment(R"DOC(
      CreateRandomDataGenerator Operator

      This Op creates a random reader.
      The reader generates random data instead of really reading from files.
      Generated data follow an uniform distribution between 'min' and 'max'.
    )DOC");
  }
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators::reader;
REGISTER_FILE_READER_OPERATOR(create_random_data_generator,
                              ops::CreateRandomDataGeneratorOp<float>,
                              ops::CreateRandomDataGeneratorOpMaker);
