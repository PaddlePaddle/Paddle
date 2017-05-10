#include <gtest/gtest.h>
#include <paddle/topology/Function.h>
#include <paddle/topology/meta/FunctionMeta.h>
#include <paddle/topology/meta/Validator.h>
#include "paddle/function/BufferArgs.h"

TEST(FunctionMeta, cosSimMeta) {
  auto err = paddle::topology::meta::FunctionMeta::registerFuncMeta(
      "test_CosSimForward",
      [](paddle::topology::meta::FunctionMetaPtr& meta) -> paddle::Error {
        meta->addAttribute<double>("scale",
                                   "The scale of cosine similarity function")
            .defaultValue(1.0)
            .largerThan(0.0);

        meta->addInput()
            ->addShape(2)
            .addDataType({paddle::topology::DataType::DENSE})
            .addSequenceType();
        meta->addInput()
            ->addShape(2)
            .addDataType({paddle::topology::DataType::DENSE})
            .addSequenceType();

        meta->addOutput()
            ->addShape(2)
            .addDataType({paddle::topology::DataType::DENSE})
            .addSequenceType();

        meta->setShapeInferer(
            [](std::vector<paddle::topology::TensorPtr>& in,
               std::vector<paddle::topology::TensorPtr>& out) {
              if (in[0]->shape() != in[1]->shape()) {
                return paddle::Error("Cosine input tensor shape mismatch");
              }

              if (in[0]->sequenceType() != in[1]->sequenceType()) {
                return paddle::Error(
                    "Cosine input tensor sequence type must be same.");
              }

              out[0]->setShape({in[0]->shape()[0], 1});
              out[0]->setDataType(paddle::topology::DataType::DENSE);
              out[0]->setSequenceType(in[0]->sequenceType());

              return paddle::Error();
            });

        return paddle::Error();
      });

  ASSERT_TRUE(err.isOK());

  paddle::topology::Function func;
  func.type = "test_CosSimForward";
  func.inputs.emplace_back(new paddle::topology::Tensor());
  auto& in0 = func.inputs.back();
  in0->setDataType(paddle::topology::DataType::DENSE)
      .setSequenceType(paddle::topology::SequenceType::SEQUENCE)
      .setShape({890, 200});
  func.inputs.emplace_back(new paddle::topology::Tensor());
  auto& in1 = func.inputs.back();
  in1->setDataType(paddle::topology::DataType::DENSE)
      .setSequenceType(paddle::topology::SequenceType::SEQUENCE)
      .setShape({890, 200});
  func.outputs.emplace_back(new paddle::topology::Tensor());

  err = paddle::topology::meta::validateAndInferShape(func);
  ASSERT_TRUE(err.isOK());
  // Default value
  ASSERT_NEAR(paddle::any_cast<double>(func.attributes["scale"]), 1.0, 1e-5);
  ASSERT_EQ(func.outputs.back()->shape(), std::vector<int>({890, 1}));
}
