#include <gtest/gtest.h>
#include <paddle/topology/Tensor.h>
#include <paddle/topology/meta/TensorMeta.h>
#include <paddle/topology/meta/Validator.h>
#include <type_traits>
TEST(TensorMeta, check) {
  paddle::topology::meta::TensorMeta meta;
  meta.supportDataTypes({paddle::topology::DataType::DENSE});
  meta.supportSequenceTypes({paddle::topology::SequenceType::NO_SEQUENCE});
  meta.setShapeDimension(2);

  paddle::topology::Tensor tensor;
  tensor.setDataType(paddle::topology::DataType::DENSE);
  tensor.setSequenceType(paddle::topology::SequenceType::NO_SEQUENCE);
  tensor.setShape({1, 200});
  auto err = paddle::topology::meta::validate(meta, tensor.attributes);
  ASSERT_TRUE(err.isOK());

  tensor.setShape({1, 200, 300});
  err = paddle::topology::meta::validate(meta, tensor.attributes);
  ASSERT_FALSE(err.isOK());
  tensor.setShape({1, 200});
  tensor.setDataType(paddle::topology::DataType::SPARSE);
  err = paddle::topology::meta::validate(meta, tensor.attributes);
  ASSERT_FALSE(err.isOK());
}
