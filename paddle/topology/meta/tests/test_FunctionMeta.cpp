#include <gtest/gtest.h>
#include <paddle/topology/Function.h>
#include <paddle/topology/meta/FunctionMeta.h>
#include <paddle/topology/meta/Validator.h>
TEST(FunctionMeta, cosSimMeta) {
  auto err = paddle::topology::meta::FunctionMeta::registerFuncMeta(
      "test_CosSimForward",
      [](paddle::topology::meta::FunctionMetaPtr& meta) -> paddle::Error {
        meta->addAttribute<double>("scale",
                                   "The scale of cosine similarity function")
            .defaultValue(1.0)
            .largerThan(0.0);
        return paddle::Error();
      });

  ASSERT_TRUE(err.isOK());

  paddle::topology::Function func;
  func.type = "test_CosSimForward";
  err = paddle::topology::meta::validate(func);
  ASSERT_TRUE(err.isOK());
  // Default value
  ASSERT_NEAR(paddle::any_cast<double>(func.attributes["scale"]), 1.0, 1e-5);
}
