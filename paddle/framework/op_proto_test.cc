#include <gtest/gtest.h>
#include <paddle/framework/op_proto.pb.h>

TEST(TestOpProto, ALL) {
  paddle::framework::OpProto proto;
  {
    auto ipt = proto.mutable_inputs()->Add();
    *ipt->mutable_name() = "a";
    *ipt->mutable_comment() = "the one input of cosine op";
  }
  {
    auto ipt = proto.mutable_inputs()->Add();
    *ipt->mutable_name() = "b";
    *ipt->mutable_comment() = "the other input of cosine op";
  }
  {
    auto opt = proto.mutable_outputs()->Add();
    *opt->mutable_name() = "output";
    *opt->mutable_comment() = "the output of cosine op";
  }
  {
    auto attr = proto.mutable_attrs()->Add();
    *attr->mutable_name() = "scale";
    attr->set_type(paddle::framework::AttrType::FLOAT);
    *attr->mutable_comment() = "the scale attribute of cosine op";
  }
  proto.set_type("cos");
  *proto.mutable_comment() = "cosine op, output = scale * cos(a, b)";

  ASSERT_TRUE(proto.IsInitialized());
}