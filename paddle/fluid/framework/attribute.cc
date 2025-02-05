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

#include "paddle/fluid/framework/attribute.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/utils/blank.h"

namespace paddle::framework {

paddle::any GetAttrValue(const Attribute& attr) {
  switch (AttrTypeID(attr)) {
    case proto::AttrType::INT:
      return PADDLE_GET_CONST(int, attr);
    case proto::AttrType::FLOAT:
      return PADDLE_GET_CONST(float, attr);
    case proto::AttrType::FLOAT64:
      return PADDLE_GET_CONST(double, attr);
    case proto::AttrType::STRING:
      return PADDLE_GET_CONST(std::string, attr);
    case proto::AttrType::INTS:
      return PADDLE_GET_CONST(std::vector<int>, attr);
    case proto::AttrType::FLOATS:
      return PADDLE_GET_CONST(std::vector<float>, attr);
    case proto::AttrType::STRINGS:
      return PADDLE_GET_CONST(std::vector<std::string>, attr);
    case proto::AttrType::BOOLEAN:
      return PADDLE_GET_CONST(bool, attr);
    case proto::AttrType::BOOLEANS:
      return PADDLE_GET_CONST(std::vector<bool>, attr);
    case proto::AttrType::LONG:
      return PADDLE_GET_CONST(int64_t, attr);
    case proto::AttrType::LONGS:
      return PADDLE_GET_CONST(std::vector<int64_t>, attr);
    case proto::AttrType::FLOAT64S:
      return PADDLE_GET_CONST(std::vector<double>, attr);
    case proto::AttrType::VAR:
      return PADDLE_GET_CONST(VarDesc*, attr);
    case proto::AttrType::VARS:
      return PADDLE_GET_CONST(std::vector<VarDesc*>, attr);
    case proto::AttrType::BLOCK:
      return PADDLE_GET_CONST(BlockDesc*, attr);
    case proto::AttrType::BLOCKS:
      return PADDLE_GET_CONST(std::vector<BlockDesc*>, attr);
    case proto::AttrType::SCALAR:
      return PADDLE_GET_CONST(paddle::experimental::Scalar, attr);
    case proto::AttrType::SCALARS:
      return PADDLE_GET_CONST(std::vector<paddle::experimental::Scalar>, attr);
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported Attribute value type `%s` for phi.",
          common::demangle(attr.type().name())));
  }
}

Attribute GetAttrValue(const proto::OpDesc::Attr& attr_desc) {
  switch (attr_desc.type()) {
    case proto::AttrType::BOOLEAN: {
      return attr_desc.b();
    }
    case proto::AttrType::INT: {
      return attr_desc.i();
    }
    case proto::AttrType::FLOAT: {
      return attr_desc.f();
    }
    case proto::AttrType::STRING: {
      return attr_desc.s();
    }
    case proto::AttrType::FLOAT64: {
      return attr_desc.float64();
    }
    case proto::AttrType::BOOLEANS: {
      std::vector<bool> val(attr_desc.bools_size());
      for (int i = 0; i < attr_desc.bools_size(); ++i) {
        val[i] = attr_desc.bools(i);
      }
      return val;
    }
    case proto::AttrType::INTS: {
      std::vector<int> val(attr_desc.ints_size());
      for (int i = 0; i < attr_desc.ints_size(); ++i) {
        val[i] = attr_desc.ints(i);
      }
      return val;
    }
    case proto::AttrType::FLOATS: {
      std::vector<float> val(attr_desc.floats_size());
      for (int i = 0; i < attr_desc.floats_size(); ++i) {
        val[i] = attr_desc.floats(i);
      }
      return val;
    }
    case proto::AttrType::STRINGS: {
      std::vector<std::string> val(attr_desc.strings_size());
      for (int i = 0; i < attr_desc.strings_size(); ++i) {
        val[i] = attr_desc.strings(i);
      }
      return val;
    }
    case proto::AttrType::LONG: {
      return attr_desc.l();
    }
    case proto::AttrType::LONGS: {
      std::vector<int64_t> val(attr_desc.longs_size());
      for (int i = 0; i < attr_desc.longs_size(); ++i) {
        val[i] = attr_desc.longs(i);
      }
      return val;
    }

    case proto::AttrType::FLOAT64S: {
      std::vector<double> val(attr_desc.float64s_size());
      for (int i = 0; i < attr_desc.float64s_size(); ++i) {
        val[i] = attr_desc.float64s(i);
      }
      return val;
    }

    case proto::AttrType::SCALAR: {
      return MakeScalarFromProto(attr_desc.scalar());
    }

    case proto::AttrType::SCALARS: {
      std::vector<paddle::experimental::Scalar> val(attr_desc.scalars_size());
      for (int i = 0; i < attr_desc.scalars_size(); ++i) {
        val[i] = MakeScalarFromProto(attr_desc.scalars(i));
      }
      return val;
    }

    default:
      PADDLE_THROW(common::errors::Unavailable("Unsupported attribute type %d.",
                                               attr_desc.type()));
  }
  return paddle::blank();
}

Attribute GetAttrValue(const proto::VarDesc::Attr& attr_desc) {
  switch (attr_desc.type()) {
    case proto::AttrType::INT: {
      return attr_desc.i();
    }
    case proto::AttrType::STRING: {
      return attr_desc.s();
    }
    case proto::AttrType::INTS: {
      std::vector<int> val(attr_desc.ints_size());
      for (int i = 0; i < attr_desc.ints_size(); ++i) {
        val[i] = attr_desc.ints(i);
      }
      return val;
    }
    default:
      PADDLE_THROW(common::errors::Unavailable("Unsupported attribute type %d.",
                                               attr_desc.type()));
  }
  return paddle::blank();
}

paddle::experimental::Scalar MakeScalarFromProto(const proto::Scalar& v) {
  auto data_type = v.type();
  switch (data_type) {
    case proto::Scalar_Type_BOOLEAN:
      return paddle::experimental::Scalar(v.b());
    case proto::Scalar_Type_LONG:
      return paddle::experimental::Scalar(v.i());
    case proto::Scalar_Type_FLOAT64:
      return paddle::experimental::Scalar(v.r());
    case proto::Scalar_Type_COMPLEX128: {
      phi::dtype::complex<double> value(v.c().r(), v.c().i());
      return paddle::experimental::Scalar(value);
    }
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Expected scalar of type boolean, "
          "integer, floating point or complex."));
      break;
  }
  return paddle::experimental::Scalar();
}

proto::Scalar MakeScalarProto(const paddle::experimental::Scalar& v) {
  proto::Scalar s;
  auto data_type = v.dtype();
  switch (data_type) {
    case phi::DataType::BOOL:
      s.set_b(v.to<bool>());
      s.set_type(proto::Scalar_Type_BOOLEAN);
      break;
    case phi::DataType::INT8:
    case phi::DataType::UINT8:
    case phi::DataType::INT16:
    case phi::DataType::UINT16:
    case phi::DataType::INT32:
    case phi::DataType::UINT32:
    case phi::DataType::INT64:
    case phi::DataType::UINT64:
      s.set_i(v.to<int64_t>());
      s.set_type(proto::Scalar_Type_LONG);
      break;
    case phi::DataType::FLOAT16:
    case phi::DataType::BFLOAT16:
    case phi::DataType::FLOAT32:
    case phi::DataType::FLOAT64:
      s.set_r(v.to<double>());
      s.set_type(proto::Scalar_Type_FLOAT64);
      break;
    case phi::DataType::COMPLEX64:
    case phi::DataType::COMPLEX128: {
      auto value = v.to<phi::dtype::complex<double>>();
      auto* complex = s.mutable_c();
      complex->set_r(value.real);
      complex->set_i(value.imag);
      s.set_type(proto::Scalar_Type_COMPLEX128);
      break;
    }
    case phi::DataType::UNDEFINED:
    case phi::DataType::PSTRING:
    case phi::DataType::NUM_DATA_TYPES:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Expected scalar of type boolean, "
          "integer, floating point or complex."));
      break;
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Expected scalar of type boolean, "
          "integer, floating point or complex."));
      break;
  }
  return s;
}

paddle::experimental::Scalar MakeScalarFromAttribute(const Attribute& v) {
  auto attr_type = static_cast<proto::AttrType>(v.index() - 1);
  switch (attr_type) {
    case proto::AttrType::SCALAR:
      return paddle::experimental::Scalar(
          PADDLE_GET_CONST(paddle::experimental::Scalar, v));
    case proto::AttrType::BOOLEAN:
      return paddle::experimental::Scalar(PADDLE_GET_CONST(bool, v));
    case proto::AttrType::INT:
      return paddle::experimental::Scalar(PADDLE_GET_CONST(int, v));
    case proto::AttrType::LONG:
      return paddle::experimental::Scalar(PADDLE_GET_CONST(int64_t, v));
    case proto::AttrType::FLOAT:
      return paddle::experimental::Scalar(PADDLE_GET_CONST(float, v));
    case proto::AttrType::FLOAT64:
      return paddle::experimental::Scalar(PADDLE_GET_CONST(double, v));
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unable to construct Scalar from given Attribute of type %s",
          attr_type));
  }
}

std::vector<paddle::experimental::Scalar> MakeScalarsFromAttribute(
    const Attribute& v) {
  auto attr_type = static_cast<proto::AttrType>(v.index() - 1);
  switch (attr_type) {
    case proto::AttrType::SCALARS:
      return PADDLE_GET_CONST(std::vector<paddle::experimental::Scalar>, v);
    case proto::AttrType::BOOLEANS:
      return experimental::WrapAsScalars(
          PADDLE_GET_CONST(std::vector<bool>, v));
    case proto::AttrType::INTS:
      return experimental::WrapAsScalars(PADDLE_GET_CONST(std::vector<int>, v));
    case proto::AttrType::LONGS:
      return experimental::WrapAsScalars(
          PADDLE_GET_CONST(std::vector<int64_t>, v));
    case proto::AttrType::FLOATS:
      return experimental::WrapAsScalars(
          PADDLE_GET_CONST(std::vector<float>, v));
    case proto::AttrType::FLOAT64S:
      return experimental::WrapAsScalars(
          PADDLE_GET_CONST(std::vector<double>, v));
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unable to construct Scalars from given Attribute of type %s",
          attr_type));
  }
}

void CanonicalizeScalarAttrs(const proto::OpProto& op_proto,
                             AttributeMap* attrs) {
  PADDLE_ENFORCE_NOT_NULL(
      attrs, common::errors::InvalidArgument("attrs can not be nullptr"));
  for (auto& attr : op_proto.attrs()) {
    proto::AttrType attr_type = attr.type();
    const std::string& attr_name = attr.name();
    auto it = attrs->find(attr_name);
    if (it == attrs->end()) {
      continue;
    }
    proto::AttrType actual_attr_type = AttrTypeID(it->second);
    if (actual_attr_type == attr_type) {
      continue;
    }
    if (actual_attr_type == proto::AttrType::VAR ||
        actual_attr_type == proto::AttrType::VARS) {
      continue;  // VAR& VARS are not proper attribute
    }
    if (attr_type == proto::AttrType::SCALAR) {
      it->second = Attribute(MakeScalarFromAttribute(it->second));
    } else if (attr_type == proto::AttrType::SCALARS) {
      it->second = Attribute(MakeScalarsFromAttribute(it->second));
    }
  }
}
}  // namespace paddle::framework
