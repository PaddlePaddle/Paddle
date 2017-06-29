# Design Doc: Operator Attributes

## Background

An operator could have attributes. For example, CosineOp could have a float typed attribute scale, which changes the output range from [-1,1] to [-scale,scale]. The default value of scale is `1.0`.

Attributes is defined by a name and a type. An instance of an attribute has a value of that type.

As part of the network description, attribute need to be serialized. So we need a protobuf message that describes an attribute, say `Attribute`.

An operator could parse the Attribute and save them into its private data member.

## Protobuf Implementation

There are two frameworks implement `Attribute` concept in `protobuf`. They are [`caffe2`](https://github.com/caffe2/caffe2/blob/master/caffe2/proto/caffe2.proto#L98) and [`tensorflow`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto#L16).

* Caffe2 uses `proto2` syntax. It treats all attributes as a list, and each attribute contains a `name`. Each time caffe2 read an attribute is searching a variable in a list. It is slow if the number of attributes is large. Caffe2 also mark all field as `optional`. It doesn't ensure `one of` attribute value is set.
* By using `proto3` syntax in tensorflow, the attribute implementation in tensorflow is using `map`, and `oneof` keywords. Looking up from attribute map in tensorflow is fast.

Paddle is using `protobuf 3` as its dependency library. By simplify `tensorflow`'s implementation, Paddle's Attribute protobuf message schema could be

```protobuf
message Attribute {
    message ListValue {
        repeated int32 ints = 1;
        repeated float floats = 2;
        repeated string strings = 3;
    }

    oneof value {
        ListValue list = 1;
        int32 i = 2;
        float f = 3;
        string s = 4;
    }
}
```

In `OperatorDescription` message, there should be a field like this:

```protobuf
message OperatorDescription {
  map<string, Attribute> attrs;
}
```

## CPP implementation

### AttributeReader

In CPP, it should be a helper class for reading `map<string, Attribute>`. The reading method should accept a template parameter, which is the type of Attribute.  If type mismatch or attribute is not found, `Get` method should return an `Error`. That helper class we named `AttributeReader`.

The interface of `AttributeReader` is like this:

```cpp
using AttributeMap = google::protobuf::Map<std::string, Attribute>;
class AttributeReader {
 public:
  explicit AttributeReader(const AttributeMap& attrs) : attrs_(attrs) {}

  template <typename T>
  T Get(const std::string& attributeName) const;

  template <typename T>
  void GetArray(const std::string& attributeName, std::vector<T>* array) const;

  template <typename T>
  bool Contains(const std::string& name) const;

 private:
  const AttributeMap& attrs_;
};
```

There are two methods in `AttributeReader`: `Get` and `GetArray`. `GetArray` is used for `ListValue`, and `Get` is used for the rests. The user should invoke either of them when he wants to get an Attribute value from `AttributeMap`.

### Attribute in Operator

Each operator stores its attributes. For faster attribute access, we should not let user parse `AttributeMap` during `Run` method in Operator. When `NetworkBase` adds an operator to computation graph, the `Attribute` could be parsed, and stored in each operator's the private member.

```cpp
class OperatorBase {
 public:
  virtual void InitializeAttribute(const AttributeReader& attrs) = 0;
};

class CosineOp : public OperatorBase {
 public:
  void InitializeAttribute(const AttributeReader& attrs) {
    if (attrs.Contain<float>("scale")) {
      scale_ = attrs.Get<float>("scale");
      PADDLE_ENFORCE(scale_ > 0.0f, "Scale of consine op should be larger than 0.0");
    }
  }

 private:
  float scale_ {1.0};
};
```

When `NetworkBase` invokes `CreateOperator(const OperatorDescription& desc)`, it create an operator first. Then `CreateOperator` will invoke `InitializeAttribute`. The implementation of `CreateOperator` could be

```cpp
std::unique_ptr<OperatorBase> CreateOperator(const OperatorDescription& desc) {
  std::unique_ptr<OperatorBase> op(OperatorRegister.Create(
          desc.type(), desc.inputs(), desc.outputs()));
  op->InitializeAttribute(AttributeReader(desc.attrs()));
  return std::move(op);
}
```
