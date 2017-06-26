# Attribute of operator

## background

In a neural network, each operator could contain some configurable attributes. For example, a cosine similarity operator may contain an attribute named `scale`. The default cosine similarity returns a value in range [-1.0, 1.0]. But the user can set range scale manually, e.g., user set `scale=5.0`, then that cosine operator will return a value in the range [-5.0, 5.0].

The configurable attributes could be various types. Some operators need `float` value to configure; some need `string` value.  We need a data structure to represent different types.

Each operator contains different configurable attributes. The names of attributes are not same.  We need an associate map from attribute name to attribute value for `Operator`.

Also as we want to use `protobuf` to serialize and deserialize our model, we need to implement the attribute value and the associated map from attribute name to attribute value in `protobuf`.

In conclusion, there are four things we know as background.

1. We need an attribute type for Operator.
1. That attribute type could represent different types.
1. That attribute value should be associated with an attribute name, like a map<string, Attribute>.
1. We need to implement them in `protobuf`.

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
