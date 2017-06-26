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
