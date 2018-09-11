# Design Doc: Functions, Operators, and Layers

In a DL system, we can compose one or more fine grained operators into a coarse grained one.  For example, the FC layer can be composed of a multiplication operator and an add operator.

Historically, some fine grained operations are known as operators, and some coarse level ones are known as layers.  But we need a well-defined separation.

In general, operators are those very fine grained operations, e.g., mul and add. In the implementation, we can write them as C++ functions:

```c++
template <typename T> T add(T x, T y) { return x + y; }
template <typename T> T mul(T x, T y) { return x * y; }
```

Then we can wrap them into operators which are C++ classes and can be created from Python bindings by name.  A C macro can do this. For example, the following macro invocation

```c++
#define MAKE_FUNCTION_OPERATOR(mul);
```

generates

```c++
template <typename T> class mulOp : public OperatorBase {...};
REGISTER_OP(mulOp<float32>, "mul");
```

so that in Python we can create operator mul by:

```python
X1 = Var()
X2 = Var()
Y = Var()
paddle.cpp.create_operator("mul", input=[X1, X2], output=Y)
```

Also, at the same time, we can compose a coarse level C++ operator class by composing functions `mul` and `add`:

```c++
template <typename T>
class FCOp : public OperatorBase {
 public:
  void Run(...) {
    add(mul(Input<T>("X"), Input<T>("W")), Input<T>("b"));
  }
};
REGISTER_OP(FCOp, "fc");
```

We need to support such composition in Python as well.  To do so, we need a higher level Python wrapping of operator creation than `paddle.cpp.create_operator`.  This higher level operator API should be compatible with the layer API.

Let's explain using an example.  Suppose that we are going to compose the FC using mul and add in Python, we'd like to have Python functions `mul` and `add` defined in module `operator`:

```python
def operator.mul(X1, X2):
    O = Var()
    paddle.cpp.create_operator("mul", input={X1, Y1}, output=O)
    return O

def operator.add(X1, X2):
    O = Var()
    paddle.cpp.create_operator("add", input={X1, X2}, output=O)
    return O
```

Above code snippets are automatically generated.  Given them, users can define

```python
def layer.fc(X):
    W = Var()
    b = Var()
    return operator.add(operator.mul(X, W), b)
```

If we don't have `operator.mul` and `operator.add`, the definiton of `layer.fc` would be complicated:

```python
def layer.fc(X):
    W = Var()
    b = Var()
    O1 = Var()
    paddle.cpp.create_operator("mul", input=[X, W], output=O1)
    O2 = Var()
    paddle.cpp.create_operator("add", input=[O1, b], output=O2)
    return O2
```

We'd like to have Python bindings to operators in package `paddle.operator`, and Python compositions of operators in package `paddle.layer`.  So we have the following concepts in above illustrative example:

<table>
<thead>
<tr>
<th>C++ functions/functors</th>
<th>mul</th>
<th>add</th>
<th></th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td>C++ operator class </td>
<td>mulOp</td>
<td>addOp </td>
<td>FCOp </td>
<td></td>
</tr>
<tr>
<td>Python binding  </td>
<td>operator.mul</td>
<td> operator.add </td>
<td>operator.fc </td>
<td></td>
</tr>
<tr>
<td>Python function   </td>
<td></td>
<td></td>
<td> </td>
<td>layer.fc</td>
</tr>
</tbody>
</table>


This is how we differentiate layer and operators in PaddlePaddle:

- those defined in C++ and have a lightweighted Python wrapper in module `operators` are operators; whereas
- those who don't have C++ implementations but a Python implementation that compose C++ operators are known as layers.
