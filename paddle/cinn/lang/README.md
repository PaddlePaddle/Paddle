# Design of CINN/DSL
This module is a simple DSL defined in CINN project.
The DSL module aims to represent the overall computation in a hardware independent way.

## Concepts
### Object
All the mutable elements in CINN are `Object`.
### Shared
The `Shared` objects are reference-count-self-contained container, which is similar to the `std::shared_ptr`.

One can pass a `Shared` object by passing a pointer and the consumer object should store it in a local `Shared` member variable.

## Tensor

The input or the temporary output node.

Every `Compute` will output a Tensor, the tensor can be sliced.



### PlaceHolder

The special tensor that represents a input slot.

```c++
PlaceHolder<float> A("A", {M, N});
PlaceHolder<float> B("B", {M, N});
```

## Operation

The Operation is the operation on tensors, including

- placeholder
- compute
- bound inference

```c++
Tensor C = Compute({M,N}/*output shape*/, [&](Var i, Var j) {
  Var k;
  return ReduceSum(A[i,k] * B[k,j], {k});
});
```

### Bound inference

The PlaceHolder should define a shape.

```c++
Var M(Int(32));
Var N(Int(32));

PlaceHolder<float> A({M, N});

Var i,j;
Expr tmp = A[i][j] + 1; // i \in {0, M}; j \in {0, N}
```

To simplify the implementation, we use ISL to generate code for basic snippets.

## Schedule

The schedule will

1. determine the order of computation, by topological sorting the computational graph composed of tensors.
2. transforming the computations

### order schedule

1. Topological sort the tensors
2. for each tensor, generate the code it needs.

## Some examples
A matrix multiplication

```c++
// Declare some iterator variables.
Var i, j, k;
Placeholder<float> A({M, K}), B({K, N});

Tensor C = Compute({M, N}/*output shape*/,
        [](Var i, Var j) {
            return ReduceSum(A(i,k) * B(k, j), k);
        }, "C");
Tensor D = Compute({M, N}, [](Var i, Var j) {
  return Map(C(i,j) + 1);
});

Schedule s = CreateSchedule(C);
auto func = Build(s, [A, B, C], target=target, name="matmul");

func(a, b, c);
```
