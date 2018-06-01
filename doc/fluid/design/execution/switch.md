# Design Doc: Switch

## Background

Many programming languages provide `switch` as a generalization of `if-elif-else`.  We want to add it to Fluid.

The following example shows the usage of `fluid.switch`.

```python
a = fluid.Var(10)
b = fluid.Var(0)

with switch() as switch:
    with switch.case(fluid.less_equal(a, 10)):
        fluid.print("Case 1")
    with switch.case(fluid.larger(a, 0)):
        fluid.print("Case 2")
    with switch.default():
        fluid.print("Case 3")
```

## The Semantics

1. A `switch` control-flow checks cases one-by-one.
1. The condition of each case is a boolean value, which is a scalar, and differs from the `fluid.if_else` control-flow, which condition could be a vector of boolean values.
1. It runs the first matched case, or the default case if there is one.
1. Once it matches a case, it runs the corresponding branch and only that branch.  It's like there is a C's `break` keyword at the end of each case.

The above program should print and print only "Case 1".

The implementation of the backward pass of the `switch` control-flow is easier than the backward of the `if_else`, because `switch` runs at most one branch, whereas `if-else` could run more than one branches.
