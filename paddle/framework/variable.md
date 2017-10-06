# Design Doc: Variable


Variable is also known as *blob* in MxNet and Caffe2.  It is the input and output type of operators, where a neural network is a graph of operators.

## Requirements: Lazy Memory Allocation

For the flexibility of a DL system, a variable should be able to contain any typed value -- a tensor in most cases, but could also be some integer IDs or a scope of other variables in the case of RNN.

To use the minimum amount of memory, we would like that a variable allocates memory only when it has to, or, lazy memory allocation.  Let's take the following example:

```cpp
Variable vr, v1, v2;

Tensor* t1 = new Tensor();
Tensor* t2 = new Tensor();

Randomize(
  /* malloc */ v1.GetMutable<Tensor>().mutable_data<float16>(DDim(100,200)),
  /* size */ t1.Size());
  
Randomize(
  /* malloc */ v2.GetMutable<Tensor>().mutable_data<float16>(DDim(200,300)),
  /* size */ t2.Size());
  
Mult(
  /*result*/ vr.GetMutable<Tensor>().mutable_data<v1.Type()>(SizeOfMult(v1, v2)),
  /*input1*/ v1.Get<Tensor>().data(),
  /*input2*/ v2.Get<Tensor>().data());
```
     
We see that a variable holds nothing until `Variable::GetMutable<Tensor>()` allocates a tensor and puts it in the variable.  Similarly, a tensor gets its memory until `Tensor::mutable_data()`.

This syntax for lazy memory allocation when we call `Randomize` and `Mult`, those functions that mutate the variable, so it saves us some line of C++ code.


## Implementation: Type Hiding

To make memory allocation lazy, we cannot assume that we know the type held by a variable at definition time.  In other words, `class Variable` cannot be a template `template <T> class Variable`.

Because we don't know the type `T`, we cannot save a `T*` as `Variable's` data member.  Instead, we save an interface object `Placeholder`, which can return the pointer to the saved object via `Placeholder::Ptr()` as `void*`.

But anyway, Variable needs to know `T` so could it `delete<T>(ptr)` and so could `Variable::Get` checks the expected type and the saved object's type.

We save `T` in `PlaceholderImpl`, the implementation of `Placeholder`.  Please be aware that `PlaceholderImpl` is a class template and `T` is passed in as a template parameter.

Because `PlaceholderImpl` knows `T`, it can save and return `typeid(T)` for the type comparison in `Variable::Get` and `Variable::GetMutable`.


## Conclusion

The technique type hiding utilizes C++ class templates, interface and derivation, and C++ RTTI (typeid).  This combination saves us from defining something like `caffe2::TypeMeta`, which takes hundreds of lines of C++ code.
