# Paddle多语言接口实现
## 背景

Paddle需要一个多语言接口，这个接口需要做到:

* 有标准的，良好的文档
    * 例如Python可以使用[Sphinx](http://www.sphinx-doc.org/en/stable/)生成API文档，golang可以使用[GoDoc](https://godoc.org/golang.org/x/tools/cmd/godoc)生成文档。这都需要这个接口按照约定俗成的规则来注释完备。
* 不同语言的接口适应不同语言的特性
    * 例如Java与Python的错误处理是直接扔出来Exception，而对于golang错误处理应该使用返回值。

## 基本要求

Paddle的多语言接口实现包括一下几个方面:

* 我们使用动态库来分发Paddle。在这个动态库中不嵌入任何其他语言的解释器，也不使用其他动态库。
* 这个动态库使用C99标准的头文件导出一些函数，不使用/导出C++符号。
* 不导出Paddle内部的结构体、类，仅仅使用`void*`指针作为类型的句柄(handler)。
* 不使用SWIG这种代码生成器，而是手写多语言绑定。


## 原因

### 使用动态库来分发Paddle

* Paddle的链接方式比较复杂
    * 如果用户要把Paddle的静态库（libpaddle.a）链接到自己的程序里，得使用 `--whole-archive` (for GCC) 或者 `--force_load` (for Clang) 参数，来确保把 libpaddle.a 里所有的符号都写入自己的程序的二进制文件里。这是因为 Paddle 的源码里使用了[object factory design pattern](http://stackoverflow.com/a/1310326/724872)。
* 编译型语言，例如C/C++使用静态库和动态库难度差不多。但是解释性语言，例如[Python](http://stackoverflow.com/questions/19560594/how-to-import-static-library-in-python)或者[Java](http://stackoverflow.com/questions/24493337/linking-static-library-with-jni)，只能调用Paddle的动态库，否则得把Paddle静态库链接到解释器里。
    * 解释性语言实际运行的二进制是解释器本身，如果调用静态库只能将静态库与解释器链接。例如对于Java来说，便是将静态库加入JVM中。这对于通常的Java的开发者来说，是不常见的做法。

### 动态库中不嵌入任何其他语言的解释器

* 目前Paddle的进程模型是C++内部驱动Python解释器进行模型配置解析和数据读取
* 我们最终的动态库中不嵌入Python或者其他任何语言的解释器。模型配置解析，数据读取均交由其他语言完成

现阶段Paddle有一个问题是，Paddle内嵌的Python解释器和外部使用的Python如果版本不同，会直接报错退出。

### Paddle动态库中，不引用其他动态库

* 即这个动态库是不依赖于其他任何文件的，可以在任何机器上执行的。

###  这个动态库使用C99标准的头文件导出一些函数，不使用/导出C++符号

* 由于C++编译器没有[名字修饰](https://en.wikipedia.org/wiki/Name_mangling#C.2B.2B)的规范，不同版本的编译器之间，对于同一段C++代码生成的符号可能不一致。而多语言接口需要直接读取生成的二进制(动态库)，需要有稳定的导出符号。
* C语言是有导出符号的标准的，并且在常见的平台上，都是ABI调用标准的。
* 大多数语言都支持使用C语言API
* 使用C99而不使用C89，是因为C99支持[Fixed-width integer types](https://en.wikipedia.org/wiki/C_data_types#Fixed-width_integer_types)和[Boolean type](https://en.wikipedia.org/wiki/C_data_types#Boolean_type)。
* 使用C99而不使用C11的原因是，[C11](https://en.wikipedia.org/wiki/C11_(C_standard_revision))并没有Paddle特别需要的特性，且C99相对于C11使用更加广泛。

### 不导出Paddle内部的结构体、类，仅仅使用`void*`指针作为类型的句柄(handler)

* Paddle内部的类为C++书写，直接导出到C的接口比较困难。
* 在C-API中使用`void*`来表示Paddle内部类。再在每一个API中自己检查类型。

在C的头文件 `paddle_matrix.h` 中:

```C
typedef void* paddle_matrix;
typedef int paddle_error;

extern "C"
paddle_error paddle_matrix_shape(paddle_matrix matrix,
                                 uint64_t* width,
                                 uint64_t* height);
```
而在CPP里面实现这个C的接口，文件 `paddle_matrix.cpp`

```cpp
#include "paddle/math/matrix.hpp"
extern "C"
paddle_error paddle_matrix_shape(paddle_matrix matrix,
                                 uint64_t *width,
                                 uint64_t *height) {
  auto m = (paddle::math::matrix*)(matrix);
  *width = m->width();
  *height = m->height();
}
```

其中`paddle/math/matrix.hpp`文件内容为:

```cpp
namespace paddle {
namespace math {  

class Matrix {
  //...
};

}  // namespace math
}  // namespace paddle
```

### 不使用SWIG这种代码生成器，而是手写多语言绑定

* [SWIG](http://www.swig.org/)是一个多语言接口的代码生成器。他的目标是使用C/C++写代码，SWIG直接读取C/C++的头文件，生成各种语言的绑定代码。
    * 对于多语言接口，SWIG需要写一个interface文件。这个文件具有独特的语法，学习成本高。且增加一个第三方语言，就需要对这个第三方语言增加一些定义。有的时候，interface文件的写法非常[tricky](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/api/Paddle.swig#L36)。社区贡献代码学习成本高。
    * SWIG暴露的接口保留了C++的接口样式，很难保证多语言代码风格的一致性。(函数命名，错误处理)
        * 因为SWIG在第三方语言中暴露的函数名，类名和C++中完全一致。C++的命名风格并不能适应其他第三方语言。如果使用SWIG我们需要将在interface文件里，将大量的`SomeCppClass`重命名成`some_python_class`，或者`SomeGoTypes`。
        * 对于不同语言，错误处理的方式也不尽相同。例如对于Java或者Python，最常见的错误处理方式是Exception，而对于Golang，错误处理方式是返回值。而SWIG只能简单的暴露C++接口，无法做到对于各种语言错误处理方式的适配。
    * 对于大多数语言，直接使用C语言的.h并不困难。例如Python的[cffi](https://cffi.readthedocs.io/en/latest/overview.html#simple-example-abi-level-in-line)或者[Cython](http://cython.org/), golang的[cgo](https://golang.org/cmd/cgo/)。
    * SWIG支持的语言或者解释器有局限。例如对于Python，使用SWIG只支持CPython解释器，而不支持PyPy解释器。


## 原因列表

| 结论 | 对比 | 原因 |
|---| --- | --- |
| 使用动态库 | 不使用静态库 | 解释型语言只能调用动态库，Paddle静态库链接复杂 |
| 不嵌入其他语言解释器 | 不嵌入Python解释器 | Paddle C++目前嵌入Python解释器，会导致不同版本Python在一个进程里的bug |
| 不引用其他动态库 | | Paddle一个动态库可以在任何Linux系统上运行 |
| 使用C99做接口 | 不使用C++做接口 | C有标准的ABI，C99是目前C最广泛的使用标准，且C99支持bool类型和定长整数(uint64_t等)类型 |
| 使用void*作为类句柄 | 不显示的写每个类具体包含什么| 实现简单，并且让接口脱离实现细节 |
| 手写多语言绑定 | 不使用SWIG | 使用SWIG需要多语言绑定的开发人员熟练掌握SWIG配置，社区参与困难。SWIG生成的代码不能保证多语言代码风格的一致性 |


## 简单实现

TBD
