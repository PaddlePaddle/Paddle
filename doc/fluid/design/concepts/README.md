A few months ago when we were trying to replace CMake with Bazel, @emailweixu suggested that we rewrite those handy Bazel functions using CMake. Now it seems that it's the right time to get this done, as we are facing problems from the porting of Majel and the development of new the parameter server using Go and C++.

Here are some initial thoughts. Your comments are welcome!

# Required CMake Function

I think we need only the following few CMake functions to make a project description mean and clean:

<table>
<thead>
<tr>
<th>C++</th>
<th>CUDA C++</th>
<th>Go</th>
</tr>
</thead>
<tbody>
<tr>
<td>cc_library </td>
<td>nv_library </td>
<td>go_library </td>
</tr>
<tr>
<td>cc_binary </td>
<td>nv_binary </td>
<td>go_binary </td>
</tr>
<tr>
<td> cc_test </td>
<td> nv_test </td>
<td> go_test </td>
</tr>
</tbody>
</table>


- The `_library` functions generate  .a files from source code.
- The `_binary` functions generate executable binary files.
- The `_test` functions generate executable unit test files. They work like `_binary` but links `-lgtest` and `-lgtest_main`.

The difference between `nv_` functions and `cc_` functions is that the former use `nvcc` instead of the system-default C++ compiler.

Both `nv_` and `cc_` functions enables C++11 (-std=c++11).

Also,

- to describe external dependencies, we need `external_library`.
- to build shared libraries, we need `shared_library`.

## An Example Project

Suppose that we have aforementioned functions defined in our `/cmake` directory.  The following example `CMakeLists.txt` describes a project including the following source files:

- tensor.h
- tensor.cc
- tensor_test.cc
- ops.h
- ops.cu
- ops_test.cu
- api.go
- api_test.go

Suppose that ops.cu depends on CUDNN.

```cmake
# cc_binary parses tensor.cc and figures out that target also depend
# on tensor.h.
cc_binary(tensor
  SRCS
  tensor.cc)

# The dependency to target tensor implies that if any of
# tensor{.h,.cc,_test.cc} is changed, tensor_test need to be re-built.
cc_test(tensor_test
  SRCS
  tensor_test.cc
  DEPS
  tensor)

# I don't have a clear idea what parameters external_library need to
# have.  @gangliao as a CMake expert would have better ideas.
external_library(cudnn
  ....)

# Suppose that ops.cu depends on external target CUDNN.  Also, ops.cu
# include global functions that take Tensor as their parameters, so
# ops depend on tensor.  This implies that if any of tensor.{h.cc},
# ops.{h,cu} is changed, ops need to be re-built.
nv_library(ops
  SRCS
  ops.cu
  DEPS
  tensor
  cudnn)  # cudnn is defined later.

nv_test(ops_test
  SRCS
  ops_test.cu
  DEPS
  ops)

# Because api.go defines a GO wrapper to ops and tensor, it depends on
# both.  This implies that if any of tensor.{h,cc}, ops.{h,cu}, or
# api.go is changed, api need to be re-built.
go_library(api
  SRCS
  api.go
  DEPS
  tensor # Because ops depend on tensor, this line is optional.
  ops)

go_test(api_test
  SRCS
  api_test.go
  DEPS
  api)


# This builds libapi.so.  shared_library might use CMake target
# api_shared so to distinguish it from above target api.
shared_library(api
  DEPS
  api)

```

## Implementation

As above example CMakeLists.txt executes, each function invocation adds "nodes" to a dependency graph.  It also use this graph to generate CMake commands including `add_executable`, `add_dependencies`, `target_link_libraries`, and `add_test`.

## Using Package Manager For Go

Building Go binaries and libraries need to satisfy their dependencies, generally
we can do `go get ./...` to download and compile all external dependencies. The
problems are:

1. `go get` will always get the latest code from the default branch of the
    remote repo, so changes of dependents might break the build. This is very
    different with what we already have in `cmake/external` which download a
    specific version or commit id of the dependency.
1. Some locations can not access external dependencies through the internet, as mentioned
   in https://github.com/PaddlePaddle/Paddle/issues/2605. Using package management
   tools can package the dependencies as a "vendor" package, which can be mirrored
   at many cloud file hosting, so users what to compile paddle by themselves can
   download this "vendor" package from a mirror site.

### Choose A Suitable Tool

As mentioned by @wangkuiyi, [Here](https://github.com/golang/go/wiki/PackageManagementTools)
list dozens of Go package managers. We choose the tool using following principles:

- Most "active" projects with more stars, more pull requests or commits
- Widely used project

After comparing all these projects, we shall choose between the most popular
tools: Godep and Glide.

Here's a brief comparison between Godep and Glide
: https://github.com/Masterminds/glide/wiki/Go-Package-Manager-Comparison. There are
also many complaints about using `Godep`. There's also a new "official" pakcage
management tool has been started at: https://github.com/golang/dep to resolve
such problems, but it's currently at Alpha stage. So the best choice now is
glide obviously.

### Manage Go Packages

- Dependencies: `go/glide.yaml` will store the dependencies and their versions which
  is directly imported by paddle. `go/glide.lock` will store all dependencies recursively
  with their commit id. Builds will "lock" to these packages if we don't `glide up`
  them
- Vendor package: `go/vendor` directory will generated when running `cmake` command. `cmake`
  will download the code corresponding to `go/glide.lock`. If we put a vendor folder
  under `go/`, cmake will just check the commit id to the packages under the folder,
  if commit id matches, there will be no download at all.
