## 堆内存分析和优化

计算机程序都可能有内存泄露的风险。**内存泄露**一般是由于程序在堆(heap)上分配了内存而没有释放，随着程序的运行占用的内存越来越大，一方面会影响程序的稳定性，可能让运行速度越来越慢，或者造成oom，甚至会影响运行程序的机器的稳定性，造成宕机。


目前有很多内存泄露分析工具，比较经典的有[valgrind](http://valgrind.org/docs/manual/quick-start.html#quick-start.intro), [gperftools](https://gperftools.github.io/gperftools/)。

因为Fluid是用Python驱动C++ core来运行，valgrind直接分析非常困难，需要自己编译debug版本的、带valgrind支持的专用版本，而且输出的信息中大部分是Python自己的符号和调用信息，分析起来很困难，所以不建议使用。

本教程主要介绍[gperftools](https://gperftools.github.io/gperftools/)的使用。

gperftool主要支持以下四个功能：

- thread-caching malloc
- heap-checking using tcmalloc
- heap-profiling using tcmalloc
- CPU profiler

Paddle也提供了基于gperftool的[CPU性能分析教程](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/howto/optimization/cpu_profiling_cn.md)。

对于堆内存的分析，主要用到thread-caching malloc和heap-profiling using tcmalloc。

## 使用流程
#### 环境
本教程基于paddle提供的Docker开发环境paddlepaddle/paddle:latest-dev，基于Ubuntu 16.04.4 LTS环境。

#### 使用流程

- 安装google-perftools

```
apt-get install libunwind-dev 
apt-get install google-perftools
```

- 安装pprof

```
go get -u github.com/google/pprof
```

- 设置运行环境

```
export PPROF_PATH=/root/gopath/bin/pprof
export PPROF_BINARY_PATH=/root/gopath/bin/pprof
export LD_PRELOAD=/usr/lib/libtcmalloc.so.4
```

- 使用heap profile来运行python程序。本质上是周期性的对堆的分配情况做一次快照。

```
# HEAPPROFILE 设置生成的堆分析文件的目录和文件前缀
# HEAP_PROFILE_ALLOCATION_INTERVAL 设置每分配多少存储dump一次dump，默认1GB
env HEAPPROFILE="./perf_log/test.log" HEAP_PROFILE_ALLOCATION_INTERVAL=209715200 python trainer.py
```

随着程序的运行，会在perf_log这个文件夹下生成很多文件，如下：

```
-rw-r--r-- 1 root root 1.0M Jun  1 15:00 test.log.0001.heap
-rw-r--r-- 1 root root 1.0M Jun  1 15:00 test.log.0002.heap
-rw-r--r-- 1 root root 1.0M Jun  1 15:00 test.log.0003.heap
-rw-r--r-- 1 root root 1.0M Jun  1 15:00 test.log.0004.heap
-rw-r--r-- 1 root root 1.0M Jun  1 15:00 test.log.0005.heap
-rw-r--r-- 1 root root 1.0M Jun  1 15:00 test.log.0006.heap
```

- 使用pprof对heap文件进行分析。分析有两种模式：
	- 完整模式。会对当前heap做一个分析，显示目前分配内存一些调用路径。

	```
	pprof --pdf python test.log.0012.heap
	```
	上述命令会生成一个profile00x.pdf的文件，可以直接打开，例如：[allocator](./memory_cpu_allocator.pdf)。从下图可以看出，在CPU版本fluid的运行过程中，分配存储最多的模块式CPUAllocator. 而别的模块相对而言分配内存较少，所以被忽略了，这对于分配内存泄露是很不方便的，因为泄露是一个缓慢的过程，在这种图中是无法看到的。
	
	![result](memory_cpu_allocator.jpg)
	
	- Diff模式。可以对两个时刻的heap做diff，把一些内存分配没有发生变化的模块去掉，而把增量部分显示出来。
	```
	pprof --pdf --base test.log.0010.heap python test.log.1045.heap
	```
	生成的结果为：[`memory_leak_protobuf`](./memory_leak_protobuf.pdf)
	
	从图中可以看出：ProgramDesc这个结构，在两个版本之间增长了200MB+，所以这里有很大的内存泄露的可能性，最终结果也确实证明是这里造成了泄露。
	
	![result](memory_program_desc.png) 
	![result](memory_protobuf_arena.png)
	
