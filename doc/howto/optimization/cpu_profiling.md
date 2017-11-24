此教程会介绍如何使用Python的cProfile包，与Python库yep，google perftools来运行性能分析(Profiling)与调优。

Paddle提供了Python语言绑定。用户使用Python进行神经网络编程，训练，测试。Python解释器通过`pybind`和`swig`调用Paddle的动态链接库，进而调用Paddle C++部分的代码。故，Paddle的性能分析与调优分为两个部分:

* Python代码的性能分析
* Python与C++混合代码的性能分析


## Python代码的性能分析

### 生成性能分析文件

Python标准库中提供了性能分析的工具包，[cProfile](https://docs.python.org/2/library/profile.html)。生成Python性能分析的命令如下:

```bash
python -m cProfile -o profile.out main.py
```

其中`-o`标识设置了一个输出文件名用来存储本次性能分析的结果。如果不指定这个文件，`cProfile`会打印一些统计信息到`stdout`。这不方便我们进行后期处理(进行`sort`, `split`, `cut`等等)。

### 查看性能分析文件

当main.py运行完毕后，性能分析结果`profile.out`就生成出来了。我们可以使用[cprofilev](https://github.com/ymichael/cprofilev)来查看性能分析结果。`cprofilev`会开启一个HTTP服务，将性能分析结果以网页的形式展示出来。

使用`pip install cprofilev`安装`cprofilev`工具，进而使用如下命令开启HTTP服务

```bash
cprofilev -a 0.0.0.0 -p 3214 -f profile.out main.py
```

其中`-a`标识HTTP服务绑定的IP。使用`0.0.0.0`允许外网访问这个HTTP服务。`-p`标识HTTP服务的端口。`-f`标识性能分析的结果文件。`main.py`标识被性能分析的源文件。

访问对应网址，即可显示性能分析的结果。性能分析结果格式如下:

```text
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.284    0.284   29.514   29.514 main.py:1(<module>)
     4696    0.128    0.000   15.748    0.003 /home/yuyang/perf_test/.env/lib/python2.7/site-packages/paddle/v2/fluid/executor.py:20(run)
     4696   12.040    0.003   12.040    0.003 {built-in method run}
        1    0.144    0.144    6.534    6.534 /home/yuyang/perf_test/.env/lib/python2.7/site-packages/paddle/v2/__init__.py:14(<module>)
```

每一列的含义是:

| 列名 | 含义 |
| --- | --- |
| ncalls | 函数的调用次数 |
| tottime | 函数实际使用的总时间。这个时间去除掉该函数调用其他函数的时间 |
| percall | tottime的每次调用平均时间 |
| cumtime | 函数总时间。包含这个函数调用其他函数的时间 |
| percall | cumtime的每次调用平均时间 |
| filename:lineno(function) | 文件名, 行号，函数名 |


### 寻找性能瓶颈
性能调优通常需要依赖性能分析的结果。因为在程序实际运行中，真正的瓶颈可能和程序员开发过程中想象的瓶颈相去甚远。通常`tottime`和`cumtime`是寻找瓶颈的关键指标。这两个指标代表了某一个函数真实的运行时间。

将性能分析结果按照tottime排序，效果如下:

```text
     4696   12.040    0.003   12.040    0.003 {built-in method run}
   300005    0.874    0.000    1.681    0.000 /home/yuyang/perf_test/.env/lib/python2.7/site-packages/paddle/v2/dataset/mnist.py:38(reader)
   107991    0.676    0.000    1.519    0.000 /home/yuyang/perf_test/.env/lib/python2.7/site-packages/paddle/v2/fluid/framework.py:219(__init__)
     4697    0.626    0.000    2.291    0.000 /home/yuyang/perf_test/.env/lib/python2.7/site-packages/paddle/v2/fluid/framework.py:428(sync_with_cpp)
        1    0.618    0.618    0.618    0.618 /home/yuyang/perf_test/.env/lib/python2.7/site-packages/paddle/v2/fluid/__init__.py:1(<module>)

```

可以看到最耗时的函数是C++端的`run`函数。这需要联合我们第二节`Python与C++混合代码的性能分析`来进行调优。而`sync_with_cpp`函数的耗时很长，每次调用的耗时也很长。于是我们可以点击`sync_with_cpp`的详细信息，了解其调用关系。

```text
Called By:

   Ordered by: internal time
   List reduced from 4497 to 2 due to restriction <'sync_with_cpp'>

Function                                                                                                 was called by...
                                                                                                             ncalls  tottime  cumtime
/home/yuyang/perf_test/.env/lib/python2.7/site-packages/paddle/v2/fluid/framework.py:428(sync_with_cpp)  <-    4697    0.626    2.291  /home/yuyang/perf_test/.env/lib/python2.7/site-packages/paddle/v2/fluid/framework.py:562(sync_with_cpp)
/home/yuyang/perf_test/.env/lib/python2.7/site-packages/paddle/v2/fluid/framework.py:562(sync_with_cpp)  <-    4696    0.019    2.316  /home/yuyang/perf_test/.env/lib/python2.7/site-packages/paddle/v2/fluid/framework.py:487(clone)
                                                                                                                  1    0.000    0.001  /home/yuyang/perf_test/.env/lib/python2.7/site-packages/paddle/v2/fluid/framework.py:534(append_backward)


Called:

   Ordered by: internal time
   List reduced from 4497 to 2 due to restriction <'sync_with_cpp'>
```

通常观察热点函数间的调用关系，和对应行的代码，就可以了解到问题代码在哪里。当我们做出性能修正后，再次进行性能分析(profiling)即可检查我们调优后的修正是否能够改善程序的性能。

通常，重复若干次性能分析 --> 寻找瓶颈 ---> 调优瓶颈 --> 性能分析确认调优效果的循环后，我们的程序性能就能够**有条不紊**的提高。
