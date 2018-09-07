# 如何使用timeline工具做性能分析

1. 在训练的主循环外加上`profiler.start_profiler(...)`和`profiler.stop_profiler(...)`。运行之后，代码会在`/tmp/profile`目录下生成一个profile的记录文件。

	**提示：**
	请不要在timeline记录信息时运行太多次迭代，因为timeline中的记录数量和迭代次数是成正比的。

	```python
    for pass_id in range(pass_num):
        for batch_id, data in enumerate(train_reader()):
            if pass_id == 0 and batch_id == 5:
                profiler.start_profiler("All")
            elif pass_id == 0 and batch_id == 10:
                profiler.stop_profiler("total", "/tmp/profile")
            exe.run(fluid.default_main_program(),
                    feed=feeder.feed(data),
                    fetch_list=[])
	            ...
	```

1. 运行`python paddle/tools/timeline.py`来处理`/tmp/profile`，这个程序默认会生成一个`/tmp/timeline`文件，你也可以用命令行参数来修改这个路径，请参考[timeline.py](https://github.com/PaddlePaddle/Paddle/blob/develop/tools/timeline.py)。
```python
python Paddle/tools/timeline.py --profile_path=/tmp/profile --timeline_path=timeline
```

1. 打开chrome浏览器，访问<chrome://tracing/>，用`load`按钮来加载生成的`timeline`文件。

	![chrome tracing](./tracing.jpeg)

1. 结果如下图所示，可以放到来查看timetime的细节信息。

	![chrome timeline](./timeline.jpeg)
