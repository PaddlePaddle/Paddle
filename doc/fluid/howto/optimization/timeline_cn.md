# 如何使用timeline工具做性能分析

1. 在训练的主循环外加上`with profiler.profiler(...)`。运行之后，代码会在`/tmp/profile`目录下生成一个profile的记录文件。

	**提示：**
	请不要在timeline记录信息时运行太多次迭代，因为timeline中的记录数量和迭代次数是成正比的。

	```python
	with profiler.profiler('All', 'total', '/tmp/profile') as prof:
	    for pass_id in range(pass_num):
	        for batch_id, data in enumerate(train_reader()):
	            exe.run(fluid.default_main_program(),
	                    feed=feeder.feed(data),
	                    fetch_list=[])
	            ...
	```

1. 运行`python paddle/tools/timeline.py`来处理`/tmp/profile`，这个程序默认会生成一个`/tmp/timeline`文件，你也可以用命令行参数来修改这个路径，请参考[timeline.py](https://github.com/PaddlePaddle/Paddle/blob/develop/tools/timeline.py)。

1. 打开chrome浏览器，访问<chrome://tracing/>，用`load`按钮来加载生成的`timeline`文件。

	![chrome tracing](./tracing.jpeg)

1. 结果如下图所示，可以放到来查看timetime的细节信息。

	![chrome timeline](./timeline.jpeg)
