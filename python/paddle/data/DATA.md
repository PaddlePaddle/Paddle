## 需求

Paddle目前提供了很多demo，且各demo运行时需要从原生网站下载其数据，并进行复杂的预处理过程，整个过程会耗费大量时间。

所以我们需要数据封装接口，采用import数据源的方式(如\：import paddle.data.amazon.review.GetJSON)来简化获取训练所需数据的时间；但是如果你习惯自己处理原生数据，我们依然提供原生数据接口来满足你的需求。

## 整体思路

数据封装接口的目的是提供数据。不论是原生数据，还是预处理数据都通过import方式导入各模型进行训练；考虑到某些模型的预处理后的数据量依然很大，或有时就仅仅想训练相对较小的网络模型，没必要考虑全量数据，自动配置数据量大小必然更符合不同需求。整个接口初步设想如下：
* 开关来控制数据来源
   * 导入数据接口时，带有开关(如:src\_from = True，来自预处理源；否则,来自原生数据源)
* 预处理数据部分添加配置train和test的数据量的大小
* 原生数据部分的数据下载数据模块化
   * 开关(src\_from = False)和<模型，数据源>对完成相关数据的下载
* 原生数据的预处理部分保持原状，通过<模型,预处理过程>对完成数据的预处理
* 在paddle的train的配置文件中修改数据源的导入方式

整个过程在tensorflow的mnist模型已有人实现，借鉴此思想，实现paddle的各demo数据接口的通用化。

```python
amazon = input_data.load_dataset(
         'Amazon',
         '/Users/baidu/git/test_package/data',
         data_unneed=False,
         src_flag=False)
batch = amazon.train.shrink_txt('train',10)
```

