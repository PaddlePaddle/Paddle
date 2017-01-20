### 数据集

Paddle目前提供了很多demo，且各demo运行时需要从原生网站下载其数据，并进行复杂的预处理过程，整个过程会耗费大量时间。同时为了方便大家用Paddle做实验的时候，可以直接访问这些预处理好的数据，我们提供一套Python库。采用import数据源的方式(如：paddle.data.amazon_product_reviews)来简化获取训练所需数据的时间；但是如果你习惯自己处理原生数据，我们依然提供原生数据接口来满足你的需求。

## 接口设计
数据集的导入通过import paddle.data.amazon_product_reviews 来实现，你可以直接通过load_data(category=None,
directory=None)获取你所需的数据集。考虑到类似Amazon的数据类型不止一种，通过category你可以选择控制所需要的数据源;如果你不指定数据源，默认为"Electronics"。directory用来指定下载路径，如果你不指定下载路径，默认为"~/paddle_data/amazon"。通过load_data()导入的数据源data为object，他是我们预处理的numpy格式数据，直接通过data.train_data()获取训练数据或者通过data.test_data()获取测试数据。你还可以打印训练数据和测试数据的数据信息，
```python
 for each_train_data in data.train_data():
     print each_train_data
```
即可。

具体的demo使用情况如下：
```python
import paddle.data.amazon_product_reviews  as raw

data = raw.load_data()
train_data = data.train_data()
test_data = data.test_data()
```
你也可以打印出各数据集的数据信息：
```python
for each_train_data in data.train_data():
    print each_train_data
```
打印出来的数据信息都是预处理之后的numpy格式的数据：
```python
(array([ 730143,  452087,  369164, 1128311, 1451292,  294749, 1370072,
       1202482, 1522860, 1055269,   39557,    1579, 1184187, 1410234,
       362445, 1133007, 1400596,  216811,  540527,  489771,  208467,
       369164,  311153,  387289,  801432,  433138,  179848,  320757,
       1410234], dtype=int32), True)
```
