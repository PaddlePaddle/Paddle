.. _cn_api_paddle_vision_models_alexnet

AlexNet
-------------------------------

.. py:function:: paddle.vision.models.AlexNet(pretrained=False, **kwargs)

 AlexNet模型，来自论文 `"ImageNet Classification with Deep Convolutional Neural Networks
" <https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>`_。

参数
:::::::::
  - **pretrained** (bool，可选) - 是否加载在imagenet数据集上的预训练权重。默认值：False。

返回
:::::::::
AlexNet模型，Layer的实例。

代码示例
:::::::::
.. code-block:: python

    import paddle
    from paddle.vision.models import AlexNet

    # build model
    model = AlexNet()

    # build model and load imagenet pretrained weight
    # model = AlexNet(pretrained=True)

    x = paddle.rand([1, 3, 224, 224])
    out = model(x)

    print(out.shape)
