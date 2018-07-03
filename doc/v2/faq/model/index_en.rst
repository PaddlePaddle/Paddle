###################
Model Configuration
###################

..  contents::

1. How to deal with error :code:`Duplicated layer name`
----------------------------------------------------------

The general reason for this error is that users may have set the same value for the attribute :code:`name` in different layers. Try to find out the :code:`name` attribute with the same value in diffrent layers and set them differently.

2. How to use :code:`paddle.layer.memory`'s attribute :code:`name`
----------------------------------------------------------------------

* :code:`paddle.layer.memory` is used to get the output of a layer's last timestep and the layer is specified by the attribute :code:`name` . Thus,  :code:`paddle.layer.memory` will associate with the layer that has the same value of attribute :code:`name` , and uses the output of the layer's last timestep as the input of its current timestep.

* All the PaddlePaddle's layers have a unique name, which is set by the attribute :code:`name` . PaddlePaddle will automatically set it for the user when it is not explicitly set. :code:`paddle.layer.memory` is not a real layer, its name is set by the attribute :code:`memory_name`  and PaddlePaddle will also automatically set it when the user does not explicitly set. The :code:`paddle.layer.memory` attribute :code:`name` is used to specify the layer it is associated with, and needs to be explicitly set by the user.


3. What is the difference between the two ways of using dropout
-----------------------------------------------------------------

* There are two ways to use dropout in PaddlePaddle

  * Set the :code:`drop_rate` parameter in the layer's :code:`layer_atter` attribute. Take :code:`paddle.layer.fc` as an example:

  ..  code-block:: python

      fc = paddle.layer.fc(input=input, layer_attr=paddle.attr.ExtraLayerAttribute(drop_rate=0.5))

  * Use :code:`paddle.layer.dropout` layer. Take :code:`paddle.layer.fc` as an example:

  ..  code-block:: python

      fc = paddle.layer.fc(input=input)
      drop_fc = paddle.layer.dropout(input=fc, dropout_rate=0.5)

* :code:`paddle.layer.dropout` actually uses the :code:`paddle.layer.add_to` layer and sets :code:`drop_rate` as the previous method. This method is very memory intensive.

* PaddlePaddle implements dropout in the activation function rather than in the layer.

* :code:`paddle.layer.lstmemory`, :code:`paddle.layer.grumemory`, :code:`paddle.layer.recurrent` implement activation of output in an unusual way, so we cannot use dropout by setting :code:`drop_rate` . To use dropout for these layers, we could use the second method, which is to use :code:`paddle.layer.dropout`.

4. The differences between different recurrent layers
--------------------------------------------------------
Take LSTM as an example. There are several kinds of recurrent layers in PaddlePaddle:

* :code:`paddle.layer.lstmemory`
* :code:`paddle.networks.simple_lstm`
* :code:`paddle.networks.lstmemory_group`
* :code:`paddle.networks.bidirectional_lstm`

According to implementations, recurrent layer can be classified into 2 types:

1. Recurrent layer implemented by recurrent_group:

  * Using this type of recurrent layers, users can access the intermediate value calculated by the recurrent unit within a timestep (eg: hidden states, memory cells, etc.)
  * :code:`paddle.networks.lstmemory_group` belongs to this type of recurrent layers.

2. Recurrent layer implemented as a complete operation：

  * Users can only access output values when using this type of recurrent layers.
  * :code:`paddle.networks.lstmemory_group` , :code:`paddle.networks.simple_lstm` and  :code:`paddle.networks.bidirectional_lstm` belong to this type of recurrent layer；

By implementing recurrent layer as a complete operation, CPU and GPU calculations can be optimized. Therefore, the second type of recurrent layer is more efficient than the first one. In practical applications, we propose to use the second type of recurrent layers if there is no need to access the intermediate variable of LSTM.

In addition, PaddlePaddle also contains a kind of LSTM calculation unit: :code:`paddle.networks.lstmemory_unit`:

  * Unlike the recurrent layer described above, :code:`paddle.networks.lstmemory_unit` defines the computational process of an LSTM unit in a timestep. It is not a complete recurrent layer, nor can it receive sequence data as input.
  * :code:`paddle.networks.lstmemory_unit` can only be used as a step function in recurrent_group.

5. Can Softmax's calculation dimension be specified？
--------------------------------------------------------------------

We can't specify calculation dimension for PaddlePaddle's softmax. It can only be calculated by rows.
In image tasks, for NCHW, if you need to calculate softmax in C dimension, you could use :code:`paddle.layer.switch_order` to change the dimension order, that is, convert NCHW to NHWC, then do the reshape operation and calculate softmax.

6. Does PaddlePaddle support variable-dimensional data inputs
----------------------------------------------------------------

PaddlePaddle provides :code:`paddle.data_type.dense_array` to support variable-dimensional data input. Simply set the dimension of the data layer to a value larger than the dimension of the input data for occupancy.
