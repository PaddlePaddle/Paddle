# API注释撰写标准

- [API注释撰写标准](#api)
    - [API注释模块](#api)
    - [格式及示例](#)
    - [完整示例](#)


## API注释模块

API文档须包含以下几个模块（排列顺序为文档撰写顺序）：

- Python API Definition

  API的代码定义。

- Function Description

  API的功能描述。描述该API的含义、作用或对输入所做的操作，及参考文献和对应链接（如果有），必要时给出公式，并解释公式中关键变量的含义。

- Args Description

  API参数介绍。按代码定义中的参数顺序逐个介绍，介绍内容包含数据类型、默认值（如果有）、含义等。

- Returns

  API返回值介绍。介绍返回值含义，必要时给出对应的形状。若返回值为包含多个参数的tuple，则按顺序逐个介绍各参数。

- Raises（如果有）

  可能抛出的异常或错误及可能的产生原因，当可能抛出多种异常或错误时应分条列出。

- Note（如果有）

  注意事项。当有多条注意事项时，应分条列出。

- Examples

  API的使用示例。


## 格式及示例

API文档须使用reStructuredText格式撰写，该格式详情请参考[链接](http://sphinx-doc-zh.readthedocs.io/en/latest/rest.html)。API文档各模块的内容格式及示例如下（以下以fc为例进行说明）：

- Python API Definition

  - 格式：

      [Python API Definition]

  - 示例

      ```
      fc(input,
         size,
         num_flatten_dims=1,
         param_attr=None,
         bias_attr=None,
         act=None,
         name=None,
         main_program=None,
         startup_program=None)
      ```

- Function Description

  - 格式

      本模块应包含以下内容（排列顺序为文档撰写顺序）：

      [Function Description]

      [Formula]

      [Symbols' Descriptions if necessary]

      [References if necessary]

  - 示例

      [Function Description]

       ```
       **Fully Connected Layer**

       The fully connected layer can take multiple tensors as its inputs. It
       creates a variable called weights for each input tensor, which represents
       a fully connected weight matrix from each input unit to each output unit.
       The fully connected layer multiplies each input tensor with its coresponding
       weight to produce an output Tensor. If multiple input tensors are given,
       the results of multiple multiplications will be sumed up. If bias_attr is
       not None, a bias variable will be created and added to the output. Finally,
       if activation is not None, it will be applied to the output as well.
       ```

      [Formula]

      ```
      This process can be formulated as follows:

      .. math::

           Out = Act({\sum_{i=0}^{N-1}X_iW_i + b})
      ```

      [Symbols' Descriptions if necessary]

      ```
      In the above equation:

      * :math:`N`: Number of the input.
      * :math:`X_i`: The input tensor.
      * :math:`W`: The weights created by this layer.
      * :math:`b`: The bias parameter created by this layer (if needed).
      * :math:`Act`: The activation function.
      * :math:`Out`: The output tensor.
      ```

      [References if necessary]

      因fc没有必要列出的参考文献，故该内容省略。其他情况下需明确给出对应的参考文献和对应连接，以 layer_norm 为例：

      ```
      Refer to `Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_ for more details.
      ```


- Args Description

  - 格式

      \[Arg's Name\][(Data Type, Default Value)][Description]

  - 示例

      fc的部分参数注释如下：

      ```
      Args:
          input (Variable|list of Variable): The input tensor(s) of this layer, and the dimension of
              the input tensor(s) is at least 2.
          param_attr (ParamAttr|list of ParamAttr, default None): The parameter attribute for learnable
              parameters/weights of this layer.
          name (str, default None): The name of this layer.
      ```

- Returns

  - 格式

      [Name][Shape]

  - 示例

      ```
      Returns:
          A tensor variable storing the transformation result.
      ```

      当返回值为包含多个参数的tuple时，应按顺序逐个介绍各参数，以dynamic_lstm为例：

      ```
      Returns:
          A tuple containing:
            The hidden state of LSTM whose shape is (T X D).
            The cell state of LSTM whose shape is (T X D).
      ```

- Raises

  - 格式

      [Exception Type][Condition]

  - 示例

      ```
      Raises:
          ValueError: If the rank of the input is less than 2.
      ```

- Note

  - 格式

     [Note]

  - 示例

      fc没有注意事项，故该模块省略不写。如有注意事项应明确给出，当有多条注意事项，须分条列出，以scaled\_dot\_product\_attention为例：

      ```
      Note:
          1. When num_heads > 1, three linear projections are learned respectively
             to map input queries, keys and values into queries', keys' and values'.
             queries', keys' and values' have the same shapes with queries, keys
             and values.
          2. When num_heads == 1, scaled_dot_product_attention has no learnable
             parameters.
      ```

- Examples

  - 格式

      \[Python Code Snipper]

  - 示例

      ```
      Examples:
          .. code-block:: python

            data = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
            fc = fluid.layers.fc(input=data, size=1000, act="tanh")
      ```

## 完整示例

fc 的完整注释见[示例](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/dev/src/fc.py)。
