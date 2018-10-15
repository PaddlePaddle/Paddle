##################
Parameter Settings
##################

.. contents::

1. How to Choose the Learning Rate of SGD Algorithm
--------------------------

An important issue when training with :code:`sgd/async_sgd` is to choose the correct value for :code:`learning_rate`. If it is too large, the training may not converge. If too small, the convergence may be slow, resulting in a long training time.

Usually, we start with a relatively large learning rate. If the training does not converge, then we need to reduce the learning rate continuously by a factor of 10 until the training converges. We examine the convergence of the training by estimating the minimum cost at a constant output of the model.

If the cost of the training process is significantly higher than the cost of the output, then we judge that the training does not converge. For example, if we have a three-class problem and use multi-class-cross-entropy as the cost, the ratio of 0, 1, and 2 in the data will be :code:`0.2, 0.5, 0.3`. The minimum cost thus will be :code:`-(0.2*log(0.2)+0.5*log(0.5)+0.3*log(0.3))=1.03`. If the cost is greater than this number after training a pass (or even before), then the training may not be converged and the learning rate should be reduced.

2. How to Implement Learning Rate Annealing
------------------------------------------------

We use the Adam algorithm as an example. Set the parameters of :code:`learning_rate_schedule` in the corresponding optimization algorithm as follows:

.. code-block:: python

    Optimizer = paddle.optimizer.Adam(
        Learning_rate=1e-3,
        Learning_rate_decay_a=0.5,
        Learning_rate_decay_b=0.75,
        Learning_rate_schedule="poly",)

PaddlePaddle currently supports 8 learning rate schedules. The 8 learning rate schedules and their corresponding learning rates are calculated as follows:

* "constant"
  
  Lr = learning_rate

* "poly"

  Lr = learning_rate * pow(1 + learning_rate_decay_a * num_samples_processed, -learning_rate_decay_b)

  Variable :code:`num_samples_processed` is the number of trained samples.

* "caffe_poly"

  Lr = learning_rate * pow(1.0 - num_samples_processed / learning_rate_decay_a, learning_rate_decay_b)

* "exp"

  Lr = learning_rate * pow(learning_rate_decay_a, num_samples_processed / learning_rate_decay_b)

* "discexp"

  Lr = learning_rate * pow(learning_rate_decay_a, floor(num_samples_processed / learning_rate_decay_b))

* "linear"

  Lr = max(learning_rate - learning_rate_decay_a * num_samples_processed, learning_rate_decay_b)

* "manual"

  This is a learning rate annealing method that is segmented by the number of trained samples. When using this learning rate schedule, we modify the learning rate attenuation factor piecewise function by changing the parameter :code:`learning_rate_args`. The current learning rate is the product of :code:`learning_rate` and the current attenuation factor. Take the Adam algorithm as an example:

  .. code-block:: python

      Optimizer = paddle.optimizer.Adam(
          Learning_rate=1e-3,
          Learning_rate_schedule="manual",
          Learning_rate_args="1000:1.0,2000:0.9,3000:0.8",)

  In this example, when the number of trained samples is less than or equal to 1000, the learning rate is: code:`1e-3*1.0`; when the number of trained samples is greater than 1000 or less than or equal to 2000, the learning rate is:code:`1e- 3 * 0.9`; when the number of trained samples is greater than 2,000, the learning rate is: code:`1e-3*0.8`.

* "pass_manual"

  This is a learning rate annealing method that piecewisely pick values according to the number of trained passes. When using this learning rate schedule, we set the learning rate attenuation factor piecewise function by the parameter :code:`learning_rate_args`. The current learning rate is the product of :code:`learning_rate` and the current attenuation factor. Take the Adam algorithm as an example:

  .. code-block:: python

      Optimizer = paddle.optimizer.Adam(
          Learning_rate=1e-3,
          Learning_rate_schedule="pass_manual",
          Learning_rate_args="1:1.0,2:0.9,3:0.8",)

  In this example, when the number of trained passes is less than or equal to 1, the learning rate is :code:`1e-3*1.0`; when the number of trained passes is greater than 1 or less than 2, the learning rate is :code:`1e- 3 * 0.9`; when the number of trained passes is greater than 2, the learning rate is :code:`1e-3*0.8`.

3. How to Initialize Parameters
-----------------

By default, PaddlePaddle initializes parameters with an average of 0 and a standard deviation of :math:`\frac{1}{\sqrt{d}}`, where :math:`d` is the width of the parameter matrix. This initialization method does not produce bad results under normal circumstances. If users want to customize the initialization method, PaddlePaddle provides two ways to initialize the parameters:

* Gaussian distribution. Set :code:`param_attr` to :code:`param_attr=ParamAttr(initial_mean=0.0, initial_std=1.0)`
* Uniform distribution. Set :code:`param_attr` to :code:`param_attr=ParamAttr(initial_max=1.0, initial_min=-1.0)`

For example, to set a full connection layer parameter initialization mode and bias initialization mode, you can use the following code:

.. code-block:: python

    Hidden = fc_layer(input=ipt, param_attr=ParamAttr(initial_max=1.0, initial_min=-1.0),
                      Bias_attr=ParamAttr(initial_mean=1.0, initial_std=0.0))

The above code initializes the bias to 1.0 and initializes the parameters to a uniform distribution of :code:`[1.0, -1.0]`.

4. How to Share Parameters
---------------

PaddlePaddle's parameters use :code:`name` as the ID. Parameters with the same name will share parameters//. We can set the name of the parameters using :code:`ParamAttr(name="YOUR_PARAM_NAME")`. More conveniently, we can make the parameters to be shared use the same :code:`ParamAttr` object.

A simple fully connected network has its configuration of parameter sharing as follows \:

.. literalinclude:: ../../python/paddle/trainer_config_helpers/tests/configs/shared_fc.py

Here :code:`hidden_a` and :code:`hidden_b` have the same parameter and bias. The two input of the softmax layer also use the same parameter :code:`softmax_param`.

5. How to Load Pre-training Parameters
------------------------
* For layers that load pre-training parameters, set :code:`is_static = True` so that the parameters of that layer remain unchanged during the training process. Take the embedding layer as an example, the code is as follows:

.. code-block:: python

    Emb_para = paddle.attr.Param(name='emb', is_static=True)
    Paddle.layer.embedding(size=word_dim, input=x, param_attr=emb_para)


* Load pre-training parameters from the model file into :code:`numpy.array`. After creating the parameters, load the pre-training parameters using :code:`parameters.set()`. The first 16 bytes of the model parameter file saved by PaddlePaddle is the header information. The user must loads : :code:`numpy.array` starting with the 17th byte. Take the embedding layer as an example, the code is as follows:

.. code-block:: python

    Def load_parameter(file_name, h, w):
        With open(file_name, 'rb') as f:
            F.read(16) # skip header.
            Return np.fromfile(f, dtype=np.float32).reshape(h, w)

    Parameters = paddle.parameters.create(my_cost)
    Parameters.set('emb', load_parameter(emb_param_file, 30000, 256))

6. Format of the Stored Parameter and How to Convert the File to Plain Text
--------------------------------------------------

The model parameter file saved by PaddlePaddle consists of 16 bytes of header information and network parameters. In the header information, the first four bytes show PaddlePaddle's version information. The user should fill in with 0s. The next four bytes represent the number of bytes occupied by each parameter. If the saved network parameter is a float type, the number is four; if it is a double, the number is eight. The third group of four bytes represents the total number of saved parameters.

When restoring the model parameters saved by PaddlePaddle back to plain text, we use the corresponding data type :code:`numpy.array` to load specific network parameters. At this time, you can skip the header information of the PaddlePaddle model parameter file. If not specified to compile with a precision for double in PaddlePaddle, then the parameter file will be caiculated with a precision for float, and the argument will be stored as a float. In this case, when using :code:`numpy.array`, generally we set :code:`dtype=float32`. An example is as follows:

.. code-block:: python

    Def read_parameter(fname, width):
        s = open(fname).read()
        # skip header
        Vec = np.fromstring(s[16:], dtype=np.float32)
        # width is the size of the corresponding layer
        Np.savetxt(fname + ".csv", vec.reshape(width, -1),
                Fmt="%.6f", delimiter=",")


When the plaintext parameters are converted into PaddlePaddle loadable model parameters, the header information is constructed first, then the network parameters are written. The following code converts the randomly generated matrix into model parameters that can be loaded by PaddlePaddle:

.. code-block:: python

    Def gen_rand_param(param_file, width, height, need_trans):
        Np.random.seed()
        Header = struct.pack("iil", 0, 4, height * width)
        Param = np.float32(np.random.rand(height, width))
        With open(param_file, "w") as fparam:
            Fparam.write(header + param.tostring())

7. A Protocol Message Rejected Because of its Large Size
-------------------------------------------------- ----------

If you are training NLP related models, and the following error occurs:

.. code-block:: bash

    [libprotobuf ERROR google/protobuf/io/coded_stream.cc:171] A protocol message was rejected because it was too big (more than 67108864 bytes). To increase the limit (or to disable these warnings), see CodedInputStream::SetTotalBytesLimit( ) in google/protobuf/io/coded_stream.h.
    F1205 14:59:50.295174 14703 TrainerConfigHelper.cpp:59] Check failed: m->conf.ParseFromString(configProtoStr)

The possible reason is that one of the args passed to the dataprovider is too large, which is usually caused by directly passing a large dictionary. A wrongly defineed `_py_data_sources2` is similar to:

.. code-block:: python

     Src_dict = dict()
     For line_count, line in enumerate(open(src_dict_path, "r")):
        Src_dict[line.strip()] = line_count

     Define_py_data_sources2(
        Train_list,
        Test_list,
        Module="dataprovider",
        Obj="process",
        Args={"src_dict": src_dict})

The solution is to pass the address of the dictionary as args to the dataprovider, and then load the dictionary according to the address in the dataprovider. Change `_py_data_sources2` to:

.. code-block:: python

     Define_py_data_sources2(
        Train_list,
        Test_list,
        Module="dataprovider",
        Obj="process",
        Args={"src_dict_path": src_dict_path})

The full source code can be found in the `sequence_recurrent <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/sequence_recurrent.py>`_ example.
