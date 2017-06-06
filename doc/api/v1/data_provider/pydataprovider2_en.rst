..  _api_pydataprovider2:

PyDataProvider2
===============

We highly recommand users to use PyDataProvider2 to provide training or testing
data to PaddlePaddle. The user only needs to focus on how to read a single
sample from the original data file by using PyDataProvider2, leaving all of the
trivial work, including, transfering data into cpu/gpu memory, shuffle, binary
serialization to PyDataProvider2. PyDataProvider2 uses multithreading and a
fanscinating but simple cache strategy to optimize the efficiency of the data
providing process.

DataProvider for the non-sequential model
-----------------------------------------

Here we use the MNIST handwriting recognition data as an example to illustrate
how to write a simple PyDataProvider.

MNIST is a handwriting classification data set. It contains 70,000 digital
grayscale images. Labels of the training sample range from 0 to 9. All the
images have been size-normalized and centered into images with the same size
of 28 x 28 pixels.

A small part of the original data as an example is shown as below:

.. literalinclude:: src/mnist_train.txt

Each line of the data contains two parts, separated by :code:`;`. The first part is
label of an image. The second part contains 28x28 pixel float values.

Just write path of the above data into train.list. It looks like this:

.. literalinclude:: src/train.list

The corresponding dataprovider is shown as below:

.. literalinclude:: src/mnist_provider.dict.py

The first line imports PyDataProvider2 package.
The main function is the process function, that has two parameters.
The first parameter is the settings, which is not used in this example.
The second parameter is the filename, that is exactly each line of train.list.
This parameter is passed to the process function by PaddlePaddle.

:code:`@provider` is a Python
`Decorator <http://www.learnpython.org/en/Decorators>`_ .
It sets some properties to DataProvider, and constructs a real PaddlePaddle
DataProvider from a very simple user implemented python function. It does not
matter if you are not familiar with `Decorator`_. You can keep it simple by
just taking :code:`@provider` as a fixed mark above the provider function you
implemented.

`input_types`_ defines the data format that a DataProvider returns.
In this example, it is set to a 28x28-dimensional dense vector and an integer
scalar, whose value ranges from 0 to 9.
`input_types`_ can be set to several kinds of input formats, please refer to the
document of `input_types`_ for more details.


The process method is the core part to construct a real DataProvider in
PaddlePaddle. It implements how to open the text file, how to read one sample
from the original text file, convert them into `input_types`_, and give them
back to PaddlePaddle process at line 23.
Note that data yielded by the process function must follow the same order that
`input_types`_ are defined.


With the help of PyDataProvider2, user can focus on how to generate ONE traning
sample by using keywords :code:`yield`.
:code:`yield` is a python keyword, and a concept related to it includes
:code:`generator`.

Only a few lines of codes need to be added into the training configuration file,
you can take this as an example.

.. literalinclude:: src/mnist_config.py

Here we specify training data by :code:`train.list`, and no testing data is specified.
The method which actually provide data is :code:`process`.

User also can use another style to provide data, which defines the
:code:`data_layer`'s name explicitly when `yield`. For example,
the :code:`dataprovider` is shown as below.

.. literalinclude:: src/mnist_provider.dict.py
   :linenos:

If user did't give the :code:`data_layer`'s name, PaddlePaddle will use
the order of :code:`data_layer` definition roughly to determine which feature to
which :code:`data_layer`. This order may be not correct, so TO DEFINE THE
:code:`data_layer`'s NAMES EXPLICITLY IS THE RECOMMANDED WAY TO PROVIDER DATA.

Now, this simple example of using PyDataProvider is finished.
The only thing that the user should know is how to generte **one sample** from
**one data file**.
And PaddlePadle will do all of the rest things\:

* Form a training batch
* Shuffle the training data
* Read data with multithreading
* Cache the training data (Optional)
* CPU-> GPU double buffering.

Is this cool?

..  _api_pydataprovider2_sequential_model:

DataProvider for the sequential model
-------------------------------------
A sequence model takes sequences as its input. A sequence is made up of several
timesteps. The so-called timestep, is not necessary to have something to do
with time. It can also be explained to that the order of data are taken into
consideration into model design and training.
For example, the sentence can be interpreted as a kind of sequence data in NLP
tasks.

Here is an example on data proivider for English sentiment classification data.
The original input data are simple English text, labeled into positive or
negative sentiment (marked by 0 and 1 respectively).

A small part of the original data as an example can be found in the path below:

.. literalinclude:: src/sentimental_train.txt

The corresponding data provider can be found in the path below:

.. literalinclude:: src/sentimental_provider.py

This data provider for sequential model is a little more complex than that
for MINST dataset.
A new initialization method is introduced here.
The method :code:`on_init` is configured to DataProvider by :code:`@provider`'s
:code:`init_hook` parameter, and it will be invoked once DataProvider is
initialized. The :code:`on_init` function has the following parameters:

* The first parameter is the settings object.
* The rest parameters are passed by key word arguments. Some of them are passed
  by PaddlePaddle, see reference for `init_hook`_.
  The :code:`dictionary` object is a python dict object passed from the trainer
  configuration file, and it maps word string to word id.

To pass these parameters into DataProvider, the following lines should be added
into trainer configuration file.

.. literalinclude:: src/sentimental_config.py

The definition is basically same as MNIST example, except:
* Load dictionary in this configuration
* Pass it as a parameter to the DataProvider

The `input_types` is configured in method :code:`on_init`. It has the same
effect to configure them by :code:`@provider`'s :code:`input_types` parameter.
However, the :code:`input_types` is set at runtime, so we can set it to
different types according to the input data. Input of the neural network is a
sequence of word id, so set :code:`seq_type` to :code:`integer_value_sequence`.

Durning :code:`on_init`, we save :code:`dictionary` variable to
:code:`settings`, and it will be used in :code:`process`. Note the settings
parameter for the process function and for the on_init's function are a same
object.

The basic processing logic is the same as MNIST's :code:`process` method. Each
sample in the data file is given back to PaddlePaddle process.

Thus, the basic usage of PyDataProvider is here.
Please refer to the following section reference for details.

Reference
---------

@provider
+++++++++

.. autofunction:: paddle.trainer.PyDataProvider2.provider

input_types
+++++++++++

PaddlePaddle has four data types, and three sequence types.
The four data types are:

* :code:`dense_vector`: dense float vector.
* :code:`sparse_binary_vector`: sparse binary vector, most of the value is 0, and
  the non zero elements are fixed to 1.
* :code:`sparse_float_vector`: sparse float vector, most of the value is 0, and some
  non zero elements can be any float value. They are given by the user.
* :code:`integer`: an integer scalar, that is especially used for label or word index.

The three sequence types are:

* :code:`SequenceType.NO_SEQUENCE` means the sample is not a sequence.
* :code:`SequenceType.SEQUENCE` means the sample is a sequence.
* :code:`SequenceType.SUB_SEQUENCE` means it is a nested sequence, that each timestep of
  the input sequence is also a sequence.

Different input type has a defferenct input format. Their formats are shown
in the above table.

+----------------------+---------------------+-----------------------------------+------------------------------------------------+
|                      | NO_SEQUENCE         | SEQUENCE                          |  SUB_SEQUENCE                                  |
+======================+=====================+===================================+================================================+
| dense_vector         | [f, f, ...]         | [[f, ...], [f, ...], ...]         | [[[f, ...], ...], [[f, ...], ...],...]         |
+----------------------+---------------------+-----------------------------------+------------------------------------------------+
| sparse_binary_vector | [i, i, ...]         | [[i, ...], [i, ...], ...]         | [[[i, ...], ...], [[i, ...], ...],...]         |
+----------------------+---------------------+-----------------------------------+------------------------------------------------+
| sparse_float_vector  | [(i,f), (i,f), ...] | [[(i,f), ...], [(i,f), ...], ...] | [[[(i,f), ...], ...], [[(i,f), ...], ...],...] |
+----------------------+---------------------+-----------------------------------+------------------------------------------------+
| integer_value        |  i                  | [i, i, ...]                       | [[i, ...], [i, ...], ...]                      |
+----------------------+---------------------+-----------------------------------+------------------------------------------------+

where f represents a float value, i represents an integer value.

init_hook
+++++++++

init_hook is a function that is invoked once the data provoder is initialized.
Its parameters lists as follows:

* The first parameter is a settings object, which is the same to :code:`settings`
  in :code:`process` method. The object contains several attributes, including:

  * :code:`settings.input_types`: the input types. Reference `input_types`_.
  * :code:`settings.logger`: a logging object.

* The rest parameters are the key word arguments. It is made up of PaddpePaddle
  pre-defined parameters and user defined parameters.

  * PaddlePaddle-defined parameters including:

    * :code:`is_train` is a bool parameter that indicates the DataProvider is used in
      training or testing.
    * :code:`file_list` is the list of all files.

  * User-defined parameters args can be set in training configuration.

Note, PaddlePaddle reserves the right to add pre-defined parameter, so please
use :code:`**kwargs` in init_hook to ensure compatibility by accepting the
parameters which your init_hook does not use.

cache
+++++
DataProvider provides two simple cache strategy. They are:

* :code:`CacheType.NO_CACHE` means do not cache any data, then data is read at runtime by
  the user implemented python module every pass.
* :code:`CacheType.CACHE_PASS_IN_MEM` means the first pass reads data by the user
  implemented python module, and the rest passes will directly read data from
  memory.
