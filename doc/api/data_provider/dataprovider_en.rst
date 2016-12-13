Introduction
==============
DataProvider is a module that loads training or testing data into cpu or gpu
memory for the following triaining or testing process.

For simple use, users can use Python :code:`PyDataProvider` to dynamically reads
the original data in any format or in any form, and then transfer them into a
data format PaddlePaddle requires. The process is extremly flexible and highly
customized, with sacrificing the efficiency only a little. This is extremly
useful when you have to dynamically generate certain kinds of data according to,
for example, the training performance.

Besides, users also can customize a C++ :code:`DataProvider` for a more
complex usage, or for a higher efficiency.

The following parameters are required to define in the PaddlePaddle network
configuration file (trainer_config.py): which DataProvider is chosen to used,
and specific parameters for DataProvider, including training file list
(train.list) and testing file list (test.list).

Train.list and test.list are simply two plain text files, which defines path
of training or testing data. It is recommended that directly placing them into
the training directory, and reference to them by using a relative path (
relative to the PaddePaddle program).

Testing or evaluating will not be performed during training if the test.list is
not set or set to None. Otherwise, PaddlePaddle will evaluate the trained model
by the specified tesing data while training, every testing period (a user
defined command line parameter in PaddlePaddle) to prevent over-fitting.

Each line of train.list and test.list is an absolute or relative path (relative
to the PaddePaddle program runtime) of data file. Fascinatingly more, each line
can also be a HDFS file path or a SQL connection string. As long as the user
assures how to access each file in DataProvider.
