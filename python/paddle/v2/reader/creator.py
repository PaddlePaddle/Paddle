# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Creator package contains some simple reader creator, which could be used in user
program.
"""

__all__ = ['np_array', 'text_file', "recordio"]


def np_array(x):
    """
    Creates a reader that yields elements of x, if it is a
    numpy vector. Or rows of x, if it is a numpy matrix.
    Or any sub-hyperplane indexed by the highest dimension.

    :param x: the numpy array to create reader from.
    :returns: data reader created from x.
    """

    def reader():
        if x.ndim < 1:
            yield x

        for e in x:
            yield e

    return reader


def text_file(path):
    """
    Creates a data reader that outputs text line by line from given text file.
    Trailing new line ('\\\\n') of each line will be removed.

    :path: path of the text file.
    :returns: data reader of text file
    """

    def reader():
        f = open(path, "r")
        for l in f:
            yield l.rstrip('\n')
        f.close()

    return reader


def recordio_local(paths, buf_size=100):
    """
    Creates a data reader from given RecordIO file paths separated by ",", 
        glob pattern is supported.
    :path: path of recordio files.
    :returns: data reader of recordio files.
    """

    import recordio as rec
    import paddle.v2.reader.decorator as dec

    def reader():
        a = ','.join(paths)
        f = rec.reader(a)
        while True:
            r = f.read()
            if r is None:
                break
            yield r
        f.close()

    return dec.buffered(reader, buf_size)


def recordio(paths, buf_size=100):
    """
    Creates a data reader that outputs record one one by one 
        from given local or cloud recordio path.
    :path: path of recordio files.
    :returns: data reader of recordio files.
    """
    import os
    import paddle.v2.master.client as cloud

    if "KUBERNETES_SERVICE_HOST" not in os.environ.keys():
        return recordio_local(paths)

    host_name = "MASTER_SERVICE_HOST"
    if host_name not in os.environ.keys():
        raise Exception('not find ' + host_name + ' in environ.')

    addr = os.environ(host)

    def reader():
        c = cloud(addr, buf_size)
        c.set_dataset(paths)

        while True:
            r, err = client.next_record()
            if err < 0:
                break
            yield r

        c.close()

    return reader
