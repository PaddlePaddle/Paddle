"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
This module provides fuzzing util functions.
"""
import atheris_no_libfuzzer as atheris
import paddle
import numpy as np

IgnoredErrors = (ValueError, RuntimeError, TypeError, AttributeError, AssertionError)


class Mutator:
    """Mutator for generating random data.

    Attributes:
        input_bytes: Input bytes for FDP.
        logging: Should log data or not.
    """

    def __init__(self, input_bytes, logging=False):
        self.fdp = atheris.FuzzedDataProvider(input_bytes)
        self.logging = logging

    def int_range(self, min_val=0, max_val=100, name=''):
        """Consume an integer in range.

        Args:
            min_val: Minimum value.
            max_val: Maximum value.
            name: Field name for logging.

        Returns:
            A consumed integer.
        """
        return self.log(name, self.fdp.ConsumeIntInRange(min_val, max_val)) if self.logging else \
            self.fdp.ConsumeIntInRange(min_val, max_val)

    def int_list(self, count=1, min_val=0, max_val=100, name=None):
        """Consume an integer list.

        Args:
            count: List length.
            min_val: Minimum value.
            max_val: Maximum value.
            name: Field name for logging.

        Returns:
            A consumed integer list.
        """
        return self.log(name, self.fdp.ConsumeIntListInRange(count, min_val, max_val)) if self.logging else \
            self.fdp.ConsumeIntListInRange(count, min_val, max_val)

    def float_range(self, min_val=0.0, max_val=100.0, name=None):
        """Consume a float in range.

        Args:
            min_val: Minimum value.
            max_val: Maximum value.
            name: Field name for logging.

        Returns:
            A consumed float.
        """
        return self.log(name, self.fdp.ConsumeFloatInRange(min_val, max_val)) if self.logging else \
            self.fdp.ConsumeFloatInRange(min_val, max_val)

    def float_list(self, count=1, min_val=0.0, max_val=100.0, name=None):
        """Consume a float list.

        Args:
            count: List length.
            min_val: Minimum value.
            max_val: Maximum value.
            name: Field name for logging.

        Returns:
            A consumed float list.
        """
        return self.log(name, self.fdp.ConsumeFloatListInRange(count, min_val, max_val)) if self.logging else \
            self.fdp.ConsumeFloatListInRange(count, min_val, max_val)

    def bool(self, name=None):
        """Consume a bool.

        Args:
            name: Field name for logging.

        Returns:
            A consumed bool.
        """
        return self.log(
            name,
            self.fdp.ConsumeBool()) if self.logging else self.fdp.ConsumeBool()

    def bool_list(self, count=1, name=None):
        """Consume a bool list.

        Args:
            count: List length.
            name: Field name for logging.

        Returns:
            A consumed bool list.
        """
        bool_list = []
        for _ in range(count):
            bool_list.append(self.fdp.ConsumeBool())
        return self.log(name, bool_list) if self.logging else bool_list

    def string(self, length=1, name=None):
        """Consume a string in length.

        Args:
            length: String length.
            name: Field name for logging.

        Returns:
            A consumed string.
        """
        s = self.fdp.ConsumeUnicodeNoSurrogates(length)
        if self.logging:
            self.log(name, s.encode('utf-8').hex())
        return s

    def string_list(self, count=1, length=1, name=None):
        """Consume a string list with same string length.

        Args:
            count: List length.
            length: String length.
            name: Field name for logging.

        Returns:
            A consumed string list.
        """
        strings = [
            self.fdp.ConsumeUnicodeNoSurrogates(length) for _ in range(count)
        ]
        if self.logging:
            self.log(name, [s.encode('utf-8').hex() for s in strings])
        return strings

    def strings(self, count=1, len_min=0, len_max=1, name=None):
        """Consume a string list with random string length.

        Args:
            count: List length.
            len_min: Minimum allowed string length.
            len_max: Maximum allowed string length.
            name: Field name for logging.

        Returns:
            A consumed string list.
        """
        strings = [
            self.fdp.ConsumeUnicodeNoSurrogates(
                self.fdp.ConsumeIntInRange(len_min, len_max))
            for _ in range(count)
        ]
        if self.logging:
            self.log(name, [s.encode('utf-8').hex() for s in strings])
        return strings

    def pick(self, candidates=None, name=None):
        """Consume a value in candidates.

        Args:
            candidates: Candidates for pick.
            name: Field name for logging.

        Returns:
            A consumed value in candidates.
        """
        if candidates is None:
            candidates = []
        return self.log(name, self.fdp.PickValueInList(candidates)) if self.logging else \
            self.fdp.PickValueInList(candidates)

    @staticmethod
    def log(name, value):
        """Log consumed data.

        Args:
            name: Data field name.
            value: Data value.

        Returns:
            The data value itself.
        """
        name = name if name else 'Unknown'
        print('{name}: {value}'.format(name=name, value=value))
        return value

    @staticmethod
    def tensor(value, rank=0, dims=None, dtype='float32', np_type=np.float32):
        """Generate a tensor.

        Args:
            value: Tensor values.
            rank: Dims length.
            dims: Dims.
            dtype: Paddle data type.
            np_type: Numpy data type.

        Returns:
            A random tensor.
        """
        if rank == 0 or not dims:
            return paddle.to_tensor(value, dtype=dtype)
        if rank > len(dims):
            return None
        if rank == 1:
            return paddle.to_tensor(value, dtype=dtype)
        elif rank == 2:
            array = np.array(value, dtype=np_type)
            return paddle.to_tensor(
                np.reshape(array, (dims[0], dims[1])), dtype=dtype)
        elif rank == 3:
            array = np.array(value, dtype=np_type)
            return paddle.to_tensor(
                np.reshape(array, (dims[0], dims[1], dims[2])), dtype=dtype)
        elif rank == 4:
            array = np.array(value, dtype=np_type)
            return paddle.to_tensor(
                np.reshape(array, (dims[0], dims[1], dims[2], dims[3])),
                dtype=dtype)
        elif rank == 5:
            array = np.array(value, dtype=np_type)
            return paddle.to_tensor(
                np.reshape(array,
                           (dims[0], dims[1], dims[2], dims[3], dims[4])),
                dtype=dtype)
        else:
            return None

    def tensor_from_dict(self, tensor_map=None):
        """Generate a tensor by using tensor_map dict.

        Args:
            tensor_map: A dict includes tensor attributes info. e.g.
                tensor_map = {
                    'name': 'x',
                    'dtype': 'float32',
                    'np_type': 'float32',
                    'min_val': 0.0,
                    'max_val': 1000.0,
                    'dims': {
                        'dim1': {
                            'min': 0,
                            'max': 20,
                        },
                        'dim2': {
                            'min': 0,
                            'max': 20,
                        }
                    }
                }

        Returns:
            A paddle tensor.
        """
        if tensor_map is None:
            return

        name = tensor_map.get('name', '')
        dtype = tensor_map.get('dtype', 'float32')
        np_type = tensor_map.get('np_type', 'float32')
        if np_type == 'float32':
            np_type = np.float32
        elif np_type == 'int32':
            np_type = np.int32
        elif np_type == 'float64':
            np_type = np.float64
        elif np_type == 'int64':
            np_type = np.int64
        else:
            raise ValueError('np_type is not supported.')
        min_val = tensor_map.get('min_val', 0.0)
        max_val = tensor_map.get('max_val', 100.0)

        dims = []
        for dim_name, dim in tensor_map['dims'].items():
            dims.append(
                self.int_range(
                    dim.get('min', 0),
                    dim.get('max', 10), name + '_' + dim_name))

        if dtype == 'float32' or dtype == 'float64':
            value = self.float_list(
                int(np.prod(dims)), min_val, max_val, name + '_' + 'val')
        elif dtype == 'int32' or dtype == 'int64':
            value = self.int_list(
                int(np.prod(dims)), min_val, max_val, name + '_' + 'val')
        else:
            raise ValueError('dtype is not supported.')

        return self.tensor(value, len(dims), dims, dtype, np_type)

    def tensor_with_diff_shape(self,
                               min_val=-10.0,
                               max_val=100.0,
                               min_dim=0,
                               max_dim=20,
                               max_rank=0,
                               dtype='float32',
                               np_type='float32'):
        """Generate a tensor whose rank value in the range of 0 to max_rank.

        Args:
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.
            min_dim: Minimum allowed dimension value.
            max_dim: Maximum allowed dimension value.
            max_rank: Max range.
            dtype: Paddle data type.
            np_type: Numpy data type.

        Returns:
            A random tensor, rank
        """

        if not isinstance(min_dim, int) or not isinstance(
                max_dim, int) or not isinstance(max_rank, int):
            raise ValueError(
                'min_dim, max_dim and max_rank should be integers.')

        if np_type == 'float32':
            np_type = np.float32
        elif np_type == 'int32':
            np_type = np.int32
        elif np_type == 'float64':
            np_type = np.float64
        elif np_type == 'int64':
            np_type = np.int64
        else:
            raise ValueError('np_type is not supported.')

        if max_rank > 5:
            max_rank = 5

        rank = self.int_range(0, max_rank, 'rank')
        dim1 = self.int_range(min_dim, max_dim, 'dim1')
        dim2 = self.int_range(min_dim, max_dim, 'dim2')
        dim3 = self.int_range(min_dim, max_dim, 'dim3')
        dim4 = self.int_range(min_dim, max_dim, 'dim4')
        dim5 = self.int_range(min_dim, max_dim, 'dim5')

        if dtype == 'float32' or dtype == 'float64':
            min_val = float(min_val)
            max_val = float(max_val)
            if rank == 0:
                return self.tensor(
                    self.float_list(dim1, min_val, max_val, 'val'),
                    dtype=dtype,
                    np_type=np_type), rank
            elif rank == 1:
                return self.tensor(
                    self.float_list(dim1, min_val, max_val, 'val'),
                    1,
                    [dim1],
                    dtype=dtype,
                    np_type=np_type), rank
            elif rank == 2:
                val = self.float_list(dim1 * dim2, min_val, max_val, 'val')
                return self.tensor(
                    val, 2, [dim1, dim2], dtype=dtype, np_type=np_type), rank
            elif rank == 3:
                val = self.float_list(dim1 * dim2 * dim3, min_val, max_val,
                                      'val')
                return self.tensor(
                    val, 3, [dim1, dim2, dim3], dtype=dtype,
                    np_type=np_type), rank
            elif rank == 4:
                val = self.float_list(dim1 * dim2 * dim3 * dim4, min_val,
                                      max_val, 'val')
                return self.tensor(
                    val,
                    4, [dim1, dim2, dim3, dim4],
                    dtype=dtype,
                    np_type=np_type), rank
            else:
                val = self.float_list(dim1 * dim2 * dim3 * dim4 * dim5, min_val,
                                      max_val, 'val')
                return self.tensor(
                    val,
                    5, [dim1, dim2, dim3, dim4, dim5],
                    dtype=dtype,
                    np_type=np_type), rank
        elif dtype == 'int32' or dtype == 'int64':
            min_val = int(min_val)
            max_val = int(max_val)
            if rank == 0:
                return self.tensor(
                    self.int_list(dim1, min_val, max_val, 'val'),
                    dtype=dtype,
                    np_type=np_type), rank
            elif rank == 1:
                return self.tensor(
                    self.int_list(dim1, min_val, max_val, 'val'),
                    1, [dim1],
                    dtype=dtype,
                    np_type=np_type), rank
            elif rank == 2:
                val = self.int_list(dim1 * dim2, min_val, max_val, 'val')
                return self.tensor(
                    val, 2, [dim1, dim2], dtype=dtype, np_type=np_type), rank
            elif rank == 3:
                val = self.int_list(dim1 * dim2 * dim3, min_val, max_val, 'val')
                return self.tensor(
                    val, 3, [dim1, dim2, dim3], dtype=dtype,
                    np_type=np_type), rank
            elif rank == 4:
                val = self.int_list(dim1 * dim2 * dim3 * dim4, min_val, max_val,
                                    'val')
                return self.tensor(
                    val,
                    4, [dim1, dim2, dim3, dim4],
                    dtype=dtype,
                    np_type=np_type), rank
            else:
                val = self.int_list(dim1 * dim2 * dim3 * dim4 * dim5, min_val,
                                    max_val, 'val')
                return self.tensor(
                    val,
                    5, [dim1, dim2, dim3, dim4, dim5],
                    dtype=dtype,
                    np_type=np_type), rank
        else:
            raise ValueError('dtype is not supported.')

    def param_attr(self):
        """Generate Paddle parameter attributes.

        User can set parameter's attributes to control training details.

        Returns:
            A Paddle parameter attributes object.
        """
        initializer = None
        learning_rate = self.float_range(0.0, 0.99, 'learning_rate')
        regularizer_val = self.float_range(-10.0, 100.0, 'regularizer_val')
        regularizer = self.pick([
            None, paddle.regularizer.L1Decay(regularizer_val),
            paddle.regularizer.L2Decay(regularizer_val)
        ], 'regularizer')
        trainable = self.bool('trainable')
        do_model_average = self.bool('do_model_average')
        need_clip = self.bool('need_clip')
        return paddle.ParamAttr(
            initializer=initializer,
            learning_rate=learning_rate,
            regularizer=regularizer,
            trainable=trainable,
            do_model_average=do_model_average,
            need_clip=need_clip)
