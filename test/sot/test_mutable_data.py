# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

from paddle.jit.sot.opcode_translator.executor.mutable_data import (
    MutableData,
    MutableDictLikeData,
    MutableListLikeData,
)


class VariableBase:
    def __init__(self): ...


class ConstVariable(VariableBase):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"ConstVariable({self.value})"

    def __eq__(self, other):
        if not isinstance(other, ConstVariable):
            return False
        return self.value == other.value


class DictVariable(VariableBase):
    def __init__(self, data):
        self.data = data
        self.proxy = MutableDictLikeData(data, DictVariable.proxy_getter)

    @staticmethod
    def proxy_getter(proxy, key):
        if key not in proxy.original_data:
            return MutableData.Empty()
        return ConstVariable(proxy.original_data[key])

    def getitem(self, key):
        res = self.proxy.get(key)
        if isinstance(res, MutableData.Empty):
            raise KeyError(f"Key {key} not found")
        return res

    def setitem(self, key, value):
        self.proxy.set(key, value)

    def delitem(self, key):
        self.proxy.delete(key)


class ListVariable(VariableBase):
    def __init__(self, data):
        self.data = data
        self.proxy = MutableListLikeData(data, ListVariable.proxy_getter)

    @staticmethod
    def proxy_getter(proxy, key):
        if key < 0 or key >= len(proxy.original_data):
            return MutableData.Empty()
        return ConstVariable(proxy.original_data[key])

    def getitem(self, key):
        if isinstance(key, int):
            res = self.proxy.get(key)
            if isinstance(res, MutableData.Empty):
                raise IndexError(f"Index {key} out of range")
            return res
        elif isinstance(key, slice):
            return self.proxy.get_all()[key]
        else:
            raise TypeError(f"Invalid key type {type(key)}")

    def __getitem__(self, key):
        return self.getitem(key)

    def setitem(self, key, value):
        if isinstance(key, int):
            self.proxy.set(key, value)
        elif isinstance(key, slice):
            start, end, step = key.indices(self.proxy.length)
            indices = list(range(start, end, step))
            if step == 1:
                # replace a continuous range
                for i, idx in enumerate(indices):
                    self.proxy.delete(idx - i)
                for i, item in enumerate(value):
                    self.proxy.insert(start + i, item)
            else:
                # replace some elements
                if len(indices) != len(value):
                    raise ValueError(
                        f"Attempt to replace {len(indices)} items with {len(value)}"
                    )
                for i, idx in enumerate(indices):
                    self.proxy.set(idx, value[i])

    def delitem(self, key):
        self.proxy.delete(key)

    def insert(self, index, value):
        self.proxy.insert(index, value)

    def append(self, value):
        self.proxy.insert(self.proxy.length, value)

    def extend(self, value):
        for item in value:
            self.append(item)

    def pop(self, index=-1):
        res = self.getitem(index)
        self.delitem(index)
        return res

    def clear(self):
        for i in range(self.proxy.length):
            self.delitem(0)

    def remove(self, value):
        for i in range(self.proxy.length):
            if self.getitem(i) == value:
                self.delitem(i)
                return
        raise ValueError(f"Value {value} not found")

    def sort(self, key=None, reverse=False):
        if key is None:
            key = lambda x: x
        permutation = list(range(self.proxy.length))
        permutation.sort(
            key=lambda x: key(self.getitem(x).value), reverse=reverse
        )
        self.proxy.permutate(permutation)

    def reverse(self):
        permutation = list(range(self.proxy.length))
        permutation.reverse()
        self.proxy.permutate(permutation)


class TestMutableDictLikeVariable(unittest.TestCase):
    def test_getitem(self):
        data = {"a": 1, "b": 2}
        var = DictVariable(data)
        self.assertEqual(var.getitem("a"), ConstVariable(1))
        self.assertEqual(var.getitem("b"), ConstVariable(2))

    def test_setitem(self):
        data = {"a": 1, "b": 2}
        var = DictVariable(data)
        var.setitem("a", ConstVariable(3))
        self.assertEqual(var.getitem("a"), ConstVariable(3))
        var.setitem("c", ConstVariable(4))
        self.assertEqual(var.getitem("c"), ConstVariable(4))

    def test_delitem(self):
        data = {"a": 1, "b": 2}
        var = DictVariable(data)
        var.delitem("a")
        with self.assertRaises(KeyError):
            var.getitem("a")

    def test_keys(self):
        data = {"a": 1, "b": 2}
        var = DictVariable(data)
        self.assertEqual(list(var.proxy.get_all().keys()), ["a", "b"])


class TestMutableListLikeVariable(unittest.TestCase):
    def test_getitem(self):
        data = [1, 2, 3]
        var = ListVariable(data)
        self.assertEqual(var.getitem(0), ConstVariable(1))
        self.assertEqual(var.getitem(1), ConstVariable(2))
        self.assertEqual(var.getitem(2), ConstVariable(3))

    def test_getitem_slice_1(self):
        data = [1, 2, 3, 4, 5, 6, 7]
        var = ListVariable(data)
        self.assertEqual(
            var.getitem(slice(0, 3)),
            [ConstVariable(1), ConstVariable(2), ConstVariable(3)],
        )
        self.assertEqual(
            var.getitem(slice(4, 1, -1)),
            [ConstVariable(5), ConstVariable(4), ConstVariable(3)],
        )
        self.assertEqual(
            var.getitem(slice(1, 5, 2)),
            [ConstVariable(2), ConstVariable(4)],
        )

    def test_getitem_slice_2(self):
        data = [1, 2, 3, 4, 5, 6, 7]
        var = ListVariable(data)
        self.assertEqual(
            var[0:3],
            [ConstVariable(1), ConstVariable(2), ConstVariable(3)],
        )
        self.assertEqual(
            var[4:1:-1],
            [ConstVariable(5), ConstVariable(4), ConstVariable(3)],
        )
        self.assertEqual(
            var[1:5:2],
            [ConstVariable(2), ConstVariable(4)],
        )

    def test_setitem(self):
        data = [1, 2, 3]
        var = ListVariable(data)
        var.setitem(0, ConstVariable(4))
        self.assertEqual(var.getitem(0), ConstVariable(4))
        var.append(ConstVariable(5))
        self.assertEqual(var.getitem(3), ConstVariable(5))

    def test_setitem_slice_1(self):
        data = [1, 2, 3, 4, 5, 6, 7]
        var = ListVariable(data)
        var.setitem(slice(0, 3), [ConstVariable(4), ConstVariable(5)])
        self.assertEqual(
            [var.getitem(i) for i in range(var.proxy.length)],
            [ConstVariable(n) for n in [4, 5, 4, 5, 6, 7]],
        )
        var.setitem(
            slice(4, 1, -1),
            [ConstVariable(8), ConstVariable(9), ConstVariable(10)],
        )
        self.assertEqual(
            [var.getitem(i) for i in range(var.proxy.length)],
            [ConstVariable(n) for n in [4, 5, 10, 9, 8, 7]],
        )

    def test_setitem_slice_2(self):
        data = [1, 2, 3, 4, 5, 6, 7]
        var = ListVariable(data)
        var.setitem(slice(2, 5, 2), [ConstVariable(8), ConstVariable(9)])
        self.assertEqual(
            [var.getitem(i) for i in range(var.proxy.length)],
            [ConstVariable(n) for n in [1, 2, 8, 4, 9, 6, 7]],
        )

    def test_delitem(self):
        data = [1, 2, 3]
        var = ListVariable(data)
        var.delitem(0)
        with self.assertRaises(IndexError):
            var.getitem(2)
        var.pop()
        with self.assertRaises(IndexError):
            var.getitem(1)

    def test_insert(self):
        data = [1, 2, 3]
        var = ListVariable(data)
        var.insert(0, ConstVariable(4))
        self.assertEqual(
            [var.getitem(i) for i in range(var.proxy.length)],
            [ConstVariable(n) for n in [4, 1, 2, 3]],
        )
        var.insert(2, ConstVariable(5))
        self.assertEqual(
            [var.getitem(i) for i in range(var.proxy.length)],
            [ConstVariable(n) for n in [4, 1, 5, 2, 3]],
        )

    def test_append(self):
        data = [1, 2, 3]
        var = ListVariable(data)
        var.append(ConstVariable(4))
        self.assertEqual(var.getitem(3), ConstVariable(4))

    def test_extend(self):
        data = [1, 2, 3]
        var = ListVariable(data)
        var.extend([ConstVariable(4), ConstVariable(5)])
        self.assertEqual(var.getitem(3), ConstVariable(4))
        self.assertEqual(var.getitem(4), ConstVariable(5))

    def test_pop(self):
        data = [1, 2, 3]
        var = ListVariable(data)
        self.assertEqual(var.pop(), ConstVariable(3))
        self.assertEqual(var.pop(0), ConstVariable(1))

    def test_clear(self):
        data = [1, 2, 3]
        var = ListVariable(data)
        var.clear()
        self.assertEqual(var.proxy.length, 0)

    def test_remove(self):
        data = [1, 2, 3]
        var = ListVariable(data)
        var.remove(ConstVariable(2))
        self.assertEqual(var.getitem(0), ConstVariable(1))
        self.assertEqual(var.getitem(1), ConstVariable(3))
        with self.assertRaises(ValueError):
            var.remove(ConstVariable(2))

    def test_sort(self):
        data = [2, 3, 0, 4, 1, 5]
        var = ListVariable(data)
        var.sort()
        self.assertEqual(
            [var.getitem(i) for i in range(var.proxy.length)],
            [ConstVariable(n) for n in [0, 1, 2, 3, 4, 5]],
        )

    def test_sort_with_key(self):
        data = [-1, -4, 2, 0, 5, -3]
        var = ListVariable(data)
        var.sort(key=lambda x: x**2)
        self.assertEqual(
            [var.getitem(i) for i in range(var.proxy.length)],
            [ConstVariable(n) for n in [0, -1, 2, -3, -4, 5]],
        )

    def test_sort_reverse(self):
        data = [2, 3, 0, 4, 1, 5]
        var = ListVariable(data)
        var.sort(reverse=True)
        self.assertEqual(
            [var.getitem(i) for i in range(var.proxy.length)],
            [ConstVariable(n) for n in [5, 4, 3, 2, 1, 0]],
        )

    def test_reverse(self):
        data = [2, 3, 0, 4, 1, 5]
        var = ListVariable(data)
        var.reverse()
        self.assertEqual(
            [var.getitem(i) for i in range(var.proxy.length)],
            [ConstVariable(n) for n in [5, 1, 4, 0, 3, 2]],
        )


if __name__ == "__main__":
    unittest.main()
