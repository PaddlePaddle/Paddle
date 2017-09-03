import unittest
import threading
import sys
sys.path.append("../")

import namespace as space


class NamespaceTest(unittest.TestCase):
    def test_hierarchy(self):
        with space.namespace("father") as father:
            self.assertEqual("father/", father)
            with space.namespace("child") as child:
                # print space.current_namespace()
                self.assertEqual("father/child/", space.current_namespace())
                self.assertEqual("father/child/", child)

    def test_sibling(self):
        with space.namespace("father"):
            with space.namespace("first"):
                self.assertEqual("father/first/", space.current_namespace())
                a = 1
            with space.namespace("second"):
                self.assertEqual("father/second/", space.current_namespace())
                a = 2
            self.assertEqual("father/", space.current_namespace())

    # def test_multithread(self):
    #   with space.namespace("father"):

    ###############
    # demo case 


import paddle as pd


def case0(x):
    with pd.namespace() as A:
        a = pd.Variable()
    with pd.namespace() as B:
        a = pd.Variable()
    c = A.a + B.a


def case1(x):
    with pd.namespace() as A:
        with pd.namespace() as AA:
            a = pd.Variable()
    with pd.namespace() as B:
        a = pd.Variable()
    c = A.AA.a + B.a


#################


def f(x):
    """ """
    return x


def g(x):
    return x * x


if __name__ == '__main__':
    unittest.main()
