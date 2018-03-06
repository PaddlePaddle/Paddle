import unittest
import threading
import sys
sys.path.append("../")

import namespace as space
from Variable import Variable


class NameprefixTest(unittest.TestCase):
    def test_hierarchy(self):
        with space.nameprefix("father") as father:
            self.assertEqual("father/", space.current_nameprefix())
            with space.nameprefix("child") as child:
                self.assertEqual("father/child/", space.current_nameprefix())

    def test_sibling(self):
        with space.nameprefix("father"):
            with space.nameprefix("first"):
                self.assertEqual("father/first/", space.current_nameprefix())
            with space.nameprefix("second"):
                self.assertEqual("father/second/", space.current_nameprefix())
            self.assertEqual("father/", space.current_nameprefix())

    def test_multithread(self):
        def _thread_test(idx):
            with space.nameprefix("father") as father:
                with space.nameprefix("%s" % (str(idx))) as child:
                    self.assertEqual("father/%s/" % (str(idx)),
                                     space.current_nameprefix())

        ths = []
        THREAD_NUM = 3
        for i in range(THREAD_NUM):
            t = threading.Thread(target=_thread_test, args=(i, ))
            ths.append(t)
        for i in range(THREAD_NUM):
            ths[i].run()

    def test_namehiding(self):
        with space.nameprefix("M1") as A:
            a = Variable("a")
        with space.nameprefix("M2") as B:
            a = Variable("a")
        c = Variable("c")
        c = Variable("M1/a") + Variable("M2/a")


if __name__ == '__main__':
    unittest.main()
