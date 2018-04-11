#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import six
import astroid

from pylint.checkers import BaseChecker, utils
from pylint.interfaces import IAstroidChecker


def register(linter):
    """Register checkers."""
    linter.register_checker(DocstringChecker(linter))


class Docstring(object):
    def __init__(self):
        from collections import defaultdict
        self.d = defaultdict(list)  #name->[]
        self.d['Args'] = []
        self.d['Examples'] = []
        self.d['Returns'] = []
        self.args = {}  #arg_name->arg_type

    def get_level(self, string, indent='    '):
        level = 0
        unit_size = len(indent)
        while string[:unit_size] == indent:
            string = string[unit_size:]
            level += 1

        return level

    def parse(self, doc):
        lines = doc.splitlines()
        state = ("others", -1)
        for l in lines:
            c = l.strip()
            if len(c) <= 0:
                continue

            level = self.get_level(l)
            if c.startswith("Args:"):
                state = ("Args", level)
            elif c.startswith("Returns:"):
                state = ("Returns", level)
            elif c.startswith("Raises:"):
                state = ("Raises", level)
            elif c.startswith("Examples:"):
                state = ("Examples", level)
            else:
                if level > state[1]:
                    self.d[state[0]].append(c)
                    continue

                state = ("others", -1)
                self.d[state[0]].append(c)

        self._arg_with_type()
        return True

    def get_returns(self):
        return self.d['Returns']

    def get_raises(self):
        return self.d['Raises']

    def get_examples(self):
        return self.d['Examples']

    def _arg_with_type(self):
        import re

        for t in self.d['Args']:
            m = re.search('([A-Za-z0-9_-]+)\s{0,4}(\(.+\))\s{0,4}:', t)
            if m:
                self.args[m.group(1)] = m.group(2)

        return self.args


class DocstringChecker(BaseChecker):
    __implements__ = (IAstroidChecker, )

    POSITIONAL_MESSAGE_ID = 'str-used-on-positional-format-argument'
    KEYWORD_MESSAGE_ID = 'str-used-on-keyword-format-argument'

    name = 'doc-string-checker'
    symbol = "doc-string"
    priority = -1
    msgs = {
        'W9001': ('One line doc string on > 1 lines', symbol + "-one-line",
                  'Used when a short doc string is on multiple lines'),
        'W9002':
        ('Doc string does not end with "." period', symbol + "-end-with",
         'Used when a doc string does not end with a period'),
        'W9003': ('All args with their types must be mentioned in doc string',
                  symbol + "-with-all-args",
                  'Used when not all arguments are in the doc string '),
        'W9005': ('Missing docstring or docstring is too shorter',
                  symbol + "-missing", 'Add docstring longer >=10'),
        'W9006': ('Docstring indent error, use 4 space for indent',
                  symbol + "-indent-error", 'Use 4 space for indent'),
        'W9007': ('You should add `Returns` in comments',
                  symbol + "-with-returns",
                  'There should be a `Returns` section in comments'),
        'W9008': ('You should add `Raises` section in comments',
                  symbol + "-with-raises",
                  'There should be a `Raises` section in comments'),
    }
    options = ()

    def visit_functiondef(self, node):
        self.check_doc_string(node)

        if node.tolineno - node.fromlineno <= 10:
            return True

        if not node.doc:
            return True

        doc = Docstring()
        doc.parse(node.doc)

        self.all_args_in_doc(node, doc)
        self.with_returns(node, doc)
        self.with_raises(node, doc)

    def visit_module(self, node):
        self.check_doc_string(node)

    def visit_classdef(self, node):
        self.check_doc_string(node)

    def check_doc_string(self, node):
        self.missing_doc_string(node)
        self.one_line_one_one_line(node)
        self.has_period(node)
        self.indent_style(node)

    def missing_doc_string(self, node):
        if node.tolineno - node.fromlineno <= 10:
            return True

        if node.doc is None or len(node.doc) < 10:
            self.add_message('W9005', node=node, line=node.fromlineno)
        return False

    # FIXME(gongwb): give the docstring line-no
    def indent_style(self, node, indent=4):
        """check doc string indent style"""
        if node.doc is None:
            return True

        doc = node.doc
        lines = doc.splitlines()

        for l in lines:
            cur_indent = len(l) - len(l.lstrip())
            if cur_indent % indent != 0:
                self.add_message('W9006', node=node, line=node.fromlineno)
                return False

        return True

    def one_line_one_one_line(self, node):
        """One line docs (len < 40) are on one line"""

        doc = node.doc
        if doc is None:
            return True

        if len(doc) > 40:
            return True
        elif sum(doc.find(nl) for nl in ('\n', '\r', '\n\r')) == -3:
            return True
        else:
            self.add_message('W9001', node=node, line=node.fromlineno)
            return False

        return True

    def has_period(self, node):
        """Doc ends in a period"""
        if node.doc is None:
            return True

        if len(node.doc.splitlines()) > 1:
            return True

        if not node.doc.strip().endswith('.'):
            self.add_message('W9002', node=node, line=node.fromlineno)
            return False

        return True

    def with_raises(self, node, doc):
        find = False
        for t in node.body:
            if not isinstance(t, astroid.Raise):
                continue

            find = True
            break

        if not find:
            return True

        if len(doc.get_raises()) == 0:
            self.add_message('W9008', node=node, line=node.fromlineno)
            return False

        return True

    def with_returns(self, node, doc):
        find = False
        for t in node.body:
            if not isinstance(t, astroid.Return):
                continue

            find = True
            break

        if not find:
            return True

        if len(doc.get_returns()) == 0:
            self.add_message('W9007', node=node, line=node.fromlineno)
            return False

        return True

    def all_args_in_doc(self, node, doc):
        """All function arguments are mentioned in doc"""
        args = []
        for arg in node.args.get_children():
            if (not isinstance(arg, astroid.AssignName)) \
                or arg.name == "self":
                continue
            args.append(arg.name)

        if len(args) <= 0:
            return True

        parsed_args = doc.args
        if len(args) > 0 and len(parsed_args) <= 0:
            print("debug:parsed args: ", parsed_args)
            self.add_message('W9003', node=node, line=node.fromlineno)
            return False

        for t in args:
            if t not in parsed_args:
                print(t, " with (type) not in ", parsed_args)
                self.add_message('W9003', node=node, line=node.fromlineno)
                return False

        return True
