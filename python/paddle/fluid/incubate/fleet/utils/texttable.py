# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# texttable - module for creating simple ASCII tables
# Copyright (C) 2003-2019 Gerome Fournier <jef(at)foutaise.org>
"""module for creating simple ASCII tables

Homepage: https://github.com/foutaise/texttable/

Example:

    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(['t',  # text
                          'f',  # float (decimal)
                          'e',  # float (exponent)
                          'i',  # integer
                          'a']) # automatic
    table.set_cols_align(["l", "r", "r", "r", "l"])
    table.add_rows([["text",    "float", "exp", "int", "auto"],
                    ["abcd",    "67",    654,   89,    128.001],
                    ["efghijk", 67.5434, .654,  89.6,  12800000000000000000000.00023],
                    ["lmn",     5e-78,   5e-78, 89.4,  .000000000000128],
                    ["opqrstu", .023,    5e+78, 92.,   12800000000000000000000]])
    print table.draw()

Result:
    text   float       exp      int     auto
    ===========================================
    abcd   67.000   6.540e+02   89    128.001
    efgh   67.543   6.540e-01   90    1.280e+22
    ijkl   0.000    5.000e-78   89    0.000
    mnop   0.023    5.000e+78   92    1.280e+22
"""

from __future__ import division

__all__ = ["Texttable", "ArraySizeError"]

__author__ = 'Gerome Fournier <jef(at)foutaise.org>'
__license__ = 'MIT'
__version__ = '1.6.2'
__credits__ = """\
Jeff Kowalczyk:
    - textwrap improved import
    - comment concerning header output

Anonymous:
    - add_rows method, for adding rows in one go

Sergey Simonenko:
    - redefined len() function to deal with non-ASCII characters

Roger Lew:
    - columns datatype specifications

Brian Peterson:
    - better handling of unicode errors

Frank Sachsenheim:
    - add Python 2/3-compatibility

Maximilian Hils:
    - fix minor bug for Python 3 compatibility

frinkelpi:
    - preserve empty lines
"""

import sys
import unicodedata

# define a text wrapping function to wrap some text
# to a specific width:
# - use cjkwrap if available (better CJK support)
# - fallback to textwrap otherwise
try:
    import cjkwrap

    def textwrapper(txt, width):
        return cjkwrap.wrap(txt, width)
except ImportError:
    try:
        import textwrap

        def textwrapper(txt, width):
            return textwrap.wrap(txt, width)
    except ImportError:
        sys.stderr.write("Can't import textwrap module!\n")
        raise

# define a function to calculate the rendering width of a unicode character
# - use wcwidth if available
# - fallback to unicodedata information otherwise
try:
    import wcwidth

    def uchar_width(c):
        """Return the rendering width of a unicode character
        """
        return max(0, wcwidth.wcwidth(c))
except ImportError:

    def uchar_width(c):
        """Return the rendering width of a unicode character
        """
        if unicodedata.east_asian_width(c) in 'WF':
            return 2
        elif unicodedata.combining(c):
            return 0
        else:
            return 1


from functools import reduce

if sys.version_info >= (3, 0):
    unicode_type = str
    bytes_type = bytes
else:
    unicode_type = unicode
    bytes_type = str


def obj2unicode(obj):
    """Return a unicode representation of a python object
    """
    if isinstance(obj, unicode_type):
        return obj
    elif isinstance(obj, bytes_type):
        try:
            return unicode_type(obj, 'utf-8')
        except UnicodeDecodeError as strerror:
            sys.stderr.write(
                "UnicodeDecodeError exception for string '%s': %s\n" %
                (obj, strerror))
            return unicode_type(obj, 'utf-8', 'replace')
    else:
        return unicode_type(obj)


def len(iterable):
    """Redefining len here so it will be able to work with non-ASCII characters
    """
    if isinstance(iterable, bytes_type) or isinstance(iterable, unicode_type):
        return sum([uchar_width(c) for c in obj2unicode(iterable)])
    else:
        return iterable.__len__()


class ArraySizeError(Exception):
    """Exception raised when specified rows don't fit the required size
    """

    def __init__(self, msg):
        self.msg = msg
        Exception.__init__(self, msg, '')

    def __str__(self):
        return self.msg


class FallbackToText(Exception):
    """Used for failed conversion to float"""
    pass


class Texttable:

    BORDER = 1
    HEADER = 1 << 1
    HLINES = 1 << 2
    VLINES = 1 << 3

    def __init__(self, max_width=80):
        """Constructor

        - max_width is an integer, specifying the maximum width of the table
        - if set to 0, size is unlimited, therefore cells won't be wrapped
        """

        self.set_max_width(max_width)
        self._precision = 3

        self._deco = Texttable.VLINES | Texttable.HLINES | Texttable.BORDER | \
            Texttable.HEADER
        self.set_chars(['-', '|', '+', '='])
        self.reset()

    def reset(self):
        """Reset the instance

        - reset rows and header
        """

        self._hline_string = None
        self._row_size = None
        self._header = []
        self._rows = []
        return self

    def set_max_width(self, max_width):
        """Set the maximum width of the table

        - max_width is an integer, specifying the maximum width of the table
        - if set to 0, size is unlimited, therefore cells won't be wrapped
        """
        self._max_width = max_width if max_width > 0 else False
        return self

    def set_chars(self, array):
        """Set the characters used to draw lines between rows and columns

        - the array should contain 4 fields:

            [horizontal, vertical, corner, header]

        - default is set to:

            ['-', '|', '+', '=']
        """

        if len(array) != 4:
            raise ArraySizeError("array should contain 4 characters")
        array = [x[:1] for x in [str(s) for s in array]]
        (self._char_horiz, self._char_vert, self._char_corner,
         self._char_header) = array
        return self

    def set_deco(self, deco):
        """Set the table decoration

        - 'deco' can be a combinaison of:

            Texttable.BORDER: Border around the table
            Texttable.HEADER: Horizontal line below the header
            Texttable.HLINES: Horizontal lines between rows
            Texttable.VLINES: Vertical lines between columns

           All of them are enabled by default

        - example:

            Texttable.BORDER | Texttable.HEADER
        """

        self._deco = deco
        return self

    def set_header_align(self, array):
        """Set the desired header alignment

        - the elements of the array should be either "l", "c" or "r":

            * "l": column flushed left
            * "c": column centered
            * "r": column flushed right
        """

        self._check_row_size(array)
        self._header_align = array
        return self

    def set_cols_align(self, array):
        """Set the desired columns alignment

        - the elements of the array should be either "l", "c" or "r":

            * "l": column flushed left
            * "c": column centered
            * "r": column flushed right
        """

        self._check_row_size(array)
        self._align = array
        return self

    def set_cols_valign(self, array):
        """Set the desired columns vertical alignment

        - the elements of the array should be either "t", "m" or "b":

            * "t": column aligned on the top of the cell
            * "m": column aligned on the middle of the cell
            * "b": column aligned on the bottom of the cell
        """

        self._check_row_size(array)
        self._valign = array
        return self

    def set_cols_dtype(self, array):
        """Set the desired columns datatype for the cols.

        - the elements of the array should be either a callable or any of
          "a", "t", "f", "e" or "i":

            * "a": automatic (try to use the most appropriate datatype)
            * "t": treat as text
            * "f": treat as float in decimal format
            * "e": treat as float in exponential format
            * "i": treat as int
            * a callable: should return formatted string for any value given

        - by default, automatic datatyping is used for each column
        """

        self._check_row_size(array)
        self._dtype = array
        return self

    def set_cols_width(self, array):
        """Set the desired columns width

        - the elements of the array should be integers, specifying the
          width of each column. For example:

                [10, 20, 5]
        """

        self._check_row_size(array)
        try:
            array = list(map(int, array))
            if reduce(min, array) <= 0:
                raise ValueError
        except ValueError:
            sys.stderr.write("Wrong argument in column width specification\n")
            raise
        self._width = array
        return self

    def set_precision(self, width):
        """Set the desired precision for float/exponential formats

        - width must be an integer >= 0

        - default value is set to 3
        """

        if not type(width) is int or width < 0:
            raise ValueError('width must be an integer greater then 0')
        self._precision = width
        return self

    def header(self, array):
        """Specify the header of the table
        """

        self._check_row_size(array)
        self._header = list(map(obj2unicode, array))
        return self

    def add_row(self, array):
        """Add a row in the rows stack

        - cells can contain newlines and tabs
        """

        self._check_row_size(array)

        if not hasattr(self, "_dtype"):
            self._dtype = ["a"] * self._row_size

        cells = []
        for i, x in enumerate(array):
            cells.append(self._str(i, x))
        self._rows.append(cells)
        return self

    def add_rows(self, rows, header=True):
        """Add several rows in the rows stack

        - The 'rows' argument can be either an iterator returning arrays,
          or a by-dimensional array
        - 'header' specifies if the first row should be used as the header
          of the table
        """

        # nb: don't use 'iter' on by-dimensional arrays, to get a
        #     usable code for python 2.1
        if header:
            if hasattr(rows, '__iter__') and hasattr(rows, 'next'):
                self.header(rows.next())
            else:
                self.header(rows[0])
                rows = rows[1:]
        for row in rows:
            self.add_row(row)
        return self

    def draw(self):
        """Draw the table

        - the table is returned as a whole string
        """

        if not self._header and not self._rows:
            return
        self._compute_cols_width()
        self._check_align()
        out = ""
        if self._has_border():
            out += self._hline()
        if self._header:
            out += self._draw_line(self._header, isheader=True)
            if self._has_header():
                out += self._hline_header()
        length = 0
        for row in self._rows:
            length += 1
            out += self._draw_line(row)
            if self._has_hlines() and length < len(self._rows):
                out += self._hline()
        if self._has_border():
            out += self._hline()
        return out[:-1]

    @classmethod
    def _to_float(cls, x):
        if x is None:
            raise FallbackToText()
        try:
            return float(x)
        except (TypeError, ValueError):
            raise FallbackToText()

    @classmethod
    def _fmt_int(cls, x, **kw):
        """Integer formatting class-method.

        - x will be float-converted and then used.
        """
        return str(int(round(cls._to_float(x))))

    @classmethod
    def _fmt_float(cls, x, **kw):
        """Float formatting class-method.

        - x parameter is ignored. Instead kw-argument f being x float-converted
          will be used.

        - precision will be taken from `n` kw-argument.
        """
        n = kw.get('n')
        return '%.*f' % (n, cls._to_float(x))

    @classmethod
    def _fmt_exp(cls, x, **kw):
        """Exponential formatting class-method.

        - x parameter is ignored. Instead kw-argument f being x float-converted
          will be used.

        - precision will be taken from `n` kw-argument.
        """
        n = kw.get('n')
        return '%.*e' % (n, cls._to_float(x))

    @classmethod
    def _fmt_text(cls, x, **kw):
        """String formatting class-method."""
        return obj2unicode(x)

    @classmethod
    def _fmt_auto(cls, x, **kw):
        """auto formatting class-method."""
        f = cls._to_float(x)
        if abs(f) > 1e8:
            fn = cls._fmt_exp
        elif f != f:  # NaN
            fn = cls._fmt_text
        elif f - round(f) == 0:
            fn = cls._fmt_int
        else:
            fn = cls._fmt_float
        return fn(x, **kw)

    def _str(self, i, x):
        """Handles string formatting of cell data

            i - index of the cell datatype in self._dtype
            x - cell data to format
        """
        FMT = {
            'a': self._fmt_auto,
            'i': self._fmt_int,
            'f': self._fmt_float,
            'e': self._fmt_exp,
            't': self._fmt_text,
        }

        n = self._precision
        dtype = self._dtype[i]
        try:
            if callable(dtype):
                return dtype(x)
            else:
                return FMT[dtype](x, n=n)
        except FallbackToText:
            return self._fmt_text(x)

    def _check_row_size(self, array):
        """Check that the specified array fits the previous rows size
        """

        if not self._row_size:
            self._row_size = len(array)
        elif self._row_size != len(array):
            raise ArraySizeError("array should contain %d elements" \
                % self._row_size)

    def _has_vlines(self):
        """Return a boolean, if vlines are required or not
        """

        return self._deco & Texttable.VLINES > 0

    def _has_hlines(self):
        """Return a boolean, if hlines are required or not
        """

        return self._deco & Texttable.HLINES > 0

    def _has_border(self):
        """Return a boolean, if border is required or not
        """

        return self._deco & Texttable.BORDER > 0

    def _has_header(self):
        """Return a boolean, if header line is required or not
        """

        return self._deco & Texttable.HEADER > 0

    def _hline_header(self):
        """Print header's horizontal line
        """

        return self._build_hline(True)

    def _hline(self):
        """Print an horizontal line
        """

        if not self._hline_string:
            self._hline_string = self._build_hline()
        return self._hline_string

    def _build_hline(self, is_header=False):
        """Return a string used to separated rows or separate header from
        rows
        """
        horiz = self._char_horiz
        if (is_header):
            horiz = self._char_header
        # compute cell separator
        s = "%s%s%s" % (horiz, [horiz, self._char_corner][self._has_vlines()],
                        horiz)
        # build the line
        l = s.join([horiz * n for n in self._width])
        # add border if needed
        if self._has_border():
            l = "%s%s%s%s%s\n" % (self._char_corner, horiz, l, horiz,
                                  self._char_corner)
        else:
            l += "\n"
        return l

    def _len_cell(self, cell):
        """Return the width of the cell

        Special characters are taken into account to return the width of the
        cell, such like newlines and tabs
        """

        cell_lines = cell.split('\n')
        maxi = 0
        for line in cell_lines:
            length = 0
            parts = line.split('\t')
            for part, i in zip(parts, list(range(1, len(parts) + 1))):
                length = length + len(part)
                if i < len(parts):
                    length = (length // 8 + 1) * 8
            maxi = max(maxi, length)
        return maxi

    def _compute_cols_width(self):
        """Return an array with the width of each column

        If a specific width has been specified, exit. If the total of the
        columns width exceed the table desired width, another width will be
        computed to fit, and cells will be wrapped.
        """

        if hasattr(self, "_width"):
            return
        maxi = []
        if self._header:
            maxi = [self._len_cell(x) for x in self._header]
        for row in self._rows:
            for cell, i in zip(row, list(range(len(row)))):
                try:
                    maxi[i] = max(maxi[i], self._len_cell(cell))
                except (TypeError, IndexError):
                    maxi.append(self._len_cell(cell))

        ncols = len(maxi)
        content_width = sum(maxi)
        deco_width = 3 * (ncols - 1) + [0, 4][self._has_border()]
        if self._max_width and (content_width + deco_width) > self._max_width:
            """ content too wide to fit the expected max_width
            let's recompute maximum cell width for each cell
            """
            if self._max_width < (ncols + deco_width):
                raise ValueError('max_width too low to render data')
            available_width = self._max_width - deco_width
            newmaxi = [0] * ncols
            i = 0
            while available_width > 0:
                if newmaxi[i] < maxi[i]:
                    newmaxi[i] += 1
                    available_width -= 1
                i = (i + 1) % ncols
            maxi = newmaxi
        self._width = maxi

    def _check_align(self):
        """Check if alignment has been specified, set default one if not
        """

        if not hasattr(self, "_header_align"):
            self._header_align = ["c"] * self._row_size
        if not hasattr(self, "_align"):
            self._align = ["l"] * self._row_size
        if not hasattr(self, "_valign"):
            self._valign = ["t"] * self._row_size

    def _draw_line(self, line, isheader=False):
        """Draw a line

        Loop over a single cell length, over all the cells
        """

        line = self._splitit(line, isheader)
        space = " "
        out = ""
        for i in range(len(line[0])):
            if self._has_border():
                out += "%s " % self._char_vert
            length = 0
            for cell, width, align in zip(line, self._width, self._align):
                length += 1
                cell_line = cell[i]
                fill = width - len(cell_line)
                if isheader:
                    align = self._header_align[length - 1]
                if align == "r":
                    out += fill * space + cell_line
                elif align == "c":
                    out += (int(fill/2) * space + cell_line \
                            + int(fill/2 + fill%2) * space)
                else:
                    out += cell_line + fill * space
                if length < len(line):
                    out += " %s " % [space, self._char_vert][self._has_vlines()]
            out += "%s\n" % ['', space + self._char_vert][self._has_border()]
        return out

    def _splitit(self, line, isheader):
        """Split each element of line to fit the column width

        Each element is turned into a list, result of the wrapping of the
        string to the desired width
        """

        line_wrapped = []
        for cell, width in zip(line, self._width):
            array = []
            for c in cell.split('\n'):
                if c.strip() == "":
                    array.append("")
                else:
                    array.extend(textwrapper(c, width))
            line_wrapped.append(array)
        max_cell_lines = reduce(max, list(map(len, line_wrapped)))
        for cell, valign in zip(line_wrapped, self._valign):
            if isheader:
                valign = "t"
            if valign == "m":
                missing = max_cell_lines - len(cell)
                cell[:0] = [""] * int(missing / 2)
                cell.extend([""] * int(missing / 2 + missing % 2))
            elif valign == "b":
                cell[:0] = [""] * (max_cell_lines - len(cell))
            else:
                cell.extend([""] * (max_cell_lines - len(cell)))
        return line_wrapped
