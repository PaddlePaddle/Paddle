# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import itertools
import re
import string

import numpy as np
import opt_einsum

from paddle import _C_ops

from ..base.data_feeder import check_type, check_variable_and_dtype
from ..base.framework import in_dynamic_or_pir_mode
from ..base.layer_helper import LayerHelper
from .linalg import matmul, transpose
from .manipulation import reshape, squeeze, unsqueeze
from .math import (
    multiply,
    sum as paddle_sum,
)

__all__ = []


def parse_op_labels(labelstr, operand):
    '''
    Parse labels for an input operand.

    Parameters
    ----------
    labelstr:
        the input label string
    operand:
        the input operand

    Returns
    -------
    the input operand's full label string in which all anonymous dimensions are
    labeled in dots.
    '''
    # Sanity checks
    for c in labelstr.replace('.', ''):
        assert (
            c.isalpha()
        ), f"Invalid equation: {c} is not a valid label, which should be letters."

    assert (
        labelstr.replace('...', '', 1).find('.') == -1
    ), "Invalid equation: `.` is found outside of an ellipsis."

    ndims = len(operand.shape)

    full_labelstr = labelstr.replace('...', '.' * (ndims - len(labelstr) + 3))

    assert (
        len(full_labelstr) == ndims
    ), f"Invalid equation: the label string '{labelstr}' misses dimensions."

    return full_labelstr


def parse_labels(labelstr, operands):
    '''
    Parse label strings for all input operands.

    Parameters
    ----------
    labelstr:
        The equation's label string
    operands:
        The input operands

    Returns
    -------
    list of full label strings for all input operands
    '''

    nop_labels = labelstr.split(',')
    assert len(nop_labels) == len(operands), (
        f"Invalid equation: the number of operands is {len(operands)}, "
        f"but found {len(nop_labels)} segments in the label equation."
    )

    return list(map(parse_op_labels, nop_labels, operands))


def validate_rhs(rhs, input_labels, n_bcast_dims):
    '''
    Check whether the equation's right hand side is valid
    '''
    # Sanity check.
    if n_bcast_dims > 0:
        assert (
            '...' in rhs
        ), "Invalid equation: missing ellipsis in output labels."

    rhs = rhs.replace('...', '')
    rhs_set = set(rhs)

    # Hidden assumption: available labels don't include '.'
    assert '.' not in input_labels

    # Verify that output labels all come from the set of input labels
    non_input_labels = rhs_set.difference(input_labels)
    assert not non_input_labels, (
        f"Invalid equation: "
        f"output label {sorted(non_input_labels)} not used by any input."
    )
    # Verify that output labels are not duplicate
    assert len(rhs) == len(
        rhs_set
    ), "Invalid equation: duplicate output labels are found."


def build_view(in_labels, out_labels):
    '''
    Build an inverse map of dimension indices. Three conditions must hold for
    the result to be meaningful.
    First, no duplicate letter labels in each label string.
    Second, the number of dots in dimout_labels >= that in in_labels.
    Third, dots are contiguous in each label string.

    Parameters
    ----------
    in_labels:
        The dimension labels to map to
    out_labels:
        The dimension labels to map from

    Returns
    -------
    The inverse map from out_labels to in_labels. The length of the inverse map equals that of
    out_labels. -1 is filled if there's no matching input dimension for a specific label.

    Examples
    --------
    in_labels = 'ij..', out_labels = '..ji'
    inv_map = [2, 3, 1, 0]
    in_labels = 'ij..', out_labels = '..kji'
    inv_map = [2, 3, -1, 1, 0]
    '''

    inv_map = [-1] * len(out_labels)

    # First build the broadcast dimension mapping
    # Find the broadcast index range in out_labels
    r = re.search(r'\.+', out_labels)
    if r:
        start, end = r.start(), r.end()
        s = re.search(r'\.+', in_labels)
        # fill the broadcast dimension indices from right to left.
        if s:
            for ax, dim in zip(
                range(start, end)[::-1], range(s.start(), s.end())[::-1]
            ):
                inv_map[ax] = dim

    # Now work on non-broadcast dimensions
    if r:
        it = itertools.chain(range(start), range(end, len(out_labels)))
    else:
        it = iter(range(len(out_labels)))

    for i in it:
        inv_map[i] = in_labels.find(out_labels[i])

    return inv_map


def build_global_view(nop_labels, rhs, n_bcast_dims):
    '''
    Build the global view, which is a layout of all dimension labels
    plus an index table that maps from the layout to the dimensions
    in each operand. In the global view, the dimensions are arranged
    such that output ones are put on the left and contraction ones
    are put on the right.

    Parameters
    ----------
    nop_labels:
        The input full label strings of all input operands
    rhs:
        The equation right hand side
    n_bcast_dims:
        The maximum number of broadcast dimensions

    Returns
    -------
    A tuple of g_labels, g_view, g_nout, g_count
    g_labels:
        the layout of all labels in a string
    g_view:
        the index table
    g_nout:
        the number of output dimensions
    g_count:
        the counter array for dimension contractions
    '''
    # Put all labels in alphabetical order
    concat = sorted(''.join(nop_labels).replace('.', ''))
    labels, count = [], []
    for a, b in zip(['.'] + concat, concat):
        if a != b:
            labels.append(b)
            count.append(1)
        else:
            count[-1] += 1

    if rhs is not None:
        validate_rhs(rhs, labels, n_bcast_dims)
        g_labels_out = rhs.replace('...', '.' * n_bcast_dims)
    else:
        g_labels_out = '.' * n_bcast_dims + ''.join(
            l for l, c in zip(labels, count) if c == 1
        )

    for i in range(len(count))[::-1]:
        if labels[i] in g_labels_out:
            labels.pop(i)
            count.pop(i)

    g_labels_sum = ''.join(labels)
    g_labels = g_labels_out + g_labels_sum
    g_view = [build_view(i, g_labels) for i in nop_labels]
    g_nout = len(g_labels_out)
    g_count = count

    return g_labels, g_view, g_nout, g_count


def build_global_shape(g_view, g_labels, op_shapes):
    '''
    The global shape is the shape of all dimensions rearranged and broadcasting
    to the global view. It's a reference data structure for einsum planning.

    Parameters
    ----------
    g_view:
        the global view
    op_shapes:
        the shapes of the all operands

    Returns
    -------
    g_shape:
        the global shape vector
    g_masks:
        list of shape masks for each operand. A dimension's shape mask is a boolean
        indicating whether its size > 1, in other words, it's not squeezable
    '''
    view_shapes = []
    g_masks = []

    for view, op_shape in zip(g_view, op_shapes):
        view_shapes.append([op_shape[dim] if dim > -1 else 1 for dim in view])

    g_shape = [set(sizes_per_ax) - {1} for sizes_per_ax in zip(*view_shapes)]

    non_bcastable = [ax for ax, sizes in enumerate(g_shape) if len(sizes) > 1]

    assert not non_bcastable, (
        f"Invalid operands: label {g_labels[non_bcastable[0]]} "
        f"corresponds to non-broadcastable dimensions."
    )

    g_shape = [sizes.pop() if len(sizes) > 0 else 1 for sizes in g_shape]

    g_masks = [
        [s > 1 or s == -1 for s in view_shape] for view_shape in view_shapes
    ]

    return g_shape, g_masks


def has_duplicated_labels(labels):
    '''
    Returns True if there is any duplicate label.
    '''
    labels = labels.replace('.', '')
    return len(labels) > len(set(labels))


def diagonalize(labels, operand):
    '''
    Merges dimensions with duplicate labels.

    For those dimensions with duplicate labels, merge them into one dimension
    which represents the diagonal elements. This requires the dimensions with
    duplicate labels are equal sized.

    Examples
    --------
    'ijj...i' would be merged into 'ij...'
    '''
    assert not has_duplicated_labels(
        labels
    ), 'Duplicate labels are not supported.'

    return labels, operand


def plan_reduce(plan, op, reduce_dims, keepdim):
    '''
    Add reduce to the plan
    '''
    varname = f'op{op}'

    f = lambda var, dims: paddle_sum(var, dims, keepdim=keepdim)
    step = f, [varname], varname, reduce_dims
    plan.add_step(step)


def plan_scalar_prod(plan, op1, op2):
    varnames = [f'op{op1}', f'op{op2}']
    f = lambda var1, var2: paddle_sum(var1) * var2
    # f = lambda var1, var2: var1 * var2
    step = f, varnames, varnames[1]
    plan.add_step(step)


def plan_matmul(plan, g_view, op1, op2, g_supports, g_shape, I, J1, J2, K):
    '''
    plan matmul
    '''
    # Transpose and re-shape op1 and op2 in I, J1, K and I, J2, K
    # Then apply matmul(x, y, transpose_x=False, transpose_y=True)
    var1, var2 = f'op{op1}', f'op{op2}'

    op1_view, op2_view = (g_view[op] for op in (op1, op2))

    I1 = [idx for idx in I if op1_view[idx] >= 0]
    I2 = [idx for idx in I if op2_view[idx] >= 0]
    op1_view = np.array(op1_view)
    op1_dims = op1_view[I1 + J1 + K]

    op2_view = np.array(op2_view)
    op2_dims = op2_view[I2 + J2 + K]

    op1_mask, op2_mask = (g_supports[op] for op in (op1, op2))
    op1_vshape = np.array([s if m else 1 for s, m in zip(g_shape, op1_mask)])
    op2_vshape = np.array([s if m else 1 for s, m in zip(g_shape, op2_mask)])
    vshape = np.maximum(op1_vshape, op2_vshape)

    i1, i2, j1, j2, k = map(len, (I1, I2, J1, J2, K))

    if any(op1_dims != np.arange(len(op1_dims))):
        # print(f'perm1: {perm1}')
        step = transpose, [var1], var1, list(op1_dims)
        plan.add_step(step)

    if any(op2_dims != np.arange(len(op2_dims))):
        # print(f'perm2: {perm2}')
        step = transpose, [var2], var2, list(op2_dims)
        plan.add_step(step)

    # Check if conditions hold for turning the operation into a matmul
    if (
        j1 + j2 > 0
        and k > 0
        and -1 not in np.concatenate((op1_vshape, op2_vshape))
    ):
        op1_shape = (
            list(op1_vshape[I])
            + [np.prod(op1_vshape[J1])]
            + [np.prod(op1_vshape[K])]
        )
        op2_shape = (
            list(op2_vshape[I])
            + [np.prod(op2_vshape[J2])]
            + [np.prod(op2_vshape[K])]
        )

        # Merge J dims and K dims by reshaping
        step = reshape, [var1], var1, op1_shape
        plan.add_step(step)
        step = reshape, [var2], var2, op2_shape
        plan.add_step(step)

        # Matmul
        step = matmul, [var1, var2], var2, False, True
        plan.add_step(step)

        # Reshape back
        shape = list(vshape[I + J1 + J2])
        step = reshape, [var2], var2, shape
        plan.add_step(step)

    elif j1 == j2 == k == 1:
        # Can still do matmul even unknown shapes are present
        step = matmul, [var1, var2], var2, False, True
        plan.add_step(step)

    # In the rest cases we opt for ops other than matmul
    else:
        # unsqueeze operands include J1...J2... dimensions
        if j2:
            fill = list(range(i1 + j1, i1 + j1 + j2))
            step = unsqueeze, [var1], var1, fill
            plan.add_step(step)
        if j1:
            fill = list(range(i2, i2 + j1))
            step = unsqueeze, [var2], var2, fill
            plan.add_step(step)
        # In case of no dimensions to contract, do an elementwise multiply
        if k == 0:
            # make broadcast
            step = multiply, [var1, var2], var2
            plan.add_step(step)
        # Contract and no join, turn into a dot
        elif j1 + j2 == 0 and k == 1:
            step = unsqueeze, [var1], var1, [-2]
            plan.add_step(step)
            step = unsqueeze, [var2], var2, [-1]
            plan.add_step(step)
            step = matmul, [var1, var2], var2
            plan.add_step(step)
            step = squeeze, [var2], var2, [-1, -2]
            plan.add_step(step)
        elif j1 + j2 == 0 and -1 not in np.concatenate(
            (op1_vshape[K], op2_vshape[K])
        ):
            assert all(op1_vshape[K] == op2_vshape[K])
            step = (
                reshape,
                [var1],
                var1,
                list(op1_vshape[I]) + [1] + [np.prod(op1_vshape[K])],
            )
            plan.add_step(step)
            step = (
                reshape,
                [var2],
                var2,
                list(op2_vshape[I]) + [1] + [np.prod(op2_vshape[K])],
            )
            plan.add_step(step)
            step = matmul, [var1, var2], var2, False, True
            plan.add_step(step)
            step = squeeze, [var2], var2, [-1, -2]
            plan.add_step(step)
        else:
            step = multiply, [var1, var2], var2
            plan.add_step(step)
            reduce_dims = list(range(-k, 0))
            plan_reduce(plan, op2, reduce_dims, keepdim=False)

    # Wrap up, updating auxiliary data
    # Updating g_mask for I and J axes
    for ax in I + J1 + J2:
        op2_mask[ax] = vshape[ax] > 1 or vshape[ax] == -1

    for ax in K:
        op2_mask[ax] = False

    for ax in range(len(op2_view)):
        op2_view[ax] = -1
    dim = 0
    for ax in I + J1 + J2:
        op2_view[ax], dim = dim, dim + 1

    g_view[op2] = list(op2_view)


def plan_summation(
    plan, g_view, op1, op2, g_supports, g_shape, g_count, n_bcast
):
    '''
    Plan various kinds of summation
    '''
    op1_view, op2_view = g_view[op1], g_view[op2]
    op1_mask, op2_mask = g_supports[op1], g_supports[op2]

    ndim = len(op1_view)
    nout = ndim - len(g_count)

    count = [0] * nout + g_count

    I, K, J1, J2 = list(range(n_bcast)), [], [], []

    for ax, dim1, dim2 in zip(
        range(n_bcast, ndim), op1_view[n_bcast:], op2_view[n_bcast:]
    ):
        if (dim1 != -1) != (dim2 != -1):
            if dim1 != -1:
                J1.append(ax)
            else:
                J2.append(ax)
        elif dim1 != -1:
            fold = int(op1_mask[ax]) + int(op2_mask[ax])
            if ax >= nout and fold == count[ax]:
                # Ready to fold the dimensions
                K.append(ax)
                count[ax] -= fold
            else:
                I.append(ax)
                count[ax] -= max(fold - 1, 0)

    # Update g_count
    g_count[:] = count[nout:]

    # Now it's OK to merge the K dims as the same shape holds
    # print(f'I: {I}   J1: {J1}    J2: {J2}   K: {K}')
    plan_matmul(plan, g_view, op1, op2, g_supports, g_shape, I, J1, J2, K)


def rearrange(axes):
    perm, fill = [], []
    for ax, dim in enumerate(axes):
        if dim < 0:
            fill.append(ax)
        else:
            perm.append(dim)
    # Trivial permutation returns []
    if all(i == dim for i, dim in enumerate(perm)):
        perm = []

    return perm, fill


def plan_broadcast(plan, operands, nop_axes):
    '''
    Plan broadcast across
    '''
    nop = len(operands)
    varnames = [f'op{i}' for i in range(nop)]

    for i, op_axes in zip(range(nop), nop_axes):
        # Re-arrange the dimensions according to the global layout
        perm, fill = rearrange(op_axes)
        var = varnames[i]
        if perm:
            step = transpose, [var], var, perm
            plan.add_step(step)
        if fill:
            step = unsqueeze, [var], var, fill
            plan.add_step(step)

    def f(*args):
        expr = ' * '.join(varnames)
        return eval(expr, dict(zip(varnames, args)))

    step = f, varnames, None
    plan.add_step(step)


class Plan:
    def __init__(self):
        self.env = {}
        self.steps = []

    def add_step(self, step):
        self.steps.append(step)

    def get_var(self, varname):
        return self.env[varname] if varname in self.env else None

    def set_var(self, varname, var):
        self.env[varname] = var

    def show(self):
        res = None
        for f, in_varnames, out_varname, *args in self.steps:
            print(repr((out_varname, f, *in_varnames, *args)))
        return res

    def execute(self):
        res = None
        for f, in_varnames, out_varname, *args in self.steps:
            res = f(*map(self.get_var, in_varnames), *args)
            if out_varname:
                self.set_var(out_varname, res)
        return res


def plan_einsum(operands, g_view, g_shape, g_supports, g_count, n_bcast):
    '''
    Plans the actual execution steps.
    Results
    -------
    the execution plan
    '''
    nop = len(operands)
    ndim = len(g_view[0])
    nout = ndim - len(g_count)

    # Initialize a plan with an environment
    plan = Plan()
    op_names = [f'op{i}' for i in range(nop)]
    list(map(plan.set_var, op_names, operands))

    # In case no dimensions to combine, do broadcast straight across
    if not g_count:
        plan_broadcast(plan, operands, g_view)
        return plan

    # Down count degenerate contraction dimensions.
    for view, support in zip(g_view, g_supports):
        # To collect the down count number, we use a type casting trick
        down_count = [
            int((d + 1) and (not s))
            for d, s in zip(view[nout:], support[nout:])
        ]
        for i, count in enumerate(down_count):
            g_count[i] -= count

    # Reduce any dimension for which g_support is set and g_count == 1
    for i, view, mask in zip(range(nop), g_view, g_supports):
        to_reduce = []
        for dim, masked, count in zip(view[nout:], mask[nout:], g_count):
            to_reduce.append(dim if (masked and count == 1) else -1)

        reduce_dims = list(filter(lambda x: x > -1, to_reduce))
        if reduce_dims:
            plan_reduce(plan, i, reduce_dims, keepdim=True)

        # Unset mask and decrease g_count for the reduced dimensions
        for i, d in enumerate(to_reduce):
            ax = i + nout
            mask[ax] = mask[ax] and (d == -1)
            g_count[i] -= 0 if d == -1 else 1

    # Plan the summations over the operand sequence
    for i in range(nop):
        # plan a single step

        if i == 0:
            continue

        # We'd like to arrange the dimensions in the following way:
        # [I...  J... K...]
        # [I...  J... K...]
        # where
        #       I... are aligned and not to be combined immediately
        #       J... are not aligned and not to be combined immediately
        #       K... are aligned and should be immediately combined
        # At this point the non-trivial broadcast dimensions in K are already reduced
        # and removed. That means all K dimensions are aligned and their sizes are not 1.
        # We then inspect the layout of I,J,K plus the above observation to make
        # specialization decisions.  The current strategy is set as follows:
        #  (1) if I... J... K... are all empty, it's multiplying a scalar
        #  (2) if K... are empty, better use a broadcast
        #  (3) if I... J... empty and K... not empty, a vector-vector multiply (or a dot)
        #  (4) Elsewise, either I... or J... not empty, and K... not empty, use a general matmul

        # Resolve the summation kind: dot, matmul or *
        if not any(g_supports[i - 1]):
            # op1 is a one element tensor.
            plan_scalar_prod(plan, i - 1, i)
        else:
            plan_summation(
                plan, g_view, i - 1, i, g_supports, g_shape, g_count, n_bcast
            )

    # for ax, dim in enumerate(g_view[nop-1][:nout]):
    #     assert dim == ax
    assert all(not masked for masked in g_supports[nop - 1][nout:])

    view = g_view[-1]
    if any(ax != dim for ax, dim in enumerate(view[:nout])):
        perm = [dim for dim in view if dim >= 0]
        if sorted(perm) != perm:
            varname = f'op{nop - 1}'
            step = transpose, [varname], varname, perm
            plan.add_step(step)
        dim = 0
        unsqueeze_dims = []
        for ax, d in enumerate(view):
            if d != -1:
                view[ax], dim = dim, dim + 1
        for ax, d in enumerate(view[:nout]):
            if d == -1:
                unsqueeze_dims.append(ax)
        if unsqueeze_dims:
            varname = f'op{nop - 1}'
            step = unsqueeze, [varname], varname, unsqueeze_dims
            plan.add_step(step)

    squeeze_dims = [dim for dim in view[nout:] if dim != -1]
    if squeeze_dims:
        # plan_reduce(plan, nop-1, reduce_dims, keepdim=False)
        varname = f'op{nop - 1}'
        step = squeeze, [varname], varname, squeeze_dims
        plan.add_step(step)

    return plan


def replace_ellipsis(left_equation, rhs, *operands):
    """
    we replace ... as unused variables to simplify the EinsumOp implementation.
    """
    ellipsis_strings = None
    max_ndim = 0
    new_operands = []
    unused_variables = {chr(c) for c in range(ord('a'), ord('z'))}
    for equ, operand in zip(left_equation.split(','), operands):
        ndims = len(operand.shape) - len(equ.replace("...", ""))
        max_ndim = max(max_ndim, ndims)
        for c in equ:
            unused_variables.discard(c)

    for equ, operand in zip(left_equation.split(','), operands):
        if '...' in equ:
            start_unsqueeze_idx = equ.index('...')
            to_squeeze_num = max_ndim - (
                len(operand.shape) - len(equ.replace("...", ""))
            )
            operand = unsqueeze(
                operand,
                axis=[i + start_unsqueeze_idx for i in range(to_squeeze_num)],
            )
        new_operands.append(operand)

    operands = new_operands
    ellipsis_strings = ''.join(unused_variables.pop() for _ in range(max_ndim))

    if ellipsis_strings is not None:
        left_equation = left_equation.replace('...', ellipsis_strings)
        rhs = rhs.replace('...', ellipsis_strings)
    return left_equation, rhs, operands


def preprocess(equation, *operands):
    """
    check equation / raise error, default right labels generation
    """
    equation = equation.replace(" ", "")
    nop = len(operands)
    assert nop > 0, (
        "Required at least one operand in Einsum API, but received %s " % nop
    )

    # Part the equation to left hand side and right hand side
    lhs, *rhs = equation.lower().split('->')
    assert len(rhs) < 2, "Invalid equation: multiple `->` were found."

    labels = parse_labels(lhs, operands)
    # Note, we distinguish between 'ij->' and 'ij' by setting rhs to '' and None
    rhs = rhs[0] if rhs else None
    if rhs is None:
        rhs = rhs_inference(lhs)

    assert len(lhs.split(',')) == len(operands), (
        f"Invalid equation: the number of operands is {len(operands)}, "
        f"but found {len(lhs.split(','))} segments in the label equation."
    )

    assert not (
        '...' in lhs and '...' not in rhs
    ), 'Invalid equation: missing ellipsis in output labels.'

    lhs, rhs, operands = replace_ellipsis(lhs, rhs, *operands)
    return lhs, rhs, labels, operands


def parse_fake_shape(equation, operands, labels):
    """

    this shape is just used for operands planning. may differ with the original shape.
    for example:
    ... is replaced by 1
    -1  is replaced by 1
    Results
    -------
    list of shape

    """
    origin_labels = (x.strip() for x in equation.split(','))
    shaped = collections.namedtuple('shaped', ['shape'])

    def fake_shape(ori_label, label, op):
        """
        1. ori_label is the original labels, not aligned by '....'
        2. if the '...' is evaluated to empty list, there is no '.' in label
        """
        assert len(op.shape) == len(label), (
            "length of shape and length of label must be the same, but received %d != %d"
            % (len(op.shape), len(label))
        )
        fakes = [s for i, (l, s) in enumerate(zip(label, op.shape)) if l != '.']
        fakes = list(map(abs, fakes))  # make -1 -> 1
        if '.' in ori_label:
            fakes.insert(ori_label.index('.'), 1)
        return shaped(fakes)

    out = list(map(fake_shape, origin_labels, labels, operands))
    return out


def rhs_inference(lhs):
    def is_free(key):
        return cnt.get(key) == 1 and key not in ['.', ',']

    cnt = collections.Counter(lhs)
    rhs = "..." if '...' in lhs else ""
    rhs = rhs + "".join(filter(is_free, sorted(cnt.elements())))
    return rhs


def gen_equation_for_opteinsum(lhs, rhs):
    """
    1. gen rhs if rhs is None
    2. '...' -> 'A'
    """

    def get_used_label(counter):
        used = set(counter.elements())
        for c in string.ascii_lowercase:
            if c not in used:
                return c
        raise ValueError(
            "You have used all `a` - `z`, there can't find a unused char for einsum optimization"
        )

    cnt = collections.Counter(lhs)
    broadcast_label = get_used_label(cnt)
    if rhs is None:
        rhs = rhs_inference(lhs)
    lhs = lhs.replace("...", broadcast_label)
    rhs = rhs.replace("...", broadcast_label)
    return lhs + "->" + rhs, broadcast_label


def einsum_v2(equation, *operands):
    """
    einsum v2 implementation.
    1. Implement C++ EinsumOp.
    2. V2 create the EinsumOp to calculate, so just a little verify work in python.
    3. V2 use opt_einsum.contract_path to optimize the multivariable einsum.
    """
    n_op = len(operands)
    lhs, rhs, labels, operands = preprocess(equation, *operands)

    if n_op <= 2:
        return gen_einsum_op(lhs + '->' + rhs, *operands)

    shapes = parse_fake_shape(lhs, operands, labels)
    opt_equation, broadcast_label = gen_equation_for_opteinsum(lhs, rhs)
    _, cons = opt_einsum.contract_path(opt_equation, *shapes, einsum_call=True)
    var_list = list(operands)
    for path in cons:
        (a, b), _, eq, *__ = path
        assert (
            a > b
        ), "Assume the first var_idx is smaller than the second_idx. opt_einsum can guarantee it."
        var_s = [var_list.pop(a), var_list.pop(b)]
        eq = eq.replace(broadcast_label, "...")
        var_list.append(gen_einsum_op(eq, *var_s))
    assert (
        len(var_list) == 1
    ), "There must be one elements in list, but received %d." % len(var_list)
    return var_list[0]


def gen_einsum_op(equation, *operands):
    """
    EinsumOp Python Interface:
    """

    if in_dynamic_or_pir_mode():
        return _C_ops.einsum(operands, equation)[0]
    else:
        assert len(operands) <= 2, "Only support two operands in EinsumOp."
        for inp in operands:
            check_variable_and_dtype(
                inp, 'dtype', ['float32', 'float64'], 'einsum'
            )
        check_type(equation, 'equation', str, 'einsum')
        helper = LayerHelper('einsum', **locals())
        out = helper.create_variable_for_type_inference(dtype=operands[0].dtype)
        attrs = {}
        attrs['equation'] = equation
        caches = [
            helper.create_variable_for_type_inference(dtype=operands[0].dtype)
            for i in range(len(operands))
        ]
        xshape = [
            helper.create_variable_for_type_inference(dtype=operands[0].dtype)
            for i in range(len(operands))
        ]
        helper.append_op(
            type='einsum',
            inputs={'Operands': operands},
            outputs={'Out': out, "InnerCache": caches, "XShape": xshape},
            attrs=attrs,
        )
        return out


def einsum(equation, *operands):
    r"""

    einsum(equation, *operands)

    The current version of this API should be used in dynamic graph only mode.

    Einsum offers a tensor operation API which allows using the Einstein summation
    convention or Einstain notation. It takes as input one or multiple tensors and
    produces as output one tensor.

    Einsum is able to perform a variety of tensor operations. Following lists a few:

        - for single operand
            - trace
            - diagonal
            - transpose
            - sum
        - for double operands
            - dot
            - outer
            - broadcasting and elementwise multiply
            - matrix multiply
            - batched matrix multiply
        - for many operads
            - broadcasting multiply
            - chained matrix multiply

    **The summation notation**

        - The tensor dimensions are labeled using uncased English letters. E.g., `ijk`
          relates to a three dimensional tensor whose dimensions are labeled i, j, and k.
        - The equation is `,` separated into terms, each being a distinct input's
          dimension label string.
        - Ellipsis `...` enables broadcasting by automatically converting the unlabeled
          dimensions into broadcasting dimensions.
        - Singular labels are called free labels, duplicate are dummy labels. Dummy labeled
          dimensions will be reduced and removed in the output.
        - Output labels can be explicitly specified on the right hand side of `->` or omitted.
            In the latter case, the output labels will be inferred from the input labels.
                - Inference of output labels
                    - Broadcasting label `...`, if present, is put on the leftmost position.
                    - Free labels are reordered alphabetically and put after `...`.
                - On explicit output labels
                    - If broadcasting is enabled, then `...` must be present.
                    - The output labels can be an empty, an indication to output as a scalar
                        the sum over the original output.
                    - Non-input labels are invalid.
                    - Duplicate labels are invalid.
                    - For any dummy label which is present for the output, it's promoted to
                        a free label.
                    - For any free label which is not present for the output, it's lowered to
                        a dummy label.

        - Examples
            - '...ij, ...jk', where i and k are free labels, j is dummy. The output label
              string is '...ik'
            - 'ij -> i', where i is a free label and j is a dummy label.
            - '...ij, ...jk -> ...ijk', where i, j and k are all free labels.
            - '...ij, ...jk -> ij', an invalid equation since `...` is not present for
              the output.

    **The summation rule**

    The summation procedure can be outlined as follows, although the actual steps taken
    may vary significantly due to implementation specific optimization.

        - Step 1: preparation for broadcasting, that is, transposing and unsqueezing
          the input operands to have each resulting dimension identically labeled across
          all the input operands.
        - Step 2: broadcasting multiply all the resulting operands from step 1.
        - Step 3: reducing dummy labeled dimensions.
        - Step 4: transposing the result tensor to match the output labels.

    **On trace and diagonal**

    The trace and diagonal are planned yet unimplemented features.

    Args:
        equation (`str`):
            The summation terms using the Einstein summation notation.
        operands (`list|Tensor`):
            The input tensors over which to compute the Einstein summation. The number of
            operands should equal the number of input terms in the equation.

    Returns:
        result (`Tensor`), the result tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(102)
            >>> x = paddle.rand([4])
            >>> y = paddle.rand([5])

            >>> # sum
            >>> print(paddle.einsum('i->', x))
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.81225157)

            >>> # dot
            >>> print(paddle.einsum('i,i->', x, x))
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.13530672)

            >>> # outer
            >>> print(paddle.einsum("i,j->ij", x, y))
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[0.26443148, 0.05962684, 0.25360870, 0.21900642, 0.56994802],
                    [0.20955276, 0.04725220, 0.20097610, 0.17355499, 0.45166403],
                    [0.35836059, 0.08080698, 0.34369346, 0.29680005, 0.77240014],
                    [0.00484230, 0.00109189, 0.00464411, 0.00401047, 0.01043695]])

            >>> A = paddle.rand([2, 3, 2])
            >>> B = paddle.rand([2, 2, 3])

            >>> # transpose
            >>> print(paddle.einsum('ijk->kji', A))
            Tensor(shape=[2, 3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[[0.50882483, 0.56067896],
                     [0.84598064, 0.36310029],
                     [0.55289471, 0.33273944]],
                    [[0.04836850, 0.73811269],
                     [0.29769155, 0.28137168],
                     [0.84636718, 0.67521429]]])

            >>> # batch matrix multiplication
            >>> print(paddle.einsum('ijk, ikl->ijl', A,B))
            Tensor(shape=[2, 3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[[0.36321065, 0.42009076, 0.40849245],
                     [0.74353045, 0.79189068, 0.81345987],
                     [0.90488225, 0.79786193, 0.93451476]],
                    [[0.12680580, 1.06945944, 0.79821426],
                     [0.07774551, 0.55068684, 0.44512171],
                     [0.08053084, 0.80583858, 0.56031936]]])

            >>> # Ellipsis transpose
            >>> print(paddle.einsum('...jk->...kj', A))
            Tensor(shape=[2, 2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[[0.50882483, 0.84598064, 0.55289471],
                     [0.04836850, 0.29769155, 0.84636718]],
                    [[0.56067896, 0.36310029, 0.33273944],
                     [0.73811269, 0.28137168, 0.67521429]]])

            >>> # Ellipsis batch matrix multiplication
            >>> print(paddle.einsum('...jk, ...kl->...jl', A,B))
            Tensor(shape=[2, 3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[[0.36321065, 0.42009076, 0.40849245],
                     [0.74353045, 0.79189068, 0.81345987],
                     [0.90488225, 0.79786193, 0.93451476]],
                    [[0.12680580, 1.06945944, 0.79821426],
                     [0.07774551, 0.55068684, 0.44512171],
                     [0.08053084, 0.80583858, 0.56031936]]])

    """
    import os

    if int(os.environ.get('FLAGS_new_einsum', "1")):
        return einsum_v2(equation, *operands)

    nop = len(operands)
    assert nop > 0, "At least one operand is expected."

    # Part the equation to left hand side and right hand side
    lhs, *rhs = equation.lower().replace(' ', '').split('->')
    assert len(rhs) < 2, "Invalid equation: multiple `->` were found."

    # Note, we distinguish between 'ij->' and 'ij' by setting rhs to '' and None
    rhs = rhs[0] if rhs else None

    # Parse labels for each operand and count the number of occurrences for each alphabet label
    nop_labels = parse_labels(lhs, operands)

    # Diagonalize the operands which have duplicate labels
    nop_labels, operands = list(zip(*map(diagonalize, nop_labels, operands)))

    # To handle broadcasting, we should first know how many dimensions are there
    # We need to use that number to generate output labels
    # e.g. 1 for ['ij', 'i.', '.k']
    n_bcast_dims = max(s.count('.') for s in nop_labels)

    # Build the data structures for planning. It's helpful to think of all the operands
    # broadcasting together from a global view. In this view, dimensions from multiple
    # operands are mapped to the same position if they are labeled uniquely. Broadcasting
    # dimensions are mapped to adjacent positions with the right bound fixed. Subject to
    # each operand, the map is injective but for all operands the map is on-to.
    # g_labels:
    #   The labels of the global view
    # g_view:
    #   Includes a list of maps from each operand's dimensions to the global view's dimensions
    #   which we refer to as ax or axes in the code to distinguish from operand's dims
    # g_shape:
    #   The shape of the global view. The size of each dimension is what the aligned dimensions
    #   should broadcast to
    # g_nout:
    #   Number of output axes
    # g_supports
    #   Booleans indicating each operand's non-trivial dimensions
    # g_count
    #   Counting how many non-trivial dimensions remain for each ax

    g_labels, g_view, g_nout, g_count = build_global_view(
        nop_labels, rhs, n_bcast_dims
    )
    g_shape, g_supports = build_global_shape(
        g_view, g_labels, [op.shape for op in operands]
    )

    # Now we're ready to build up an execution plan
    args = operands, g_view, g_shape, g_supports, g_count, n_bcast_dims
    plan = plan_einsum(*args)
    result = plan.execute()

    return result
