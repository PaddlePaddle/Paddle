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

import itertools
import re

from ..fluid.layers import reshape, transpose
from .linalg import matmul
from .manipulation import squeeze, unsqueeze
from .math import multiply
from .math import sum as paddle_sum

from paddle.common_ops_import import dygraph_only

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
        assert c.isalpha(), (
            f"Invalid equation: {c} is not a valid label, which should be letters."
        )

    assert labelstr.replace('...', '', 1).find('.') == -1, (
        f"Invalid equation: `.` is found outside of an ellipsis."
    )

    # Check shape. Note, in Paddle a tensor rank is always nonzero
    ndims = len(operand.shape)
    assert ndims > 0

    full_labelstr = labelstr.replace('...', '.' * (ndims - len(labelstr) + 3))

    assert len(full_labelstr) == ndims, (
        f"Invalid equation: the label string '{labelstr}' misses dimensions.")

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
        f"but found {len(nop_labels)} segments in the label equation.")

    return list(map(parse_op_labels, nop_labels, operands))


def validate_rhs(rhs, input_labels, n_bcast_dims):
    '''
    Check whether the equation's right hand side is valid 
    '''
    # Sanity check.
    if n_bcast_dims > 0:
        assert '...' in rhs, (
            f"Invalid equation: missing ellipsis in output labels.")

    rhs = rhs.replace('...', '')
    rhs_set = set(rhs)

    # Hidden assumption: availble labels don't include '.'
    assert '.' not in input_labels

    # Verify that output labels all come from the set of input labels
    non_input_labels = rhs_set.difference(input_labels)
    assert not non_input_labels, (
        f"Invalid equation: "
        f"output label {sorted(non_input_labels)} not used by any input.")
    # Verify that output labels are not duplicate
    assert len(rhs) == len(rhs_set), (
        f"Invalid equation: duplicate output labels are found.")


# def bcastable_test(args, f=None):
#     '''
#     Tests if the two operands can perform a broadcast operation on the given ranges of dimensions. 
#     We follow the Numpy broadcasting convention which states that, by lining up the shape arrays
#     starting from the right most dimension, all the aligned dimensions either have equal sizes or
#     one of them is sized one.
#     Parameters
#     ----------
#     args:
#         *args unpacks into operand one's axes range, shape, operand two's axes range, shape
#     f: 
#         if available, is used as a callback for postprocessing the aligned operand dimensions.
#     '''
#     xran, xshape, yran, yshape = args
#
#     xran_inv, yran_inv = xran[::-1], yran[::-1]
#
#     for xi, yi in zip(xran_inv, yran_inv):
#         xs, ys = xshape[xi], yshape[yi]
#         cond = xs == ys or xs == 1 or ys == 1
#         if not cond:
#             return False
#
#     if not f:
#         return True
#
#     # Apply the callback to each aligned dimension pair
#     for xi, yi in zip(xran_inv, yran_inv):
#         f(xi, yi)


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
    out_labels. -1 is filled if there's no matching intput dimension for a specific label.

    Examples
    --------
    in_labels = 'ij..', out_labels = '..ji'
    inv_map = [2, 3, 1, 0]
    in_labels = 'ij..', out_labels = '..kji'
    inv_map = [2, 3, -1, 1, 0]
    '''
    # print(f"in labels: '{in_labels}'     out labels: '{out_labels}'")

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
                    range(start, end)[::-1], range(s.start(), s.end())[::-1]):
                inv_map[ax] = dim

    # Now work on non-broadcast dimensions 
    if r:
        it = itertools.chain(range(start), range(end, len(out_labels)))
    else:
        it = iter(range(len(out_labels)))

    for i in it:
        inv_map[i] = in_labels.find(out_labels[i])

    return inv_map


# def part_labels(nop_labels, rhs, n_bcast_dims):
#     '''
#     Part labels in two strings, the output label string and the combined label string. 
#     Infer output labels in case no explicit right hand side is given. In this case
#     the output label string is formed by labels that count only once, in alphabetical order.
#     Returns
#     -------
#     output:
#         output label string
#     combine:
#         combine label string
#     count:
#         combine labels' count
#     '''
#     # Put all labels in alphabetical order
#     concat = sorted(''.join(nop_labels).replace('.', ''))
#     labels, count = [], []
#     for a, b in zip(['.'] + concat, concat):
#         if a != b:
#             labels.append(b)
#             count.append(1)
#         else:
#             count[-1] += 1
#     if rhs != None:
#         validate_rhs(rhs, labels, n_bcast_dims)
#         output = rhs.replace('...', '.' * n_bcast_dims)
#     else:
#         output = '.' * n_bcast_dims + ''.join(l for l, c in zip(labels, count) if c == 1)
#
#     for i in range(len(count))[::-1]:  # Ouch ... it hurts to get confused with [-1:]
#         if labels[i] in output:
#             labels.pop(i)
#             count.pop(i)
#
#     combine = ''.join(labels)
#
#     return output, combine, count


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
        The maxium number of broadcast dimensions
    
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

    if rhs != None:
        validate_rhs(rhs, labels, n_bcast_dims)
        g_labels_out = rhs.replace('...', '.' * n_bcast_dims)
    else:
        g_labels_out = '.' * n_bcast_dims + ''.join(
            l for l, c in zip(labels, count) if c == 1)

    for i in range(len(count))[::-1]:
        if labels[i] in g_labels_out:
            labels.pop(i)
            count.pop(i)

    g_labels_sum = ''.join(labels)
    g_labels = g_labels_out + g_labels_sum
    g_view = list(map(lambda i: build_view(i, g_labels), nop_labels))
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

    nonbcastable_axes = [ax for ax, sizes in enumerate(g_shape) if len(sizes) > 1]

    assert not nonbcastable_axes, (
        f"Invalid operands: label {g_labels[nonbcastable_axes[0]]} "
        f"corresponds to non-broadcastable dimensions.")

    g_shape = [sizes.pop() if len(sizes) > 0 else 1 for sizes in g_shape]

    g_masks = [[s > 1 for s in view_shape] for view_shape in view_shapes]

    return g_shape, g_masks


def dim_strides(shape):
    '''
    Returns the dimension strides for a tensor shape
    '''
    strides = []
    stride = 1
    for size in shape[::-1]:
        strides.append(stride)
        stride = stride * size
    return strides


def create_view(operand, *view_def):
    '''
    Create and materialize a view.
    
    Parameters
    ----------
    operand:
        the base tensor operand
    view_def: 
        include two lists which define the view's dimension sizes and strides
    '''
    assert False, f'Diagonal and trace not implemented yet.'
    view_shape, view_strides = view_def
    return operand.create_view(view_shape, view_strides)


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
    which represents the diagonal elements. That requires the duplicate labeled
    dimensions equal sized. The order of dimensions is kept unchanged up to 
    the left-most appearance of each label.
    
    Examples
    -------- 
    'ijj...i' would be merged into 'ij...'
    '''
    if not has_duplicated_labels(labels):
        return labels, operand

    strides = dim_strides(operand.shape)
    shape = operand.shape
    new_labels = []
    new_shape = []
    new_strides = []

    for ax, l in enumerate(labels):
        if l == '.' or l not in new_labels:
            # not duplicate
            new_labels.append(l)
            new_strides.append(strides[ax])
            new_shape.append(shape[ax])
        else:
            # duplicate label
            diag_ax = new_labels.index(l)
            new_strides[diag_ax] += strides[ax]

    # call framework API to build a new tensor
    new_op = create_view(operand, new_shape, new_strides)
    return new_labels, new_op


def prod(iter, default=1):
    if len(iter):
        res = 1
        for s in iter:
            res *= s
        return res
    return default


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
    step = f, varnames, varnames[1]
    plan.add_step(step)


def plan_matmul(plan, g_view, op1, op2, g_op_masks, g_shape, I, J1, J2, K):
    '''
    plan matmul
    '''
    # Transpose and re-shape op1 and op2 in I, J1, K and I, J2, K
    # Then apply matmul(x, y, transpose_x=False, tranpose_y=True)
    var1, var2 = f'op{op1}', f'op{op2}'

    op1_view, op2_view = [g_view[op] for op in (op1, op2)]

    # Note, I may index into -1
    I1_dims = [op1_view[ax] for ax in I if op1_view[ax] >= 0]
    I2_dims = [op2_view[ax] for ax in I if op2_view[ax] >= 0]
    J1_dims = [op1_view[ax] for ax in J1]
    J2_dims = [op2_view[ax] for ax in J2]
    K1_dims = [op1_view[ax] for ax in K]
    K2_dims = [op2_view[ax] for ax in K]

    op1_mask, op2_mask = [g_op_masks[op] for op in (op1, op2)]
    op1_vshape = [s if m else 1 for s, m in zip(g_shape, op1_mask)]
    op2_vshape = [s if m else 1 for s, m in zip(g_shape, op2_mask)]

    I1_shape, J1_shape, K1_shape = [[op1_vshape[ax] for ax in axes]
                                    for axes in (I, J1, K)]
    I2_shape, J2_shape, K2_shape = [[op2_vshape[ax] for ax in axes]
                                    for axes in (I, J2, K)]

    K1_size, J1_size, J2_size = prod(K1_shape), prod(J1_shape), prod(J2_shape)

    perm1 = I1_dims + J1_dims + K1_dims
    perm2 = I2_dims + J2_dims + K2_dims

    if any(i != dim for i, dim in enumerate(perm1)):
        # print(f'perm1: {perm1}')
        step = transpose, [var1], var1, perm1
        plan.add_step(step)

    if any(i != dim for i, dim in enumerate(perm2)):
        # print(f'perm2: {perm2}')
        step = transpose, [var2], var2, perm2
        plan.add_step(step)

    # In case of no K... dimensions, do a broadcast
    if not K:
        # unsqueeze operands include J1...J2... dimensions
        if J2:
            fill_start = len(I2_dims) + len(J1)
            fill_end = fill_start + len(J2)
            fill = list(range(fill_start, fill_end))
            step = unsqueeze, [var1], var1, fill
            plan.add_step(step)
        if J1:
            fill_start = len(I2_dims)
            fill_end = fill_start + len(J1)
            fill = list(range(fill_start, fill_end))
            step = unsqueeze, [var2], var2, fill
            plan.add_step(step)
        # make broadcast
        step = multiply, [var1, var2], var2
        plan.add_step(step)
    # K... are there, let's reason about I... and J...
    # In case I... and J... are empty, do the vector-vector version of matmul
    elif not I and not J1 and not J2:
        # merge K dimensions
        if len(K) > 1:
            for var in var1, var2:
                step = reshape, [var], var, [K1_size]
                plan.add_step(step)
        # Build vector-vector matmul
        step = matmul, [var1, var2], var2
        plan.add_step(step)
    # General case, there are K... and some I... and J..., the actual operation will be 
    # matrix-vector or matrix-matrix multiplies, depending on the operands' shapes.
    else:
        # Merge J dims and K dims by reshaping
        merged_shape1 = I1_shape + [J1_size] + [K1_size]
        merged_shape2 = I2_shape + [J2_size] + [K1_size]

        step = reshape, [var1], var1, merged_shape1
        plan.add_step(step)
        step = reshape, [var2], var2, merged_shape2
        plan.add_step(step)

        # Matmul
        step = matmul, [var1, var2], var2, False, True
        plan.add_step(step)

    # The result shape is in I..., J1, J2. Let's reshape back to known dimensions
    # Note, this is static deduction, not by reading the tensor shape at runtime
    result_shape = [1] * len(I)
    for i, ax in enumerate(I):
        result_shape[i] = max(op1_vshape[ax], op2_vshape[ax])
    if J1:
        result_shape += J1_shape
    if J2:
        result_shape += J2_shape

    # Need a scalar dimension somehow
    if result_shape:
        step = reshape, [var2], var2, result_shape
        plan.add_step(step)

    # Wrap up, updating auxiliary data
    # Updating g_mask for I and J axes
    for i, ax in enumerate(I + J1 + J2):
        op2_mask[ax] = (result_shape[i] > 1)

    for ax in K:
        op2_mask[ax] = False

    for ax in range(len(op2_view)):
        op2_view[ax] = -1
    dim = 0
    for ax in I + J1 + J2:
        op2_view[ax], dim = dim, dim + 1


def plan_summation(plan, g_view, op1, op2, g_op_masks, g_shape, g_count,
                   n_bcast):
    '''
    Plan various kinds of summation
    '''
    op1_view, op2_view = g_view[op1], g_view[op2]
    op1_mask, op2_mask = g_op_masks[op1], g_op_masks[op2]

    ndim = len(op1_view)
    nout = ndim - len(g_count)

    count = [0] * nout + g_count

    I, K, J1, J2 = list(range(n_bcast)), [], [], []

    for ax, dim1, dim2 in zip(
            range(n_bcast, ndim), op1_view[n_bcast:], op2_view[n_bcast:]):

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
    plan_matmul(plan, g_view, op1, op2, g_op_masks, g_shape, I, J1, J2, K)


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
        # Re-arrange the dimesions according to the global layout
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


def plan_einsum(operands, g_view, g_shape, g_op_masks, g_count, n_bcast):
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

    # Down count axis >= nout and degenerate dimensions (masked is not set)
    for view, mask in zip(g_view, g_op_masks):
        down_count = [
            1 if (dim > -1 and not masked) else 0
            for dim, masked in zip(view[nout:], mask[nout:])
        ]
        for i, d in enumerate(down_count):
            g_count[i] -= d

    # Reduce any dimension for which g_mask is set and g_count == 1
    for i, view, mask in zip(range(nop), g_view, g_op_masks):
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
        # At this point the non-trivial broadcast dimensinos in K are already reduced
        # and removed. That means all K dimensions are aligned and their sizes are not 1.
        # We then inspect the layout of I,J,K plus the above observation to make
        # specializatoin decisions.  The current strategy is set as follows:
        #  (1) if I... J... K... are all empty, it's multiplying a scalar
        #  (2) if K... are empty, better use a broadcast
        #  (3) if I... J... empty and K... not empty, a vector-vector multiply (or a dot)
        #  (4) Elsewise, either I... or J... not empty, and K... not empty, use a general matmul

        # Resolve the summation kind: dot, matmul or *
        if not any(g_op_masks[i - 1]):
            # op1 is a scalar
            plan_scalar_prod(plan, i - 1, i)
        else:
            plan_summation(plan, g_view, i - 1, i, g_op_masks, g_shape, g_count,
                           n_bcast)

    # for ax, dim in enumerate(g_view[nop-1][:nout]):
    #     assert dim == ax
    assert all(not masked for masked in g_op_masks[nop - 1][nout:])

    view = g_view[-1]
    if any(ax != dim for ax, dim in enumerate(view[:nout])):
        perm = [dim for dim in view if dim >= 0]
        varname = f'op{nop-1}'
        step = transpose, [varname], varname, perm
        plan.add_step(step)
        dim = 0
        for ax, d in enumerate(view):
            if d != -1:
                view[ax], dim = dim, dim + 1

    squeeze_dims = [dim for dim in view[nout:] if dim != -1]
    if squeeze_dims:
        # plan_reduce(plan, nop-1, reduce_dims, keepdim=False)
        varname = f'op{nop-1}'
        step = squeeze, [varname], varname, squeeze_dims
        plan.add_step(step)

    return plan


@dygraph_only
def einsum(equation, *operands):
    r"""
    einsum(equation, *operands)

    This function performs the Einstein summation on input operands.

    Einsum provides a unified interface to a variety of operations including element-wise
    multiplication, vector-vector multiplication, vector multiplication, 
    matrix multiplication, batch matrix multiplication, transpose, reduce_sum, trace, 
    diagonal, etc. 

    Args:
        equation (`str`):
            The equation uses uncased letters to indicate dimensions. These letters are called 
            dimension labels. The dimension labels are comma separated into groups the number of which
            equals the number of the input operands. The equation uses `->` to indicate 
            explicitly the output dimensions which otherwise would be collapsed as the result of summation.
            In case the explicit output is not given, Einsum will deduce the output dimensions automatically.
            Dimensions with the same label should be broadcastable. The equation uses ellipsis ('...') to 
            specify broadcast dimensions.
        operands (`Tensor`):
            The operands to compute the Einstein sum of. The number of operands should be the same as the
            the operands described in input equation.
    
    Returns:
        `Tensor`: The result of Einstein sum product.
    
    Example:
    .. code-block::
        import paddle
        import paddlenlp
        import numpy as np
        np.random.seed(102)
        x = paddle.to_tensor(np.random.rand(4))
        y = paddle.to_tensor(np.random.rand(5))
        # sum
        print(paddlenlp.ops.einsum('i->', x))
        # Tensor(shape=[], dtype=float64, place=CUDAPlace(0), stop_gradient=True, 2.30369050)
        # dot
        print(paddlenlp.ops.einsum('i,i->', x, x))
        # Tensor(shape=[], dtype=float64, place=CUDAPlace(0), stop_gradient=True, 1.43773247)
        # outer
        print(paddlenlp.ops.einsum("i,j->ij", x, y)),
        # Tensor(shape=[4, 5], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #         [[0.34590188, 0.48353496, 0.09996135, 0.18656330, 0.21392910],
        #         [0.39122025, 0.54688535, 0.11305780, 0.21100591, 0.24195704],
        #         [0.17320613, 0.24212422, 0.05005442, 0.09341929, 0.10712238],
        #         [0.42290818, 0.59118179, 0.12221522, 0.22809690, 0.26155500]])
        A = paddle.to_tensor(np.random.rand(2, 3, 2))
        B = paddle.to_tensor(np.random.rand(2, 2, 3))
        # transpose
        print(paddlenlp.ops.einsum('ijk->kji', A))
        #  Tensor(shape=[2, 3, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #        [[[0.49174730, 0.33344683],
        #          [0.89440989, 0.26162022],
        #          [0.36116209, 0.12241719]],
        #         [[0.49019824, 0.51895050],
        #          [0.18241053, 0.13092809],
        #          [0.81059146, 0.55165734]]])
        # batch matrix multiplication
        print(paddlenlp.ops.einsum('ijk, ikl->ijl', A,B))
        # Tensor(shape=[2, 3, 3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #     [[[0.13654339, 0.39331432, 0.65059661],
        #      [0.07171420, 0.57518653, 0.77629221],
        #      [0.21250688, 0.37793541, 0.73643411]],
        #     [[0.56925339, 0.65859030, 0.57509818],
        #      [0.30368265, 0.25778348, 0.21630400],
        #      [0.39587265, 0.58031243, 0.51824755]]])
        # Ellipsis transpose
        print(paddlenlp.ops.einsum('...jk->...kj', A))
        # Tensor(shape=[2, 2, 3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #     [[[0.49174730, 0.89440989, 0.36116209],
        #         [0.49019824, 0.18241053, 0.81059146]],
        #         [[0.33344683, 0.26162022, 0.12241719],
        #         [0.51895050, 0.13092809, 0.55165734]]])
        # Ellipsis batch matrix multiplication
        print(paddlenlp.ops.einsum('...jk, ...kl->...jl', A,B))
        # Tensor(shape=[2, 3, 3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        # [[[0.13654339, 0.39331432, 0.65059661],
        #     [0.07171420, 0.57518653, 0.77629221],
        #     [0.21250688, 0.37793541, 0.73643411]],
        #     [[0.56925339, 0.65859030, 0.57509818],
        #     [0.30368265, 0.25778348, 0.21630400],
        #     [0.39587265, 0.58031243, 0.51824755]]])
    """

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
    n_bcast_dims = max(map(lambda s: s.count('.'), nop_labels))

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
    # g_op_masks
    #   A list of masks that specify each operand's non-trivial dimensions
    # g_count
    #   Counting how many non-trivial dimensions remain for each ax

    g_labels, g_view, g_nout, g_count = build_global_view(nop_labels, rhs,
                                                          n_bcast_dims)
    g_shape, g_op_masks = build_global_shape(g_view, g_labels,
                                             [op.shape for op in operands])

    # Now we're ready to build up an execution plan
    args = operands, g_view, g_shape, g_op_masks, g_count, n_bcast_dims
    plan = plan_einsum(*args)
    result = plan.execute()

    return result
