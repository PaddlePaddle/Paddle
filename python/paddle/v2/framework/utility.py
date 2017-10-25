import paddle.v2.framework.core as core
import paddle.v2.framework.proto.framework_pb2 as framework_pb2
import paddle.v2.framework.framework as framework


def add_feed_components(block, feeded_var_names, feed_var_name):
    feed_var = block.create_var(
        name=feed_var_name,
        type=core.VarDesc.VarType.FEED_MINIBATCH,
        persistable=True)
    feed_indexes = dict()
    for i, name in enumerate(feeded_var_names):
        out = block.var(name)
        block.prepend_op(
            type="feed",
            inputs={"X": [feed_var]},
            outputs={"Out": [out]},
            attrs={"col": i})
        feed_indexes[name] = i
    return feed_indexes


def add_fetch_components(block, fetched_vars, fetch_var_name):
    fetch_var = block.create_var(
        name=fetch_var_name,
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True)
    fetch_indexes = dict()
    for i, var in enumerate(fetched_vars):
        block.append_op(
            type="fetch",
            inputs={"X": [var]},
            outputs={"Out": [fetch_var]},
            attrs={"col": i})
        fetch_indexes[var.name] = i
    return fetch_indexes
