import paddle.v2.framework.core as core


def unique_name(prefix):
    uid = core.unique_integer()  # unique during whole process.
    return "{0}{1}".format(prefix, uid)
