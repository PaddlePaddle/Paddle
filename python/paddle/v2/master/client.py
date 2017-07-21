import ctypes
import os

path = os.path.join(os.path.dirname(__file__), "libpaddle_master.so")
lib = ctypes.cdll.LoadLibrary(path)


class client(object):
    """
    client is a client to the master server.
    """

    def __init__(self, etcd_endpoints, timeout, buf_size):
        self.c = lib.paddle_new_etcd_master_client(etcd_endpoints, timeout,
                                                   buf_size)

    def close(self):
        lib.paddle_release_master_client(self.c)
        self.c = None

    def set_dataset(self, paths):
        holder_type = ctypes.c_char_p * len(paths)
        holder = holder_type()
        print paths
        for idx, path in enumerate(paths):
            c_ptr = ctypes.c_char_p(path)
            holder[idx] = c_ptr
        lib.paddle_set_dataset(self.c, holder, len(paths))

    # return format: (record, errno)
    # errno =  0: ok
    #       <  0: error
    def next_record(self):
        p = ctypes.c_char_p()
        ret = ctypes.pointer(p)
        size = lib.paddle_next_record(self.c, ret)
        if size < 0:
            # Error
            return None, size

        if size == 0:
            # Empty record
            return "", 0

        record = ret.contents.value[:size]
        # Memory created from C should be freed.
        lib.mem_free(ret.contents)
        return record, 0
