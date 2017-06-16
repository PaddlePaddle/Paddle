import ctypes
import os

path = os.path.join(os.path.dirname(__file__), "libpaddle_master.so")
lib = ctypes.cdll.LoadLibrary(path)


class client(object):
    """
    client is a client to the master server.
    """

    def __init__(self, addr, buf_size):
        self.c = lib.paddle_new_master_client(addr, buf_size)

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

    def next_record(self):
        p = ctypes.c_char_p()
        ret = ctypes.pointer(p)
        size = lib.paddle_next_record(self.c, ret)
        if size == 0:
            # Empty record
            return ""
        record = ret.contents.value[:size]
        # Memory created from C should be freed.
        lib.mem_free(ret.contents)
        return record
