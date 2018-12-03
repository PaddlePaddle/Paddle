from mpi4py import MPI

class MPIHelper(object):
    def __init__(self):
        self.comm = MPI.COMM_WORLD

    def get_rank(self):
        return self.comm.Get_rank()

    def get_size(self):
        return self.comm.Get_size()

    def get_ip(self):
        import socket
        local_ip = socket.gethostbyname(socket.gethostname())
        return local_ip

    def get_hostname(self):
        import socket
        return socket.gethostname()
