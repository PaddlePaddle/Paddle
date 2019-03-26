

class P2PRoleMakers(object):
    def __init__(self):
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD
        self.MPI = MPI

    def get_endpoints(self, port_start):
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        import socket
        local_ip = socket.gethostbyname(socket.gethostname())
        hostname = socket.gethostname()
        all_ips = self.comm.allgather(local_ip)
        all_ports = [str(port_start + rank) for ]
        return all_ports
