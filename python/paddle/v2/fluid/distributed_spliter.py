def hash_name(varlist, pserver_endpoints):
    """
    hash variable names to several endpoints.

    :param varlist: a list of Variables
    :return: a map of pserver endpoint -> varname
    """

    def _hash_block(block_str, total):
        return hash(block_str) % total

    eplist = []
    for var in varlist:
        server_id = _hash_block(var.name(), len(pserver_endpoints))
        server_for_param = pserver_endpoints[server_id]
        eplist.append(server_for_param)
    return eplist


def round_robin(varlist, pserver_endpoints):
    """
    distribute variables to several endpoints.
    """
    assert (len(varlist) > len(pserver_endpoints))

    eplist = []
    pserver_idx = 0
    for var in varlist:
        server_for_param = pserver_endpoints[pserver_idx]
        eplist.append(server_for_param)

        pserver_idx += 1
        if pserver_idx >= len(pserver_endpoints):
            pserver_idx = 0
    return eplist
