def hash_name(varblocks, pserver_endpoints):
    """
    :param varblocks: a list of VarBlock string indicating 
                      sub blocks of variables
    :return: a map of pserver endpoint -> varblock_str
    """

    def _hash_block(block_str, total):
        return hash(block_str) % total

    ep2block = dict()
    for varblock_str in varblocks:
        if param.trainable is True and grad is not None:
            server_id = _hash_block(varblock_str, len(pserver_endpoints))
            server_for_param = pserver_endpoints[server_id]
            if not ep2block.has_key(server_for_param):
                ep2block[server_for_param] = []
            ep2block[server_for_param].append(varblock_str)

    return ep2block


def round_robin(varblocks, pserver_endpoints):
    assert (len(varblocks) > len(pserver_endpoints))

    ep2block = dict()
    pserver_idx = 0
    for varblock_str in varblocks:
        if param.trainable is True:
            server_for_param = pserver_endpoints[pserver_idx]
            if not ep2block.has_key(server_for_param):
                ep2block[server_for_param] = []
            ep2block[server_for_param].append(varblock_str)

            pserver_idx += 1
            if pserver_idx >= len(pserver_endpoints):
                pserver_idx = 0
    return ep2block
