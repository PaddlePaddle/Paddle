import os
import sys
import logging
import functools

from paddle.fluid.dygraph.parallel import ParallelEnv


@functools.lru_cache()
def setup_logger(output=None, name="hapi", log_level=logging.INFO):
    """
    Initialize logger of hapi and set its verbosity level to "INFO".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger. Default: 'hapi'.
        log_level (enum): log level. eg.'INFO', 'DEBUG', 'ERROR'. Default: logging.INFO.
    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.propagate = False

    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=format_str, level=log_level)

    # stdout logging: only local rank==0
    local_rank = ParallelEnv().local_rank
    if local_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)

        ch.setFormatter(logging.Formatter(format_str))
        logger.addHandler(ch)

    # file logging if output is not None: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if local_rank > 0:
            filename = filename + ".rank{}".format(local_rank)

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        fh = logging.StreamHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(format_str))
        logger.addHandler(fh)

    return logger
