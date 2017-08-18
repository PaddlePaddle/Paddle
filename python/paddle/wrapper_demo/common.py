import logging
import paddle.v2.framework.core as core

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)

g_scope = core.Scope()
g_device = core.CPUPlace()
