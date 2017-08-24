import logging
import paddle.v2.framework.core as core

g_scope = core.Scope()

logger = logging.getLogger("paddle python")
logger.setLevel(logging.INFO)
