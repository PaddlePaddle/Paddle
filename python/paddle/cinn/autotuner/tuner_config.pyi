from typing import *  # noqa: F403
from typing_extensions import *  # type: ignore # noqa: F403
from paddle._typing import *  # noqa: F403
from paddle.cinn.autotuner import tuner_config


def _tuner_add_config_helper(
          candidate: List[int], bucket_info: tuner_config.BucketInfo) -> None:
     ...

def _env_set_tile_config_policy(policy: str) -> None:
     ...

def _env_get_tile_config_policy() -> str:
     ...