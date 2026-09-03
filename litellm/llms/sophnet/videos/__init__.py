from litellm.llms.base_llm.videos.transformation import BaseVideoConfig

from .transformation import SophnetVideoConfig

__all__ = ["SophnetVideoConfig"]


def get_sophnet_video_config() -> BaseVideoConfig:
    return SophnetVideoConfig()
