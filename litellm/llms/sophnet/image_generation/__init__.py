from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import SophnetImageGenerationConfig

__all__ = ["SophnetImageGenerationConfig"]


def get_sophnet_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return SophnetImageGenerationConfig()
