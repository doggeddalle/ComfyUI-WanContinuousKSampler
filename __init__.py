"""ComfyUI entrypoint for the Wan sampling toolkit."""

from comfy_api.latest import ComfyExtension
from typing_extensions import override

from .nodes import (
    Wan22CombinedKSampler,
    Wan22CombinedSamplerExtension,
    Wan22LivePreview,
)
from .scail_identity import WanSCAILIdentityControl
from .scail_nodes import WanSCAILAutoExtendSampler, WanSCAILMediaPrep


WEB_DIRECTORY = "./web"


class WanSamplerToolkitExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [
            Wan22CombinedKSampler,
            Wan22LivePreview,
            WanSCAILMediaPrep,
            WanSCAILIdentityControl,
            WanSCAILAutoExtendSampler,
        ]


async def comfy_entrypoint() -> WanSamplerToolkitExtension:
    return WanSamplerToolkitExtension()


__all__ = [
    "Wan22CombinedKSampler",
    "Wan22CombinedSamplerExtension",
    "Wan22LivePreview",
    "WanSCAILAutoExtendSampler",
    "WanSCAILIdentityControl",
    "WanSCAILMediaPrep",
    "WanSamplerToolkitExtension",
    "WEB_DIRECTORY",
    "comfy_entrypoint",
]
