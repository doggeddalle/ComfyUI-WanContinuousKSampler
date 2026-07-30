"""Build the checked-in Wan 2.2 LightX2V I2V example from testwan.json."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "testwan.json"
OUTPUT = ROOT / "example_workflows" / "Wan22_Combined_I2V_LightX2V_4step.json"

HIGH_LORA = (
    r"wan\wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors"
)
LOW_LORA = (
    r"wan\wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
)

LORA_HIGH_ID = 679
LORA_LOW_ID = 680
SAMPLER_ID = 681
PREVIEW_ID = 682
LIVE_PREVIEW_TYPE = "WAN22_LIVE_PREVIEW"

# PreviewAny is retained as a diagnostics sink. Everything else here belonged to
# the old split-sampler controls, the disconnected int8 alternative, or the
# workflow-wide GPU purge.
REMOVE_NODE_IDS = {522, 523, 535, 541, 551, 655, 656, 673, 674, 675}

# ComfyUI serializes the seed's control-after-generate mode immediately after
# the seed itself. The remaining values follow define_schema() in nodes.py.
SAMPLER_WIDGETS = [
    0,
    "fixed",
    "I2V",
    "LightX2V 4-step distilled",
    4,
    1.0,
    1.0,
    5.0,
    "euler",
    "simple",
]

LIVE_PREVIEW_WIDGETS = [16.0, 24, 1, 70, 512]


def _lora_node(node_id: int, title: str, filename: str, pos: list[int]) -> dict:
    return {
        "id": node_id,
        "type": "LoraLoaderModelOnly",
        "pos": pos,
        "size": [420, 100],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [{"name": "model", "type": "MODEL", "link": None}],
        "outputs": [{"name": "MODEL", "type": "MODEL", "links": []}],
        "title": title,
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.3.46",
            "Node name for S&R": "LoraLoaderModelOnly",
        },
        "widgets_values": [filename, 1.0],
        "color": "#223",
        "bgcolor": "#335",
    }


def _sampler_node() -> dict:
    return {
        "id": SAMPLER_ID,
        "type": "Wan22CombinedKSampler",
        "pos": [-1060, 2800],
        "size": [390, 400],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "high_noise_model", "type": "MODEL", "link": None},
            {"name": "low_noise_model", "type": "MODEL", "link": None},
            {"name": "positive", "type": "CONDITIONING", "link": None},
            {"name": "negative", "type": "CONDITIONING", "link": None},
            {"name": "latent_image", "type": "LATENT", "link": None},
            {"name": "live_preview", "type": LIVE_PREVIEW_TYPE, "link": None},
        ],
        "outputs": [
            {"name": "LATENT", "type": "LATENT", "links": []},
            {"name": "diagnostics", "type": "STRING", "links": []},
        ],
        "properties": {
            "cnr_id": "comfyui-wan22-combined-ksampler",
            "ver": "0.2.0",
            "Node name for S&R": "Wan22CombinedKSampler",
        },
        "widgets_values": list(SAMPLER_WIDGETS),
    }


def _live_preview_node() -> dict:
    return {
        "id": PREVIEW_ID,
        "type": "Wan22LivePreview",
        "pos": [-1060, 3240],
        "size": [390, 360],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [],
        "outputs": [
            {
                "name": "live_preview",
                "type": LIVE_PREVIEW_TYPE,
                "links": [],
            }
        ],
        "properties": {
            "cnr_id": "comfyui-wan22-combined-ksampler",
            "ver": "0.2.0",
            "Node name for S&R": "Wan22LivePreview",
        },
        "widgets_values": list(LIVE_PREVIEW_WIDGETS),
    }


def _new_links() -> list[list]:
    return [
        [1641, 61, 0, LORA_HIGH_ID, 0, "MODEL"],
        [1642, 62, 0, LORA_LOW_ID, 0, "MODEL"],
        [1643, LORA_HIGH_ID, 0, 67, 0, "MODEL"],
        [1644, LORA_LOW_ID, 0, 68, 0, "MODEL"],
        [1645, 67, 0, SAMPLER_ID, 0, "MODEL"],
        [1646, 68, 0, SAMPLER_ID, 1, "MODEL"],
        [1647, 50, 0, SAMPLER_ID, 2, "CONDITIONING"],
        [1648, 50, 1, SAMPLER_ID, 3, "CONDITIONING"],
        [1649, 50, 2, SAMPLER_ID, 4, "LATENT"],
        [1650, SAMPLER_ID, 0, 8, 0, "LATENT"],
        [1651, SAMPLER_ID, 1, 542, 0, "STRING"],
        [1652, PREVIEW_ID, 0, SAMPLER_ID, 5, LIVE_PREVIEW_TYPE],
    ]


def _rebuild_connection_metadata(workflow: dict) -> None:
    nodes = {node["id"]: node for node in workflow["nodes"]}
    for node in nodes.values():
        for input_spec in node.get("inputs", []):
            input_spec["link"] = None
        for output_spec in node.get("outputs", []):
            output_spec["links"] = []

    seen_link_ids: set[int] = set()
    for link in workflow["links"]:
        link_id, source_id, source_slot, target_id, target_slot, link_type = link
        if link_id in seen_link_ids:
            raise ValueError(f"Duplicate link id: {link_id}")
        seen_link_ids.add(link_id)
        if source_id not in nodes or target_id not in nodes:
            raise ValueError(f"Link {link_id} refers to a missing node")

        source_outputs = nodes[source_id].get("outputs", [])
        target_inputs = nodes[target_id].get("inputs", [])
        if not 0 <= source_slot < len(source_outputs):
            raise ValueError(f"Link {link_id} has an invalid source slot")
        if not 0 <= target_slot < len(target_inputs):
            raise ValueError(f"Link {link_id} has an invalid target slot")

        source = source_outputs[source_slot]
        target = target_inputs[target_slot]
        if source["type"] != link_type:
            raise ValueError(f"Link {link_id} disagrees with its source type")
        accepted_types = str(target["type"]).split(",")
        if target["type"] != "*" and link_type not in accepted_types:
            raise ValueError(f"Link {link_id} disagrees with its target type")
        if target.get("link") is not None:
            raise ValueError(f"Input {target_id}:{target_slot} has multiple links")

        target["link"] = link_id
        source["links"].append(link_id)


def _validate(workflow: dict) -> None:
    nodes = {node["id"]: node for node in workflow["nodes"]}
    if len(nodes) != len(workflow["nodes"]):
        raise ValueError("Duplicate node id")
    if REMOVE_NODE_IDS & nodes.keys():
        raise ValueError("An obsolete node survived the migration")

    expected_ids = {
        6,
        7,
        8,
        38,
        39,
        50,
        61,
        62,
        67,
        68,
        82,
        92,
        542,
        641,
        642,
        676,
        677,
        678,
        LORA_HIGH_ID,
        LORA_LOW_ID,
        SAMPLER_ID,
        PREVIEW_ID,
    }
    if nodes.keys() != expected_ids:
        raise ValueError(f"Unexpected node set: {sorted(nodes.keys() ^ expected_ids)}")

    if nodes[67]["widgets_values"] != [5.0] or nodes[68]["widgets_values"] != [5.0]:
        raise ValueError("Both ModelSamplingSD3 nodes must use shift 5")
    if nodes[LORA_HIGH_ID]["widgets_values"] != [HIGH_LORA, 1.0]:
        raise ValueError("High-noise LightX2V LoRA settings changed")
    if nodes[LORA_LOW_ID]["widgets_values"] != [LOW_LORA, 1.0]:
        raise ValueError("Low-noise LightX2V LoRA settings changed")

    sampler = nodes[SAMPLER_ID]
    if [item["name"] for item in sampler["inputs"]] != [
        "high_noise_model",
        "low_noise_model",
        "positive",
        "negative",
        "latent_image",
        "live_preview",
    ]:
        raise ValueError("Sampler socket order no longer matches nodes.py")
    if sampler["widgets_values"] != SAMPLER_WIDGETS:
        raise ValueError("Sampler widget order no longer matches nodes.py")
    if [item["type"] for item in sampler["outputs"]] != ["LATENT", "STRING"]:
        raise ValueError("Sampler output order no longer matches nodes.py")

    links = {link[0]: link for link in workflow["links"]}
    if links[1650][1:5] != [SAMPLER_ID, 0, 8, 0]:
        raise ValueError("Sampler latent must connect directly to VAEDecode")
    if links[1651][1:5] != [SAMPLER_ID, 1, 542, 0]:
        raise ValueError("Sampler diagnostics must connect to PreviewAny")
    if links[1652][1:5] != [PREVIEW_ID, 0, SAMPLER_ID, 5]:
        raise ValueError("Live preview must connect once to the combined sampler")

    preview = nodes[PREVIEW_ID]
    if preview["widgets_values"] != LIVE_PREVIEW_WIDGETS:
        raise ValueError("Live-preview widget order no longer matches nodes.py")
    if preview["outputs"][0]["type"] != LIVE_PREVIEW_TYPE:
        raise ValueError("Live-preview socket type changed")

    for node in workflow["nodes"]:
        for slot, input_spec in enumerate(node.get("inputs", [])):
            link_id = input_spec.get("link")
            if link_id is not None and links[link_id][3:5] != [node["id"], slot]:
                raise ValueError(f"Input metadata mismatch at {node['id']}:{slot}")
        for slot, output_spec in enumerate(node.get("outputs", [])):
            for link_id in output_spec.get("links") or []:
                if links[link_id][1:3] != [node["id"], slot]:
                    raise ValueError(f"Output metadata mismatch at {node['id']}:{slot}")


def build() -> dict:
    workflow = json.loads(SOURCE.read_text(encoding="utf-8"))
    required_source_ids = REMOVE_NODE_IDS | {6, 7, 8, 38, 39, 50, 61, 62, 67, 68, 82}
    source_ids = {node["id"] for node in workflow["nodes"]}
    missing = required_source_ids - source_ids
    if missing:
        raise ValueError(f"testwan.json is missing expected nodes: {sorted(missing)}")

    workflow["id"] = "b4b92c6d-36d6-53ac-b297-ff7864c383ca"
    workflow["revision"] = 0
    workflow["nodes"] = [
        node for node in workflow["nodes"] if node["id"] not in REMOVE_NODE_IDS
    ]

    nodes = {node["id"]: node for node in workflow["nodes"]}
    nodes[61]["title"] = "Wan 2.2 I2V high-noise GGUF"
    nodes[62]["title"] = "Wan 2.2 I2V low-noise GGUF"
    nodes[67]["title"] = "High-noise ModelSamplingSD3 (shift 5)"
    nodes[68]["title"] = "Low-noise ModelSamplingSD3 (shift 5)"
    nodes[67]["widgets_values"] = [5.0]
    nodes[68]["widgets_values"] = [5.0]
    nodes[542]["title"] = "Combined sampler diagnostics"

    nodes[61]["pos"] = [-2370, 2770]
    nodes[62]["pos"] = [-2370, 2920]
    nodes[67]["pos"] = [-1460, 2760]
    nodes[68]["pos"] = [-1460, 2910]
    nodes[542]["pos"] = [-620, 2920]
    nodes[542]["size"] = [300, 120]
    nodes[8]["pos"] = [-620, 3090]
    nodes[82]["pos"] = [-360, 3270]

    workflow["nodes"].extend(
        [
            _lora_node(
                LORA_HIGH_ID,
                "LightX2V rank-64 LoRA (high noise)",
                HIGH_LORA,
                [-1900, 2760],
            ),
            _lora_node(
                LORA_LOW_ID,
                "LightX2V rank-64 LoRA (low noise)",
                LOW_LORA,
                [-1900, 2910],
            ),
            _sampler_node(),
            _live_preview_node(),
        ]
    )

    # Remove the old sampling branch links while preserving every upstream I2V,
    # source-image, preview, VAE, decode and video-combine connection.
    migrated_links = []
    for link in workflow["links"]:
        _, source_id, _, target_id, target_slot, _ = link
        if source_id in REMOVE_NODE_IDS or target_id in REMOVE_NODE_IDS:
            continue
        if source_id in {50, 61, 62, 67, 68}:
            continue
        if target_id == 8 and target_slot == 0:
            continue
        if target_id == 542:
            continue
        migrated_links.append(link)
    workflow["links"] = migrated_links + _new_links()

    execution_order = [
        38,
        39,
        61,
        62,
        642,
        676,
        LORA_HIGH_ID,
        LORA_LOW_ID,
        6,
        7,
        677,
        641,
        67,
        68,
        678,
        92,
        50,
        PREVIEW_ID,
        SAMPLER_ID,
        542,
        8,
        82,
    ]
    by_id = {node["id"]: node for node in workflow["nodes"]}
    for order, node_id in enumerate(execution_order):
        by_id[node_id]["order"] = order

    workflow["last_node_id"] = PREVIEW_ID
    workflow["last_link_id"] = 1652
    _rebuild_connection_metadata(workflow)
    _validate(workflow)
    return workflow


def main() -> None:
    source_before = SOURCE.read_bytes()
    workflow = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(workflow, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if SOURCE.read_bytes() != source_before:
        raise RuntimeError("The source testwan.json was modified")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
