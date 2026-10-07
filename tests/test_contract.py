import ast
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def test_v1_contract_and_outputs(package):
    cls = package.NODE_CLASS_MAPPINGS["FrameAuditAnalyze"]
    assert cls.RETURN_TYPES == ("IMAGE", "STRING", "INT", "STRING", "STRING")
    assert cls.FUNCTION == "analyze" and cls.OUTPUT_NODE
    assert cls.INPUT_TYPES()["required"]["images"] == ("IMAGE",)
    images = torch.full((5, 8, 8, 3), 0.3)
    result = cls().analyze(images)
    assert len(result["result"]) == len(cls.RETURN_TYPES)
    preview, encoded, verdict, flagged, cuts = result["result"]
    report = json.loads(encoded)
    assert preview.dtype == torch.float32 and preview.shape[-1] == 3
    assert verdict == report["verdict"] == 1
    assert json.loads(flagged) == report["flagged_indices"]
    assert json.loads(cuts) == report["heuristic_cut_indices"]
    assert result["ui"]["text"] == [encoded]


def test_runtime_has_no_dynamic_import_network_env_database_or_commands():
    root = Path(__file__).resolve().parents[1]
    prohibited_modules = {
        "importlib",
        "os",
        "subprocess",
        "requests",
        "urllib",
        "socket",
        "sqlite3",
    }
    for name in ["__init__.py", "core.py", "nodes.py"]:
        tree = ast.parse((root / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    alias.name.split(".")[0] not in prohibited_modules
                    for alias in node.names
                )
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                assert (node.module or "").split(".")[0] not in prohibited_modules
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec", "__import__", "open"}


def test_api_fixture_prompt_uses_registered_node(package):
    root = Path(__file__).resolve().parents[1]
    prompt = json.loads((root / "examples" / "audit_api.json").read_text())
    assert prompt["6"]["class_type"] in package.NODE_CLASS_MAPPINGS
    required = package.NODE_CLASS_MAPPINGS[prompt["6"]["class_type"]].INPUT_TYPES()[
        "required"
    ]
    assert set(required) == set(prompt["6"]["inputs"])
    for key in ["1", "2", "3"]:
        assert (
            root / "examples" / "fixtures" / prompt[key]["inputs"]["image"]
        ).is_file()


def test_bundled_fixtures_produce_documented_flicker_result(package):
    root = Path(__file__).resolve().parents[1]
    prompt = json.loads((root / "examples" / "audit_api.json").read_text())
    frames = []
    for key in ["1", "2", "3"]:
        with Image.open(
            root / "examples" / "fixtures" / prompt[key]["inputs"]["image"]
        ) as image:
            frames.append(
                torch.from_numpy(
                    np.asarray(image.convert("RGB"), dtype=np.float32) / 255
                )
            )
    kwargs = {
        key: value for key, value in prompt["6"]["inputs"].items() if key != "images"
    }
    output = package.NODE_CLASS_MAPPINGS["FrameAuditAnalyze"]().analyze(
        torch.stack(frames), **kwargs
    )
    report = json.loads(output["result"][1])
    assert report["flagged_indices"] == [1]
    assert report["strong_flicker_indices"] == [1]
    assert report["heuristic_cut_indices"] == []
    assert report["frames"][1]["timestamp_seconds"] == 1 / 24
    with Image.open(root / "examples" / "frames.gif") as gif:
        assert gif.n_frames == 3
        assert gif.size == (48, 32)
