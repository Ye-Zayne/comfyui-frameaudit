"""ComfyUI V1 adapter for bounded-resolution, scene-aware frame inspection."""

import json

from .core import analyze_frames, contact_sheet


class FrameAuditAnalyze:
    CATEGORY = "FrameAudit"
    FUNCTION = "analyze"
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = (
        "contact_sheet",
        "report_json",
        "verdict",
        "flagged_indices_json",
        "cut_indices_json",
    )
    OUTPUT_NODE = True
    DESCRIPTION = "Locate heuristic exposure/chroma flicker and near-identical frame runs; input frames are never corrected or removed."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 1000}),
                "max_analysis_side": ("INT", {"default": 128, "min": 16, "max": 256}),
                "exposure_threshold": (
                    "FLOAT",
                    {"default": 0.035, "min": 0.001, "max": 1},
                ),
                "chroma_threshold": (
                    "FLOAT",
                    {"default": 0.04, "min": 0.001, "max": 2},
                ),
                "repeat_threshold": (
                    "FLOAT",
                    {"default": 0.0015, "min": 0, "max": 0.1},
                ),
                "min_repeat_frames": ("INT", {"default": 3, "min": 2, "max": 1000}),
                "cut_threshold": ("FLOAT", {"default": 0.7, "min": 0.05, "max": 3}),
                "cut_guard_frames": ("INT", {"default": 1, "min": 0, "max": 5}),
                "max_preview_frames": ("INT", {"default": 12, "min": 1, "max": 24}),
            }
        }

    def analyze(
        self,
        images,
        fps=24.0,
        max_analysis_side=128,
        exposure_threshold=0.035,
        chroma_threshold=0.04,
        repeat_threshold=0.0015,
        min_repeat_frames=3,
        cut_threshold=0.7,
        cut_guard_frames=1,
        max_preview_frames=12,
    ):
        report = analyze_frames(
            images,
            fps,
            max_analysis_side,
            exposure_threshold,
            chroma_threshold,
            repeat_threshold,
            min_repeat_frames,
            cut_threshold,
            cut_guard_frames,
        )
        preview = contact_sheet(images, report, max_preview_frames)
        encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
        result = (
            preview,
            encoded,
            report["verdict"],
            json.dumps(report["flagged_indices"]),
            json.dumps(report["heuristic_cut_indices"]),
        )
        return {
            "ui": {"text": [encoded], "frameaudit_verdict": [report["verdict"]]},
            "result": result,
        }
