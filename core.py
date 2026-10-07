"""Frame-localization heuristics; no frame correction, files or external services."""

from __future__ import annotations

import heapq
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

VERSION = "0.1.0"
FAIL, REVIEW, PASS = 0, 1, 2
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
MAX_FRAMES = 4096
MAX_PIXELS_PER_FRAME = 16_777_216
MAX_TOTAL_PIXELS = 536_870_912


def _number(name, value, low, high, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}].")
    if integer and int(value) != value:
        raise ValueError(f"{name} must be an integer.")
    return int(value) if integer else float(value)


def _small_frame(frame: torch.Tensor, side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(1.0, side / max(h, w))
    size = (max(1, round(h * scale)), max(1, round(w * scale)))
    # Some MPS versions cannot area-pool non-divisible spatial dimensions.
    # Move only this frame to CPU before pooling on that backend.
    source = frame.detach()
    if source.device.type == "mps":
        source = source.to(device="cpu", dtype=torch.float32)
    value = source.movedim(-1, 0).unsqueeze(0).float()
    if size != (h, w):
        value = F.interpolate(value, size=size, mode="area")
    return value[0].movedim(0, -1).cpu().numpy()


def _validate_batch(images):
    if not isinstance(images, torch.Tensor):
        raise ValueError("images must be a ComfyUI torch IMAGE tensor.")
    if images.ndim != 4 or images.shape[-1] != 3 or min(images.shape[:3]) < 1:
        raise ValueError("images must have shape [N,H,W,3] with N,H,W >= 1.")
    if images.device.type == "meta":
        raise ValueError("Meta tensors have no pixels and cannot be audited.")
    count, height, width, _ = images.shape
    if count > MAX_FRAMES:
        raise ValueError(
            f"Clip exceeds {MAX_FRAMES} frames; split it into shorter clips with overlapping boundary frames."
        )
    if height * width > MAX_PIXELS_PER_FRAME:
        raise ValueError(
            f"A frame exceeds {MAX_PIXELS_PER_FRAME} pixels; resize before auditing."
        )
    if count * height * width > MAX_TOTAL_PIXELS:
        raise ValueError(
            f"Clip exceeds {MAX_TOTAL_PIXELS} total pixels; split or resize before auditing."
        )
    if not images.is_floating_point():
        raise ValueError("images must contain floating point RGB values in [0,1].")
    # Full input validation precedes downsampling, including unsampled pixels.
    # Validate one frame at a time rather than allocate a whole-clip boolean copy.
    for frame in images:
        if not bool(torch.isfinite(frame).all()):
            raise ValueError("images contain NaN or infinity.")
        if float(frame.min()) < 0 or float(frame.max()) > 1:
            raise ValueError("images values must be in [0,1].")


def analyze_frames(
    images,
    fps=24.0,
    max_analysis_side=128,
    exposure_threshold=0.035,
    chroma_threshold=0.04,
    repeat_threshold=0.0015,
    min_repeat_frames=3,
    cut_threshold=0.7,
    cut_guard_frames=1,
):
    """Return O(N) metrics; retain only two bounded-resolution frame arrays."""
    fps = _number("fps", fps, 0.001, 1000)
    side = _number("max_analysis_side", max_analysis_side, 16, 256, integer=True)
    exposure_threshold = _number("exposure_threshold", exposure_threshold, 0.001, 1)
    chroma_threshold = _number("chroma_threshold", chroma_threshold, 0.001, 2)
    repeat_threshold = _number("repeat_threshold", repeat_threshold, 0, 0.1)
    min_repeat_frames = _number(
        "min_repeat_frames", min_repeat_frames, 2, 1000, integer=True
    )
    cut_threshold = _number("cut_threshold", cut_threshold, 0.05, 3)
    guard = _number("cut_guard_frames", cut_guard_frames, 0, 5, integer=True)
    _validate_batch(images)
    n, h, w, _ = images.shape
    means = np.zeros(n, dtype=np.float64)
    chromas = np.zeros((n, 2), dtype=np.float64)
    deltas = np.zeros(n, dtype=np.float64)
    cut_scores = np.zeros(n, dtype=np.float64)
    previous, previous_structure = None, None
    analyzed_shape = None
    for index, frame in enumerate(images):
        rgb = _small_frame(frame, side)
        analyzed_shape = list(rgb.shape[:2])
        gray = rgb @ LUMA
        means[index] = float(gray.mean())
        mean_rgb = rgb.mean(axis=(0, 1), dtype=np.float64)
        chromas[index] = [mean_rgb[0] - mean_rgb[1], mean_rgb[2] - mean_rgb[1]]
        structure = np.clip((gray - gray.mean()) / max(float(gray.std()), 0.05), -3, 3)
        if previous is not None:
            deltas[index] = float(np.mean(np.abs(rgb - previous), dtype=np.float64))
            cut_scores[index] = float(
                np.mean(np.abs(structure - previous_structure), dtype=np.float64)
            )
        previous, previous_structure = rgb, structure

    cuts = [i for i in range(1, n) if cut_scores[i] >= cut_threshold]
    is_cut = np.zeros(n, dtype=bool)
    is_cut[cuts] = True
    scene_ids = np.cumsum(is_cut).tolist()
    excluded = set()
    for index in cuts:
        excluded.update(range(max(0, index - guard), min(n, index + guard + 1)))
    frames = []
    for index in range(n):
        reasons = []
        if index == 0 or index == n - 1:
            reasons.append("missing_two_temporal_neighbors")
        if index in excluded or is_cut[index] or (index + 1 < n and is_cut[index + 1]):
            reasons.append("heuristic_cut_exclusion")
        brightness, color = None, None
        flags = []
        if not reasons:
            before, after = (
                means[index] - means[index - 1],
                means[index + 1] - means[index],
            )
            brightness = (
                float(min(abs(before), abs(after))) if before * after < 0 else 0.0
            )
            a, b = (
                chromas[index] - chromas[index - 1],
                chromas[index + 1] - chromas[index],
            )
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            reversal = (
                max(0.0, -float(np.dot(a, b)) / (na * nb)) if na * nb > 1e-12 else 0.0
            )
            color = float(min(na, nb) / math.sqrt(2) * reversal)
            if brightness >= exposure_threshold:
                flags.append("exposure_flicker")
            if color >= chroma_threshold:
                flags.append("chroma_flicker")
        frames.append(
            {
                "index": index,
                "timestamp_seconds": index / fps,
                "scene_id": int(scene_ids[index]),
                "mean_luminance": float(means[index]),
                "mean_opponent_chroma": [float(x) for x in chromas[index]],
                "rgb_change_from_previous": float(deltas[index]) if index else None,
                "cut_score_from_previous": float(cut_scores[index]) if index else None,
                "heuristic_cut_before": bool(is_cut[index]),
                "flicker_evaluated": not reasons,
                "flicker_exclusion_reasons": reasons,
                "exposure_flicker_score": brightness,
                "chroma_flicker_score": color,
                "flags": flags,
            }
        )

    # A run includes its first frame and the following near-identical transitions.
    # Runs never cross a heuristic cut. A static clip remains a REVIEW candidate.
    runs = []
    start = 0
    for stop in range(1, n + 1):
        continues = stop < n and not is_cut[stop] and deltas[stop] <= repeat_threshold
        if continues:
            continue
        end = stop - 1
        if end - start + 1 >= min_repeat_frames:
            context = []
            for transition in range(max(1, start - 3), min(n, end + 5)):
                if start < transition <= end:
                    continue
                if (
                    not is_cut[transition]
                    and scene_ids[transition] == scene_ids[start]
                    and scene_ids[transition - 1] == scene_ids[start]
                ):
                    context.append(float(deltas[transition]))
            context_motion = max(context, default=0.0)
            motion_evidence = context_motion > max(0.01, repeat_threshold * 4)
            kind = "freeze_suspect" if motion_evidence else "near_identical_run"
            runs.append(
                {
                    "start_frame": start,
                    "end_frame_inclusive": end,
                    "frame_count": end - start + 1,
                    "start_seconds": start / fps,
                    "end_seconds_exclusive": (end + 1) / fps,
                    "scene_id": int(scene_ids[start]),
                    "classification": kind,
                    "surrounding_pixel_change_evidence": motion_evidence,
                    "maximum_context_rgb_change": context_motion,
                    "interpretation": "Pixel similarity cannot distinguish a freeze from an intentional static shot, hold, or repeated animation frame.",
                }
            )
            for index in range(start, end + 1):
                frames[index]["flags"].append("repeated_frame_run")
                if motion_evidence:
                    frames[index]["flags"].append("freeze_suspect")
        start = stop

    flagged = [frame["index"] for frame in frames if frame["flags"]]
    strong = [
        frame["index"]
        for frame in frames
        if (
            frame["exposure_flicker_score"] is not None
            and frame["exposure_flicker_score"] >= 2 * exposure_threshold
        )
        or (
            frame["chroma_flicker_score"] is not None
            and frame["chroma_flicker_score"] >= 2 * chroma_threshold
        )
    ]
    evaluated = sum(frame["flicker_evaluated"] for frame in frames)
    warnings = [
        "All findings are pixel/statistical heuristics, not semantic defects or optical flow.",
        "Global means can miss local flicker or cancel opposing changes; moving subjects and intentional flashes may be flagged.",
        "Cut detection is heuristic: large motion can resemble a cut, and flat-color cuts or gradual transitions can be missed.",
        "Cut exclusion prevents cross-shot flicker tests; it does not certify excluded frames or modify any input frame.",
        "Repeat scores are measured after downsampling; tiny moving details can disappear. Static shots and intentional holds need human review.",
    ]
    if n < 3:
        warnings.append(
            "Fewer than three frames: two-sided flicker cannot be evaluated."
        )
    if evaluated == 0:
        warnings.append(
            "No frame has two usable temporal neighbors; temporal quality is unverified."
        )
    if strong:
        verdict = FAIL
        verdict_reason = "Strong exposure/chroma reversals exceeded twice their configured thresholds; inspect the flagged frames."
    elif flagged or evaluated == 0:
        verdict = REVIEW
        verdict_reason = "Suspect frames, near-identical runs, or insufficient temporal coverage require human review."
    else:
        verdict = PASS
        verdict_reason = "No configured heuristic threshold was exceeded on evaluated frames; this is not a semantic quality guarantee."
    return {
        "schema_version": "1.0",
        "package_version": VERSION,
        "verdict": verdict,
        "verdict_label": {FAIL: "FAIL", REVIEW: "REVIEW", PASS: "PASS"}[verdict],
        "verdict_reason": verdict_reason,
        "fps": fps,
        "frame_count": n,
        "duration_seconds": n / fps,
        "input_shape": [n, h, w, 3],
        "analysis_shape": analyzed_shape,
        "limits": {
            "max_frames": MAX_FRAMES,
            "max_pixels_per_frame": MAX_PIXELS_PER_FRAME,
            "max_total_pixels": MAX_TOTAL_PIXELS,
        },
        "parameters": {
            "max_analysis_side": side,
            "exposure_threshold": exposure_threshold,
            "chroma_threshold": chroma_threshold,
            "repeat_threshold": repeat_threshold,
            "min_repeat_frames": min_repeat_frames,
            "cut_threshold": cut_threshold,
            "cut_guard_frames": guard,
        },
        "metrics": {
            "exposure": "minimum adjacent mean-luminance change at a temporal sign reversal",
            "chroma": "mean (R-G,B-G) change, weighted by reversed direction; not perceptual Delta E",
            "repeat": "mean absolute downsampled RGB difference",
            "cut": "mean absolute contrast-normalized luminance structure difference; not a semantic shot detector",
        },
        "flagged_indices": flagged,
        "strong_flicker_indices": strong,
        "heuristic_cut_indices": cuts,
        "cut_excluded_indices": sorted(excluded),
        "flicker_evaluated_frame_count": evaluated,
        "repeat_runs": runs,
        "frames": frames,
        "warnings": warnings,
    }


def contact_sheet(images, report, max_preview_frames=12):
    count = _number("max_preview_frames", max_preview_frames, 1, 24, integer=True)
    n = report["frame_count"]
    priorities = []
    for frame in report["frames"]:
        score = max(
            (frame["exposure_flicker_score"] or 0)
            / report["parameters"]["exposure_threshold"],
            (frame["chroma_flicker_score"] or 0)
            / report["parameters"]["chroma_threshold"],
        )
        if frame["flags"]:
            priorities.append((-score, frame["index"]))
    selected = []
    for index in (
        [i for _, i in heapq.nsmallest(count, priorities)]
        + report["heuristic_cut_indices"]
        + [0, n // 2, n - 1]
    ):
        if index not in selected:
            selected.append(index)
        if len(selected) >= count:
            break
    selected.sort()
    columns = min(3, len(selected))
    rows = math.ceil(len(selected) / columns)
    tile_w, tile_h = 240, 174
    canvas = Image.new("RGB", (columns * tile_w, rows * tile_h + 24), (20, 20, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 6),
        f"FrameAudit {VERSION} | {report['verdict_label']}",
        fill="white",
    )
    for position, index in enumerate(selected):
        rgb = _small_frame(images[index], 224)
        image = Image.fromarray(np.rint(rgb * 255).astype(np.uint8))
        image.thumbnail((224, 120))
        x, y = (position % columns) * tile_w + 8, (position // columns) * tile_h + 30
        canvas.paste(image, (x + (224 - image.width) // 2, y))
        frame = report["frames"][index]
        label = " + ".join(frame["flags"]) or (
            "CUT excluded" if frame["heuristic_cut_before"] else "no heuristic flag"
        )
        draw.text(
            (x, y + 124), f"frame {index} | {index / report['fps']:.3f}s", fill="white"
        )
        draw.text(
            (x, y + 138),
            label[:35],
            fill=(255, 160, 110) if frame["flags"] else (140, 220, 180),
        )
        draw.text(
            (x, y + 152),
            "excluded"
            if not frame["flicker_evaluated"]
            else f"L:{frame['exposure_flicker_score']:.4f} C:{frame['chroma_flicker_score']:.4f}",
            fill=(185, 185, 190),
        )
    report["preview_frame_indices"] = selected
    return torch.from_numpy(
        np.asarray(canvas, dtype=np.float32).copy() / 255
    ).unsqueeze(0)
