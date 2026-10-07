import json

import numpy as np
import pytest
import torch


def textured(count=9, height=24, width=32):
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    gray = 0.25 + 0.2 * ((xx + yy) % 2).float()
    return gray[..., None].repeat(1, 1, 3).unsqueeze(0).repeat(count, 1, 1, 1)


def test_exposure_alternation_localizes_interior(core):
    images = textured()
    images[1::2] += 0.16
    report = core.analyze_frames(images, fps=20)
    assert report["heuristic_cut_indices"] == []
    assert report["verdict"] == core.FAIL
    assert report["flagged_indices"] == list(range(1, 8))
    assert all("exposure_flicker" in report["frames"][i]["flags"] for i in range(1, 8))
    assert report["frames"][3]["timestamp_seconds"] == pytest.approx(0.15)


def test_single_flash_localizes_only_flash(core):
    images = textured(7)
    images[3] += 0.2
    report = core.analyze_frames(images)
    exposure = [
        f["index"] for f in report["frames"] if "exposure_flicker" in f["flags"]
    ]
    assert exposure == [3]


def test_chroma_alternation_without_exposure(core):
    images = torch.full((9, 12, 16, 3), 0.5)
    # Opposite color changes with equal weighted luminance.
    shift = torch.tensor([0.12, -0.12 * 0.2126 / 0.7152, 0.0])
    images[0::2] += shift
    images[1::2] -= shift
    report = core.analyze_frames(images)
    assert report["verdict"] == core.FAIL
    assert report["heuristic_cut_indices"] == []
    assert all("chroma_flicker" in f["flags"] for f in report["frames"][1:-1])
    assert not any("exposure_flicker" in f["flags"] for f in report["frames"])


def test_monotonic_exposure_ramp_is_not_flicker(core):
    images = textured(9) + torch.linspace(0, 0.2, 9)[:, None, None, None]
    report = core.analyze_frames(images)
    assert report["verdict"] == core.PASS
    assert report["flagged_indices"] == []
    assert report["heuristic_cut_indices"] == []


def test_monotonic_chroma_ramp_is_not_flicker(core):
    images = torch.full((9, 12, 16, 3), 0.4)
    images[:, :, :, 0] += torch.linspace(0, 0.3, 9)[:, None, None]
    report = core.analyze_frames(images)
    assert report["flagged_indices"] == []


def test_constant_clip_reported_as_review_not_definite_freeze(core):
    report = core.analyze_frames(textured(6))
    assert report["verdict"] == core.REVIEW
    assert report["strong_flicker_indices"] == []
    assert report["repeat_runs"][0]["start_frame"] == 0
    assert report["repeat_runs"][0]["end_frame_inclusive"] == 5
    assert report["repeat_runs"][0]["classification"] == "near_identical_run"
    assert report["repeat_runs"][0]["surrounding_pixel_change_evidence"] is False


def test_freeze_inside_changing_clip_and_timestamp(core):
    base = textured(1)[0]
    images = torch.stack(
        [base + offset for offset in [0, 0.04, 0.08, 0.08, 0.08, 0.08, 0.12, 0.16]]
    )
    report = core.analyze_frames(images, fps=10)
    assert len(report["repeat_runs"]) == 1
    run = report["repeat_runs"][0]
    assert (run["start_frame"], run["end_frame_inclusive"], run["frame_count"]) == (
        2,
        5,
        4,
    )
    assert (run["start_seconds"], run["end_seconds_exclusive"]) == pytest.approx(
        (0.2, 0.6)
    )
    assert run["classification"] == "freeze_suspect"
    assert report["verdict"] == core.REVIEW


def test_heuristic_cut_excludes_cross_scene_flicker_and_splits_runs(core):
    a = textured(4)
    b = 1 - a
    images = torch.cat([a, b])
    report = core.analyze_frames(images)
    assert report["heuristic_cut_indices"] == [4]
    assert report["cut_excluded_indices"] == [3, 4, 5]
    assert report["strong_flicker_indices"] == []
    assert [
        (r["start_frame"], r["end_frame_inclusive"]) for r in report["repeat_runs"]
    ] == [(0, 3), (4, 7)]
    assert all(
        r["classification"] == "near_identical_run" for r in report["repeat_runs"]
    )
    assert not report["frames"][3]["flicker_evaluated"]
    assert report["frames"][4]["scene_id"] == 1


@pytest.mark.parametrize("count", [1, 2])
def test_short_and_tiny_batches_are_finite_review(core, count):
    images = torch.full((count, 1, 1, 3), 0.5)
    report = core.analyze_frames(images)
    assert report["verdict"] == core.REVIEW
    assert report["analysis_shape"] == [1, 1]
    assert report["flicker_evaluated_frame_count"] == 0
    json.dumps(report, allow_nan=False)
    preview = core.contact_sheet(images, report)
    assert preview.shape[0] == 1 and preview.shape[-1] == 3
    assert bool(torch.isfinite(preview).all())


def test_features_bounded_and_input_unchanged(core):
    images = textured(7, 96, 160)
    images[1::2] += 0.15
    before = images.clone()
    report = core.analyze_frames(images, max_analysis_side=32)
    assert report["analysis_shape"] == [19, 32]
    assert torch.equal(images, before)
    preview = core.contact_sheet(images, report, 2)
    assert len(report["preview_frame_indices"]) <= 2
    assert bool((preview >= 0).all() and (preview <= 1).all())


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), 1001])
def test_invalid_fps_rejected(core, fps):
    with pytest.raises(ValueError, match="fps"):
        core.analyze_frames(textured(), fps=fps)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_full_resolution_pixels_rejected_before_downsample(core, value):
    images = textured(3, 64, 64)
    images[1, 63, 63, 1] = value
    with pytest.raises(ValueError):
        core.analyze_frames(images, max_analysis_side=16)


@pytest.mark.parametrize(
    "images",
    [
        torch.empty(0, 2, 2, 3),
        torch.zeros(2, 2, 3),
        torch.zeros(1, 2, 2, 4),
        torch.zeros(1, 2, 2, 3, dtype=torch.uint8),
    ],
)
def test_invalid_shape_or_dtype(core, images):
    with pytest.raises(ValueError):
        core.analyze_frames(images)


def test_threshold_boundary_repeat_length(core):
    images = textured(4)
    assert core.analyze_frames(images, min_repeat_frames=5)["repeat_runs"] == []
    assert (
        core.analyze_frames(images, min_repeat_frames=4)["repeat_runs"][0][
            "frame_count"
        ]
        == 4
    )


def test_parameter_integer_and_finiteness_checks(core):
    for kwargs in [
        {"cut_threshold": float("nan")},
        {"repeat_threshold": -1},
        {"min_repeat_frames": 2.5},
        {"cut_guard_frames": True},
    ]:
        with pytest.raises(ValueError):
            core.analyze_frames(textured(), **kwargs)


def test_cpu_half_noncontiguous_supported(core):
    images = textured(6).transpose(1, 2).contiguous().transpose(1, 2).half()
    report = core.analyze_frames(images)
    assert len(report["frames"]) == 6
    assert report["repeat_runs"][0]["frame_count"] == 6
    assert np.isfinite(report["frames"][0]["mean_luminance"])


def test_odd_size_downsampling(core):
    images = textured(3, 129, 257)
    images[1] += 0.15
    report = core.analyze_frames(images, max_analysis_side=128)
    assert report["analysis_shape"] == [64, 128]
    assert "exposure_flicker" in report["frames"][1]["flags"]
    json.dumps(report, allow_nan=False)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS hardware unavailable"
)
def test_mps_odd_size_downsampling_matches_cpu(core):
    images = textured(3, 129, 257)
    images[1] += 0.15
    cpu = core.analyze_frames(images, max_analysis_side=128)
    mps = core.analyze_frames(images.to("mps"), max_analysis_side=128)
    assert mps["analysis_shape"] == cpu["analysis_shape"]
    assert mps["flagged_indices"] == cpu["flagged_indices"]
    assert mps["frames"][1]["exposure_flicker_score"] == pytest.approx(
        cpu["frames"][1]["exposure_flicker_score"]
    )


@pytest.mark.parametrize(
    "shape, message",
    [
        ((4097, 1, 1, 3), "frames"),
        ((1, 4097, 4097, 3), "pixels"),
        ((1024, 1024, 1024, 3), "total pixels"),
    ],
)
def test_input_budgets_reject_before_pixel_work(core, shape, message):
    images = torch.zeros(1, 1, 1, 3).expand(shape)
    with pytest.raises(ValueError, match=message):
        core.analyze_frames(images)


def test_meta_tensor_rejected_clearly(core):
    with pytest.raises(ValueError, match="Meta"):
        core.analyze_frames(torch.empty(3, 8, 8, 3, device="meta"))
