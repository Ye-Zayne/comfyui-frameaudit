# FrameAudit

Frame-indexed video quality inspection for ComfyUI IMAGE batches. Version **0.1.0**.
Locate suspect exposure/color reversals and near-identical frame runs, then inspect
the contact sheet and JSON before choosing frames for rework. Every input frame
is retained; this node does not smooth, repair, trim, or remove scenes.

## Install and use

Clone this repository into `ComfyUI/custom_nodes/`, restart ComfyUI, then search
**FrameAudit · Analyze Clip** (`FrameAuditAnalyze`). Uses the host's PyTorch,
NumPy and Pillow. No extra model, automatic dependency installation, network,
database, environment-variable access or persistent file output is performed.

Connect a decoded video's `IMAGE` frame batch to `images`, enter the actual FPS,
and connect `contact_sheet` to **Preview Image**. Native `VIDEO` objects must first
be decoded to IMAGE frames by a video-loading node. This node expects a single
constant-FPS batch at one resolution; it cannot infer original timestamps after
frame skipping, resampling, or cropping by an upstream loader.

| Output | Meaning |
| --- | --- |
| `contact_sheet` | One bounded preview IMAGE, prioritizing suspicious frames and showing indices/timestamps |
| `report_json` | Per-frame scores/flags, repeat intervals, heuristic cuts, evaluated/excluded frames, settings and warnings |
| `verdict` | `0=FAIL`, `1=REVIEW`, `2=PASS`, interpreted below |
| `flagged_indices_json` | JSON array of zero-based suspect frame indices |
| `cut_indices_json` | JSON array of frame indices where a heuristic cut begins |

The node is an output node. API history also contains the same JSON in
`outputs["<node id>"]["text"][0]`. No report is written to disk by FrameAudit.
For a flagged frame, use its original batch index to select the source frame in
your existing workflow. The thumbnails and cut markers are inspection aids.

## Metrics and controls

- **Exposure:** a mean-luminance change must reverse sign at an interior frame;
  the score is the smaller adjacent change. A monotonic luminance ramp does not
  trigger this reversal test. Default threshold: `0.035` on RGB normalized to 0–1.
- **Chroma:** uses mean `(R-G, B-G)` changes and their reversed direction. Default
  threshold `0.04`. This is an opponent-RGB heuristic, not perceptual Delta E.
- **Repeats:** mean absolute RGB change at the analysis resolution is at most
  `repeat_threshold=0.0015` for a run of at least `min_repeat_frames=3` frames.
  Runs include both endpoints. Surrounding same-scene pixel changes can label
  a run `freeze_suspect`; no motion or freeze is semantically proved.
- **Cuts:** contrast-normalized spatial luminance change at least
  `cut_threshold=0.7` marks the transition before that frame. Cut-adjacent
  two-sided tests are excluded, plus `cut_guard_frames=1` nearby frame on each
  side. Repeated runs do not cross a marked cut. No frames are removed.
- **Resolution:** `max_analysis_side=128`, range 16–256; no upscaling. Features
  are computed from one frame at a time, with only the preceding thumbnail
  retained. Per-frame JSON is O(N); no all-pairs or optical-flow computation.
  MPS frames are moved individually to CPU before area downsampling to support
  non-divisible dimensions; CPU/CUDA inputs downsample on their original device.
- **Preview:** `max_preview_frames=12`, range 1–24. At most 24 small thumbnails;
  this is not a full-resolution overlay batch.

`FAIL` means a reversal score reached at least twice its configured threshold.
It is a configurable QC decision, not proof that an intentional flash is wrong.
`REVIEW` means weaker flags, repeat runs, or no usable two-sided temporal tests.
An entirely static clip is reported as a near-identical run and REVIEW, not a
definite freeze. `PASS` means no configured heuristic threshold was exceeded on
evaluated frames; excluded frames are not certified. One or two input frames
return REVIEW because two-sided flicker cannot be assessed.

Indices start at zero. Frame timestamps are `index/fps`. Repeat intervals use
`start_frame` / `end_frame_inclusive` and `[start_seconds, end_seconds_exclusive)`.
The contact sheet is sampled for review; consult all frame entries in JSON.

## Limits

RGB floating-point `[N,H,W,3]` values must be finite and in `[0,1]`; every original
pixel is checked before downsampling. FPS must be finite in `[0.001,1000]`.
Meta tensors and empty dimensions are rejected. Hard limits checked before pixel
processing: **4096 frames**, **16,777,216 pixels per frame**, and **536,870,912
total pixels** (RGB channel count excluded). Split long clips with overlapping
boundary frames, or resize them before this node; timestamps then refer to each
input chunk and need the chunk's original start offset added externally.

The cut detector may exclude large camera/object motion and miss flat-color cuts,
gradual transitions, or cuts with similar structure. Global frame averages can
miss small local flicker, cancel opposing changes, and are diluted by borders.
Exposure changes can affect freeze-context evidence. Downsampling can hide tiny
moving objects. Static shots, animation holds, intentional flashes, object motion
and color changes all require human judgment. No learned quality score, semantic
scene identity, optical flow, or physical motion speed is claimed.

## Reproducible API example

Copy the three tiny synthetic PNG files from `examples/fixtures/` into ComfyUI's
`input/` directory. `examples/audit_api.json` is a real API-format **prompt object**:
submit it as the `prompt` property to ComfyUI's `/prompt` endpoint, not as an
editor-format workflow import. Its LoadImage/ImageBatch nodes form a three-frame
low/high/low exposure sequence; node 6 should flag frame 1 (`1/24` second).
The generated `examples/frames.gif` illustrates those same fixtures; it is not
required by the API prompt. No third-party media is bundled.

## Related work

[comfyUi-deflicker](https://github.com/karcsiha/comfyUi-deflicker) offers brightness/
color correction, temporal smoothing and boundary correction. FrameAudit is an
independent inspection/localization tool and changes no input frames.
[SeamRank](https://github.com/Ye-Zayne/comfyui-seamrank) ranks continuation candidates
at a join; FrameAudit inspects a clip's interior frames. These references do not
establish that no other QC nodes exist. No competitor code was copied.

## Development

Use the ComfyUI Python environment with pytest and Ruff:

```sh
python -m pytest --rootdir=tests --confcutdir=tests tests --import-mode=importlib -q
ruff check --select S102,S307,E702 .
```

Tests cover exposure/chroma alternation, a single flash, monotonic ramps, static
shots, localized repeated runs, scene exclusion, original-pixel validation,
budgets, tiny/short inputs, input preservation and the V1/API fixture contract.
Tests include an optional MPS odd-resolution check when hardware is available.
Other GPU configurations and real production videos need separate validation.
MIT license.

---

## 中文说明

FrameAudit 为 ComfyUI 视频帧批次提供**问题帧定位与人工验收**，版本 0.1.0。
输入 `IMAGE [N,H,W,3]`，输出接触表、逐帧 JSON、验收等级和问题/切换帧索引。
节点不修改、平滑、裁切或删除原帧；原生 VIDEO 需要先解码为 IMAGE。

安装到 `custom_nodes` 后重启，搜索 **FrameAudit · Analyze Clip**。使用宿主已有
PyTorch、NumPy、Pillow，不下载模型、不自动安装依赖，不访问网络、数据库或环境变量，
不把结果永久保存到文件。输入需为同分辨率、恒定 FPS 的帧批次，FPS 由用户提供。

**检测含义：** 亮度需要相邻变化发生反向才标记，因此单调亮度渐变不会被该规则视为
交替闪烁。色度使用均值 `(R-G,B-G)` 的反向变化，并非感知色差或语义分析。
默认重复阈值 0.0015，至少 3 帧的近似相同段会被报告；附近同镜头像素变化可生成
`freeze_suspect` 提示，但真实静止镜头、动画保持和故障冻结不能仅凭像素区分。

切换判定依赖对比度归一化的空间亮度变化，默认阈值 0.7、保护范围 1 帧。跨切换的
双侧闪烁检查会排除，重复段不会跨越该边界。大幅运动可能被误判为切换，纯色切换、
渐变切换或结构相似的切换可能漏检；排除帧并不代表合格，也不会从视频中删除。

**等级：** `0=FAIL` 表示反向变化至少达到配置阈值的两倍，仍需确认是否为有意闪光；
`1=REVIEW` 表示轻度问题、重复段或时间邻居不足；`2=PASS` 只表示被检查帧未超过这些
启发式阈值。整段静止视频会要求 REVIEW，不会被断言为冻结。单帧/两帧也返回 REVIEW。

索引从 0 开始，时间戳为 `index/fps`；重复区间末帧包含，结束秒数不包含。JSON 包含
全部帧，接触表最多展示 24 个缩略图。使用索引在现有工作流选择原始帧进行定向返工。

每个原像素都会进行有限值与 `[0,1]` 范围检查，分析缩略图最长边最大 256（默认 128），
不做全帧两两匹配。硬限额为 **4096 帧、每帧 16,777,216 像素、整段 536,870,912 像素**，
RGB 通道数不计入；超限请先分段或缩放。分段可保留重叠邻居帧，时间戳需自行加上分段
在原视频中的起始偏移。FPS 有效范围 `[0.001,1000]`，空输入和 meta tensor 明确拒绝。
MPS 输入逐帧转到 CPU 后缩小，兼容非整除尺寸；CPU/CUDA 输入在原设备缩小。

局部闪烁可能被全局均值掩盖，黑边会稀释指标，缩略图可能漏掉小物体运动。真实闪光、
物体运动、曝光变化和静止镜头均需人工判断；节点不宣称语义质量、光流或真实运动速度。

示例：把 `examples/fixtures/` 三张 PNG 放入 ComfyUI 的 input，再把
`examples/audit_api.json` 作为 `/prompt` 请求的 `prompt` 对象提交。它是 API 格式，
不是编辑器工作流导入格式；会生成低/高/低曝光三帧，预期标记帧 1。附带小 GIF 仅供展示。

现有 [comfyUi-deflicker](https://github.com/karcsiha/comfyUi-deflicker) 面向曝光/颜色纠正与
平滑；FrameAudit 面向检查和定位。[SeamRank](https://github.com/Ye-Zayne/comfyui-seamrank)
面向接片候选优选。本项目不宣称市场上没有其他 QC 节点，未复制上述项目代码。
