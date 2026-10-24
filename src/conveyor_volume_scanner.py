#!/usr/bin/env python3
"""
Conveyor volume scanner - OAK-D Pro (RVC2), DepthAI v3.

WHAT IT DOES
  Measures the volume of an object lying under a downward-looking OAK-D Pro stereo
  camera. Stereo matching runs on the camera. The host reads the depth map, crops a
  square region of interest (ROI) around the image centre, and integrates
  (height above the belt) x (ground area seen by each pixel) over the ROI.
  STATUS: static objects only. Accumulating volume for objects that move through the
  ROI on a running belt is NOT implemented yet (see DESIGN NOTES in the source).

SETUP IT ASSUMES
  * Camera mounted vertically above the belt, optical axis perpendicular to the belt.
  * Belt is flat and its distance to the camera is a single number (--baseline).
  * OAK-D Pro (RVC2): CAM_B = left mono, CAM_C = right mono, CAM_A = RGB.
    The IR dot projector is switched on (helps stereo matching on a plain belt) and
    the IR flood light is off.
  * DepthAI v3 API (dai.Pipeline, node.Camera, node.StereoDepth), not v2.

UNITS
  Depth, baseline, threshold: millimetres.  Volume: cm^3.  Focal length: pixels.

TWO MODES
  GUI (default)  Colour depth map, valid-pixel mask window, optional RGB / 3D plot.
  --headless     No windows. Only the ROI math runs and a status line is printed every
                 --print-interval seconds. Use it to measure the real processing FPS and
                 for production runs. Stop with Ctrl+C or --duration.

USAGE
    python conveyor_volume_scanner.py // 400p 60fps FAST_ACCURACY 
    python conveyor_volume_scanner.py --fps 90
    python conveyor_volume_scanner.py --headless --fps 120 --preset fast_density --no-subpixel
    python conveyor_volume_scanner.py --headless --tare-on-start --duration 60
    python conveyor_volume_scanner.py --subpixel-bits 3 --median 5 --fps 60
    python conveyor_volume_scanner.py --calib my_calibration.json

KEYS (GUI mode only)
    q = quit
    t = tare: set the baseline to the median depth of the ROI (belt must be empty)
"""
import argparse
import time

import cv2
import numpy as np
import depthai as dai

# =============================================================================
# DESIGN NOTES  (read this first when changing the script)
# =============================================================================
# DATA FLOW
#   CAM_B (left mono)  --\
#                         >-- StereoDepth (on the camera) --XLink--> q_depth --> process_depth()
#   CAM_C (right mono) --/    preset + overrides                    (non-blocking, maxSize=4)
#   The depth frame is uint16 in millimetres, 0 = invalid pixel ("hole"), aligned to the
#   rectified LEFT camera (DepthAI v3 default). Its size equals the stereo output size:
#   640x400 for 400P input, unless a preset decimates it (DEFAULT/ROBOTICS/HIGH_DETAIL
#   halve it). The code always reads the real size from depth.shape.
#
# CALIBRATION
#   Stereo rectification and the depth values depend on the camera calibration. By default
#   the pipeline uses the calibration stored on the device. --calib FILE.json replaces it
#   for this run only (pipeline.setCalibrationData); nothing is written to the device.
#   The startup line "calibration: ..." prints the stereo baseline and left-camera
#   intrinsics so you can tell which calibration is active. To compare calibrations, run
#   the same scene with and without --calib and compare "ROI holes" and the empty-belt
#   volume noise. Calibration files come from the Luxonis calibration tool (depthai_calib
#   JSON format).
#
# HOST LOOP  (one pass of `while pipeline.isRunning()`)
#   1. Drain q_depth completely, so every frame the camera delivered gets processed.
#   2. For each frame: crop the ROI -> volume (process_depth) -> optional tare sampling.
#   3. GUI only: draw at most --display-fps times per second, using the NEWEST frame.
#   4. Headless only: print a status line every --print-interval seconds.
#   The FPS shown counts frames pulled from the queue. Frames the camera produced but the
#   non-blocking queue discarded (host too slow) are not counted. Run the same settings
#   with and without --headless to see how much the display costs.
#
# VOLUME MODEL  (code in process_depth)
#   For every ROI pixel whose depth Z is above the belt:
#       height    = baseline - Z                  [mm]
#       footprint = (Z / fx) * (Z / fy)           [mm^2] ground area seen by one pixel
#   volume = sum(height * footprint)
#   The footprint uses each pixel's own Z, so an object's top (closer to the camera) is
#   not credited with the same area as the belt. Assumes: camera looks straight down,
#   object sides are vertical, belt is flat at one fixed distance.
#
# KNOWN LIMITATIONS / NOT IMPLEMENTED
#   * Moving objects: the ROI sum is a snapshot. An object only partly inside the ROI
#     gives a partial volume. Accumulation over a moving belt is not written yet (idea:
#     cross-section area x belt speed x device timestamps, not loop time).
#   * Single flat baseline: tilt or sag of the belt shows up as height error on every ROI
#     pixel. Re-tare after moving the camera or the belt.
#   * Invalid pixels (depth 0) count as belt, i.e. height 0. That is right for the
#     occlusion strips next to tall objects, but invalid pixels on the object itself (edge
#     pixels, dark or shiny spots) make the volume read low. There is no hole filling.
#   * Belt noise larger than --threshold counts as load and adds phantom volume.
#   * Depth steps: with subpixel OFF one disparity step is about 10 mm at 58 cm; with 3
#     subpixel bits about 1.3 mm (estimate for ~440 px focal length and a 7.5 cm baseline,
#     check your own calibration). Keep --threshold above the measured noise floor.
#   * Minimum distance: with the default 96-pixel disparity search, anything closer than
#     about 35 cm to the camera returns holes (an object ~23 cm tall on a 58 cm belt).
#     --disparity-shift moves the search window to lower this limit. At --res 800 the
#     minimum distance is about 70 cm, which is farther than the default --baseline, so
#     800P needs a disparity shift.
#
# HOW TO CHECK THE NUMBERS
#   * Empty belt: volume should be ~0. Log it for a minute: the mean is the bias and the
#     standard deviation is the noise floor. Tune --threshold, subpixel, median, preset
#     against it.
#   * Known object fully inside the ROI, e.g. a 12 x 8 x 5.5 cm box = 528 cm^3. Try several
#     box sizes: an error that grows with perimeter x height is an edge problem, one that
#     grows with volume is a scale or baseline problem.
#   * The "stereo config:" line printed at startup shows what the preset really set.
# =============================================================================

# Mono sensor mode: CLI value (rows) -> (width, height) in pixels.
RES = {400: (640, 400), 800: (1280, 800)}

# CLI preset name -> attribute name of dai.node.StereoDepth.PresetMode.
#   fast_accuracy / fast_density: no post-processing, light on resources. Accuracy gives
#     fewer outliers but more invalid pixels, density the opposite.
#   default / robotics / high_detail: enable post-processing, may halve the output size.
# The exact contents of a preset depend on the depthai version; the startup print of
# "stereo config:" is the reliable reference.
PRESETS = {
    "fast_accuracy": "FAST_ACCURACY",
    "fast_density": "FAST_DENSITY",
    "robotics": "ROBOTICS",
    "default": "DEFAULT",
    "high_detail": "HIGH_DETAIL",
    "accuracy": "ACCURACY",      # newer names; FAST_* are marked deprecated in recent docs
    "density": "DENSITY",
}
# keep only the presets this depthai build actually has
PRESETS = {k: v for k, v in PRESETS.items() if hasattr(dai.node.StereoDepth.PresetMode, v)}

# Display only (GUI mode): size of the shown depth image, and the depth that maps to the
# far end of the colour scale. These do not affect the volume.
DISPLAY_W, DISPLAY_H = 800, 600
MAX_DISTANCE_MM = 3000


# ---- helpers ---------------------------------------------------------------
def draw_detailed_legend(frame, max_distance_mm, step_size=500):
    """Draw a vertical colour bar with distance labels onto `frame`, in place.

    The bar sits near the right edge. Colours use COLORMAP_JET with near = red at the top
    and far = blue at the bottom, matching how the depth image is coloured in main().
    Expects a BGR uint8 image. main() draws it once on a black canvas and then copies it
    onto every frame, so its cost is paid only once. Returns `frame`.
    """
    frame_h, frame_w = frame.shape[:2]
    legend_w, legend_h = 35, int(frame_h * 0.85)
    start_x = frame_w - legend_w - 110       # leave room on the right for the labels
    start_y = (frame_h - legend_h) // 2

    # Gradient 255 (top) -> 0 (bottom), repeated across the bar width, then coloured.
    gradient_1d = np.linspace(255, 0, legend_h).astype(np.uint8)
    gradient_2d = np.tile(gradient_1d, (legend_w, 1)).T
    legend_colored = cv2.applyColorMap(gradient_2d, cv2.COLORMAP_JET)
    frame[start_y:start_y + legend_h, start_x:start_x + legend_w] = legend_colored
    cv2.rectangle(frame, (start_x, start_y), (start_x + legend_w, start_y + legend_h),
                  (255, 255, 255), 2)

    # Tick marks and labels every `step_size` mm; the last label reads ">max".
    font = cv2.FONT_HERSHEY_SIMPLEX
    num_steps = max_distance_mm // step_size
    for i in range(num_steps + 1):
        val = i * step_size
        y_pos = start_y + int((i / num_steps) * legend_h)
        cv2.line(frame, (start_x + legend_w, y_pos), (start_x + legend_w + 10, y_pos),
                 (255, 255, 255), 2)
        text_y = y_pos + 5
        if i == 0:
            text_y += 8       # nudge the first and last labels so they stay inside the bar
        elif i == num_steps:
            text_y -= 5
        text = f"{val}mm" if i < num_steps else f">{val}mm"
        cv2.putText(frame, text, (start_x + legend_w + 15, text_y), font, 0.6,
                    (255, 255, 255), 2)
    return frame


def build_cam(pipeline, socket, size, fps):
    """Create a Camera node on `socket` with the sensor mode (resolution + FPS) fixed.

    Fixing sensorFps matters: the mono cameras otherwise run at 30 FPS, which would cap
    the stereo output. Same pattern as the benchmark script.
    Falls back to setting only the FPS if this depthai build rejects `sensorResolution`.
    """
    cam = pipeline.create(dai.node.Camera)
    try:
        return cam.build(socket, sensorResolution=size, sensorFps=fps)
    except TypeError:
        return cam.build(socket, sensorFps=fps)


def apply_pool(cam, pool):
    """Set the number of buffers in the camera's RAW and ISP frame pools.

    A pool of 16 raised FPS in the benchmark runs (together with XLink chunk size 0).
    A failing setter is reported and skipped, not fatal.
    """
    for setter in ("setRawNumFramesPool", "setIspNumFramesPool"):
        try:
            getattr(cam, setter)(pool)
        except Exception as e:
            print(f"  warning: {setter}({pool}) failed: {e}")


def get_focal_px(depth_msg, calib, depth_w, req_size):
    """Return (fx, fy, source) - focal lengths in pixels for the depth frame received.

    The volume formula needs the focal length of the frame it is applied to, and the depth
    frame size can differ from the requested size (decimating presets).
    Source 1: the intrinsics attached to the frame itself (exact).
    Source 2 (fallback): factory calibration of the left camera at the requested size,
    scaled by depth_w / requested width. Approximate: these are unrectified intrinsics.
    `source` is a short string for the startup print.
    """
    try:
        K = depth_msg.getTransformation().getIntrinsicMatrix()
        return float(K[0][0]), float(K[1][1]), "frame transformation"
    except Exception:
        pass
    # Fallback: calibration at the requested size, scaled to the depth frame width.
    # In DepthAI v3 the default depth alignment is RECTIFIED_LEFT, so use the left
    # camera (CAM_B). These are the unrectified intrinsics, so this is only approximate.
    K = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, req_size[0], req_size[1])
    s = depth_w / req_size[0]
    return float(K[0][0]) * s, float(K[1][1]) * s, "calibration (scaled)"


def describe_calibration(calib, source, size):
    """Print which calibration is in use and numbers that identify it.

    Shows the stereo baseline [cm] and the left camera's intrinsics at the requested mono
    size, so a factory calibration and a recalibrated one can be told apart at a glance.
    `source` is a label for the startup print. A read failure is reported, not fatal.
    """
    try:
        baseline_cm = calib.getBaselineDistance()
        K = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, size[0], size[1])
        print(f"calibration: {source} | stereo baseline {baseline_cm:.3f} cm | left "
              f"fx={K[0][0]:.1f}px fy={K[1][1]:.1f}px cx={K[0][2]:.1f} cy={K[1][2]:.1f} "
              f"(at {size[0]}x{size[1]})")
    except Exception as e:
        print(f"calibration: {source} (could not read details: {e})")


def describe_stereo(stereo):
    """Print the stereo settings that will actually be used (after preset + overrides).

    Call it after all setters. A value that this depthai version does not expose is shown
    as "?". Warns about a combination the device silently ignores (median filter with 4
    or 5 subpixel bits).
    """
    ic = stereo.initialConfig

    def g(fn):
        # Read one setting; "?" if the getter/attribute is missing in this version.
        try:
            return fn()
        except Exception:
            return "?"

    items = {
        "subpixel": g(ic.getSubpixel),
        "bits": g(ic.getSubpixelFractionalBits),
        "lr_check": g(ic.getLeftRightCheck),
        "lr_thr": g(ic.getLeftRightCheckThreshold),
        "extended": g(ic.getExtendedDisparity),
        "conf_thr": g(ic.getConfidenceThreshold),
        "median": g(ic.getMedianFilter),
        "disp_shift": g(lambda: ic.algorithmControl.disparityShift),
        "decimation": g(lambda: ic.postProcessing.decimationFilter.decimationFactor),
        "speckle": g(lambda: ic.postProcessing.speckleFilter.enable),
        "temporal": g(lambda: ic.postProcessing.temporalFilter.enable),
        "spatial": g(lambda: ic.postProcessing.spatialFilter.enable),
        "align": g(lambda: ic.algorithmControl.depthAlign),
    }
    print("stereo config: " + ", ".join(f"{k}={v}" for k, v in items.items()))
    median_on = "OFF" not in str(items["median"]).upper() and items["median"] != "?"
    if items["subpixel"] is True and items["bits"] in (4, 5) and median_on:
        print("warning: the device disables the median filter when subpixel bits are 4 or 5")


def process_depth(depth, baseline, threshold, roi_frac, fx, fy):
    """Crop the ROI from a depth frame and compute the volume inside it.

    Shared by GUI and headless mode. See VOLUME MODEL in the design notes for the formula.

    Args:
        depth:     2-D uint16 array, full depth frame in mm (0 = invalid).
        baseline:  distance from the camera to the empty belt [mm].
        threshold: minimum height above the belt for a pixel to count as load [mm].
        roi_frac:  ROI side length as a fraction of the frame WIDTH. Using the width keeps
                   the physical ROI the same when the frame size changes.
        fx, fy:    focal lengths of this depth frame [px].

    The ROI is a square centred on the frame; its side is made even and at least 8 px.

    Returns:
        roi        - view into `depth` (no copy), uint16 mm
        valid      - bool mask over the ROI: pixel is valid AND higher than the threshold
        volume_cm3 - float, volume of the load inside the ROI [cm^3]
        holes_pct  - float, percent of ROI pixels with depth 0 (invalid)
        geom       - (cx, cy, half): ROI centre and half side in depth-frame pixels
    """
    h, w = depth.shape
    side = max(8, int(w * roi_frac)) // 2 * 2
    half = side // 2
    cy, cx = h // 2, w // 2
    roi = depth[cy - half:cy + half, cx - half:cx + half]

    z = roi.astype(np.float32)
    # Load pixels: a valid depth that is closer than (belt - threshold). Holes (z == 0)
    # and belt-level pixels are both excluded, i.e. they contribute zero height.
    valid = (z > 0) & (z < (baseline - threshold))
    zv = z[valid]
    # sum(height * footprint) with footprint = zv^2 / (fx * fy); mm^3 here, cm^3 on return.
    vol_mm3 = float(np.sum((baseline - zv) * zv ** 2) / (fx * fy))
    holes_pct = 100.0 * float(np.mean(z == 0))
    return roi, valid, vol_mm3 / 1000.0, holes_pct, (cx, cy, half)


def parse_args():
    """Command-line options. `None` defaults mean "leave the preset's value untouched"."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- camera / stereo ---
    ap.add_argument("--res", type=int, choices=[400, 800], default=400,
                    help="mono resolution (800P has MinZ ~70 cm without extended disparity)")
    ap.add_argument("--fps", type=int, default=60, help="requested mono/stereo FPS")
    ap.add_argument("--preset", choices=list(PRESETS),
                    default="accuracy" if "accuracy" in PRESETS else "fast_accuracy",
                    help="stereo preset; the base configuration the overrides below modify")
    ap.add_argument("--no-subpixel", action="store_true",
                    help="disable subpixel (same as --subpixel-bits 0)")
    ap.add_argument("--calib", default="dynamic_calib.json", metavar="FILE.json",
                    help="calibration JSON to use instead of the one stored on the device. "
                         "Applied to this run only; nothing is flashed to the device")
    ap.add_argument("--subpixel-bits", type=int, choices=[0, 3, 4, 5], default=None,
                    help="subpixel fractional bits: 0 = off, 3/4/5 = on with that many "
                         "bits. Default: leave whatever the chosen preset sets")
    ap.add_argument("--disparity-shift", type=int, default=None,
                    help="shift the disparity search window [shift, shift+95]. Raises the "
                         "usable height of objects (lowers MinZ) when nothing is farther "
                         "than the belt. Default: leave the preset value")
    ap.add_argument("--median", type=int, choices=[0, 3, 5, 7], default=None,
                    help="hardware median filter kernel (0 = off). The device only applies "
                         "it with 3 subpixel bits. Default: leave the preset value")
    ap.add_argument("--confidence", type=int, default=None,
                    help="confidence threshold 0..255 (lower = stricter: fewer outliers, "
                         "more holes). Default: leave the preset value")

    # --- throughput tuning (found with the benchmark script) ---
    ap.add_argument("--no-tuning", action="store_true",
                    help="do not apply --xlink-chunk 0 / --pool 16")
    ap.add_argument("--xlink-chunk", type=int, default=0,
                    help="XLink packet chunk size in bytes; 0 disables chunking")
    ap.add_argument("--pool", type=int, default=16,
                    help="frames in each mono camera's RAW and ISP buffer pools")

    # --- optional extras (GUI mode only) ---
    ap.add_argument("--rgb", action="store_true", help="also stream the RGB camera")
    ap.add_argument("--rgb-fps", type=int, default=60,
                    help="RGB FPS; kept low because RGB costs USB bandwidth and ISP time")
    ap.add_argument("--plot3d", action="store_true", help="live matplotlib 3D view (slow)")
    ap.add_argument("--plot-every", type=int, default=30,
                    help="redraw the 3D plot every N depth frames")

    # --- measurement ---
    ap.add_argument("--baseline", type=float, default=578.0,
                    help="distance to the empty belt [mm]; use 't' to tare")
    ap.add_argument("--threshold", type=float, default=15.0,
                    help="min height above the belt to count as load [mm]")
    ap.add_argument("--roi-frac", type=float, default=0.25,
                    help="square ROI side as a fraction of the depth frame width")

    # --- headless / run control ---
    ap.add_argument("--headless", action="store_true",
                    help="no windows/visualization; print status to the console")
    ap.add_argument("--print-interval", type=float, default=1.0,
                    help="seconds between console status lines in headless mode")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after this many seconds (0 = run until Ctrl+C / 'q')")
    ap.add_argument("--tare-on-start", action="store_true",
                    help="set the baseline from the median ROI depth of the first "
                         "--tare-frames frames (belt must be empty)")
    ap.add_argument("--tare-frames", type=int, default=30,
                    help="number of frames used by --tare-on-start")
    ap.add_argument("--display-fps", type=float, default=60.0,
                    help="GUI refresh rate. The volume math still runs on every frame; "
                         "only drawing/imshow/waitKey are limited to this rate")
    return ap.parse_args()


# ---- main ------------------------------------------------------------------
def main():
    args = parse_args()
    size = RES[args.res]
    baseline = args.baseline        # mm; may be replaced by a tare during the run
    headless = args.headless

    # Windows and the RGB stream are GUI features; drop them rather than fail.
    if headless and (args.rgb or args.plot3d):
        print("note: --rgb and --plot3d are ignored in --headless mode")
        args.rgb = False
        args.plot3d = False

    # matplotlib is imported only when asked for (slow import, extra dependency).
    plt = ax = X = Y = None
    if args.plot3d:
        import matplotlib.pyplot as plt
        plt.ion()
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")

    with dai.Pipeline() as pipeline:
        # Calibration override. Must happen before the nodes are created, as in the
        # official calibration_load example. A bad path or file raises on purpose.
        calib_override = None
        if args.calib:
            calib_override = dai.CalibrationHandler(args.calib)
            pipeline.setCalibrationData(calib_override)

        # ---- 1. Build the pipeline (nothing runs until pipeline.start()) ----
        if not args.no_tuning:
            try:
                pipeline.setXLinkChunkSize(args.xlink_chunk)
            except Exception as e:
                print(f"warning: setXLinkChunkSize failed: {e}")

        # Mono pair. Sensor FPS is set explicitly, same as in the benchmark.
        left = build_cam(pipeline, dai.CameraBoardSocket.CAM_B, size, args.fps)
        right = build_cam(pipeline, dai.CameraBoardSocket.CAM_C, size, args.fps)
        if not args.no_tuning:
            apply_pool(left, args.pool)
            apply_pool(right, args.pool)
        left_out = left.requestOutput(size, type=dai.ImgFrame.Type.GRAY8, fps=args.fps)
        right_out = right.requestOutput(size, type=dai.ImgFrame.Type.GRAY8, fps=args.fps)

        # Stereo: start from a preset, then apply the command-line overrides on top.
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(getattr(dai.node.StereoDepth.PresetMode,
                                               PRESETS[args.preset]))
        sub_bits = 0 if args.no_subpixel else args.subpixel_bits   # None = preset default
        if sub_bits == 0:
            stereo.setSubpixel(False)
        elif sub_bits is not None:
            stereo.setSubpixel(True)
            try:
                stereo.setSubpixelFractionalBits(sub_bits)
            except Exception:
                # some builds only expose the setter on the config object
                try:
                    stereo.initialConfig.setSubpixelFractionalBits(sub_bits)
                except Exception as e:
                    print(f"warning: could not set subpixel fractional bits: {e}")
        try:
            if args.disparity_shift is not None:
                stereo.initialConfig.setDisparityShift(args.disparity_shift)
            if args.confidence is not None:
                stereo.initialConfig.setConfidenceThreshold(args.confidence)
            if args.median is not None:
                kernels = {0: dai.MedianFilter.MEDIAN_OFF,
                           3: dai.MedianFilter.KERNEL_3x3,
                           5: dai.MedianFilter.KERNEL_5x5,
                           7: dai.MedianFilter.KERNEL_7x7}
                stereo.initialConfig.setMedianFilter(kernels[args.median])
        except Exception as e:
            print(f"warning: could not apply a stereo override: {e}")
        describe_stereo(stereo)
        left_out.link(stereo.left)
        right_out.link(stereo.right)
        # Non-blocking: when the host is slow the OLDEST queued frame is dropped instead of
        # stalling the camera. The loop below drains this queue every pass.
        q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

        q_rgb = None
        if args.rgb:
            rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            # BGR888i (interleaved) is what getCvFrame() needs to give an OpenCV image.
            rgb_out = rgb.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888i,
                                        fps=args.rgb_fps)
            q_rgb = rgb_out.createOutputQueue(maxSize=2, blocking=False)

        # ---- 2. Start the device and configure it ----
        print("Starting pipeline...")
        pipeline.start()
        device = pipeline.getDefaultDevice()
        device.setIrLaserDotProjectorIntensity(1.0)   # 0..1
        device.setIrFloodLightIntensity(0.0)          # flood off, projector only
        # Calibration object for the focal-length fallback and the startup print.
        calib = calib_override if calib_override is not None else device.readCalibration()
        describe_calibration(calib, f"file {args.calib}" if calib_override is not None
                             else "device EEPROM", size)
        print(f"mono {args.res}P @ {args.fps} fps | preset {args.preset} | "
              f"subpixel {'preset default' if sub_bits is None else ('OFF' if sub_bits == 0 else f'{sub_bits} bits')} | "
              f"tuning {'OFF' if args.no_tuning else f'xlink-chunk {args.xlink_chunk}, pool {args.pool}'}"
              f" | rgb {'ON' if args.rgb else 'OFF'} | {'HEADLESS' if headless else 'GUI'}")

        window = "OAK-D Conveyor Dashboard"
        if not headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            # RGB (if on) is shown side by side with depth, so the window is twice as wide.
            cv2.resizeWindow(window, DISPLAY_W * (2 if args.rgb else 1), DISPLAY_H)

        # ---- 3. Loop state ----
        roi = None                      # latest ROI view; None until the first frame ('t' checks this)
        fx = fy = None                  # focal lengths [px]; read once from the first depth frame
        volume_cm3, holes_pct = 0.0, 0.0    # results for the most recent frame
        fps, frames_in_window, total_frames = 0.0, 0, 0   # FPS = frames pulled per second
        t_start = t_window = t_print = time.monotonic()   # run start, FPS window start, last status print
        latest_rgb = latest_depth = None    # last rendered images (GUI mode)
        render_interval = 1.0 / max(args.display_fps, 1e-3)   # seconds between redraws
        last_render = 0.0
        legend_layer = legend_mask = legend_box = None   # legend is drawn once, then copied
        # Newest processed frame's ROI geometry and load mask. They live here (not reset
        # each pass) so a redraw that happens on a pass without new frames still has them.
        geom = valid = None
        # True while a processed frame has not been drawn yet. A render tick is only used
        # up when there really is something new to draw (this was the cause of the lag).
        draw_pending = False
        last_plot_total = 0             # total_frames at the last 3D plot redraw

        # --tare-on-start: collect the median ROI depth of N frames, then average them.
        tare_samples = []
        tare_left = args.tare_frames if args.tare_on_start else 0
        if tare_left:
            print(f"Taring from the first {tare_left} frames (belt must be empty)...")

        if headless:
            print("Running headless. Ctrl+C to stop.")
        else:
            print("Running. 'q' = quit, 't' = tare (empty belt).")

        # ---- 4. Main loop ----
        try:
            while pipeline.isRunning():
                # Take everything waiting in the queue so no frame is skipped by the math.
                frames = []
                while (m := q_depth.tryGet()) is not None:
                    frames.append(m)
                in_rgb = q_rgb.tryGet() if q_rgb is not None else None

                now = time.monotonic()
                if now - t_window > 1:          # refresh the FPS figure about once a second
                    fps = frames_in_window / (now - t_window)
                    frames_in_window, t_window = 0, now

                if args.duration > 0 and now - t_start >= args.duration:
                    print(f"Duration {args.duration:.0f}s reached.")
                    break

                # ---- depth + volume (every frame) ----
                # `depth`, `w`, `h`, `roi`, `valid`, `geom` and the two result values
                # belong to the LAST frame processed; drawing uses them.
                for in_depth in frames:
                    frames_in_window += 1
                    total_frames += 1
                    depth = in_depth.getFrame()           # uint16, mm
                    h, w = depth.shape

                    if fx is None:      # first frame only: read focal length and report it
                        fx, fy, src = get_focal_px(in_depth, calib, w, size)
                        print(f"depth frame {w}x{h} | fx={fx:.1f}px fy={fy:.1f}px ({src}) | "
                              f"footprint at baseline = {baseline / fx:.2f} mm/px")

                    roi, valid, volume_cm3, holes_pct, geom = process_depth(
                        depth, baseline, args.threshold, args.roi_frac, fx, fy)

                    # Auto-tare: take the median of the valid ROI pixels of each frame, and
                    # after N frames set the baseline to the median of those medians.
                    if tare_left:
                        v = roi[roi > 0]
                        if v.size:
                            tare_samples.append(float(np.median(v)))
                            tare_left -= 1
                            if tare_left == 0:
                                baseline = float(np.median(tare_samples))
                                print(f"[TARE] baseline = {baseline:.1f} mm "
                                      f"({len(tare_samples)} frames)")
                if frames:
                    draw_pending = True

                # ---- visualization (GUI only, newest frame, limited to --display-fps) ----
                # A render tick is consumed only when a new frame is waiting to be drawn.
                do_render = ((not headless) and draw_pending
                             and (now - last_render >= render_interval))
                if do_render:
                    last_render = now
                    draw_pending = False
                if do_render:
                    cx, cy, half = geom
                    # Mask window: white = pixels counted as load (what the volume uses).
                    mask = cv2.resize((valid * 255).astype(np.uint8), (400, 400),
                                      interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("Valid Pixels Mask (White = Load)", mask)

                    # Depth image: clip to 0..MAX_DISTANCE_MM, scale to 0..255, invert so
                    # near = red and far = blue, and paint invalid pixels (0 mm) black.
                    vis = np.clip(depth, 0, MAX_DISTANCE_MM)
                    vis = (vis * (255.0 / MAX_DISTANCE_MM)).astype(np.uint8)
                    vis = 255 - vis
                    vis[depth == 0] = 0
                    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                    vis = cv2.resize(vis, (DISPLAY_W, DISPLAY_H))

                    # ROI rectangle, converted from depth-frame pixels to display pixels.
                    sx, sy = DISPLAY_W / w, DISPLAY_H / h
                    cv2.rectangle(vis, (int((cx - half) * sx), int((cy - half) * sy)),
                                  (int((cx + half) * sx), int((cy + half) * sy)),
                                  (255, 255, 255), 2)
                    cv2.putText(vis, f"Depth {w}x{h} | FPS: {fps:.1f}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(vis, f"Volume: {volume_cm3:.2f} cm^3", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    cv2.putText(vis, f"ROI holes: {holes_pct:.0f}%  baseline {baseline:.0f} mm",
                                (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    # Legend: rendered once on a black canvas, cropped to its bounding box,
                    # then copied where it has pixels. Masking only the small box is much
                    # cheaper than masking the whole 800x600 image on every redraw.
                    if legend_layer is None:
                        full_layer = draw_detailed_legend(np.zeros_like(vis),
                                                          MAX_DISTANCE_MM, 500)
                        full_mask = full_layer.any(axis=2)
                        ys, xs = np.nonzero(full_mask)
                        ly0, ly1 = int(ys.min()), int(ys.max()) + 1
                        lx0, lx1 = int(xs.min()), int(xs.max()) + 1
                        legend_box = (ly0, ly1, lx0, lx1)
                        legend_layer = full_layer[ly0:ly1, lx0:lx1].copy()
                        legend_mask = full_mask[ly0:ly1, lx0:lx1].copy()
                    ly0, ly1, lx0, lx1 = legend_box
                    vis_box = vis[ly0:ly1, lx0:lx1]          # view, edits go into vis
                    vis_box[legend_mask] = legend_layer[legend_mask]
                    latest_depth = vis

                    # 3D plot every --plot-every frames (counted since the last redraw, so
                    # it still fires when the redraw lands on a pass without new frames).
                    if args.plot3d and total_frames - last_plot_total >= args.plot_every:
                        last_plot_total = total_frames
                        if X is None or X.shape != roi.shape:     # rebuild grid only on size change
                            X, Y = np.meshgrid(np.arange(roi.shape[1]), np.arange(roi.shape[0]))
                        ax.clear()
                        top = baseline + 500
                        ax.set_zlim(0, top)
                        ax.set_axis_off()
                        ax.set_title(f"Live 3D Profile (Vol: {volume_cm3:.0f} cm3)")
                        ax.plot_surface(X, Y, top - np.clip(roi, 0, top),
                                        cmap="viridis", edgecolor="none")
                        plt.pause(0.001)

                # ---- rgb (optional, GUI only) ----
                if in_rgb is not None:
                    frame = in_rgb.getCvFrame()
                    latest_rgb = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
                    cv2.putText(latest_rgb, "RGB Feed", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # ---- output ----
                if headless:
                    if now - t_print >= args.print_interval:
                        t_print = now
                        print(f"t={now - t_start:7.1f}s  fps={fps:5.1f}  "
                              f"volume={volume_cm3:9.2f} cm^3  holes={holes_pct:4.1f}%  "
                              f"baseline={baseline:.1f} mm  frames={total_frames}")
                    if not frames:
                        time.sleep(0.0005)   # avoid a 100% CPU spin while waiting
                    continue                 # headless never reaches the window/key code below

                if not do_render:
                    if not frames:
                        time.sleep(0.0005)   # idle wait, same as headless
                    continue                 # imshow/waitKey only run on render ticks

                if latest_depth is not None:
                    if args.rgb and latest_rgb is not None:
                        cv2.imshow(window, np.hstack((latest_rgb, latest_depth)))
                    elif not args.rgb:
                        cv2.imshow(window, latest_depth)

                # Keys are only polled on render ticks, so they react at --display-fps.
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
                elif key == ord("t"):
                    # Manual tare: median of the valid pixels in the current ROI.
                    if roi is None:
                        print("[TARE] No depth frame yet.")
                    else:
                        v = roi[roi > 0]
                        if v.size:
                            baseline = float(np.median(v))
                            print(f"[TARE] baseline = {baseline:.1f} mm")
                        else:
                            print("[TARE] Failed: no valid depth in ROI.")
        except KeyboardInterrupt:
            pass
        finally:
            print("Stopping...")
            elapsed = time.monotonic() - t_start
            if elapsed > 0 and total_frames:
                # Whole-run average (includes startup), so it reads slightly below the
                # steady-state FPS shown while running.
                print(f"Average over run: {total_frames / elapsed:.1f} fps "
                      f"({total_frames} frames in {elapsed:.1f} s)")
            if plt is not None:
                plt.ioff()
                plt.close("all")
            if not headless:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()