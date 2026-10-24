#!/usr/bin/env python3
"""
Conveyor volume scanner - OAK-D Pro (RVC2), DepthAI v3.

Changes vs. the previous version:
  * --headless: no windows, no colormaps, no legend, no waitKey. Only the ROI volume
    math runs. Status is printed to the console every --print-interval seconds.
    --rgb and --plot3d are ignored in headless mode.
  * In headless mode the loop does not use keys, so:
      - quit with Ctrl+C, or set --duration SECONDS to stop automatically
      - use --tare-on-start (belt must be empty) or --baseline to set the baseline
  * Every frame waiting in the queue is now processed for the volume math (before, only
    one frame per loop pass was taken). Only the newest one is drawn in GUI mode.
  * Volume math moved into process_depth() so both modes share it.

Usage:
    python conveyor_volume_scanner.py
    python conveyor_volume_scanner.py --fps 90
    python conveyor_volume_scanner.py --headless --fps 120 --preset fast_density --no-subpixel
    python conveyor_volume_scanner.py --headless --tare-on-start --duration 60

Keys (GUI mode): q = quit, t = tare (set baseline from the median of the ROI; belt must be empty).
"""
import argparse
import time

import cv2
import numpy as np
import depthai as dai

RES = {400: (640, 400), 800: (1280, 800)}
PRESETS = {
    "fast_accuracy": "FAST_ACCURACY",
    "fast_density": "FAST_DENSITY",
    "robotics": "ROBOTICS",
    "default": "DEFAULT",
    "high_detail": "HIGH_DETAIL",
}

DISPLAY_W, DISPLAY_H = 800, 600
MAX_DISTANCE_MM = 3000


# ---- helpers ---------------------------------------------------------------
def draw_detailed_legend(frame, max_distance_mm, step_size=500):
    frame_h, frame_w = frame.shape[:2]
    legend_w, legend_h = 35, int(frame_h * 0.85)
    start_x = frame_w - legend_w - 110
    start_y = (frame_h - legend_h) // 2

    gradient_1d = np.linspace(255, 0, legend_h).astype(np.uint8)
    gradient_2d = np.tile(gradient_1d, (legend_w, 1)).T
    legend_colored = cv2.applyColorMap(gradient_2d, cv2.COLORMAP_JET)
    frame[start_y:start_y + legend_h, start_x:start_x + legend_w] = legend_colored
    cv2.rectangle(frame, (start_x, start_y), (start_x + legend_w, start_y + legend_h),
                  (255, 255, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    num_steps = max_distance_mm // step_size
    for i in range(num_steps + 1):
        val = i * step_size
        y_pos = start_y + int((i / num_steps) * legend_h)
        cv2.line(frame, (start_x + legend_w, y_pos), (start_x + legend_w + 10, y_pos),
                 (255, 255, 255), 2)
        text_y = y_pos + 5
        if i == 0:
            text_y += 8
        elif i == num_steps:
            text_y -= 5
        text = f"{val}mm" if i < num_steps else f">{val}mm"
        cv2.putText(frame, text, (start_x + legend_w + 15, text_y), font, 0.6,
                    (255, 255, 255), 2)
    return frame


def build_cam(pipeline, socket, size, fps):
    """Camera node with explicit sensor mode/FPS (same pattern as the benchmark)."""
    cam = pipeline.create(dai.node.Camera)
    try:
        return cam.build(socket, sensorResolution=size, sensorFps=fps)
    except TypeError:
        return cam.build(socket, sensorFps=fps)


def apply_pool(cam, pool):
    for setter in ("setRawNumFramesPool", "setIspNumFramesPool"):
        try:
            getattr(cam, setter)(pool)
        except Exception as e:
            print(f"  warning: {setter}({pool}) failed: {e}")


def get_focal_px(depth_msg, calib, depth_w, req_size):
    """(fx, fy) in pixels for the depth frame actually received."""
    try:
        K = depth_msg.getTransformation().getIntrinsicMatrix()
        return float(K[0][0]), float(K[1][1]), "frame transformation"
    except Exception:
        pass
    # Fallback: calibration at the requested size, scaled to the depth frame width
    # (depth is aligned to the right camera, CAM_C, by default).
    K = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, req_size[0], req_size[1])
    s = depth_w / req_size[0]
    return float(K[0][0]) * s, float(K[1][1]) * s, "calibration (scaled)"


def process_depth(depth, baseline, threshold, roi_frac, fx, fy):
    """ROI crop + volume. Shared by GUI and headless mode.

    Depth is Z (distance along the optical axis). A pixel at depth Zs covers
    (Zs/fx)*(Zs/fy) mm^2 of the object's top surface. With the camera looking straight
    down and vertical object sides, the column below that patch has the same
    cross-section, so: volume = sum(height * patch area).

    Returns roi (uint16 view), valid mask, volume [cm^3], holes [%], (cx, cy, half).
    """
    h, w = depth.shape
    side = max(8, int(w * roi_frac)) // 2 * 2
    half = side // 2
    cy, cx = h // 2, w // 2
    roi = depth[cy - half:cy + half, cx - half:cx + half]

    z = roi.astype(np.float32)
    valid = (z > 0) & (z < (baseline - threshold))
    zv = z[valid]
    vol_mm3 = float(np.sum((baseline - zv) * zv ** 2) / (fx * fy))
    holes_pct = 100.0 * float(np.mean(z == 0))
    return roi, valid, vol_mm3 / 1000.0, holes_pct, (cx, cy, half)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", type=int, choices=[400, 800], default=400,
                    help="mono resolution (800P has MinZ ~70 cm without extended disparity)")
    ap.add_argument("--fps", type=int, default=60, help="requested mono/stereo FPS")
    ap.add_argument("--preset", choices=list(PRESETS), default="fast_accuracy")
    ap.add_argument("--no-subpixel", action="store_true",
                    help="disable subpixel (integer disparity only)")
    ap.add_argument("--no-tuning", action="store_true",
                    help="do not apply --xlink-chunk 0 / --pool 16")
    ap.add_argument("--xlink-chunk", type=int, default=0)
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--rgb", action="store_true", help="also stream the RGB camera")
    ap.add_argument("--rgb-fps", type=int, default=15)
    ap.add_argument("--plot3d", action="store_true", help="live matplotlib 3D view (slow)")
    ap.add_argument("--plot-every", type=int, default=30)
    ap.add_argument("--baseline", type=float, default=578.0,
                    help="distance to the empty belt [mm]; use 't' to tare")
    ap.add_argument("--threshold", type=float, default=15.0,
                    help="min height above the belt to count as load [mm]")
    ap.add_argument("--roi-frac", type=float, default=0.20,
                    help="square ROI side as a fraction of the depth frame width")
    # headless
    ap.add_argument("--headless", action="store_true",
                    help="no windows/visualization; print status to the console")
    ap.add_argument("--print-interval", type=float, default=1.0,
                    help="seconds between console status lines in headless mode")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after this many seconds (0 = run until Ctrl+C / 'q')")
    ap.add_argument("--tare-on-start", action="store_true",
                    help="set the baseline from the median ROI depth of the first "
                         "--tare-frames frames (belt must be empty)")
    ap.add_argument("--tare-frames", type=int, default=30)
    return ap.parse_args()


# ---- main ------------------------------------------------------------------
def main():
    args = parse_args()
    size = RES[args.res]
    baseline = args.baseline
    headless = args.headless

    if headless and (args.rgb or args.plot3d):
        print("note: --rgb and --plot3d are ignored in --headless mode")
        args.rgb = False
        args.plot3d = False

    plt = ax = X = Y = None
    if args.plot3d:
        import matplotlib.pyplot as plt
        plt.ion()
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")

    with dai.Pipeline() as pipeline:
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

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(getattr(dai.node.StereoDepth.PresetMode,
                                               PRESETS[args.preset]))
        if args.no_subpixel:
            stereo.setSubpixel(False)
        left_out.link(stereo.left)
        right_out.link(stereo.right)
        q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

        q_rgb = None
        if args.rgb:
            rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            rgb_out = rgb.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888i,
                                        fps=args.rgb_fps)
            q_rgb = rgb_out.createOutputQueue(maxSize=2, blocking=False)

        print("Starting pipeline...")
        pipeline.start()
        device = pipeline.getDefaultDevice()
        device.setIrLaserDotProjectorIntensity(1.0)   # 0..1
        device.setIrFloodLightIntensity(0.0)          # flood off, projector only
        calib = device.readCalibration()
        print(f"mono {args.res}P @ {args.fps} fps | preset {args.preset} | "
              f"subpixel {'OFF' if args.no_subpixel else 'preset default'} | "
              f"tuning {'OFF' if args.no_tuning else f'xlink-chunk {args.xlink_chunk}, pool {args.pool}'}"
              f" | rgb {'ON' if args.rgb else 'OFF'} | {'HEADLESS' if headless else 'GUI'}")

        window = "OAK-D Conveyor Dashboard"
        if not headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, DISPLAY_W * (2 if args.rgb else 1), DISPLAY_H)

        roi = None
        fx = fy = None
        volume_cm3, holes_pct = 0.0, 0.0
        fps, frames_in_window, total_frames = 0.0, 0, 0
        t_start = t_window = t_print = time.monotonic()
        latest_rgb = latest_depth = None

        tare_samples = []
        tare_left = args.tare_frames if args.tare_on_start else 0
        if tare_left:
            print(f"Taring from the first {tare_left} frames (belt must be empty)...")

        if headless:
            print("Running headless. Ctrl+C to stop.")
        else:
            print("Running. 'q' = quit, 't' = tare (empty belt).")

        try:
            while pipeline.isRunning():
                # Take everything waiting in the queue so no frame is skipped by the math.
                frames = []
                while (m := q_depth.tryGet()) is not None:
                    frames.append(m)
                in_rgb = q_rgb.tryGet() if q_rgb is not None else None

                now = time.monotonic()
                if now - t_window > 1:
                    fps = frames_in_window / (now - t_window)
                    frames_in_window, t_window = 0, now

                if args.duration > 0 and now - t_start >= args.duration:
                    print(f"Duration {args.duration:.0f}s reached.")
                    break

                # ---- depth + volume (every frame) ----
                geom = valid = None
                for in_depth in frames:
                    frames_in_window += 1
                    total_frames += 1
                    depth = in_depth.getFrame()           # uint16, mm
                    h, w = depth.shape

                    if fx is None:
                        fx, fy, src = get_focal_px(in_depth, calib, w, size)
                        print(f"depth frame {w}x{h} | fx={fx:.1f}px fy={fy:.1f}px ({src}) | "
                              f"footprint at baseline = {baseline / fx:.2f} mm/px")

                    roi, valid, volume_cm3, holes_pct, geom = process_depth(
                        depth, baseline, args.threshold, args.roi_frac, fx, fy)

                    if tare_left:
                        v = roi[roi > 0]
                        if v.size:
                            tare_samples.append(float(np.median(v)))
                            tare_left -= 1
                            if tare_left == 0:
                                baseline = float(np.median(tare_samples))
                                print(f"[TARE] baseline = {baseline:.1f} mm "
                                      f"({len(tare_samples)} frames)")

                # ---- visualization (GUI only, newest frame only) ----
                if not headless and frames:
                    cx, cy, half = geom
                    mask = cv2.resize((valid * 255).astype(np.uint8), (400, 400),
                                      interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("Valid Pixels Mask (White = Load)", mask)

                    vis = np.clip(depth, 0, MAX_DISTANCE_MM)
                    vis = (vis * (255.0 / MAX_DISTANCE_MM)).astype(np.uint8)
                    vis = 255 - vis
                    vis[depth == 0] = 0
                    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                    vis = cv2.resize(vis, (DISPLAY_W, DISPLAY_H))

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
                    latest_depth = draw_detailed_legend(vis, MAX_DISTANCE_MM, 500)

                    if args.plot3d and total_frames % args.plot_every < len(frames):
                        if X is None or X.shape != roi.shape:
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
                    continue

                if latest_depth is not None:
                    if args.rgb and latest_rgb is not None:
                        cv2.imshow(window, np.hstack((latest_rgb, latest_depth)))
                    elif not args.rgb:
                        cv2.imshow(window, latest_depth)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
                elif key == ord("t"):
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
                print(f"Average over run: {total_frames / elapsed:.1f} fps "
                      f"({total_frames} frames in {elapsed:.1f} s)")
            if plt is not None:
                plt.ioff()
                plt.close("all")
            if not headless:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()