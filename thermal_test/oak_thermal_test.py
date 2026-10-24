#!/usr/bin/env python3
"""
OAK-D Pro thermal evaluation - DepthAI v3.

Separates the heat contribution of the IR dot projector from the VPU stereo-depth
compute, over a 10-run matrix (RAW mono pair / STEREO depth  x  IR 0, .25, .5, .75, 1.0).
Camera settings (400P, 60 FPS, accuracy preset, xlink-chunk 0, pool 16, calibration file)
are the defaults of the conveyor volume scanner script.

TEN LAUNCH COMMANDS (one run per launch; each waits until chip <= 40.0 C first)
    python oak_thermal_test.py --mode raw    --ir 0.00     # R01_RAW_IR000
    python oak_thermal_test.py --mode raw    --ir 0.25     # R02_RAW_IR025
    python oak_thermal_test.py --mode raw    --ir 0.50     # R03_RAW_IR050
    python oak_thermal_test.py --mode raw    --ir 0.75     # R04_RAW_IR075
    python oak_thermal_test.py --mode raw    --ir 1.00     # R05_RAW_IR100
    python oak_thermal_test.py --mode stereo --ir 0.00     # R06_STR_IR000
    python oak_thermal_test.py --mode stereo --ir 0.25     # R07_STR_IR025
    python oak_thermal_test.py --mode stereo --ir 0.50     # R08_STR_IR050
    python oak_thermal_test.py --mode stereo --ir 0.75     # R09_STR_IR075
    python oak_thermal_test.py --mode stereo --ir 1.00     # R10_STR_IR100
  or the whole matrix in one process:
    python oak_thermal_test.py --all

OUTPUT (in --outdir, default ./thermal_results)
    thermal_timeseries_log_<RUN_ID>.csv   one per run, 5 s sampling
    thermal_summary_results.csv           one line appended per run

DEFINITIONS
    rate_of_change_c_min : instantaneous (T_now - T_prev) / 5 s, in C/min (first row 0.0)
    steady state         : |linear-regression slope| over the last 120 s (25 samples)
                           < 0.2 C/min. Not evaluated until the window is full.
    spike                : T_now - T_prev_5s > 2.0 C
    cutoff               : T_now >= 80.0 C  -> run aborted, UNSTABLE
    is_stable            : steady state reached AND cutoff not hit
    Run also ends (not stable) on --max-duration, Ctrl+C, or pipeline stop.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import depthai as dai

# ---- test constants (from the specification) --------------------------------
T_INIT_MAX_C = 50.0        # must cool to this before each run
T_CUTOFF_C = 80.0          # instant abort
SLOPE_STEADY_C_MIN = 0.2   # steady-state criterion
WINDOW_S = 120             # rolling window for regression
SAMPLE_S = 5               # sampling interval
SPIKE_C = 2.0              # jump within one sample interval
WINDOW_N = WINDOW_S // SAMPLE_S + 1   # 25 samples span 120 s

TS_FIELDS = ["timestamp_s", "run_id", "mode", "ir_level", "temp_chip_c",
             "rate_of_change_c_min", "spike_flag"]
SUM_FIELDS = ["run_id", "mode", "ir_level", "t_init_c", "t_final_c", "t_max_c",
              "duration_s", "steady_state_reached", "is_stable", "spike_detected"]

# ---- test matrix --------------------------------------------------------------
IR_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
MATRIX = []
for _i, _ir in enumerate(IR_LEVELS):
    MATRIX.append((f"R{_i + 1:02d}_RAW_IR{int(round(_ir * 100)):03d}", "RAW", _ir))
for _i, _ir in enumerate(IR_LEVELS):
    MATRIX.append((f"R{_i + 6:02d}_STR_IR{int(round(_ir * 100)):03d}", "STEREO", _ir))

# ---- camera settings shared with the volume scanner ---------------------------
RES = {400: (640, 400), 800: (1280, 800)}
PRESETS = {
    "fast_accuracy": "FAST_ACCURACY", "fast_density": "FAST_DENSITY",
    "robotics": "ROBOTICS", "default": "DEFAULT", "high_detail": "HIGH_DETAIL",
    "accuracy": "ACCURACY", "density": "DENSITY",
}
PRESETS = {k: v for k, v in PRESETS.items() if hasattr(dai.node.StereoDepth.PresetMode, v)}


def run_id_for(mode, ir):
    """Matrix run id if (mode, ir) is in the matrix, else a CUSTOM_ id."""
    for rid, m, level in MATRIX:
        if m == mode and abs(level - ir) < 1e-6:
            return rid
    return f"CUSTOM_{'RAW' if mode == 'RAW' else 'STR'}_IR{int(round(ir * 100)):03d}"


# ---- device helpers -------------------------------------------------------------
def read_chip_temp(device, retries=3):
    """Average chip temperature [C]; falls back to the mean of the sub-sensors."""
    last = None
    for _ in range(retries):
        try:
            ct = device.getChipTemperature()
            avg = float(ct.average)
            if avg > 0:
                return avg
            parts = [float(getattr(ct, n)) for n in ("css", "mss", "upa", "dss")
                     if hasattr(ct, n)]
            parts = [p for p in parts if p > 0]
            if parts:
                return float(np.mean(parts))
        except Exception as e:
            last = e
        time.sleep(0.2)
    raise RuntimeError(f"could not read chip temperature: {last}")


def build_cam(pipeline, socket, size, fps):
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


def configure_stereo(stereo, args):
    stereo.setDefaultProfilePreset(getattr(dai.node.StereoDepth.PresetMode,
                                           PRESETS[args.preset]))
    sub_bits = 0 if args.no_subpixel else args.subpixel_bits
    if sub_bits == 0:
        stereo.setSubpixel(False)
    elif sub_bits is not None:
        stereo.setSubpixel(True)
        try:
            stereo.setSubpixelFractionalBits(sub_bits)
        except Exception:
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
            kernels = {0: dai.MedianFilter.MEDIAN_OFF, 3: dai.MedianFilter.KERNEL_3x3,
                       5: dai.MedianFilter.KERNEL_5x5, 7: dai.MedianFilter.KERNEL_7x7}
            stereo.initialConfig.setMedianFilter(kernels[args.median])
    except Exception as e:
        print(f"warning: could not apply a stereo override: {e}")


def build_pipeline(pipeline, args, mode):
    """RAW: mono pair -> host queues. STEREO: mono pair -> StereoDepth -> host queue.
    Returns the queues that must be drained so USB/XLink load stays realistic."""
    size = RES[args.res]
    if not args.no_tuning:
        try:
            pipeline.setXLinkChunkSize(args.xlink_chunk)
        except Exception as e:
            print(f"warning: setXLinkChunkSize failed: {e}")
    left = build_cam(pipeline, dai.CameraBoardSocket.CAM_B, size, args.fps)
    right = build_cam(pipeline, dai.CameraBoardSocket.CAM_C, size, args.fps)
    if not args.no_tuning:
        apply_pool(left, args.pool)
        apply_pool(right, args.pool)
    left_out = left.requestOutput(size, type=dai.ImgFrame.Type.GRAY8, fps=args.fps)
    right_out = right.requestOutput(size, type=dai.ImgFrame.Type.GRAY8, fps=args.fps)

    if mode == "RAW":
        return [left_out.createOutputQueue(maxSize=4, blocking=False),
                right_out.createOutputQueue(maxSize=4, blocking=False)]
    stereo = pipeline.create(dai.node.StereoDepth)
    configure_stereo(stereo, args)
    left_out.link(stereo.left)
    right_out.link(stereo.right)
    return [stereo.depth.createOutputQueue(maxSize=4, blocking=False)]


# ---- UI helpers -----------------------------------------------------------------
def bar(frac, width=28):
    frac = min(max(frac, 0.0), 1.0)
    n = int(frac * width)
    return "#" * n + "-" * (width - n)


def status(line):
    sys.stdout.write("\r" + line.ljust(118)[:118])
    sys.stdout.flush()


def wait_for_cooldown(device, run_id, t_max, timeout, poll=5.0):
    """Block until chip temperature <= t_max. Device is connected but idle (no pipeline)."""
    t0 = time.monotonic()
    first = read_chip_temp(device)
    start_T = first
    T = first
    while T > t_max:
        waited = time.monotonic() - t0
        if waited > timeout:
            print()
            raise RuntimeError(f"{run_id}: still {T:.1f} C after {timeout:.0f} s of cooling "
                               f"(limit {t_max} C). Check ambient temperature / raise "
                               f"--cooldown-timeout.")
        span = max(start_T - t_max, 1e-6)
        status(f"[{run_id}] COOLING [{bar((start_T - T) / span)}] "
               f"T={T:5.1f} C -> <= {t_max:.1f} C | waited {waited:5.0f} s")
        time.sleep(poll)
        T = read_chip_temp(device)
    print()
    print(f"[{run_id}] start temperature OK: {T:.1f} C (waited {time.monotonic() - t0:.0f} s)")
    return T


def append_summary(path, row):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUM_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def tf(b):
    return "TRUE" if b else "FALSE"


# ---- one run ----------------------------------------------------------------------
def run_test(args, run_id, mode, ir):
    os.makedirs(args.outdir, exist_ok=True)
    ts_path = os.path.join(args.outdir, f"thermal_timeseries_log_{run_id}.csv")
    sum_path = os.path.join(args.outdir, "thermal_summary_results.csv")
    mode_label = "RAW" if mode == "RAW" else "STEREO"
    print(f"\n=== {run_id} | {mode_label} | IR {ir:.2f} ===")

    calib_override = None
    if args.calib:
        if os.path.exists(args.calib):
            calib_override = dai.CalibrationHandler(args.calib)
        else:
            print(f"warning: calibration file '{args.calib}' not found, using device calibration")

    # 1. connect and cool down (device booted, no pipeline, IR off)
    device = dai.Device()
    reason = "UNKNOWN"
    t_init = t_final = t_max = None
    last_ts = 0.0
    steady = cutoff = spike_any = False
    ts_file = None
    try:
        wait_for_cooldown(device, run_id, T_INIT_MAX_C, args.cooldown_timeout)

        # 2. build + start pipeline on the already-connected device
        with dai.Pipeline(device) as pipeline:
            if calib_override is not None:
                pipeline.setCalibrationData(calib_override)
            queues = build_pipeline(pipeline, args, mode)
            pipeline.start()
            try:
                device.setIrLaserDotProjectorIntensity(float(ir))
                device.setIrFloodLightIntensity(0.0)
            except Exception as e:
                print(f"warning: could not set IR: {e}")

            ts_file = open(ts_path, "w", newline="")
            writer = csv.writer(ts_file)
            writer.writerow(TS_FIELDS)

            win_t, win_T = [], []
            T_prev = None
            k = 0
            t0 = time.monotonic()
            last_refresh = 0.0
            slope = None
            T = None
            try:
                while True:
                    now = time.monotonic()
                    # keep USB/XLink traffic realistic: drain host queues
                    for q in queues:
                        while q.tryGet() is not None:
                            pass

                    if now - t0 >= k * SAMPLE_S:
                        ts = k * SAMPLE_S
                        T = read_chip_temp(device)
                        rate = 0.0 if T_prev is None else (T - T_prev) / SAMPLE_S * 60.0
                        spike = T_prev is not None and (T - T_prev) > SPIKE_C
                        spike_any |= spike
                        T_prev = T
                        if t_init is None:
                            t_init = T
                        t_max = T if t_max is None else max(t_max, T)
                        t_final, last_ts = T, float(ts)

                        win_t.append(ts)
                        win_T.append(T)
                        while win_t[-1] - win_t[0] > WINDOW_S:
                            win_t.pop(0)
                            win_T.pop(0)
                        slope = None
                        if len(win_t) >= WINDOW_N:
                            slope = float(np.polyfit(win_t, win_T, 1)[0]) * 60.0

                        writer.writerow([f"{ts:.1f}", run_id, mode_label, f"{ir:.2f}",
                                         f"{T:.2f}", f"{rate:.2f}", tf(spike)])
                        ts_file.flush()

                        if T >= T_CUTOFF_C:
                            cutoff, reason = True, "CUTOFF (>= 80 C)"
                            break
                        if slope is not None and abs(slope) < SLOPE_STEADY_C_MIN:
                            steady, reason = True, "STEADY STATE"
                            break
                        if args.max_duration and ts >= args.max_duration:
                            reason = "MAX DURATION"
                            break
                        k += 1
                        last_refresh = 0.0  # force a bar refresh after a new sample

                    if not pipeline.isRunning():
                        reason = "PIPELINE STOPPED"
                        break

                    if now - last_refresh >= 1.0 and T is not None:
                        last_refresh = now
                        el = now - t0
                        sl = "  n/a " if slope is None else f"{slope:+6.2f}"
                        status(f"[{run_id}] [{bar(el / args.max_duration if args.max_duration else 0)}] "
                               f"{el:5.0f}s | T={T:5.1f} C peak={t_max:5.1f} | "
                               f"slope120={sl} C/min | spikes={'YES' if spike_any else 'no'}")
                    time.sleep(0.02)
            except KeyboardInterrupt:
                reason = "USER ABORT (Ctrl+C)"
            finally:
                print()
                try:
                    device.setIrLaserDotProjectorIntensity(0.0)   # IR off before stopping
                except Exception:
                    pass
                try:
                    pipeline.stop()
                except Exception:
                    pass
                if ts_file is not None:
                    ts_file.close()
    finally:
        try:
            device.close()
        except Exception:
            pass

    if t_init is None:      # never sampled
        print(f"{run_id}: no samples recorded ({reason}); no summary line written")
        return None

    row = {
        "run_id": run_id, "mode": mode_label, "ir_level": f"{ir:.1f}",
        "t_init_c": f"{t_init:.1f}", "t_final_c": f"{t_final:.1f}",
        "t_max_c": f"{t_max:.1f}", "duration_s": int(round(last_ts)),
        "steady_state_reached": tf(steady), "is_stable": tf(steady and not cutoff),
        "spike_detected": tf(spike_any),
    }
    append_summary(sum_path, row)
    print(f"{run_id}: {reason} | T {t_init:.1f} -> {t_final:.1f} C (peak {t_max:.1f}) in "
          f"{int(round(last_ts))} s | stable={tf(steady and not cutoff)} "
          f"spike={tf(spike_any)}")
    print(f"  {ts_path}\n  {sum_path}")
    return row


# ---- CLI --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("test selection")
    g.add_argument("--mode", choices=["raw", "stereo"], help="stream mode")
    g.add_argument("--ir", type=float, default=0.0, help="IR projector level 0.0..1.0")
    g.add_argument("--all", action="store_true", help="run the full 10-run matrix in order")
    g.add_argument("--outdir", default="thermal_results", help="folder for the CSV files")
    g.add_argument("--max-duration", type=float, default=3600.0,
                   help="safety limit per run [s] (0 = none); run ends as not steady")
    g.add_argument("--cooldown-timeout", type=float, default=3600.0,
                   help="give up if the chip does not reach 40 C within this time [s]")

    c = ap.add_argument_group("camera / stereo (defaults = volume scanner)")
    c.add_argument("--res", type=int, choices=[400, 800], default=400)
    c.add_argument("--fps", type=int, default=60)
    c.add_argument("--preset", choices=list(PRESETS),
                   default="accuracy" if "accuracy" in PRESETS else "fast_accuracy")
    c.add_argument("--no-subpixel", action="store_true")
    c.add_argument("--subpixel-bits", type=int, choices=[0, 3, 4, 5], default=None)
    c.add_argument("--disparity-shift", type=int, default=None)
    c.add_argument("--median", type=int, choices=[0, 3, 5, 7], default=None)
    c.add_argument("--confidence", type=int, default=None)
    c.add_argument("--calib", default="dynamic_calib.json", metavar="FILE.json",
                   help="calibration JSON (falls back to device calibration if missing; "
                        "'' to disable)")
    c.add_argument("--no-tuning", action="store_true")
    c.add_argument("--xlink-chunk", type=int, default=0)
    c.add_argument("--pool", type=int, default=16)
    args = ap.parse_args()
    if not args.all and args.mode is None:
        ap.error("give --mode raw|stereo (and --ir), or --all")
    if not 0.0 <= args.ir <= 1.0:
        ap.error("--ir must be between 0.0 and 1.0")
    return args


def main():
    args = parse_args()
    try:
        if args.all:
            rows = []
            for rid, mode, ir in MATRIX:
                r = run_test(args, rid, mode, ir)
                if r:
                    rows.append(r)
            print("\n=== MATRIX COMPLETE ===")
            for r in rows:
                print(f"{r['run_id']:<14} peak {r['t_max_c']:>5} C  {r['duration_s']:>5} s  "
                      f"stable={r['is_stable']}")
        else:
            mode = "RAW" if args.mode == "raw" else "STEREO"
            run_test(args, run_id_for(mode, args.ir), mode, args.ir)
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()