#!/usr/bin/env python3
"""
bench_oak.py - benchmark of the OAK-D Pro's INTERNAL stereo (DepthAI v3) + volume scan.

WHAT IT DOES
  Streams ONLY the stereo depth from the camera (no raw pair, no RGB, no window) and, for every
  combination below, records a short run and measures performance, depth quality and the
  volume from your conveyor_volume_scanner.py (same ROI / volume formula, same helper code).
  During the timed run the host only pulls + copies depth frames; all maths happens after
  the run, so the FPS you see is the stream FPS, not "stream + processing".

WHAT IS SWEPT (edit CONFIGS below, or select with --configs / --calibs / --only / --res)
  setups     2 motions x 4 scenes = 8 physical setups (same ids as record_raw.py)
  resolution 800p, 400p
  calibration factory (device EEPROM) and dynamic (--calib file)
  stereo cfg  presets / subpixel / median in CONFIGS
  Per setup you set up the scene once, press ENTER, and the script runs every
  (resolution x calibration x config) combination on it, restarting the camera between runs.
  Use --dry-run to see the number of runs and the estimated time.

SAME EXPOSURE AS THE RAW RECORDING
  --match-raw recordings/<session>  uses the exposure/ISO measured in each raw recording
  (stats.json), so the OAK sees the same image brightness as the raw frames. Without it the
  cameras run on auto exposure like your current script.

FOLDER LAYOUT
  benchmarks/<session>/session.json     settings, calibration, plan, configs (NO measurements)
  benchmarks/<session>/runs.csv         ONE ROW PER RUN - the table for the report
  benchmarks/<session>/baselines.json   belt distance tared from the static_empty runs
  benchmarks/<session>/<setup>_<res>p/<calib>__<config>/
        frames.csv   per frame: seq, timestamps, latency, holes, volume
        run.json     everything about this run (stereo config the device really used, summary)
        depth/*.png  --save-depth N evenly spaced depth frames, uint16 mm (0 = invalid)

BASELINE (belt distance)
  The static_empty run of every (resolution, calibration, config) sets its own baseline
  (median ROI depth) and saves it in baselines.json; the other setups reuse it. Run the
  static_empty setup first (it is first in the plan).

TRUTH VOLUMES (static scenes only)
  --box-volume-cm3 (default 528 = 12 x 8 x 5.5 cm), --shape-volume-cm3, --rocks-volume-cm3
  give vol_err_pct in runs.csv. Moving scenes have no snapshot truth; their volume series is
  in frames.csv for the later accumulation work.

USAGE
  python bench_oak.py --session site1 --calib dynamic_calib.json --match-raw recordings/site1
  python bench_oak.py --session site1 --calib dynamic_calib.json --configs fastdens_nosub
  python bench_oak.py --session site1 --only static_box --res 400
  python bench_oak.py --dry-run --calib dynamic_calib.json
Rerun with the same --session to resume (runs with frames.csv are skipped; --redo overrides).
"""
import argparse
import csv
import datetime as dt
import json
import math
import platform
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import depthai as dai

# ----------------------------------------------------------------------------- plan
RES = {400: (640, 400), 800: (1280, 800)}
RESOLUTIONS = (800, 400)
MOTIONS = ("static", "moving")
SCENES = {
    "empty": "EMPTY belt (nothing in the view)",
    "box": "the KNOWN BOX, fully inside the centre of the view",
    "shape": "the OBJECT WITH A SPECIFIC SHAPE, inside the centre of the view",
    "rocks": "the SCATTERED ROCKS",
}
MOTION_TEXT = {"static": "belt STOPPED", "moving": "belt RUNNING at working speed"}

# CLI preset name -> attribute of dai.node.StereoDepth.PresetMode (same table as the scanner script)
PRESETS = {
    "fast_accuracy": "FAST_ACCURACY", "fast_density": "FAST_DENSITY", "robotics": "ROBOTICS",
    "default": "DEFAULT", "high_detail": "HIGH_DETAIL", "accuracy": "ACCURACY", "density": "DENSITY",
}
PRESETS = {k: v for k, v in PRESETS.items() if hasattr(dai.node.StereoDepth.PresetMode, v)}

# ---- stereo configurations to test (edit freely) ----------------------------------------
# keys: preset (name from PRESETS), subpixel_bits (0 = off, 3/4/5), median (0/3/5/7),
#       confidence (0..255), disparity shift is added automatically (see --disparity-shift)
CONFIGS = {
    "fastdens_nosub":     dict(preset="fast_density", subpixel_bits=0),
    "fastacc_nosub":      dict(preset="fast_accuracy", subpixel_bits=0),
    "fastdens_sub3_med7": dict(preset="fast_density", subpixel_bits=3, median=7),
    "fastacc_sub3_med7":  dict(preset="fast_accuracy", subpixel_bits=3, median=7),
    "accuracy_sub3":      dict(preset="accuracy", subpixel_bits=3),
    "default_preset":     dict(preset="default"),
}
CONFIGS = {k: v for k, v in CONFIGS.items() if v["preset"] in PRESETS}
DEFAULT_CONFIGS = [c for c in ("fastdens_nosub", "fastacc_nosub", "fastdens_sub3_med7") if c in CONFIGS]

RUN_FIELDS = [
    "setup", "motion", "scene", "res", "calib", "config", "depth_w", "depth_h", "exposure", "ir_dots",
    "disp_shift", "baseline_mm", "baseline_source", "frames", "seconds", "fps_host", "fps_device",
    "seq_gaps", "interval_ms_mean", "interval_ms_std", "interval_ms_max", "latency_ms_mean",
    "latency_ms_p95", "volume_calc_ms", "fill_frame_pct", "roi_holes_pct", "roi_holes_pct_std",
    "roi_median_mm", "plane_resid_std_mm", "temporal_std_mm", "always_hole_pct", "flicker_pct",
    "vol_mean_cm3", "vol_std_cm3", "vol_min_cm3", "vol_max_cm3", "vol_drift_cm3_per_s",
    "truth_cm3", "vol_err_cm3", "vol_err_pct", "folder",
]


def build_plan(args):
    plan = []
    for motion in MOTIONS:
        for scene in SCENES:
            for res in RESOLUTIONS:
                plan.append(dict(setup=f"{motion}_{scene}", motion=motion, scene=scene, res=res,
                                 id=f"{motion}_{scene}_{res}p"))
    if args.res:
        plan = [p for p in plan if p["res"] in args.res]
    if args.only:
        toks = [t.strip() for t in args.only.split(",") if t.strip()]
        plan = [p for p in plan if any(t in p["id"] for t in toks)]
    return plan


# ----------------------------------------------------------------------------- helpers from the tested scanner script
def build_cam(pipeline, socket, size, fps):
    """Camera node on `socket` with sensor mode (resolution + FPS) fixed."""
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
    """(fx, fy, source): from the frame's own intrinsics, else calibration scaled to the depth width."""
    try:
        K = depth_msg.getTransformation().getIntrinsicMatrix()
        return float(K[0][0]), float(K[1][1]), "frame transformation"
    except Exception:
        pass
    K = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, req_size[0], req_size[1])
    s = depth_w / req_size[0]
    return float(K[0][0]) * s, float(K[1][1]) * s, "calibration (scaled)"


def process_depth(depth, baseline, threshold, roi_frac, fx, fy):
    """Crop the ROI and compute its volume. Identical to conveyor_volume_scanner.py.
    Returns roi (view), valid mask, volume_cm3, holes_pct, (cx, cy, half)."""
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


def crop_roi(depth, roi_frac):
    h, w = depth.shape
    half = max(8, int(w * roi_frac)) // 2 * 2 // 2
    return depth[h // 2 - half:h // 2 + half, w // 2 - half:w // 2 + half]


def stereo_config_dict(stereo):
    """What the device will really use after preset + overrides (same getters as describe_stereo)."""
    ic = stereo.initialConfig

    def g(fn):
        try:
            return fn()
        except Exception:
            return "?"
    return {
        "subpixel": g(ic.getSubpixel), "bits": g(ic.getSubpixelFractionalBits),
        "lr_check": g(ic.getLeftRightCheck), "lr_thr": g(ic.getLeftRightCheckThreshold),
        "extended": g(ic.getExtendedDisparity), "conf_thr": g(ic.getConfidenceThreshold),
        "median": str(g(ic.getMedianFilter)), "disp_shift": g(lambda: ic.algorithmControl.disparityShift),
        "decimation": g(lambda: ic.postProcessing.decimationFilter.decimationFactor),
        "speckle": g(lambda: ic.postProcessing.speckleFilter.enable),
        "temporal": g(lambda: ic.postProcessing.temporalFilter.enable),
        "spatial": g(lambda: ic.postProcessing.spatialFilter.enable),
        "align": str(g(lambda: ic.algorithmControl.depthAlign)),
    }


def configure_stereo(stereo, cfg, shift):
    """Preset first, then overrides (same logic as the scanner script)."""
    stereo.setDefaultProfilePreset(getattr(dai.node.StereoDepth.PresetMode, PRESETS[cfg["preset"]]))
    bits = cfg.get("subpixel_bits")
    if bits == 0:
        stereo.setSubpixel(False)
    elif bits is not None:
        stereo.setSubpixel(True)
        try:
            stereo.setSubpixelFractionalBits(bits)
        except Exception:
            try:
                stereo.initialConfig.setSubpixelFractionalBits(bits)
            except Exception as e:
                print(f"  warning: could not set subpixel bits: {e}")
    try:
        if shift:
            stereo.initialConfig.setDisparityShift(int(shift))
        if cfg.get("confidence") is not None:
            stereo.initialConfig.setConfidenceThreshold(cfg["confidence"])
        if cfg.get("median") is not None:
            kernels = {0: dai.MedianFilter.MEDIAN_OFF, 3: dai.MedianFilter.KERNEL_3x3,
                       5: dai.MedianFilter.KERNEL_5x5, 7: dai.MedianFilter.KERNEL_7x7}
            stereo.initialConfig.setMedianFilter(kernels[cfg["median"]])
    except Exception as e:
        print(f"  warning: could not apply a stereo override: {e}")


def auto_shift(calib, res, z_mm, margin=10):
    """Disparity shift so the belt (at z_mm) sits at the near edge of the 96 px search window.
    Needed when the belt disparity f*B/Z exceeds the default window (800p at ~58 cm)."""
    w, h = RES[res]
    fx = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, w, h)[0][0]
    d = fx * calib.getBaselineDistance() * 10.0 / z_mm
    return 0 if d <= 80 else int(math.ceil(d - margin))


def device_info(dev):
    info = {}
    for name in ("getDeviceName", "getProductName", "getDeviceId", "getPlatformAsString"):
        try:
            info[name[3:]] = str(getattr(dev, name)())
        except Exception as e:
            info[name[3:]] = f"? ({e})"
    try:
        info["UsbSpeed"] = str(dev.getUsbSpeed())
    except Exception as e:
        info["UsbSpeed"] = f"? ({e})"
    return info


def calib_summary(calib):
    out = {"baseline_cm": calib.getBaselineDistance()}
    for res, (w, h) in RES.items():
        out[f"K_left_{res}p"] = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, w, h)
        out[f"K_right_{res}p"] = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, w, h)
    out["extrinsics_left_to_right"] = calib.getCameraExtrinsics(
        dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)
    return out


def calib_diff(a, b):
    Ea, Eb = np.array(a["extrinsics_left_to_right"]), np.array(b["extrinsics_left_to_right"])
    cosang = np.clip((np.trace(Ea[:3, :3].T @ Eb[:3, :3]) - 1) / 2, -1, 1)
    return {"rotation_diff_deg": float(np.degrees(np.arccos(cosang))),
            "translation_diff_cm": (Eb[:3, 3] - Ea[:3, 3]).tolist(),
            "baseline_diff_cm": b["baseline_cm"] - a["baseline_cm"],
            "fx_left_diff_px_800p": b["K_left_800p"][0][0] - a["K_left_800p"][0][0],
            "fx_left_diff_px_400p": b["K_left_400p"][0][0] - a["K_left_400p"][0][0]}


def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


# ----------------------------------------------------------------------------- one run
def run_once(rec, calib_name, calib_path, calib_obj, cfg_name, cfg, args, exposure):
    """Start the camera with this calibration + config, discard the warm-up, record --seconds of
    depth. Returns a dict with the raw frames (nothing is analysed here), or None on failure."""
    res = rec["res"]
    size = RES[res]
    shift = (auto_shift(calib_obj, res, args.baseline) if args.disparity_shift == "auto"
             else (0 if args.disparity_shift == "none" else int(args.disparity_shift)))
    with dai.Pipeline() as pipeline:
        if calib_path:
            pipeline.setCalibrationData(dai.CalibrationHandler(calib_path))
        if not args.no_tuning:
            try:
                pipeline.setXLinkChunkSize(args.xlink_chunk)
            except Exception as e:
                print(f"  warning: setXLinkChunkSize failed: {e}")
        left = build_cam(pipeline, dai.CameraBoardSocket.CAM_B, size, args.fps)
        right = build_cam(pipeline, dai.CameraBoardSocket.CAM_C, size, args.fps)
        if not args.no_tuning:
            apply_pool(left, args.pool)
            apply_pool(right, args.pool)
        if exposure:
            for cam in (left, right):
                cam.initialControl.setManualExposure(int(exposure[0]), int(exposure[1]))
        left_out = left.requestOutput(size, type=dai.ImgFrame.Type.GRAY8, fps=args.fps)
        right_out = right.requestOutput(size, type=dai.ImgFrame.Type.GRAY8, fps=args.fps)
        stereo = pipeline.create(dai.node.StereoDepth)
        configure_stereo(stereo, cfg, shift)
        cfg_real = stereo_config_dict(stereo)
        left_out.link(stereo.left)
        right_out.link(stereo.right)
        q = stereo.depth.createOutputQueue(maxSize=args.queue, blocking=False)
        pipeline.start()
        dev = pipeline.getDefaultDevice()
        dev.setIrLaserDotProjectorIntensity(args.ir_dots)
        dev.setIrFloodLightIntensity(args.ir_flood)

        t_end = time.monotonic() + args.warmup          # warm-up: stereo start, exposure settling
        while time.monotonic() < t_end:
            if q.tryGet() is None:
                time.sleep(0.001)
        while q.tryGet() is not None:
            pass

        frames, focal = [], None
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.seconds:
            got = False
            while (m := q.tryGet()) is not None:
                got = True
                now = dai.Clock.now()
                d = m.getFrame()
                if focal is None:
                    focal = get_focal_px(m, calib_obj, d.shape[1], size)
                frames.append((m.getSequenceNum(), m.getTimestampDevice().total_seconds(),
                               (now - m.getTimestamp()).total_seconds() * 1000.0, d.copy()))
            if not got:
                time.sleep(0.0002)
        elapsed = time.monotonic() - t0
    if not frames:
        return None
    return {"frames": frames, "elapsed": elapsed, "focal": focal, "shift": shift, "cfg_real": cfg_real}


# ----------------------------------------------------------------------------- analysis (after the run)
def plane_resid_std(roi, step=4):
    """Std of the residual of a least-squares plane fitted to the valid ROI pixels [mm]."""
    z = roi[::step, ::step].astype(np.float64)
    yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    m = z > 0
    if m.sum() < 50:
        return float("nan")
    A = np.c_[xx[m], yy[m], np.ones(m.sum())]
    coef, *_ = np.linalg.lstsq(A, z[m], rcond=None)
    return float((z[m] - A @ coef).std())


def temporal_stats(rois, max_frames=100):
    """Static scenes: per-pixel temporal std (median over stable pixels) and hole persistence."""
    idx = np.linspace(0, len(rois) - 1, min(len(rois), max_frames)).astype(int)
    S = np.stack([rois[i].astype(np.float32) for i in idx])
    valid = S > 0
    cnt = valid.sum(0)
    mean = np.where(cnt > 0, (S * valid).sum(0) / np.maximum(cnt, 1), np.nan)
    var = np.where(cnt > 1, (((S - mean) ** 2) * valid).sum(0) / np.maximum(cnt - 1, 1), np.nan)
    stable = cnt >= 0.8 * len(idx)
    tstd = float(np.nanmedian(np.sqrt(var[stable]))) if stable.any() else float("nan")
    frac = valid.mean(0)
    return tstd, float((frac == 0).mean() * 100), float(((frac > 0) & (frac < 1)).mean() * 100)


def analyze_run(cap, rec, baseline, baseline_source, args, truth):
    """Per-frame table + one summary dict (a row of runs.csv)."""
    frames = cap["frames"]
    fx, fy, _src = cap["focal"]
    n = len(frames)
    seq = np.array([f[0] for f in frames])
    ts = np.array([f[1] for f in frames])
    lat = np.array([f[2] for f in frames])
    depths = [f[3] for f in frames]

    vol, holes, fill, med, calc = (np.zeros(n) for _ in range(5))
    rois = []
    for i, d in enumerate(depths):
        t = time.perf_counter()
        roi, _valid, vol[i], holes[i], _geom = process_depth(d, baseline, args.threshold, args.roi_frac, fx, fy)
        calc[i] = time.perf_counter() - t
        fill[i] = 100.0 * np.count_nonzero(d) / d.size
        v = roi[roi > 0]
        med[i] = float(np.median(v)) if v.size else np.nan
        rois.append(roi)

    dts = np.diff(ts) * 1000 if n > 1 else np.array([0.0])
    fps_dev = (n - 1) / (ts[-1] - ts[0]) if n > 1 and ts[-1] > ts[0] else 0.0
    drift = float(np.polyfit(ts - ts[0], vol, 1)[0]) if n > 2 and ts[-1] > ts[0] else float("nan")
    s = {
        "depth_w": depths[0].shape[1], "depth_h": depths[0].shape[0], "frames": n,
        "seconds": round(cap["elapsed"], 3), "fps_host": round(n / cap["elapsed"], 2),
        "fps_device": round(fps_dev, 2), "seq_gaps": int(np.maximum(np.diff(seq) - 1, 0).sum()),
        "interval_ms_mean": float(dts.mean()), "interval_ms_std": float(dts.std()),
        "interval_ms_max": float(dts.max()), "latency_ms_mean": float(lat.mean()),
        "latency_ms_p95": float(np.percentile(lat, 95)), "volume_calc_ms": float(calc.mean() * 1000),
        "fill_frame_pct": float(fill.mean()), "roi_holes_pct": float(holes.mean()),
        "roi_holes_pct_std": float(holes.std()), "roi_median_mm": float(np.nanmean(med)),
        "plane_resid_std_mm": float("nan"), "temporal_std_mm": float("nan"),
        "always_hole_pct": float("nan"), "flicker_pct": float("nan"),
        "vol_mean_cm3": float(vol.mean()), "vol_std_cm3": float(vol.std()),
        "vol_min_cm3": float(vol.min()), "vol_max_cm3": float(vol.max()),
        "vol_drift_cm3_per_s": drift, "truth_cm3": truth if truth else float("nan"),
        "vol_err_cm3": float("nan"), "vol_err_pct": float("nan"),
    }
    if rec["scene"] == "empty":                      # flat belt: spatial noise
        sel = range(0, n, max(1, n // 20))
        s["plane_resid_std_mm"] = float(np.nanmean([plane_resid_std(rois[i]) for i in sel]))
    if rec["motion"] == "static":                    # nothing changes: temporal noise + hole persistence
        s["temporal_std_mm"], s["always_hole_pct"], s["flicker_pct"] = temporal_stats(rois)
    if truth:
        s["vol_err_cm3"] = s["vol_mean_cm3"] - truth
        s["vol_err_pct"] = 100.0 * s["vol_err_cm3"] / truth
    table = {"seq": seq, "ts": ts, "lat": lat, "fill": fill, "holes": holes, "med": med, "vol": vol}
    return table, s, depths


def tare_baseline(cap, roi_frac):
    """Baseline from an empty belt: median over frames of the median valid ROI depth."""
    meds = []
    for f in cap["frames"]:
        v = crop_roi(f[3], roi_frac)
        v = v[v > 0]
        if v.size:
            meds.append(float(np.median(v)))
    return float(np.median(meds)) if meds else None


def write_run(run_dir, table, summary, depths, meta, args):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "frames.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "seq", "ts_device_s", "latency_ms", "fill_frame_pct", "roi_holes_pct",
                    "roi_median_mm", "volume_cm3"])
        for i in range(len(table["seq"])):
            w.writerow([i, table["seq"][i], f"{table['ts'][i]:.6f}", f"{table['lat'][i]:.2f}",
                        f"{table['fill'][i]:.2f}", f"{table['holes'][i]:.2f}",
                        f"{table['med'][i]:.1f}", f"{table['vol'][i]:.3f}"])
    if args.save_depth and depths:
        (run_dir / "depth").mkdir(exist_ok=True)
        k = len(depths) if args.save_depth < 0 else min(args.save_depth, len(depths))
        for i in np.linspace(0, len(depths) - 1, k).astype(int):
            cv2.imwrite(str(run_dir / "depth" / f"{i:05d}.png"), depths[i])
    dump_json(run_dir / "run.json", {**meta, "summary": summary})


def append_row(path, row):
    new = not Path(path).exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RUN_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


# ----------------------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=f"bench_{dt.date.today():%Y%m%d}", help="session folder; reuse to resume")
    ap.add_argument("--out", default="benchmarks")
    ap.add_argument("--calib", default=None, metavar="FILE.json", help="dynamic calibration json (adds the 'dynamic' calibration)")
    ap.add_argument("--calibs", default=None, help="comma list of calibrations to run: factory,dynamic (default: all available)")
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS), help=f"comma list from: {', '.join(CONFIGS)}")
    ap.add_argument("--match-raw", default=None, metavar="DIR", help="raw session folder: reuse its exposure/ISO per setup+resolution")
    ap.add_argument("--seconds", type=float, default=5.0, help="measured seconds per run")
    ap.add_argument("--warmup", type=float, default=1.0, help="discarded seconds after camera start")
    ap.add_argument("--fps", type=int, default=120, help="requested mono/stereo FPS (the stereo output is what limits it)")
    ap.add_argument("--ir-dots", type=float, default=1.0, help="IR dot projector 0..1 (same as the raw recording)")
    ap.add_argument("--ir-flood", type=float, default=0.0)
    ap.add_argument("--no-tuning", action="store_true", help="no xlink-chunk / pool tuning")
    ap.add_argument("--xlink-chunk", type=int, default=0)
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--queue", type=int, default=64, help="host queue size for depth frames")
    ap.add_argument("--disparity-shift", default="auto",
                    help="auto (computed from --baseline, only when the belt is outside the default window), none, or an integer")
    ap.add_argument("--baseline", type=float, default=578.0, help="nominal belt distance [mm]; used before/without a tare")
    ap.add_argument("--threshold", type=float, default=15.0, help="min height above belt to count as load [mm]")
    ap.add_argument("--roi-frac", type=float, default=0.25, help="ROI side as fraction of depth frame width")
    ap.add_argument("--box-volume-cm3", type=float, default=528.0, help="true volume of the box (12x8x5.5 cm = 528)")
    ap.add_argument("--shape-volume-cm3", type=float, default=None)
    ap.add_argument("--rocks-volume-cm3", type=float, default=None)
    ap.add_argument("--save-depth", type=int, default=20, help="depth PNGs saved per run (evenly spaced; -1 all, 0 none)")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between runs (device release)")
    ap.add_argument("--res", type=int, nargs="+", choices=[400, 800], default=None)
    ap.add_argument("--only", default=None, help="comma list; keep ids containing any of these substrings")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def exposure_from_raw(raw_dir, rec_id):
    if not raw_dir:
        return None
    p = Path(raw_dir) / rec_id / "stats.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return (s["exposure_us_median"], s["iso_median"])


def main():
    args = parse_args()
    plan = build_plan(args)
    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    bad = [c for c in names if c not in CONFIGS]
    if bad:
        raise SystemExit(f"unknown config(s) {bad}; available: {list(CONFIGS)}")
    calib_names = ["factory"] + (["dynamic"] if args.calib else [])
    if args.calibs:
        calib_names = [c for c in args.calibs.split(",") if c in calib_names]
    setups = list(dict.fromkeys(p["setup"] for p in plan))
    n_runs = len(plan) * len(calib_names) * len(names)
    est_min = n_runs * (args.seconds + args.warmup + 9) / 60
    print(f"{len(setups)} setups, {len(plan)} (setup,res) pairs x {len(calib_names)} calibrations x "
          f"{len(names)} configs = {n_runs} runs, ~{est_min:.0f} min (+ your setup time)")
    print("calibrations:", calib_names, "| configs:", names)
    if args.dry_run:
        for p in plan:
            print("  ", p["id"])
        return

    root = Path(args.out) / args.session
    root.mkdir(parents=True, exist_ok=True)
    with dai.Device() as dev:
        factory = dev.readCalibration()
        dinfo = device_info(dev)
    cal_obj = {"factory": (None, factory)}
    cal_sum = {"factory": calib_summary(factory)}
    if args.calib:
        dyn = dai.CalibrationHandler(args.calib)
        cal_obj["dynamic"] = (args.calib, dyn)
        cal_sum["dynamic"] = calib_summary(dyn)
        cal_sum["dynamic_minus_factory"] = calib_diff(cal_sum["factory"], cal_sum["dynamic"])
        shutil.copy(args.calib, root / "dynamic_calib.json")
        print("dynamic vs factory:", json.dumps(cal_sum["dynamic_minus_factory"]))
    if "SUPER" not in dinfo.get("UsbSpeed", ""):
        print(f"WARNING: USB speed is {dinfo.get('UsbSpeed')} - use a USB3 port and cable")
    sp = root / "session.json"
    created = json.loads(sp.read_text()).get("created_at") if sp.exists() else None
    dump_json(sp, {
        "created_at": created or dt.datetime.now().isoformat(timespec="seconds"),
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "note": "settings and context only; measurements are in runs.csv, <run>/run.json, <run>/frames.csv",
        "args": vars(args),
        "host": {"platform": platform.platform(), "python": platform.python_version(),
                 "depthai": dai.__version__, "opencv": cv2.__version__, "numpy": np.__version__},
        "device": dinfo, "calibration": cal_sum,
        "configs": {n: CONFIGS[n] for n in names}, "plan": [p["id"] for p in plan],
    })

    bl_path = root / "baselines.json"
    baselines = json.loads(bl_path.read_text()) if bl_path.exists() else {}
    truths = {"box": args.box_volume_cm3, "shape": args.shape_volume_cm3, "rocks": args.rocks_volume_cm3}
    done_runs = 0
    try:
        for setup in setups:
            recs = [p for p in plan if p["setup"] == setup]
            first = recs[0]
            print(f"\n=== {setup}: {MOTION_TEXT[first['motion']]}, {SCENES[first['scene']]} ===")
            ans = input("Set the scene up, then ENTER to start (s = skip this setup, q = quit): ").strip().lower()
            if ans == "q":
                break
            if ans == "s":
                continue
            for rec in recs:
                exposure = exposure_from_raw(args.match_raw, rec["id"])
                for calib_name in calib_names:
                    calib_path, calib_obj = cal_obj[calib_name]
                    for cname in names:
                        folder = root / rec["id"] / f"{calib_name}__{cname}"
                        if (folder / "frames.csv").exists() and not args.redo:
                            print(f"  [{rec['id']} {calib_name} {cname}] done, skipping")
                            continue
                        key = f"{rec['res']}p|{calib_name}|{cname}"
                        try:
                            cap = run_once(rec, calib_name, calib_path, calib_obj, cname, CONFIGS[cname],
                                           args, exposure)
                        except Exception as e:
                            print(f"  [{rec['id']} {calib_name} {cname}] FAILED: {type(e).__name__}: {e}")
                            time.sleep(args.pause)
                            continue
                        if cap is None:
                            print(f"  [{rec['id']} {calib_name} {cname}] no frames received")
                            continue
                        if rec["setup"] == "static_empty":
                            bl = tare_baseline(cap, args.roi_frac)
                            if bl:
                                baselines[key] = bl
                                dump_json(bl_path, baselines)
                        baseline = baselines.get(key, args.baseline)
                        bl_src = "tare" if key in baselines else "nominal --baseline"
                        table, summ, depths = analyze_run(cap, rec, baseline, bl_src, args,
                                                          truths.get(rec["scene"]) if rec["motion"] == "static" else None)
                        meta = {"setup": rec["setup"], "id": rec["id"], "calib": calib_name, "config": cname,
                                "config_requested": CONFIGS[cname], "stereo_config_on_device": cap["cfg_real"],
                                "disparity_shift": cap["shift"], "baseline_mm": baseline,
                                "baseline_source": bl_src, "focal_px": cap["focal"][:2],
                                "focal_source": cap["focal"][2], "exposure_us_iso": exposure,
                                "ir_dots": args.ir_dots}
                        write_run(folder, table, summ, depths, meta, args)
                        row = {"setup": rec["setup"], "motion": rec["motion"], "scene": rec["scene"],
                               "res": rec["res"], "calib": calib_name, "config": cname,
                               "exposure": f"{exposure[0]:.0f}us/iso{exposure[1]}" if exposure else "auto",
                               "ir_dots": args.ir_dots, "disp_shift": cap["shift"], "baseline_mm": baseline,
                               "baseline_source": bl_src, "folder": str(folder.relative_to(root)), **summ}
                        append_row(root / "runs.csv", row)
                        done_runs += 1
                        err = f" err {summ['vol_err_pct']:+.1f}%" if not math.isnan(summ["vol_err_pct"]) else ""
                        print(f"  [{rec['id']} {calib_name} {cname}] {summ['fps_device']:.1f} fps | holes "
                              f"{summ['roi_holes_pct']:.1f}% | vol {summ['vol_mean_cm3']:.1f}"
                              f"+-{summ['vol_std_cm3']:.1f} cm3{err} | shift {cap['shift']} | base {baseline:.0f}mm")
                        time.sleep(args.pause)
    except KeyboardInterrupt:
        print("\ninterrupted - finished runs are kept; rerun with the same --session to resume")
    print(f"\n{done_runs} runs this time. Table: {root / 'runs.csv'}")


if __name__ == "__main__":
    main()