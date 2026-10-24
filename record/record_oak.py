#!/usr/bin/env python3
"""
record_raw.py - RAW mono-pair recorder for the conveyor volume scanner (OAK-D Pro, DepthAI v3).

WHAT IT DOES
  Records the two mono cameras only (no StereoDepth node, no RGB), as UNRECTIFIED 8-bit
  frames, at the highest FPS the camera can deliver. This is the data that every other
  stereo method (SGBM, FoundationStereo, light networks, Hailo) will be fed later, so it is
  recorded once, with the calibration saved next to it.
  During the timed capture nothing else runs: no window, no processing, only "pull frame,
  keep a copy". Disk writing happens afterwards.

THE 16 RECORDINGS (plan is fixed, in this order)
  2 motions (static, moving) x 4 scenes (empty, box, shape, rocks) x 2 resolutions (800p, 400p)
  Setup order: static_empty, static_box, static_shape, static_rocks, moving_empty, ...
  For every setup you record 800p first, then 400p, WITHOUT touching the scene in between.
  id example: static_box_800p

FOLDER LAYOUT
  recordings/<session>/session.json        settings + calibration + device + plan (NO measurements)
  recordings/<session>/summary.csv         one row per recording (measurements, see stats.json)
  recordings/<session>/calib/factory_eeprom.json   calibration stored on the device
  recordings/<session>/calib/dynamic.json          copy of --calib file (if given)
  recordings/<session>/<id>/left/000000.png        raw left  (CAM_B), 8-bit
  recordings/<session>/<id>/right/000000.png       raw right (CAM_C), 8-bit
  recordings/<session>/<id>/meta.csv               per frame pair: seq, timestamps, exposure, iso
  recordings/<session>/<id>/stats.json             measurements of this recording

  Frames are NOT rectified. Rectify offline with factory_eeprom.json or dynamic.json
  (so both calibrations can be tested on the same frames).

KEYS (preview window, shown before every recording)
  SPACE  record now (after --delay s, once earlier recordings finished writing)
  n      skip this recording          q  quit (finished recordings are kept, rerun to resume)
  w / s  exposure x1.25 / x0.8 (switches both cameras to manual exposure)
  l      lock the current auto exposure as manual     a  back to auto exposure
  i / k  ISO up / down (only used while exposure is manual)
  The exposure you set is remembered for the next recording with the same motion + resolution.

USAGE
  python record_raw.py --session site1 --calib dynamic_calib.json
  python record_raw.py --session site1 --calib dynamic_calib.json --exposure-us-moving 1000
  python record_raw.py --session site1 --only static_box          # only these ids (substring match)
  python record_raw.py --session site1 --res 400 --redo           # redo all 400p recordings
  python record_raw.py --dry-run                                  # print the plan and exit

Rerunning with the same --session resumes: recordings that already have stats.json are skipped.
The plan and the helper functions are kept identical in bench_oak.py.
"""
import argparse
import csv
import datetime as dt
import json
import platform
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import depthai as dai

# ----------------------------------------------------------------------------- plan
RES = {400: (640, 400), 800: (1280, 800)}
RESOLUTIONS = (800, 400)                 # recording order inside one setup
MOTIONS = ("static", "moving")
SCENES = {
    "empty": "EMPTY belt (nothing in the view)",
    "box": "the KNOWN BOX, fully inside the centre of the view",
    "shape": "the OBJECT WITH A SPECIFIC SHAPE, inside the centre of the view",
    "rocks": "the SCATTERED ROCKS",
}
MOTION_TEXT = {"static": "belt STOPPED", "moving": "belt RUNNING at working speed"}

SUMMARY_FIELDS = [
    "id", "setup", "motion", "scene", "res", "fps_requested", "pairs", "seconds",
    "fps_device", "fps_host", "seq_gaps_left", "seq_gaps_right", "unpaired_left",
    "unpaired_right", "pairing", "interval_ms_mean", "interval_ms_std", "interval_ms_max",
    "lr_skew_ms_max", "exposure_us_median", "iso_median", "brightness_mean",
    "saturated_pct", "dark_pct",
]


def build_plan(args):
    plan = []
    for motion in MOTIONS:
        for scene in SCENES:
            for res in RESOLUTIONS:
                plan.append(dict(id=f"{motion}_{scene}_{res}p", setup=f"{motion}_{scene}",
                                 motion=motion, scene=scene, res=res))
    if args.res:
        plan = [p for p in plan if p["res"] in args.res]
    if args.only:
        toks = [t.strip() for t in args.only.split(",") if t.strip()]
        plan = [p for p in plan if any(t in p["id"] for t in toks)]
    return plan


# ----------------------------------------------------------------------------- helpers
# build_cam / apply_pool are the same as in the tested conveyor_volume_scanner.py
def build_cam(pipeline, socket, size, fps):
    """Camera node on `socket` with sensor mode (resolution + FPS) fixed."""
    cam = pipeline.create(dai.node.Camera)
    try:
        return cam.build(socket, sensorResolution=size, sensorFps=fps)
    except TypeError:
        return cam.build(socket, sensorFps=fps)


def apply_pool(cam, pool):
    """Number of buffers in the camera's RAW and ISP frame pools (benchmark tuning)."""
    for setter in ("setRawNumFramesPool", "setIspNumFramesPool"):
        try:
            getattr(cam, setter)(pool)
        except Exception as e:
            print(f"  warning: {setter}({pool}) failed: {e}")


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
    """Numbers that identify a calibration: baseline, intrinsics per resolution, L->R extrinsics."""
    out = {"baseline_cm": calib.getBaselineDistance()}
    for res, (w, h) in RES.items():
        out[f"K_left_{res}p"] = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, w, h)
        out[f"K_right_{res}p"] = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, w, h)
    out["extrinsics_left_to_right"] = calib.getCameraExtrinsics(
        dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)
    return out


def calib_diff(a, b):
    """How much calibration b differs from a (rotation between the cameras, translation, f)."""
    Ea, Eb = np.array(a["extrinsics_left_to_right"]), np.array(b["extrinsics_left_to_right"])
    cosang = np.clip((np.trace(Ea[:3, :3].T @ Eb[:3, :3]) - 1) / 2, -1, 1)
    return {
        "rotation_diff_deg": float(np.degrees(np.arccos(cosang))),
        "translation_diff_cm": (Eb[:3, 3] - Ea[:3, 3]).tolist(),
        "baseline_diff_cm": b["baseline_cm"] - a["baseline_cm"],
        "fx_left_diff_px_800p": b["K_left_800p"][0][0] - a["K_left_800p"][0][0],
        "fx_left_diff_px_400p": b["K_left_400p"][0][0] - a["K_left_400p"][0][0],
    }


def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def host_info():
    return {"platform": platform.platform(), "python": platform.python_version(),
            "depthai": dai.__version__, "opencv": cv2.__version__, "numpy": np.__version__}


# ----------------------------------------------------------------------------- session
def init_session(root, args, plan):
    """Create the session folder, save calibrations, write session.json."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "calib").mkdir(exist_ok=True)
    with dai.Device() as dev:
        factory = dev.readCalibration()
        dinfo = device_info(dev)
    try:
        dump_json(root / "calib" / "factory_eeprom.json", factory.eepromToJson())
    except Exception as e:
        print("warning: could not export factory calibration json:", e)
    calibs = {"factory": calib_summary(factory)}
    if args.calib:
        shutil.copy(args.calib, root / "calib" / "dynamic.json")
        calibs["dynamic"] = calib_summary(dai.CalibrationHandler(args.calib))
        calibs["dynamic_minus_factory"] = calib_diff(calibs["factory"], calibs["dynamic"])
    if "SUPER" not in dinfo.get("UsbSpeed", ""):
        print(f"WARNING: USB speed is {dinfo.get('UsbSpeed')} - use a USB3 port and cable for max FPS")
    sp = root / "session.json"
    created = json.loads(sp.read_text()).get("created_at") if sp.exists() else None
    sess = {
        "created_at": created or dt.datetime.now().isoformat(timespec="seconds"),
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "note": "settings and context only; measurements are in <id>/stats.json and summary.csv",
        "args": vars(args), "host": host_info(), "device": dinfo, "calibration": calibs,
        "frame_format": "raw unrectified GRAY8 png; left=CAM_B right=CAM_C; sizes " + str(RES),
        "plan": [p["id"] for p in plan],
    }
    dump_json(sp, sess)
    return sess


# ----------------------------------------------------------------------------- capture
def make_preview(l, r, lines):
    def fit(img):
        h, w = img.shape
        return img if w <= 640 else cv2.resize(img, (640, int(h * 640 / w)))
    img = cv2.cvtColor(np.hstack([fit(l), fit(r)]), cv2.COLOR_GRAY2BGR)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return img


def send_ctrl(ctrl_qs, exp=None, iso=None, auto=False):
    c = dai.CameraControl()
    if auto:
        c.setAutoExposureEnable()
    else:
        c.setManualExposure(int(exp), int(iso))
    for q in ctrl_qs:
        q.send(c)


def pack(m):
    return (m.getSequenceNum(), m.getTimestampDevice().total_seconds(),
            m.getExposureTime().total_seconds() * 1e6, int(m.getSensitivity()),
            m.getFrame().copy())


def wait_writers(writers):
    pending = [t for t in writers if t.is_alive()]
    if pending:
        print("  waiting for earlier recordings to finish writing to disk ...")
        for t in pending:
            t.join()


def capture_one(rec, args, state, writers):
    """Open the camera at rec['res'], show a preview until the user decides, then record.
    Returns ("record", capture_dict) | ("skip", None) | ("quit", None)."""
    res = rec["res"]
    size = RES[res]
    fps = args.fps_800 if res == 800 else args.fps_400
    key = f"{rec['motion']}_{res}"
    st = state.setdefault(key, {
        "exp": args.exposure_us_moving if (rec["motion"] == "moving" and args.exposure_us_moving)
        else args.exposure_us,
        "iso": args.iso})
    header = f"{rec['id']} | {MOTION_TEXT[rec['motion']]} | {SCENES[rec['scene']]}"

    with dai.Pipeline() as pipeline:
        if not args.no_tuning:
            try:
                pipeline.setXLinkChunkSize(args.xlink_chunk)
            except Exception as e:
                print(f"warning: setXLinkChunkSize failed: {e}")
        left = build_cam(pipeline, dai.CameraBoardSocket.CAM_B, size, fps)
        right = build_cam(pipeline, dai.CameraBoardSocket.CAM_C, size, fps)
        if not args.no_tuning:
            apply_pool(left, args.pool)
            apply_pool(right, args.pool)
        if st["exp"]:
            for cam in (left, right):
                cam.initialControl.setManualExposure(int(st["exp"]), int(st["iso"]))
        out_l = left.requestOutput(size, type=dai.ImgFrame.Type.GRAY8, fps=fps)
        out_r = right.requestOutput(size, type=dai.ImgFrame.Type.GRAY8, fps=fps)
        qL = out_l.createOutputQueue(maxSize=args.queue, blocking=False)
        qR = out_r.createOutputQueue(maxSize=args.queue, blocking=False)
        ctrl_qs = [left.inputControl.createInputQueue(), right.inputControl.createInputQueue()]
        print(f"[{rec['id']}] starting camera {res}p @ {fps} fps ...")
        pipeline.start()
        dev = pipeline.getDefaultDevice()
        dev.setIrLaserDotProjectorIntensity(args.ir_dots)
        dev.setIrFloodLightIntensity(args.ir_flood)

        # ---- preview until SPACE / n / q ----
        win = "raw recorder"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        last_draw, n_cnt, t_cnt, stream_fps = 0.0, 0, time.monotonic(), 0.0
        lf = rf = None
        cur_exp, cur_iso = 0.0, 0
        action = None
        while action is None:
            got = False
            while (m := qL.tryGet()) is not None:
                lf, got = m, True
                n_cnt += 1
            while (m := qR.tryGet()) is not None:
                rf, got = m, True
            now = time.monotonic()
            if now - t_cnt >= 1.0:
                stream_fps, n_cnt, t_cnt = n_cnt / (now - t_cnt), 0, now
            if lf is not None and rf is not None and now - last_draw >= 1.0 / args.display_fps:
                last_draw = now
                l, r = lf.getFrame(), rf.getFrame()
                cur_exp = lf.getExposureTime().total_seconds() * 1e6
                cur_iso = int(lf.getSensitivity())
                sat = float((l[::4, ::4] >= 250).mean()) * 100
                mode = "MANUAL" if st["exp"] else "AUTO"
                lines = [header,
                         f"stream {stream_fps:5.1f} fps (preview limited; recording is not)  "
                         f"exp {cur_exp:.0f}us iso {cur_iso} [{mode}]  mean {l[::4, ::4].mean():.0f} "
                         f"sat {sat:.1f}%",
                         "SPACE record | n skip | q quit | w/s exposure | l lock | a auto | i/k iso"]
                cv2.imshow(win, make_preview(l, r, lines))
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    action = "record"
                elif key == ord("n"):
                    action = "skip"
                elif key == ord("q"):
                    action = "quit"
                elif key in (ord("w"), ord("s")):
                    st["exp"] = int(np.clip((st["exp"] or cur_exp) * (1.25 if key == ord("w") else 0.8),
                                            50, 33000))
                    send_ctrl(ctrl_qs, st["exp"], st["iso"])
                elif key == ord("l"):
                    st["exp"] = int(cur_exp)
                    send_ctrl(ctrl_qs, st["exp"], st["iso"])
                elif key == ord("a"):
                    st["exp"] = None
                    send_ctrl(ctrl_qs, auto=True)
                elif key in (ord("i"), ord("k")):
                    st["iso"] = int(np.clip(st["iso"] * (1.25 if key == ord("i") else 0.8), 100, 1600))
                    if st["exp"]:
                        send_ctrl(ctrl_qs, st["exp"], st["iso"])
            elif not got:
                time.sleep(0.0005)
        if action != "record":
            return action, None

        # ---- timed capture: pull + copy only ----
        wait_writers(writers)
        time.sleep(args.delay)
        while qL.tryGet() is not None:
            pass
        while qR.tryGet() is not None:
            pass
        print(f"[{rec['id']}] RECORDING {args.seconds:.1f} s ...")
        L, R = [], []
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.seconds:
            got = False
            while (m := qL.tryGet()) is not None:
                L.append(pack(m))
                got = True
            while (m := qR.tryGet()) is not None:
                R.append(pack(m))
                got = True
            if not got:
                time.sleep(0.0002)
        elapsed = time.monotonic() - t0
    print(f"[{rec['id']}] captured {len(L)} left / {len(R)} right frames in {elapsed:.2f} s")
    return "record", {"L": L, "R": R, "elapsed": elapsed, "fps_requested": fps}


# ----------------------------------------------------------------------------- analysis + writing
def pair_frames(L, R):
    """Pair left/right frames by sequence number; fall back to nearest timestamp."""
    if not L or not R:
        return [], "none"
    sl = {f[0]: i for i, f in enumerate(L)}
    sr = {f[0]: i for i, f in enumerate(R)}
    common = sorted(set(sl) & set(sr))
    if common and len(common) >= 0.5 * min(len(L), len(R)):
        return [(sl[s], sr[s]) for s in common], "seq"
    tl = np.array([f[1] for f in L])
    tr = np.array([f[1] for f in R])
    tol = 0.5 * float(np.median(np.diff(tl))) if len(tl) > 1 else 0.005
    pairs = []
    for i, t in enumerate(tl):
        j = int(np.argmin(np.abs(tr - t)))
        if abs(tr[j] - t) <= tol:
            pairs.append((i, j))
    return pairs, "timestamp"


def seq_gaps(frames):
    s = np.array([f[0] for f in frames])
    return int(np.maximum(np.diff(s) - 1, 0).sum()) if len(s) > 1 else 0


def compute_stats(rec, args, cap, pairs, method):
    L, R = cap["L"], cap["R"]
    ts = np.array([L[i][1] for i, _ in pairs]) if pairs else np.array([])
    dts = np.diff(ts) * 1000 if len(ts) > 1 else np.array([0.0])
    skew = np.array([abs(L[i][1] - R[j][1]) for i, j in pairs]) * 1000 if pairs else np.array([0.0])
    exps = np.array([L[i][2] for i, _ in pairs]) if pairs else np.array([0.0])
    isos = np.array([L[i][3] for i, _ in pairs]) if pairs else np.array([0])
    sample = [L[i][4][::4, ::4] for i, _ in pairs[::10]] or [np.zeros((2, 2), np.uint8)]
    s = np.concatenate([x.ravel() for x in sample])
    fps_dev = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 0.0
    return {
        "id": rec["id"], "setup": rec["setup"], "motion": rec["motion"], "scene": rec["scene"],
        "res": rec["res"], "fps_requested": cap["fps_requested"], "pairs": len(pairs),
        "seconds": round(cap["elapsed"], 3), "fps_device": round(fps_dev, 2),
        "fps_host": round(len(pairs) / cap["elapsed"], 2) if cap["elapsed"] > 0 else 0,
        "seq_gaps_left": seq_gaps(L), "seq_gaps_right": seq_gaps(R),
        "unpaired_left": len(L) - len(pairs), "unpaired_right": len(R) - len(pairs),
        "pairing": method, "interval_ms_mean": round(float(dts.mean()), 3),
        "interval_ms_std": round(float(dts.std()), 3), "interval_ms_max": round(float(dts.max()), 3),
        "lr_skew_ms_max": round(float(skew.max()), 3),
        "exposure_us_median": round(float(np.median(exps)), 1), "iso_median": int(np.median(isos)),
        "brightness_mean": round(float(s.mean()), 1),
        "saturated_pct": round(float((s >= 250).mean() * 100), 2),
        "dark_pct": round(float((s <= 5).mean() * 100), 2),
    }


def write_recording(rec_dir, L, R, pairs, stats, args, summary_path, lock):
    """Runs in a background thread: PNGs, meta.csv, stats.json (last = 'done' marker), summary row."""
    rec_dir = Path(rec_dir)
    for sub in ("left", "right"):
        (rec_dir / sub).mkdir(parents=True, exist_ok=True)
    png = [cv2.IMWRITE_PNG_COMPRESSION, args.png_level]
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = []
        for k, (i, j) in enumerate(pairs):
            futs.append(ex.submit(cv2.imwrite, str(rec_dir / "left" / f"{k:06d}.png"), L[i][4], png))
            futs.append(ex.submit(cv2.imwrite, str(rec_dir / "right" / f"{k:06d}.png"), R[j][4], png))
        for f in futs:
            f.result()
    with open(rec_dir / "meta.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "seq_left", "seq_right", "ts_left_s", "ts_right_s", "exposure_us", "iso"])
        for k, (i, j) in enumerate(pairs):
            w.writerow([k, L[i][0], R[j][0], f"{L[i][1]:.6f}", f"{R[j][1]:.6f}",
                        f"{L[i][2]:.0f}", L[i][3]])
    dump_json(rec_dir / "stats.json", stats)
    with lock:
        new = not Path(summary_path).exists()
        with open(summary_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(stats)
    print(f"[{stats['id']}] saved {len(pairs)} pairs -> {rec_dir}")


# ----------------------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=f"raw_{dt.date.today():%Y%m%d}",
                    help="session folder name; reuse the same name to resume")
    ap.add_argument("--out", default="recordings", help="root output folder")
    ap.add_argument("--calib", default=None, metavar="FILE.json",
                    help="your dynamic calibration json; copied into the session and compared with factory")
    ap.add_argument("--seconds", type=float, default=5.0, help="length of every recording")
    ap.add_argument("--fps-800", type=int, default=120, help="requested FPS at 800p (use your measured max)")
    ap.add_argument("--fps-400", type=int, default=120, help="requested FPS at 400p (use your measured max)")
    ap.add_argument("--exposure-us", type=int, default=None, help="start manual exposure (default: auto)")
    ap.add_argument("--exposure-us-moving", type=int, default=None,
                    help="start manual exposure for moving scenes (short = less motion blur)")
    ap.add_argument("--iso", type=int, default=400, help="ISO used while exposure is manual")
    ap.add_argument("--ir-dots", type=float, default=1.0, help="IR dot projector 0..1 (1.0 = like your scanner script)")
    ap.add_argument("--ir-flood", type=float, default=0.0, help="IR flood light 0..1")
    ap.add_argument("--no-tuning", action="store_true", help="do not apply xlink-chunk / pool tuning")
    ap.add_argument("--xlink-chunk", type=int, default=0, help="XLink chunk size, 0 = no chunking")
    ap.add_argument("--pool", type=int, default=16, help="RAW/ISP frame pool size per camera")
    ap.add_argument("--queue", type=int, default=128, help="host queue size per camera (frames)")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between SPACE and capture start")
    ap.add_argument("--png-level", type=int, default=1, help="PNG compression 0-9 (lossless, 1 = fast)")
    ap.add_argument("--display-fps", type=float, default=20.0, help="preview refresh rate")
    ap.add_argument("--res", type=int, nargs="+", choices=[400, 800], default=None, help="only these resolutions")
    ap.add_argument("--only", default=None, help="comma list; keep ids containing any of these substrings")
    ap.add_argument("--redo", action="store_true", help="record again even if stats.json exists")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return ap.parse_args()


def main():
    args = parse_args()
    plan = build_plan(args)
    root = Path(args.out) / args.session
    est_gb = sum(RES[p["res"]][0] * RES[p["res"]][1] * 2 * (args.fps_800 if p["res"] == 800 else args.fps_400)
                 * args.seconds for p in plan) / 1e9
    print(f"Session '{args.session}': {len(plan)} recordings x {args.seconds:.0f} s "
          f"(~{est_gb:.0f} GB uncompressed, PNG is roughly half)")
    for p in plan:
        done = (root / p["id"] / "stats.json").exists()
        print(f"  {p['id']:<22} {'[done]' if done else ''}")
    if args.dry_run:
        return
    free_gb = shutil.disk_usage(Path(args.out).resolve().anchor or ".").free / 1e9
    print(f"Free disk: {free_gb:.0f} GB")

    sess = init_session(root, args, plan)
    if "dynamic_minus_factory" in sess["calibration"]:
        print("dynamic vs factory calibration:", json.dumps(sess["calibration"]["dynamic_minus_factory"]))

    state, writers, lock = {}, [], threading.Lock()
    last_setup = None
    try:
        for n, rec in enumerate(plan, 1):
            if (root / rec["id"] / "stats.json").exists() and not args.redo:
                print(f"[{rec['id']}] already done, skipping")
                continue
            if rec["setup"] != last_setup:
                last_setup = rec["setup"]
                print(f"\n=== {rec['setup']}: {MOTION_TEXT[rec['motion']]}, {SCENES[rec['scene']]} ===")
            action, cap = capture_one(rec, args, state, writers)
            if action == "quit":
                break
            if action == "skip":
                print(f"[{rec['id']}] skipped")
                continue
            pairs, method = pair_frames(cap["L"], cap["R"])
            stats = compute_stats(rec, args, cap, pairs, method)
            print(f"[{rec['id']}] {stats['fps_device']} fps (camera clock), {stats['pairs']} pairs, "
                  f"gaps L/R {stats['seq_gaps_left']}/{stats['seq_gaps_right']}, "
                  f"exp {stats['exposure_us_median']:.0f}us, sat {stats['saturated_pct']}%")
            t = threading.Thread(target=write_recording,
                                 args=(root / rec["id"], cap["L"], cap["R"], pairs, stats, args,
                                       root / "summary.csv", lock))
            t.start()
            writers.append(t)
    finally:
        wait_writers(writers)
        cv2.destroyAllWindows()
        sp = root / "session.json"
        if sp.exists():
            s = json.loads(sp.read_text())
            s["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
            s["recorded"] = sorted(p.parent.name for p in root.glob("*/stats.json"))
            s["last_exposure_state"] = state
            dump_json(sp, s)
        print(f"\nDone. Summary: {root / 'summary.csv'}")


if __name__ == "__main__":
    main()