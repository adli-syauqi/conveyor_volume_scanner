#!/usr/bin/env python3
"""
OAK-D Pro (RVC2) mono throughput sweep - DepthAI v3.

One run = one configuration (single or pair, 800P or 400P) swept over 5
requested FPS values. Each point: fresh pipeline, 45 s capture, 15 s cooldown.
All runs append to one CSV so the 4 configurations end up in one file.

Run the 4 configurations:
    python oak_mono_sweep.py --cams single --res 800
    python oak_mono_sweep.py --cams single --res 400
    python oak_mono_sweep.py --cams pair   --res 800
    python oak_mono_sweep.py --cams pair   --res 400

Useful options:
    --fps 30,60,90,110,129      override the 5 requested FPS values
    --exposure auto             compare against manual exposure (3A load)
    --type none                 let DepthAI pick the frame type (if GRAY8 fails)
    --type raw10                sensor RAW10-packed via the Camera 'raw' output
                                (bypasses the ISP output path, so it changes
                                bit depth AND processing; frame size is measured)
    --frames                    also write a per-frame CSV per point (arrival and
                                device-timestamp intervals, latency) to locate jitter
    --link-mbps 2192            link ceiling for the % column (health check value)

Definitions (thresholds are constants below):
    delivered FPS : frames arriving on the host / second
    sensor FPS    : frames the camera produced / second (from sequence numbers
                    and device timestamps), independent of delivery
    drops         : gaps in the frame sequence numbers
    sustained     : delivered >= 98% of requested AND zero drops AND
                    p99 frame interval <= 1.5x nominal (all streams must pass)
    latency       : host arrival time - frame capture timestamp (host-synced)
"""
import argparse
import bisect
import csv
import math
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
import depthai as dai

try:
    import psutil
except ImportError:
    psutil = None

# ---- constants -------------------------------------------------------------
RES = {800: (1280, 800), 400: (640, 400)}
DEFAULT_FPS = {800: [30, 60, 90, 110, 129], 400: [30, 90, 150, 200, 255]}
PASS_FPS_RATIO = 0.98
PASS_JITTER_FACTOR = 1.5
ISP_3A_NOTE_FPS = 250  # Luxonis docs: RVC2 3A tops out ~250 fps aggregated
MIB = 1024 * 1024
BOOT_ESTIMATE_S = 6

CSV_FIELDS = [
    "run_id", "timestamp", "cams", "res", "exposure", "frame_type_req",
    "requested_fps", "agg_requested_fps", "stream", "status",
    "sensor_mode", "sensor_fps_reported", "frame_type", "bytes_pp",
    "window_s", "frames", "delivered_fps", "sensor_fps", "fps_min_1s",
    "fps_1pct_low", "drops", "drop_pct", "interval_p99_ms", "interval_std_ms",
    "lat_avg_ms", "lat_p50_ms", "lat_p99_ms", "lat_max_ms",
    "data_mbps", "link_pct", "point_mbps", "point_link_pct",
    "skew_avg_ms", "skew_p99_ms", "skew_max_ms", "unpaired",
    "stream_pass", "point_sustained", "fail_reasons",
    "temp_start_c", "temp_end_c", "temp_max_c", "temp_delta_c",
    "cpu_css_avg", "cpu_css_max", "cpu_mss_avg", "cpu_mss_max",
    "ddr_used_mib_avg", "host_proc_cpu_avg", "host_proc_cpu_max",
    "host_sys_cpu_avg",
]


# ---- helpers ---------------------------------------------------------------
def now_s():
    return dai.Clock.now().total_seconds()


def num(x, spec=".1f"):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return format(x, spec)


def safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def pct(a, q):
    return float(np.percentile(a, q))


def build_cam(pipeline, socket, size, fps):
    cam = pipeline.create(dai.node.Camera)
    try:
        return cam.build(socket, sensorResolution=size, sensorFps=fps)
    except TypeError:
        if not getattr(build_cam, "warned", False):
            print("  note: build(sensorResolution=...) not accepted; "
                  "sensor mode will be auto-selected from the output request")
            build_cam.warned = True
        return cam.build(socket, sensorFps=fps)


def frame_meta(f):
    w = safe(f.getWidth, 0)
    h = safe(f.getHeight, 0)
    bpp = safe(f.getBytesPerPixel, 1.0)
    nbytes = safe(lambda: f.getData().nbytes)  # real size as received
    if nbytes and w and h:
        bpp = nbytes / (w * h)  # measured bytes per pixel (e.g. 1.25 for RAW10)
    return {
        "type": str(safe(f.getType, "?")).split(".")[-1],
        "bpp": bpp,
        "w": w,
        "h": h,
        "sensor_mode": safe(f.getSensorMode),
        "sensor_fps": safe(f.getFps),
    }


# ---- analysis --------------------------------------------------------------
def stream_stats(samples, req_fps, bpf, link_mbps, t_lo, t_hi):
    """samples: list of (arrival_s, ts_sync_s, ts_dev_s, seq). Window-filtered."""
    w = [s for s in samples if t_lo <= s[0] <= t_hi]
    if len(w) < 3:
        return None
    arr = np.array([s[0] for s in w])
    ts = np.array([s[1] for s in w])
    dev = np.array([s[2] for s in w])
    seq = np.array([s[3] for s in w], dtype=np.int64)

    n = len(w)
    span = arr[-1] - arr[0]
    iv = np.diff(arr)
    lat = (arr - ts) * 1000.0
    expected = int(seq[-1] - seq[0] + 1)
    drops = expected - n
    dev_span = dev[-1] - dev[0]
    counts = np.bincount(np.floor(arr - arr[0]).astype(int))[:-1]
    iv_p99 = pct(iv, 99)

    mbps = (n - 1) * bpf * 8 / span / 1e6 if span > 0 else float("nan")
    st = {
        "window_s": span,
        "frames": n,
        "delivered_fps": (n - 1) / span if span > 0 else float("nan"),
        "sensor_fps": (seq[-1] - seq[0]) / dev_span if dev_span > 0 else float("nan"),
        "fps_min_1s": float(counts.min()) if len(counts) else float("nan"),
        "fps_1pct_low": 1.0 / iv_p99 if iv_p99 > 0 else float("nan"),
        "drops": drops,
        "drop_pct": 100.0 * drops / expected if expected > 0 else float("nan"),
        "interval_p99_ms": iv_p99 * 1000,
        "interval_std_ms": float(iv.std()) * 1000,
        "lat_avg_ms": float(lat.mean()),
        "lat_p50_ms": pct(lat, 50),
        "lat_p99_ms": pct(lat, 99),
        "lat_max_ms": float(lat.max()),
        "data_mbps": mbps,
        "link_pct": 100.0 * mbps / link_mbps if link_mbps else float("nan"),
        "ts": ts,
    }
    reasons = []
    if st["delivered_fps"] < PASS_FPS_RATIO * req_fps:
        reasons.append("fps")
    if drops > 0:
        reasons.append("drops")
    if iv_p99 > PASS_JITTER_FACTOR / req_fps:
        reasons.append("jitter")
    st["reasons"] = reasons
    st["pass"] = not reasons
    return st


def pair_skew(ts_left_window, ts_right_all, req_fps):
    """Nearest-timestamp skew of each left frame vs the right stream."""
    right = sorted(ts_right_all)
    if not right:
        return None
    skews = []
    for t in ts_left_window:
        i = bisect.bisect_left(right, t)
        cands = []
        if i < len(right):
            cands.append(abs(right[i] - t))
        if i > 0:
            cands.append(abs(right[i - 1] - t))
        skews.append(min(cands))
    sk = np.array(skews) * 1000.0
    half_interval_ms = 0.5 / req_fps * 1000.0
    return {
        "skew_avg_ms": float(sk.mean()),
        "skew_p99_ms": pct(sk, 99),
        "skew_max_ms": float(sk.max()),
        "unpaired": int((sk > half_interval_ms).sum()),
    }


def usage_stats(usage, host, t_lo, t_hi):
    u = [x for x in usage if t_lo <= x[0] <= t_hi]
    h = [x for x in host if t_lo <= x[0] <= t_hi]
    out = {}
    if u:
        temp = [x[1] for x in u]
        out.update({
            "temp_start_c": temp[0], "temp_end_c": temp[-1],
            "temp_max_c": max(temp), "temp_delta_c": temp[-1] - temp[0],
            "cpu_css_avg": float(np.mean([x[2] for x in u])),
            "cpu_css_max": max(x[2] for x in u),
            "cpu_mss_avg": float(np.mean([x[3] for x in u])),
            "cpu_mss_max": max(x[3] for x in u),
            "ddr_used_mib_avg": float(np.mean([x[4] for x in u])),
        })
    if h:
        out.update({
            "host_proc_cpu_avg": float(np.mean([x[1] for x in h])),
            "host_proc_cpu_max": max(x[1] for x in h),
            "host_sys_cpu_avg": float(np.mean([x[2] for x in h])),
        })
    return out


# ---- one sweep point -------------------------------------------------------
def run_point(args, fps, streams, size):
    """Returns dict with raw samples, meta, usage, host, t0 (or raises)."""
    store = {name: [] for name, _ in streams}
    meta = {}
    usage, host = [], []
    stop = threading.Event()

    with dai.Pipeline() as pipeline:
        device = pipeline.getDefaultDevice()
        print(f"  device {device.getDeviceInfo().getDeviceId()}  "
              f"USB {device.getUsbSpeed()}")

        queues = {}
        for name, sock in streams:
            cam = build_cam(pipeline, sock, size, fps)
            if args.exposure == "manual":
                exp_us = min(args.exp_us, int(1e6 / fps) - 100)
                try:
                    cam.initialControl.setManualExposure(exp_us, args.iso)
                except Exception as e:
                    print(f"  warning: manual exposure not applied ({e})")
            if args.type == "raw10":
                out = cam.raw  # RAW10-packed straight from the sensor
            else:
                kw = {"size": size, "fps": fps}
                if args.type == "gray8":
                    kw["type"] = dai.ImgFrame.Type.GRAY8
                out = cam.requestOutput(**kw)
            queues[name] = out.createOutputQueue(maxSize=8, blocking=False)

        sys_log = pipeline.create(dai.node.SystemLogger)
        sys_log.setRate(1.0)
        sys_q = sys_log.out.createOutputQueue(maxSize=4, blocking=False)

        def reader(name, q):
            while not stop.is_set():
                try:
                    f = q.get()  # blocks until a frame arrives
                except Exception:
                    return
                t = now_s()  # stamp immediately, before any other work
                if f is None:
                    continue
                if name not in meta:
                    meta[name] = frame_meta(f)
                store[name].append((
                    t,
                    f.getTimestamp().total_seconds(),
                    f.getTimestampDevice().total_seconds(),
                    f.getSequenceNum(),
                ))

        pipeline.start()
        threads = [threading.Thread(target=reader, args=(n, q), daemon=True)
                   for n, q in queues.items()]
        for th in threads:
            th.start()

        # wait for every stream to deliver its first frame
        t_wait = now_s()
        while not all(store[n] for n in store):
            if now_s() - t_wait > 20:
                stop.set()
                hint = (" (raw-only output may need another output requested)"
                        if args.type == "raw10" else "")
                raise RuntimeError("no frames within 20 s" + hint)
            time.sleep(0.05)
        t0 = max(store[n][0][0] for n in store)
        t_end = t0 + args.duration

        proc = None
        if psutil:
            proc = psutil.Process()
            proc.cpu_percent(None)
            psutil.cpu_percent(None)
        next_host, next_print = t0 + 1, t0 + 5

        while True:
            t = now_s()
            if t >= t_end:
                break
            while True:
                m = sys_q.tryGet()
                if m is None:
                    break
                usage.append((
                    now_s(),
                    m.chipTemperature.average,
                    m.leonCssCpuUsage.average * 100,
                    m.leonMssCpuUsage.average * 100,
                    m.ddrMemoryUsage.used / MIB,
                ))
            if proc and t >= next_host:
                host.append((t, proc.cpu_percent(None), psutil.cpu_percent(None)))
                next_host += 1
            if t >= next_print:
                parts = []
                for n in store:
                    recent = sum(1 for s in store[n] if s[0] >= t - 5)
                    parts.append(f"{n} {recent / 5:6.1f} fps")
                temp = f"{usage[-1][1]:.1f}C" if usage else "n/a"
                print(f"    t={t - t0:4.0f}s  {'  '.join(parts)}  temp {temp}")
                next_print += 5
            time.sleep(0.05)

        stop.set()
        for th in threads:
            th.join(timeout=2)

    return {"store": store, "meta": meta, "usage": usage, "host": host,
            "t0": t0, "t_end": t_end}


# ---- reporting -------------------------------------------------------------
def analyse_point(args, fps, streams, data, link_mbps):
    t_lo = data["t0"] + args.warmup
    t_hi = data["t_end"]
    per = {}
    for name, _ in streams:
        m = data["meta"].get(name, {})
        bpf = m.get("w", 0) * m.get("h", 0) * m.get("bpp", 1.0)
        per[name] = stream_stats(data["store"][name], fps, bpf, link_mbps, t_lo, t_hi)
    if any(v is None for v in per.values()):
        return None
    skew = None
    if len(streams) == 2:
        skew = pair_skew(per["L"]["ts"], [s[1] for s in data["store"]["R"]], fps)
    use = usage_stats(data["usage"], data["host"], t_lo, t_hi)
    point_ok = all(v["pass"] for v in per.values())
    reasons = sorted({r for v in per.values() for r in v["reasons"]})
    point_mbps = sum(v["data_mbps"] for v in per.values())
    return {"per": per, "skew": skew, "use": use, "ok": point_ok,
            "reasons": reasons, "point_mbps": point_mbps,
            "point_link_pct": 100.0 * point_mbps / link_mbps if link_mbps else float("nan")}


TABLE_HDR = (f"{'req':>4} {'strm':>4} {'sensFPS':>8} {'delivFPS':>9} {'min1s':>6} "
             f"{'low1%':>6} {'drop%':>6} {'lat p50/p99/max ms':>22} "
             f"{'Mbps':>7} {'%link':>6}  pass")


def print_point(fps, res):
    for name, st in res["per"].items():
        verdict = "PASS" if st["pass"] else "FAIL(" + ",".join(st["reasons"]) + ")"
        lat = f"{st['lat_p50_ms']:6.1f}/{st['lat_p99_ms']:6.1f}/{st['lat_max_ms']:6.1f}"
        print(f"{fps:>4} {name:>4} {num(st['sensor_fps']):>8} "
              f"{st['delivered_fps']:9.2f} {num(st['fps_min_1s']):>6} "
              f"{num(st['fps_1pct_low']):>6} {st['drop_pct']:6.2f} {lat:>22} "
              f"{st['data_mbps']:7.0f} {st['link_pct']:6.1f}  {verdict}")
    u, sk = res["use"], res["skew"]
    line = [f"total {res['point_mbps']:.0f} Mbps ({res['point_link_pct']:.1f}% link)"]
    if sk:
        line.append(f"L-R skew avg/p99/max {sk['skew_avg_ms']:.2f}/"
                    f"{sk['skew_p99_ms']:.2f}/{sk['skew_max_ms']:.2f} ms, "
                    f"unpaired {sk['unpaired']}")
    if "temp_delta_c" in u:
        line.append(f"temp {u['temp_start_c']:.1f}->{u['temp_end_c']:.1f}C "
                    f"({u['temp_delta_c']:+.1f})")
    if "cpu_css_avg" in u:
        line.append(f"LeonCSS {u['cpu_css_avg']:.1f}/{u['cpu_css_max']:.1f}% "
                    f"MSS {u['cpu_mss_avg']:.1f}/{u['cpu_mss_max']:.1f}%")
    if "host_proc_cpu_avg" in u:
        line.append(f"host proc CPU {u['host_proc_cpu_avg']:.0f}/"
                    f"{u['host_proc_cpu_max']:.0f}% (100% = 1 core)")
    print("     " + " | ".join(line))


def csv_rows(args, run_id, fps, streams, res, status, error=""):
    base = {
        "run_id": run_id, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cams": args.cams, "res": args.res, "exposure": args.exposure,
        "frame_type_req": args.type, "requested_fps": fps,
        "agg_requested_fps": fps * len(streams), "status": status,
    }
    if res is None:
        return [dict(base, stream=n, fail_reasons=error) for n, _ in streams]
    rows = []
    for name, st in res["per"].items():
        row = dict(base)
        row.update({k: v for k, v in st.items() if k in CSV_FIELDS})
        row.update(res["use"])
        if res["skew"]:
            row.update(res["skew"])
        row.update({
            "stream": name, "stream_pass": st["pass"],
            "point_sustained": res["ok"], "fail_reasons": ",".join(res["reasons"]),
            "point_mbps": res["point_mbps"], "point_link_pct": res["point_link_pct"],
        })
        rows.append(row)
    return rows


def dump_frames(args, run_id, fps, data):
    """Per-frame log: shows whether long gaps come from the sensor side
    (device_interval) or from delivery (arrival_interval)."""
    path = f"frames_{args.cams}_{args.res}p_{args.type}_{fps}fps_{run_id}.csv"
    t0, t_lo, t_hi = data["t0"], data["t0"] + args.warmup, data["t_end"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream", "t_rel_s", "seq", "arrival_interval_ms",
                    "device_interval_ms", "latency_ms", "in_window"])
        for name, samples in data["store"].items():
            prev = None
            for arr, ts, dev, seq in samples:
                ai = di = ""
                if prev is not None:
                    ai = round((arr - prev[0]) * 1000, 3)
                    di = round((dev - prev[1]) * 1000, 3)
                w.writerow([name, round(arr - t0, 4), seq, ai, di,
                            round((arr - ts) * 1000, 3), int(t_lo <= arr <= t_hi)])
                prev = (arr, dev)
    print(f"  per-frame log: {path}")


def write_csv(path, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, restval="", extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})


# ---- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cams", choices=["single", "pair"], default="pair")
    ap.add_argument("--res", type=int, choices=[800, 400], default=800)
    ap.add_argument("--fps", type=str, default=None, help="comma list, 5 values by default")
    ap.add_argument("--duration", type=float, default=45.0)
    ap.add_argument("--cooldown", type=float, default=15.0)
    ap.add_argument("--warmup", type=float, default=3.0, help="seconds discarded from stats")
    ap.add_argument("--exposure", choices=["manual", "auto"], default="manual")
    ap.add_argument("--exp-us", type=int, default=2000)
    ap.add_argument("--iso", type=int, default=400)
    ap.add_argument("--type", choices=["gray8", "none", "raw10"], default="gray8")
    ap.add_argument("--frames", action="store_true",
                    help="write a per-frame CSV for each point")
    ap.add_argument("--link-mbps", type=float, default=2192.0)
    ap.add_argument("--csv", default="throughput_sweep.csv")
    args = ap.parse_args()

    size = RES[args.res]
    fps_list = [float(x) if "." in x else int(x) for x in args.fps.split(",")] \
        if args.fps else DEFAULT_FPS[args.res]
    streams = [("L", dai.CameraBoardSocket.CAM_B)]
    if args.cams == "pair":
        streams.append(("R", dai.CameraBoardSocket.CAM_C))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    n = len(fps_list)
    est_min = (n * (args.duration + BOOT_ESTIMATE_S) + (n - 1) * args.cooldown) / 60

    print("=" * 100)
    print(f"MONO THROUGHPUT SWEEP  |  {args.cams} {args.res}P ({size[0]}x{size[1]})  |  "
          f"exposure {args.exposure}  |  type {args.type}")
    print(f"requested FPS: {fps_list}  |  {args.duration:.0f} s each "
          f"(first {args.warmup:.0f} s discarded)  |  cooldown {args.cooldown:.0f} s  "
          f"|  ~{est_min:.1f} min")
    print(f"sustained = delivered >= {PASS_FPS_RATIO:.0%} of requested, 0 drops, "
          f"p99 interval <= {PASS_JITTER_FACTOR}x nominal  |  link ceiling "
          f"{args.link_mbps:.0f} Mbps")
    if psutil is None:
        print("note: psutil not installed -> host CPU columns will be blank "
              "(pip install psutil)")
    over = [f for f in fps_list if f * len(streams) > ISP_3A_NOTE_FPS]
    if over:
        print(f"note: aggregate requested FPS > {ISP_3A_NOTE_FPS} at {over}; Luxonis docs say "
              f"RVC2 3A tops out around {ISP_3A_NOTE_FPS} fps across all camera streams "
              f"(compare --exposure auto vs manual)")
    print("=" * 100)
    print(TABLE_HDR)
    print("-" * 100)

    summary = []
    for i, fps in enumerate(fps_list, 1):
        print(f"[{i}/{n}] requesting {fps} fps ...")
        res, status, err = None, "OK", ""
        try:
            data = run_point(args, fps, streams, size)
            m = data["meta"]
            print("  frame: " + ", ".join(
                f"{k}: {v['type']} {v['w']}x{v['h']} {v['bpp']}B/px "
                f"sensor_mode={v['sensor_mode']} sensor_fps={num(v['sensor_fps'])}"
                for k, v in m.items()))
            if args.type == "gray8":
                for k, v in m.items():
                    if v["type"] != "GRAY8":
                        print(f"  WARNING: requested GRAY8 but stream {k} "
                              f"delivers {v['type']}")
            if args.frames:
                dump_frames(args, run_id, fps, data)
            res = analyse_point(args, fps, streams, data, args.link_mbps)
            if res is None:
                status, err = "NO_DATA", "too few frames in window"
        except Exception as e:  # keep sweeping even if one point fails
            status, err = "ERROR", f"{type(e).__name__}: {e}"

        if res is not None:
            for k, v in data["meta"].items():
                res["per"][k].update({"sensor_mode": v["sensor_mode"],
                                      "sensor_fps_reported": v["sensor_fps"],
                                      "frame_type": v["type"], "bytes_pp": v["bpp"]})
            print_point(fps, res)
            summary.append((fps, res["ok"], res["reasons"],
                            {k: v["delivered_fps"] for k, v in res["per"].items()}))
        else:
            print(f"{fps:>4}  {status}: {err}")
            summary.append((fps, False, [status], {}))

        write_csv(args.csv, csv_rows(args, run_id, fps, streams, res, status, err))
        if i < n:
            print(f"  cooling down {args.cooldown:.0f} s ...")
            time.sleep(args.cooldown)

    print("=" * 100)
    print(f"SUMMARY  {args.cams} {args.res}P  (exposure {args.exposure})")
    print(f"{'requested':>10}  {'delivered (per stream)':<26} result")
    for fps, ok, reasons, deliv in summary:
        d = " / ".join(f"{k} {v:.1f}" for k, v in deliv.items()) or "-"
        print(f"{fps:>10}  {d:<26} {'PASS' if ok else 'FAIL (' + ','.join(reasons) + ')'}")
    passed = [s[0] for s in summary if s[1]]
    print()
    if passed:
        best = max(passed)
        print(f"Max sustained requested FPS: {best}")
        higher = [s for s in summary if s[0] > best]
        if higher:
            print(f"First failing point above it: {higher[0][0]} fps "
                  f"({','.join(higher[0][2])})")
    else:
        print("No point met the sustained criteria.")
    print(f"Rows appended to {args.csv}")


if __name__ == "__main__":
    main()