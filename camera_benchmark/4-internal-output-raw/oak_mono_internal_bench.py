#!/usr/bin/env python3
"""
OAK-D Pro (RVC2) mono internal benchmark - DepthAI v3.

Same cameras and settings as oak_mono_sweep.py, but frames are counted by
Luxonis's own BenchmarkIn node instead of my Python readers. Adding one stage
at a time tells you where the pair breaks:

    Test A  (--where device): camera -> ISP -> BenchmarkIn ON THE DEVICE
            (no USB, no Python)
    Test B  (--where host)  : camera -> ISP -> USB -> BenchmarkIn ON THE HOST
            (USB + C++ receive path, but no Python reading frames)
    Chapter-3 result        : camera -> ISP -> USB -> Python threads
            (read automatically from throughput_sweep.csv if present)

Reading guide (verdict = delivered FPS >= 98% of requested):
    A fails            -> limit is inside the camera / ISP
    A ok, B fails      -> frames are lost between device and host
    A ok, B ok, sweep fails -> my Python readers were the bottleneck
Drops and frame-interval jitter cannot be measured here (BenchmarkIn only counts).

Usage:
    python oak_mono_internal_bench.py                       # pair 800P, A and B
    python oak_mono_internal_bench.py --fps 60,90,110       # fewer points
    python oak_mono_internal_bench.py --where device        # only Test A
    python oak_mono_internal_bench.py --cams single --res 400

Output: terminal tables + internal_bench.csv (appended).
"""
import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import depthai as dai

try:
    import psutil
except ImportError:
    psutil = None

RES = {800: (1280, 800), 400: (640, 400)}
DEFAULT_FPS = {800: [30, 60, 90, 110, 129], 400: [30, 90, 150, 200, 255]}
PASS_RATIO = 0.98

CSV_FIELDS = [
    "run_id", "timestamp", "where", "cams", "res", "exposure", "requested_fps",
    "agg_requested_fps", "stream", "status", "reports", "delivered_fps",
    "delivered_pct_of_req", "report_fps_min", "report_fps_max",
    "lat_avg_ms", "lat_max_ms", "stream_pass", "point_pass", "sweep_delivered_fps",
    "temp_start_c", "temp_end_c", "cpu_css_avg", "cpu_css_max",
    "cpu_mss_avg", "cpu_mss_max", "host_proc_cpu_avg", "host_proc_cpu_max",
    "host_sys_cpu_avg", "error",
]


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


def summarize_reports(reports, req_fps, skip):
    """reports: list of (fps, n_msgs, time_total_s, avg_latency_s)."""
    r = reports[skip:]
    if len(r) < 2:
        return None
    msgs = sum(x[1] for x in r)
    secs = sum(x[2] for x in r)
    if secs <= 0:
        return None
    delivered = msgs / secs
    lat = [x[3] * 1000 for x in r]
    return {
        "reports": len(r),
        "delivered_fps": delivered,
        "delivered_pct_of_req": 100.0 * delivered / req_fps,
        "report_fps_min": min(x[0] for x in r),
        "report_fps_max": max(x[0] for x in r),
        "lat_avg_ms": float(np.mean(lat)),
        "lat_max_ms": float(max(lat)),
        "pass": delivered >= PASS_RATIO * req_fps,
    }


def usage_summary(usage, host):
    out = {}
    if usage:
        temp = [u[0] for u in usage]
        out.update({
            "temp_start_c": temp[0], "temp_end_c": temp[-1],
            "cpu_css_avg": float(np.mean([u[1] for u in usage])),
            "cpu_css_max": max(u[1] for u in usage),
            "cpu_mss_avg": float(np.mean([u[2] for u in usage])),
            "cpu_mss_max": max(u[2] for u in usage),
        })
    if host:
        out.update({
            "host_proc_cpu_avg": float(np.mean([h[0] for h in host])),
            "host_proc_cpu_max": max(h[0] for h in host),
            "host_sys_cpu_avg": float(np.mean([h[1] for h in host])),
        })
    return out


def load_reference(path, cams, res):
    """Latest oak_mono_sweep.py result per requested FPS (min over streams)."""
    if not os.path.exists(path):
        return {}
    by = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("cams") == cams and r.get("res") == str(res)
                    and r.get("exposure") == "manual"
                    and r.get("frame_type_req") == "gray8"
                    and r.get("status") == "OK" and r.get("delivered_fps")):
                key = (float(r["requested_fps"]), r["run_id"])
                by.setdefault(key, []).append(float(r["delivered_fps"]))
    latest = {}
    for (fps, run), vals in by.items():
        if fps not in latest or run > latest[fps][0]:
            latest[fps] = (run, min(vals))
    return {fps: v[1] for fps, v in latest.items()}


def hint(fps, a, b, c):
    ok = lambda v: v is not None and v >= PASS_RATIO * fps
    if a is not None and not ok(a):
        return "limit inside camera/ISP"
    if a is not None and ok(a) and b is not None and not ok(b):
        return "lost between device and host"
    if ok(a) and ok(b) and c is not None and not ok(c):
        return "Python readers were the bottleneck"
    if b is not None and not ok(b) and a is None:
        return "fails once transfer is included"
    if (a is None or ok(a)) and (b is None or ok(b)) and (c is None or ok(c)):
        return "all stages deliver"
    return "-"


def run_point(args, where, fps, streams, size):
    reports = {n: [] for n, _ in streams}
    usage, host = [], []

    with dai.Pipeline() as pipeline:
        device = pipeline.getDefaultDevice()
        print(f"  device {device.getDeviceInfo().getDeviceId()}  "
              f"USB {device.getUsbSpeed()}  |  BenchmarkIn on {where.upper()}")

        per_report = max(10, int(0.5 * fps))
        queues = {}
        for name, sock in streams:
            cam = build_cam(pipeline, sock, size, fps)
            if args.exposure == "manual":
                exp_us = min(args.exp_us, int(1e6 / fps) - 100)
                try:
                    cam.initialControl.setManualExposure(exp_us, args.iso)
                except Exception as e:
                    print(f"  warning: manual exposure not applied ({e})")
            out = cam.requestOutput(size=size, fps=fps, type=dai.ImgFrame.Type.GRAY8)
            bi = pipeline.create(dai.node.BenchmarkIn)
            bi.setRunOnHost(where == "host")
            bi.sendReportEveryNMessages(per_report)
            bi.logReportsAsWarnings(False)
            out.link(bi.input)
            queues[name] = bi.report.createOutputQueue(maxSize=16, blocking=False)

        sys_log = pipeline.create(dai.node.SystemLogger)
        sys_log.setRate(1.0)
        sys_q = sys_log.out.createOutputQueue(maxSize=4, blocking=False)

        pipeline.start()

        def poll():
            for name, q in queues.items():
                while True:
                    r = q.tryGet()
                    if r is None:
                        break
                    reports[name].append((r.fps, r.numMessagesReceived,
                                          r.timeTotal, r.averageLatency))
            while True:
                m = sys_q.tryGet()
                if m is None:
                    break
                usage.append((m.chipTemperature.average,
                              m.leonCssCpuUsage.average * 100,
                              m.leonMssCpuUsage.average * 100))

        t_wait = time.monotonic()
        while not all(reports[n] for n in reports):
            poll()
            if time.monotonic() - t_wait > 20:
                raise RuntimeError("no benchmark reports within 20 s")
            time.sleep(0.05)

        proc = None
        if psutil:
            proc = psutil.Process()
            proc.cpu_percent(None)
            psutil.cpu_percent(None)
        usage.clear()
        t_end = time.monotonic() + args.duration
        next_host = time.monotonic() + 1
        next_print = time.monotonic() + 5
        t_start = time.monotonic()
        while time.monotonic() < t_end:
            poll()
            now = time.monotonic()
            if proc and now >= next_host:
                host.append((proc.cpu_percent(None), psutil.cpu_percent(None)))
                next_host += 1
            if now >= next_print:
                last = "  ".join(f"{n} {reports[n][-1][0]:6.1f} fps" for n in reports)
                print(f"    t={now - t_start:4.0f}s  {last}")
                next_print += 5
            time.sleep(0.05)
        poll()

    return reports, usage, host


def csv_rows(args, run_id, where, fps, streams, per, use, point_pass, ref, status, err):
    base = {
        "run_id": run_id, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "where": where, "cams": args.cams, "res": args.res, "exposure": args.exposure,
        "requested_fps": fps, "agg_requested_fps": fps * len(streams),
        "status": status, "error": err, "sweep_delivered_fps": ref,
    }
    if not per:
        return [dict(base, stream=n) for n, _ in streams]
    rows = []
    for name, st in per.items():
        row = dict(base, stream=name, stream_pass=st["pass"], point_pass=point_pass)
        row.update({k: v for k, v in st.items() if k in CSV_FIELDS})
        row.update(use)
        rows.append(row)
    return rows


def write_csv(path, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, restval="", extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--where", choices=["device", "host", "both"], default="both")
    ap.add_argument("--cams", choices=["single", "pair"], default="pair")
    ap.add_argument("--res", type=int, choices=[800, 400], default=800)
    ap.add_argument("--fps", type=str, default=None)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--skip-reports", type=int, default=2,
                    help="warm-up reports discarded per stream")
    ap.add_argument("--exposure", choices=["manual", "auto"], default="manual")
    ap.add_argument("--exp-us", type=int, default=2000)
    ap.add_argument("--iso", type=int, default=400)
    ap.add_argument("--csv", default="internal_bench.csv")
    ap.add_argument("--ref-csv", default="throughput_sweep.csv",
                    help="mono sweep CSV to compare against")
    args = ap.parse_args()

    size = RES[args.res]
    fps_list = ([float(x) if "." in x else int(x) for x in args.fps.split(",")]
                if args.fps else DEFAULT_FPS[args.res])
    wheres = ["device", "host"] if args.where == "both" else [args.where]
    streams = [("L", dai.CameraBoardSocket.CAM_B)]
    if args.cams == "pair":
        streams.append(("R", dai.CameraBoardSocket.CAM_C))
    ref = load_reference(args.ref_csv, args.cams, args.res)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_runs = len(fps_list) * len(wheres)
    est = (n_runs * (args.duration + 8) + (n_runs - 1) * args.cooldown) / 60

    print("=" * 100)
    print(f"MONO INTERNAL BENCHMARK  |  {args.cams} {args.res}P ({size[0]}x{size[1]})  |  "
          f"exposure {args.exposure}  |  BenchmarkIn on: {', '.join(wheres)}")
    print(f"requested FPS: {fps_list}  |  {args.duration:.0f} s each  |  "
          f"cooldown {args.cooldown:.0f} s  |  ~{est:.1f} min")
    print(f"verdict: delivered >= {PASS_RATIO:.0%} of requested FPS (drops/jitter not measurable here)")
    print(f"sweep reference: {len(ref)} point(s) from {args.ref_csv}"
          if ref else f"sweep reference: none found in {args.ref_csv}")
    if psutil is None:
        print("note: psutil not installed -> host CPU columns blank (pip install psutil)")
    print("=" * 100)
    print(f"{'where':>6} {'req':>4} {'strm':>4} {'delivFPS':>9} {'%req':>6} "
          f"{'rep-fps min/max':>16} {'lat avg/max ms':>15}  verdict")
    print("-" * 100)

    results = {}
    k = 0
    for fps in fps_list:
        for where in wheres:
            k += 1
            print(f"[{k}/{n_runs}] {where} | requesting {fps} fps ...")
            per, use, status, err, point_pass = {}, {}, "OK", "", False
            try:
                reports, usage, host = run_point(args, where, fps, streams, size)
                for name in reports:
                    s = summarize_reports(reports[name], fps, args.skip_reports)
                    if s is None:
                        raise RuntimeError(f"too few reports for stream {name}")
                    per[name] = s
                use = usage_summary(usage, host)
                point_pass = all(s["pass"] for s in per.values())
                results[(fps, where)] = min(s["delivered_fps"] for s in per.values())
            except Exception as e:
                status, err, per = "ERROR", f"{type(e).__name__}: {e}", {}

            if per:
                for name, s in per.items():
                    print(f"{where:>6} {fps:>4} {name:>4} {s['delivered_fps']:9.2f} "
                          f"{s['delivered_pct_of_req']:6.1f} "
                          f"{s['report_fps_min']:7.1f}/{s['report_fps_max']:<7.1f} "
                          f"{s['lat_avg_ms']:7.1f}/{s['lat_max_ms']:<7.1f}  "
                          f"{'PASS' if s['pass'] else 'FAIL'}")
                line = []
                if "temp_end_c" in use:
                    line.append(f"temp {use['temp_start_c']:.1f}->{use['temp_end_c']:.1f}C")
                    line.append(f"LeonCSS {use['cpu_css_avg']:.1f}/{use['cpu_css_max']:.1f}% "
                                f"MSS {use['cpu_mss_avg']:.1f}/{use['cpu_mss_max']:.1f}%")
                if "host_proc_cpu_avg" in use:
                    line.append(f"host proc CPU {use['host_proc_cpu_avg']:.0f}/"
                                f"{use['host_proc_cpu_max']:.0f}% (100% = 1 core)")
                if line:
                    print("       " + " | ".join(line))
            else:
                print(f"{where:>6} {fps:>4}  {status}: {err}")

            write_csv(args.csv, csv_rows(args, run_id, where, fps, streams, per, use,
                                         point_pass, ref.get(float(fps), ""), status, err))
            if k < n_runs:
                time.sleep(args.cooldown)

    print("=" * 100)
    print(f"SUMMARY  {args.cams} {args.res}P - delivered FPS (min over streams)")
    print(f"{'requested':>10} {'A: device':>11} {'B: host':>10} {'sweep (Py)':>11}  reading")
    for fps in fps_list:
        a, b = results.get((fps, "device")), results.get((fps, "host"))
        c = ref.get(float(fps))
        fmt = lambda v: f"{v:.1f}" if v is not None else "-"
        print(f"{fps:>10} {fmt(a):>11} {fmt(b):>10} {fmt(c):>11}  {hint(fps, a, b, c)}")
    print(f"\nRows appended to {args.csv}")


if __name__ == "__main__":
    main()