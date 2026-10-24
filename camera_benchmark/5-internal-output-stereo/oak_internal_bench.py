#!/usr/bin/env python3
"""
OAK-D Pro (RVC2) internal benchmark, 4 stages - DepthAI v3. (Chapter 4)

Frames are counted by Luxonis's own BenchmarkIn node. Stages add one piece at a
time, using the same camera pair and settings as oak_mono_sweep.py:

    A  mono-device   : cameras -> ISP -> BenchmarkIn ON THE DEVICE   (no USB)
    B  mono-host     : cameras -> ISP -> USB -> BenchmarkIn ON THE HOST
    C  stereo-device : cameras -> StereoDepth -> BenchmarkIn ON THE DEVICE
    D  stereo-host   : cameras -> StereoDepth -> USB -> BenchmarkIn ON THE HOST

The mono-sweep (Python readers) result is shown next to A/B when found in
throughput_sweep.csv. For stereo, the published Luxonis RVC2 FPS for the chosen
preset is shown for reference (measured with BenchmarkIn on device, same as C).

Verdict = delivered FPS >= 98% of requested (drops/jitter are not measurable here).

Tuning (applies to every stage; both off by default):
    --xlink-chunk 0   Pipeline.setXLinkChunkSize: size of the pieces a device-sent
                      message is split into (default 64 KB; 0 = no splitting).
                      Luxonis's published performance tables used 0.
    --pool 16         Camera.setRawNumFramesPool / setIspNumFramesPool: number of
                      frame buffers per camera stage.
    --tuned           shorthand for --xlink-chunk 0 --pool 16

Usage:
    python oak_internal_bench.py                          # all 4 stages, 800P
    python oak_internal_bench.py --tuned                  # same with tuning
    python oak_internal_bench.py --stages mono-device,mono-host --fps 60,90,110
    python oak_internal_bench.py --stages stereo-device,stereo-host --stereo-preset robotics
    python oak_internal_bench.py --res 400

Output: terminal tables + chapter4_internal_bench.csv (appended).
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
DEFAULT_FPS = {
    "mono": {800: [30, 60, 90, 110, 129], 400: [30, 90, 150, 200, 255]},
    "stereo": {800: [15, 30, 40, 50, 60], 400: [30, 60, 90, 110, 130]},
}
PRESETS = ["default", "face", "high_detail", "robotics", "fast_accuracy",
           "fast_density", "fast_density_nosubpixel"]
PRESET_ENUM = {"default": "DEFAULT", "face": "FACE", "high_detail": "HIGH_DETAIL",
               "robotics": "ROBOTICS", "fast_accuracy": "FAST_ACCURACY",
               "fast_density": "FAST_DENSITY", "fast_density_nosubpixel": "FAST_DENSITY"}
# Luxonis docs, "Depth - Performance Metrics, RVC2 Stereo": (preset, input res) -> FPS
DOCS_STEREO_FPS = {
    ("default", 800): 16, ("default", 400): 30,
    ("face", 800): 16, ("face", 400): 30,
    ("high_detail", 800): 16, ("high_detail", 400): 30,
    ("robotics", 800): 16, ("robotics", 400): 34,
    ("fast_accuracy", 800): 30, ("fast_accuracy", 400): 90,
    ("fast_density", 800): 30, ("fast_density", 400): 90,
    ("fast_density_nosubpixel", 800): 50, ("fast_density_nosubpixel", 400): 110,
}
STAGES = {  # name -> (label, family, where)
    "mono-device": ("A", "mono", "device"),
    "mono-host": ("B", "mono", "host"),
    "stereo-device": ("C", "stereo", "device"),
    "stereo-host": ("D", "stereo", "host"),
}
PASS_RATIO = 0.98

CSV_FIELDS = [
    "run_id", "timestamp", "stage", "family", "where", "res", "preset", "exposure",
    "xlink_chunk", "pool", "requested_fps", "stream", "status", "reports",
    "delivered_fps", "delivered_pct_of_req", "report_fps_min", "report_fps_max",
    "lat_avg_ms", "lat_max_ms", "stream_pass", "point_pass", "ref_fps", "ref_source",
    "frame_bytes", "data_mbps", "temp_start_c", "temp_end_c", "cpu_css_avg",
    "cpu_css_max", "cpu_mss_avg", "cpu_mss_max", "ddr_used_mib_avg",
    "host_proc_cpu_avg", "host_proc_cpu_max", "host_sys_cpu_avg", "error",
]


# ---- helpers ---------------------------------------------------------------
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


def apply_pipeline_tuning(pipeline, args):
    if args.xlink_chunk is not None:
        try:
            pipeline.setXLinkChunkSize(args.xlink_chunk)
        except Exception as e:
            print(f"  warning: setXLinkChunkSize({args.xlink_chunk}) failed: {e}")


def apply_cam_tuning(cam, args):
    if not args.pool:
        return
    for setter in ("setRawNumFramesPool", "setIspNumFramesPool"):
        try:
            getattr(cam, setter)(args.pool)
        except Exception as e:
            print(f"  warning: {setter}({args.pool}) failed: {e}")


def configure_stereo(stereo, preset):
    mode = getattr(dai.node.StereoDepth.PresetMode, PRESET_ENUM[preset])
    stereo.setDefaultProfilePreset(mode)
    if preset == "fast_density_nosubpixel":
        stereo.setSubpixel(False)


def add_benchmark(pipeline, out, where, per_report, want_passthrough):
    bi = pipeline.create(dai.node.BenchmarkIn)
    bi.setRunOnHost(where == "host")
    bi.sendReportEveryNMessages(per_report)
    bi.logReportsAsWarnings(False)
    out.link(bi.input)
    rq = bi.report.createOutputQueue(maxSize=16, blocking=False)
    pq = None
    if want_passthrough and where == "host":  # device stages must not touch USB
        try:
            pq = bi.passthrough.createOutputQueue(maxSize=1, blocking=False)
        except Exception:
            pq = None
    return rq, pq


def tuning_key(chunk, pool):
    return (str(chunk) if chunk is not None else "", str(pool) if pool else "")


# ---- analysis --------------------------------------------------------------
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
        out.update({
            "temp_start_c": usage[0][0], "temp_end_c": usage[-1][0],
            "cpu_css_avg": float(np.mean([u[1] for u in usage])),
            "cpu_css_max": max(u[1] for u in usage),
            "cpu_mss_avg": float(np.mean([u[2] for u in usage])),
            "cpu_mss_max": max(u[2] for u in usage),
            "ddr_used_mib_avg": float(np.mean([u[3] for u in usage])),
        })
    if host:
        out.update({
            "host_proc_cpu_avg": float(np.mean([h[0] for h in host])),
            "host_proc_cpu_max": max(h[0] for h in host),
            "host_sys_cpu_avg": float(np.mean([h[1] for h in host])),
        })
    return out


def load_reference(path, res, chunk, pool):
    """Latest oak_mono_sweep.py pair result per requested FPS with matching tuning."""
    if not os.path.exists(path):
        return {}
    want = tuning_key(chunk, pool)
    by = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("cams") == "pair" and r.get("res") == str(res)
                    and r.get("exposure") == "manual"
                    and r.get("frame_type_req") == "gray8"
                    and r.get("status") == "OK" and r.get("delivered_fps")
                    and (r.get("xlink_chunk", "") or "", r.get("pool", "") or "") == want):
                key = (float(r["requested_fps"]), r["run_id"])
                by.setdefault(key, []).append(float(r["delivered_fps"]))
    latest = {}
    for (fps, run), vals in by.items():
        if fps not in latest or run > latest[fps][0]:
            latest[fps] = (run, min(vals))
    return {fps: v[1] for fps, v in latest.items()}


def ok_at(v, fps):
    return v is not None and v >= PASS_RATIO * fps


def hint_mono(fps, a, b, c):
    if a is not None and not ok_at(a, fps):
        return "limit inside camera/ISP"
    if ok_at(a, fps) and b is not None and not ok_at(b, fps):
        return "lost between device and host"
    if ok_at(a, fps) and ok_at(b, fps) and c is not None and not ok_at(c, fps):
        return "Python readers were the bottleneck"
    if a is None and b is not None and not ok_at(b, fps):
        return "fails once transfer is included"
    if all(v is None or ok_at(v, fps) for v in (a, b, c)):
        return "all stages deliver"
    return "-"


def hint_stereo(fps, c, d):
    if c is not None and not ok_at(c, fps):
        return "stereo node / camera limit"
    if ok_at(c, fps) and d is not None and not ok_at(d, fps):
        return "lost between device and host"
    if all(v is None or ok_at(v, fps) for v in (c, d)):
        return "all stages deliver"
    return "-"


def stage_ceiling(rows, fps_list):
    """rows: {fps: (passed, min_delivered)} -> (best_passing_fps, first_failing_fps)."""
    passed = [f for f in fps_list if rows.get(f, (False, None))[0]]
    best = max(passed) if passed else None
    fails = [f for f in fps_list
             if f in rows and not rows[f][0] and (best is None or f > best)]
    return best, (min(fails) if fails else None)


# ---- one measurement point -------------------------------------------------
def run_point(args, stage, fps, size):
    label, family, where = STAGES[stage]
    names = ["L", "R"] if family == "mono" else ["D"]
    reports = {n: [] for n in names}
    usage, host = [], []
    frame_info = None

    with dai.Pipeline() as pipeline:
        apply_pipeline_tuning(pipeline, args)
        device = pipeline.getDefaultDevice()
        print(f"  device {device.getDeviceInfo().getDeviceId()}  "
              f"USB {device.getUsbSpeed()}  |  {stage} (BenchmarkIn on {where.upper()})")

        per_report = max(10, int(0.5 * fps))
        cam_outs = []
        for name, sock in (("L", dai.CameraBoardSocket.CAM_B),
                           ("R", dai.CameraBoardSocket.CAM_C)):
            cam = build_cam(pipeline, sock, size, fps)
            apply_cam_tuning(cam, args)
            if args.exposure == "manual":
                exp_us = min(args.exp_us, int(1e6 / fps) - 100)
                try:
                    cam.initialControl.setManualExposure(exp_us, args.iso)
                except Exception as e:
                    print(f"  warning: manual exposure not applied ({e})")
            if family == "mono":
                out = cam.requestOutput(size=size, fps=fps, type=dai.ImgFrame.Type.GRAY8)
            else:
                out = cam.requestOutput(size=size, fps=fps)  # type left to DepthAI
            cam_outs.append((name, out))

        queues, pq = {}, None
        if family == "mono":
            for i, (name, out) in enumerate(cam_outs):
                queues[name], p = add_benchmark(pipeline, out, where, per_report, i == 0)
                pq = pq or p
        else:
            stereo = pipeline.create(dai.node.StereoDepth)
            configure_stereo(stereo, args.stereo_preset)
            cam_outs[0][1].link(stereo.left)
            cam_outs[1][1].link(stereo.right)
            queues["D"], pq = add_benchmark(pipeline, stereo.depth, where, per_report, True)

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
                              m.leonMssCpuUsage.average * 100,
                              m.ddrMemoryUsage.used / (1024 * 1024)))

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
        t_start = time.monotonic()
        t_end = t_start + args.duration
        next_host, next_print = t_start + 1, t_start + 5
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

        if pq is not None:  # host stages only: real frame size, for the data rate
            try:
                m = pq.tryGet()
                if m is not None:
                    frame_info = {"bytes": int(m.getData().nbytes), "w": m.getWidth(),
                                  "h": m.getHeight(),
                                  "type": str(m.getType()).split(".")[-1]}
            except Exception:
                frame_info = None

    return reports, usage, host, frame_info


# ---- output ----------------------------------------------------------------
def csv_rows(args, run_id, stage, fps, per, use, point_pass, ref, ref_source,
             frame_info, status, err):
    label, family, where = STAGES[stage]
    base = {
        "run_id": run_id, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": label + " " + stage, "family": family, "where": where, "res": args.res,
        "preset": args.stereo_preset if family == "stereo" else "",
        "exposure": args.exposure,
        "xlink_chunk": args.xlink_chunk if args.xlink_chunk is not None else "",
        "pool": args.pool or "", "requested_fps": fps, "status": status,
        "ref_fps": ref if ref is not None else "", "ref_source": ref_source, "error": err,
    }
    if not per:
        names = ["L", "R"] if family == "mono" else ["D"]
        return [dict(base, stream=n) for n in names]
    rows = []
    for name, st in per.items():
        row = dict(base, stream=name, stream_pass=st["pass"], point_pass=point_pass)
        row.update({k: v for k, v in st.items() if k in CSV_FIELDS})
        row.update(use)
        if frame_info:
            row["frame_bytes"] = frame_info["bytes"]
            row["data_mbps"] = st["delivered_fps"] * frame_info["bytes"] * 8 / 1e6
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
    ap.add_argument("--stages", type=str, default="mono-device,mono-host,stereo-device,stereo-host")
    ap.add_argument("--res", type=int, choices=[800, 400], default=800)
    ap.add_argument("--fps", type=str, default=None, help="requested FPS list, mono stages")
    ap.add_argument("--stereo-fps", type=str, default=None, help="requested FPS list, stereo stages")
    ap.add_argument("--stereo-preset", choices=PRESETS, default="fast_density_nosubpixel")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--skip-reports", type=int, default=2)
    ap.add_argument("--exposure", choices=["manual", "auto"], default="manual")
    ap.add_argument("--exp-us", type=int, default=2000)
    ap.add_argument("--iso", type=int, default=400)
    ap.add_argument("--xlink-chunk", type=int, default=None)
    ap.add_argument("--pool", type=int, default=None)
    ap.add_argument("--tuned", action="store_true", help="--xlink-chunk 0 --pool 16")
    ap.add_argument("--csv", default="chapter4_internal_bench.csv")
    ap.add_argument("--ref-csv", default="throughput_sweep.csv",
                    help="mono sweep CSV to compare against")
    args = ap.parse_args()
    if args.tuned:
        if args.xlink_chunk is None:
            args.xlink_chunk = 0
        if args.pool is None:
            args.pool = 16

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in STAGES:
            raise SystemExit(f"unknown stage '{s}'; choose from {list(STAGES)}")

    def parse_fps(txt, default):
        return ([float(x) if "." in x else int(x) for x in txt.split(",")]
                if txt else default)

    fps_for = {
        "mono": parse_fps(args.fps, DEFAULT_FPS["mono"][args.res]),
        "stereo": parse_fps(args.stereo_fps, DEFAULT_FPS["stereo"][args.res]),
    }
    size = RES[args.res]
    ref = load_reference(args.ref_csv, args.res, args.xlink_chunk, args.pool)
    docs_fps = DOCS_STEREO_FPS.get((args.stereo_preset, args.res))

    n_points = sum(len(fps_for[STAGES[s][1]]) for s in stages)
    est = (n_points * (args.duration + 8) + (n_points - 1) * args.cooldown) / 60
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 104)
    print(f"INTERNAL BENCHMARK  |  pair {args.res}P ({size[0]}x{size[1]})  |  "
          f"exposure {args.exposure}  |  stages: {', '.join(stages)}")
    print(f"xlink chunk: {'default (64 KB)' if args.xlink_chunk is None else args.xlink_chunk}"
          f"  |  frame pools: {'default' if not args.pool else f'raw+isp = {args.pool}'}"
          f"  |  {args.duration:.0f} s per point, ~{n_points} points, ~{est:.0f} min")
    if any(STAGES[s][1] == "stereo" for s in stages):
        print(f"stereo preset: {args.stereo_preset}  |  Luxonis docs max FPS for this "
              f"preset at {args.res}P: {docs_fps}")
    print(f"verdict: delivered >= {PASS_RATIO:.0%} of requested FPS  |  "
          f"mono-sweep reference points found: {len(ref)}")
    if psutil is None:
        print("note: psutil not installed -> host CPU columns blank (pip install psutil)")
    print("=" * 104)
    print(f"{'stage':>14} {'req':>5} {'strm':>4} {'delivFPS':>9} {'%req':>6} "
          f"{'rep-fps min/max':>16} {'lat avg/max ms':>15} {'MB/s':>7}  verdict")
    print("-" * 104)

    results = {s: {} for s in stages}
    k = 0
    for stage in stages:
        label, family, where = STAGES[stage]
        for fps in fps_for[family]:
            k += 1
            print(f"[{k}/{n_points}] {label} {stage} | requesting {fps} fps ...")
            per, use, finfo, status, err, point_pass = {}, {}, None, "OK", "", False
            try:
                reports, usage, host, finfo = run_point(args, stage, fps, size)
                for name in reports:
                    s = summarize_reports(reports[name], fps, args.skip_reports)
                    if s is None:
                        raise RuntimeError(f"too few reports for stream {name}")
                    per[name] = s
                use = usage_summary(usage, host)
                point_pass = all(s["pass"] for s in per.values())
                results[stage][fps] = (point_pass, min(s["delivered_fps"] for s in per.values()))
            except Exception as e:
                status, err, per = "ERROR", f"{type(e).__name__}: {e}", {}
                results[stage][fps] = (False, None)

            if per:
                for name, s in per.items():
                    mbs = (f"{s['delivered_fps'] * finfo['bytes'] / 1e6:7.1f}"
                           if finfo else f"{'-':>7}")
                    print(f"{label + ' ' + stage:>14} {fps:>5} {name:>4} "
                          f"{s['delivered_fps']:9.2f} {s['delivered_pct_of_req']:6.1f} "
                          f"{s['report_fps_min']:7.1f}/{s['report_fps_max']:<7.1f} "
                          f"{s['lat_avg_ms']:7.1f}/{s['lat_max_ms']:<7.1f} {mbs}  "
                          f"{'PASS' if s['pass'] else 'FAIL'}")
                line = []
                if "temp_end_c" in use:
                    line.append(f"temp {use['temp_start_c']:.1f}->{use['temp_end_c']:.1f}C")
                if "cpu_css_avg" in use:
                    line.append(f"LeonCSS {use['cpu_css_avg']:.1f}/{use['cpu_css_max']:.1f}% "
                                f"MSS {use['cpu_mss_avg']:.1f}/{use['cpu_mss_max']:.1f}%")
                if "host_proc_cpu_avg" in use:
                    line.append(f"host proc CPU {use['host_proc_cpu_avg']:.0f}/"
                                f"{use['host_proc_cpu_max']:.0f}% (100% = 1 core)")
                if finfo:
                    line.append(f"frame {finfo['w']}x{finfo['h']} {finfo['type']} "
                                f"{finfo['bytes']} B")
                if line:
                    print("               " + " | ".join(line))
            else:
                print(f"{label + ' ' + stage:>14} {fps:>5}  {status}: {err}")

            if family == "mono":
                r_val, r_src = ref.get(float(fps)), "sweep"
            else:
                r_val, r_src = docs_fps, "docs"
            write_csv(args.csv, csv_rows(args, run_id, stage, fps, per, use, point_pass,
                                         r_val, r_src, finfo, status, err))
            if k < n_points:
                time.sleep(args.cooldown)

    # ---- summaries ----------------------------------------------------------
    print("=" * 104)
    print("STAGE CEILINGS (highest requested FPS that still delivers >= 98%)")
    print(f"{'stage':>16} {'max sustained':>14} {'first failing':>14} {'delivered there':>16}")
    for stage in stages:
        label, family, where = STAGES[stage]
        best, ff = stage_ceiling(results[stage], fps_for[family])
        dl = results[stage].get(ff, (False, None))[1] if ff is not None else None
        print(f"{label + ' ' + stage:>16} {str(best) if best is not None else '-':>14} "
              f"{str(ff) if ff is not None else '-':>14} "
              f"{(f'{dl:.1f}' if dl is not None else '-'):>16}")

    def cell(stage, fps):
        v = results.get(stage, {}).get(fps)
        return f"{v[1]:.1f}" if v and v[1] is not None else "-"

    if any(STAGES[s][1] == "mono" for s in stages):
        print()
        print("MONO PAIR - delivered FPS per camera (min over L/R)")
        print(f"{'requested':>10} {'A: device':>10} {'B: host':>9} {'sweep (Py)':>11}  reading")
        for fps in fps_for["mono"]:
            a = results.get("mono-device", {}).get(fps, (None, None))[1]
            b = results.get("mono-host", {}).get(fps, (None, None))[1]
            c = ref.get(float(fps))
            print(f"{fps:>10} {cell('mono-device', fps):>10} {cell('mono-host', fps):>9} "
                  f"{(f'{c:.1f}' if c is not None else '-'):>11}  {hint_mono(fps, a, b, c)}")
    if any(STAGES[s][1] == "stereo" for s in stages):
        print()
        print(f"STEREO ({args.stereo_preset}) - delivered depth FPS; Luxonis docs max: {docs_fps}")
        print(f"{'requested':>10} {'C: device':>10} {'D: host':>9}  reading")
        for fps in fps_for["stereo"]:
            c = results.get("stereo-device", {}).get(fps, (None, None))[1]
            d = results.get("stereo-host", {}).get(fps, (None, None))[1]
            print(f"{fps:>10} {cell('stereo-device', fps):>10} {cell('stereo-host', fps):>9}  "
                  f"{hint_stereo(fps, c, d)}")
    print(f"\nRows appended to {args.csv}")


if __name__ == "__main__":
    main()