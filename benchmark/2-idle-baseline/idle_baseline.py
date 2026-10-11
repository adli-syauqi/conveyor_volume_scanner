#!/usr/bin/env python3
"""
OAK-D Pro (RVC2) idle baseline.

Runs a pipeline containing only a SystemLogger node (no cameras, no NN, no
stereo) and logs chip temperature, CPU load and memory use at 1 Hz.
Field names follow the DepthAI v3 SystemInformation docs.

Usage:
    python oak_idle_baseline.py            # 60 s
    python oak_idle_baseline.py 120        # custom duration (seconds)

Output:
    idle_baseline_samples.csv   (raw samples)
    summary printed to the terminal
"""
import csv
import statistics
import sys
import time

import depthai as dai

CSV_PATH = "idle_baseline_samples.csv"
DURATION_S = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
LOG_RATE_HZ = 1.0
MIB = 1024 * 1024

# (csv column, label, unit) in display order
METRICS = [
    ("temp_avg_c", "Chip temp (average)", "C"),
    ("temp_css_c", "Chip temp (CSS)", "C"),
    ("temp_mss_c", "Chip temp (MSS)", "C"),
    ("temp_upa_c", "Chip temp (UPA / SHAVE)", "C"),
    ("temp_dss_c", "Chip temp (DSS / DDR)", "C"),
    ("cpu_css_pct", "Leon CSS CPU", "%"),
    ("cpu_mss_pct", "Leon MSS CPU", "%"),
    ("ddr_used_mib", "DDR used", "MiB"),
    ("cmx_used_mib", "CMX used", "MiB"),
    ("leon_css_heap_mib", "Leon CSS heap used", "MiB"),
    ("leon_mss_heap_mib", "Leon MSS heap used", "MiB"),
]
COLUMNS = ["t_s"] + [m[0] for m in METRICS]


def read_sample(info: dai.SystemInformation, t_s: float) -> dict:
    t = info.chipTemperature
    return {
        "t_s": t_s,
        "temp_avg_c": t.average,
        "temp_css_c": t.css,
        "temp_mss_c": t.mss,
        "temp_upa_c": t.upa,
        "temp_dss_c": t.dss,
        # CPU usage .average is a 0..1 fraction
        "cpu_css_pct": info.leonCssCpuUsage.average * 100,
        "cpu_mss_pct": info.leonMssCpuUsage.average * 100,
        "ddr_used_mib": info.ddrMemoryUsage.used / MIB,
        "cmx_used_mib": info.cmxMemoryUsage.used / MIB,
        "leon_css_heap_mib": info.leonCssMemoryUsage.used / MIB,
        "leon_mss_heap_mib": info.leonMssMemoryUsage.used / MIB,
    }


def collect():
    samples = []
    with dai.Pipeline() as pipeline:
        device = pipeline.getDefaultDevice()
        device_id = device.getDeviceInfo().getDeviceId()
        usb_speed = str(device.getUsbSpeed())

        sys_log = pipeline.create(dai.node.SystemLogger)
        sys_log.setRate(LOG_RATE_HZ)
        queue = sys_log.out.createOutputQueue(maxSize=4, blocking=False)

        pipeline.start()
        print(f"Idle baseline on {device_id} ({usb_speed}) for {DURATION_S:.0f} s")
        print(f"{'t(s)':>6} {'temp(C)':>8} {'cssCPU%':>8} {'mssCPU%':>8} {'DDR(MiB)':>9}")

        t0 = time.monotonic()
        while pipeline.isRunning() and time.monotonic() - t0 < DURATION_S:
            info = queue.get()  # blocks until the next message (~1 per second)
            s = read_sample(info, time.monotonic() - t0)
            samples.append(s)
            print(
                f"{s['t_s']:6.1f} {s['temp_avg_c']:8.2f} {s['cpu_css_pct']:8.2f} "
                f"{s['cpu_mss_pct']:8.2f} {s['ddr_used_mib']:9.2f}"
            )
    return device_id, usb_speed, samples


def write_csv(samples):
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for s in samples:
            w.writerow({k: round(v, 4) for k, v in s.items()})


def print_summary(device_id, usb_speed, samples):
    print()
    print("=" * 64)
    print("IDLE BASELINE SUMMARY")
    print("=" * 64)
    print(f"Device : {device_id}")
    print(f"USB    : {usb_speed}")
    print(f"Run    : {DURATION_S:.0f} s at {LOG_RATE_HZ:g} Hz, {len(samples)} samples")
    print("Load   : SystemLogger only (no cameras / NN / stereo)")
    print()
    print(f"{'Metric':<26}{'min':>10}{'avg':>10}{'max':>10}  unit")
    print("-" * 64)
    for key, label, unit in METRICS:
        vals = [s[key] for s in samples]
        print(
            f"{label:<26}{min(vals):10.2f}{statistics.fmean(vals):10.2f}"
            f"{max(vals):10.2f}  {unit}"
        )

    temps = [s["temp_avg_c"] for s in samples]
    if len(temps) >= 2:
        print()
        print(
            f"Temperature drift: {temps[0]:.2f} C -> {temps[-1]:.2f} C "
            f"({temps[-1] - temps[0]:+.2f} C over {samples[-1]['t_s']:.0f} s)"
        )
    print()
    print(f"Raw samples saved to {CSV_PATH}")


def main():
    device_id, usb_speed, samples = collect()
    if not samples:
        print("No samples collected.")
        return
    write_csv(samples)
    print_summary(device_id, usb_speed, samples)


if __name__ == "__main__":
    main()