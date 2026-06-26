"""Reproducible OCR benchmark: time the SAME frames on GPU vs CPU.

GPU  -> the live isolated OCR service (:8090, onnxruntime CUDA), real deployment path.
CPU  -> a standalone RapidOCR (onnxruntime CPU, intra_op=4) in this process, i.e. the
        'isolated CPU' configuration (the best CPU result we ever got).

Outputs CSV: frame,line_count,gpu_ms,cpu_ms  -> paper/bench_ocr.csv
"""
import glob, os, time, csv, urllib.request
import cv2

FRAMES_DIR = r"C:\Users\Pedro\pc_server_sessions\20260626-150802_2a8ac5\frames"
N = 25
OUT = r"C:\Users\Pedro\glass_diag\paper\bench_ocr.csv"

def gpu_ocr(raw):
    req = urllib.request.Request("http://127.0.0.1:8090/ocr?lang=en", data=raw,
                                 method="POST", headers={"Content-Type": "application/octet-stream"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=40) as r:
        import json
        d = json.loads(r.read().decode())
    return (time.time() - t) * 1000.0, len(d.get("lines", []))

def main():
    files = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
    if not files:
        print("no frames in", FRAMES_DIR); return
    step = max(1, len(files) // N)
    picks = files[::step][:N]
    print(f"{len(files)} frames, benchmarking {len(picks)}")

    # CPU engine (standalone, isolated config)
    from rapidocr_onnxruntime import RapidOCR
    cpu = RapidOCR(intra_op_num_threads=4)
    # warm both paths
    raw0 = open(picks[0], "rb").read()
    gpu_ocr(raw0); cpu(cv2.imread(picks[0]))

    rows = []
    for p in picks:
        raw = open(p, "rb").read()
        img = cv2.imread(p)
        g_ms, g_lines = gpu_ocr(raw)
        t = time.time(); c_res, _ = cpu(img); c_ms = (time.time() - t) * 1000.0
        c_lines = len(c_res) if c_res else 0
        lines = max(g_lines, c_lines)
        rows.append((os.path.basename(p), lines, round(g_ms, 1), round(c_ms, 1)))
        print(f"  {os.path.basename(p)}  lines={lines:2d}  gpu={g_ms:6.0f}ms  cpu={c_ms:7.0f}ms")

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "line_count", "gpu_ms", "cpu_ms"]); w.writerows(rows)
    import statistics as st
    g = [r[2] for r in rows]; c = [r[3] for r in rows]
    print(f"\nGPU  median={st.median(g):.0f}ms  mean={st.mean(g):.0f}ms  max={max(g):.0f}ms")
    print(f"CPU  median={st.median(c):.0f}ms  mean={st.mean(c):.0f}ms  max={max(c):.0f}ms")
    print("saved", OUT)

if __name__ == "__main__":
    main()
