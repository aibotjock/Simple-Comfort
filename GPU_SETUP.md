# Dual GPU Setup — NVIDIA RTX 3090 × 2 (Ubuntu 26)

This guide gets **both** RTX 3090s detected and sharing the load **evenly (50/50)**
for Simple-Comfort, and explains how to run **LM Studio / other LLM tools**
alongside them without fighting over the cards.

> TL;DR — run the doctor first: `bash check-gpu.sh`. It tells you exactly what's
> wrong and how to fix it.

---

## How the "even split" actually works here

Simple-Comfort does **not** shard one model across two cards (no
`device_map="auto"`, no tensor parallelism). Splitting a diffusion/video model
across GPUs over PCIe is slow and fragile. Instead it uses the robust approach:

* **One ComfyUI instance per GPU** — GPU 0 → port `8188`, GPU 1 → port `8189`,
  each pinned with `CUDA_VISIBLE_DEVICES`.
* **Round-robin scheduling** — the `WorkerPool` alternates jobs across the two
  endpoints (job 1 → GPU 0, job 2 → GPU 1, …).

Net effect: over any batch of jobs each RTX 3090 does ~half the work and uses
~half the total VRAM budget — a true 50/50 split, with **full 24 GB available to
each model** (far better than a cross-card shard that would leave each model
starved).

Verify it live:

```bash
nvidia-smi -l 1        # watch both cards — both should get busy under load
```

or hit the app's status API, which now reports per-card memory/utilization:

```bash
curl -s http://127.0.0.1:7788/api/status | python3 -m json.tool
# -> "gpus": [ {index:0, memory_used_mb, utilization_percent, ...}, {index:1, ...} ]
```

---

## 1. Install / verify the NVIDIA driver

On Ubuntu the reliable way to get a driver matched to your kernel + RTX 3090
(Ampere) is:

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

`ubuntu-drivers autoinstall` picks the correct recommended driver branch for
your hardware automatically. (If you prefer a specific branch, list what's
offered with `ubuntu-drivers devices` and install e.g.
`sudo apt install nvidia-driver-570`.)

After reboot, confirm **both** cards are seen:

```bash
nvidia-smi                      # should list 2× "NVIDIA GeForce RTX 3090"
lspci | grep -i nvidia          # both cards on the bus
nvidia-smi --query-gpu=index,name --format=csv,noheader
# 0, NVIDIA GeForce RTX 3090
# 1, NVIDIA GeForce RTX 3090
```

If only one card shows up: check the second card's PCIe seating, riser, and PSU
power cables, then reboot. Both cards must appear in `nvidia-smi` before
dual-GPU mode can work.

---

## 2. Make sure ComfyUI's torch is a CUDA build

Simple-Comfort itself is a thin FastAPI front-end — **it does not import torch**.
The GPU work happens inside **ComfyUI**, so torch+CUDA must be correct in
**ComfyUI's** virtual environment (not this app's `venv`).

Check it:

```bash
# adjust the path to your ComfyUI install
/path/to/ComfyUI/.venv/bin/python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
print("device_count:", torch.cuda.device_count())   # want 2
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

If `device_count` is 0 or torch reports a CPU-only build, reinstall a CUDA
build in the **ComfyUI** venv:

```bash
/path/to/ComfyUI/.venv/bin/python -m pip install --force-reinstall \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

(Use the `cu124` index — or the wheel matching your driver's CUDA version shown
in `nvidia-smi`.) See `requirements-gpu.txt` for the exact lines.

---

## 3. Launch in dual-GPU mode

```bash
bash check-gpu.sh          # confirm both cards + torch are healthy
./launch-dual-gpu.sh       # starts ComfyUI on GPU0:8188 and GPU1:8189, then the app
```

The launcher pins each ComfyUI instance to one card
(`CUDA_VISIBLE_DEVICES=0` / `=1`), waits for both to answer, writes
`COMFY_ENDPOINTS=http://127.0.0.1:8188,http://127.0.0.1:8189` to `.env`, and
starts Simple-Comfort. Open <http://localhost:7788> — the header shows a status
chip per GPU.

`.env` (or `.env.example`) equivalent:

```ini
COMFY_ENDPOINTS=http://127.0.0.1:8188,http://127.0.0.1:8189
```

---

## 4. Running LM Studio (or other LLMs) alongside — how to divide the cards

LM Studio, ComfyUI, and Simple-Comfort all draw from the **same** 2×24 GB of
VRAM, so plan how they share it:

**A. Time-share (simplest).** The RTX 3090's 24 GB is mostly consumed by a
single video model, so **unload the model in LM Studio before rendering** and
vice-versa. This is the recommended default when you switch between chatting and
video generation.

**B. Space-share — give each tool its own card.** Pin LM Studio to one GPU and
Simple-Comfort to the other so they never collide:

* In LM Studio, restrict CUDA visibility to a single card by launching it with
  `CUDA_VISIBLE_DEVICES=1` (that pins it to GPU 1), and set its GPU offload so
  the model fits in that one card's 24 GB.
* Run Simple-Comfort in **single-GPU** mode on the *other* card:

  ```bash
  # ComfyUI on GPU 0 only:
  CUDA_VISIBLE_DEVICES=0 python /path/to/ComfyUI/main.py --port 8188 --listen 127.0.0.1 &
  # point the app at just that one endpoint:
  echo 'COMFY_ENDPOINTS=http://127.0.0.1:8188' >> .env
  ./launch.sh
  ```

**C. Split a big LLM evenly across both 3090s (LM Studio).** For a model too
large for one 24 GB card, LM Studio (llama.cpp backend) can split layers across
both GPUs:

* Enable **multi-GPU** and set the **tensor split** to `50,50` (equal split
  across GPU 0 and GPU 1) in LM Studio's model load settings.
* Set **GPU offload / `n_gpu_layers`** to the max (offload all layers) so the
  model stays on the GPUs rather than spilling to CPU RAM.
* Leave headroom on each card if you also want Simple-Comfort running — in that
  case combine with option **B** rather than giving the LLM both full cards.

> If you split an LLM `50,50` across both cards **and** try to run dual-GPU video
> generation at the same time, you will run out of VRAM. Pick one heavy workload
> per card, or time-share.

---

## 5. Troubleshooting quick table

| Symptom | Fix |
|---------|-----|
| `nvidia-smi: command not found` | Driver not installed → `sudo ubuntu-drivers autoinstall`, reboot. |
| Only 1 GPU in `nvidia-smi` | Reseat card / check riser + PSU cables, reboot. |
| `launch-dual-gpu.sh` said "no CUDA devices" but `nvidia-smi` works | Fixed in this branch — the driver check is now authoritative; the torch probe is advisory. Re-run `./launch-dual-gpu.sh`. |
| ComfyUI runs on CPU (very slow) | ComfyUI venv has CPU-only torch → reinstall the `cu124` build (section 2). |
| Load only ever hits GPU 0 | You're in single-endpoint mode. Ensure `COMFY_ENDPOINTS` lists **both** ports and both ComfyUI instances are up (`bash check-gpu.sh`). |
| LM Studio can't load a model onto GPU | Increase GPU offload / `n_gpu_layers`; free VRAM by unloading video models; confirm LM Studio sees CUDA (its runtime must be the CUDA build). |
