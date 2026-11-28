# User & Developer Guide — Cortivra-Apex

This guide explains how Cortivra-Apex installs on your computer, how its automatic background detection works for everyday users and gamers, and how developers integrate it into PyTorch, HuggingFace, and multi-GPU systems.

---

## 1. Download & First Launch (Windows, macOS, Linux)

1. Download and unzip `cortivra-apex-v2.0.0-release.zip` into any folder on your computer.
2. Run the launcher for your operating system:
   - **Windows**: Double-click `Cortivra-Apex.exe` (or `Install-Windows.bat`).
   - **macOS**: Double-click `Install-Mac.command` (or run `python3 cortivra-apex.pyz verify`).
   - **Linux**: Run `bash install_linux.sh` (or run `python3 cortivra-apex.pyz verify`).

### What Happens Automatically on First Launch
1. **Runs the 15-Second GPU Speed Test**: Measures your graphics card's exact before-and-after millisecond timings across all 11 GPU acceleration modules (10 hardware scorecard benchmarks) and saves a signed report to `logs/APEX_HARDWARE_PROOF.md`.
2. **Registers `cortivra_apex` Globally in Python**: Writes a lightweight `.pth` file (`cortivra_apex_autoload.pth`) into your Python `site-packages` folder. From that moment on, you can run `import cortivra_apex as apex` from **any folder on your computer** without manually setting `PYTHONPATH`.
3. **Opens the Local Control Center (`http://127.0.0.1:7860`)**: Shows live GPU clocks, temperature, free video memory, your verified speedup scorecard, optimization mode buttons, and license activation.
4. **Registers the `0.0 MB` Idle Startup Guardian** *(when you run `Install-Windows.bat`, `Install-Mac.command`, or `install_linux.sh`)*: Starts quietly in normal user mode whenever you log into your computer, using `0.0 MB` of GPU memory when idle.

---

## 2. How Automatic Background Boost Works (Gamers, Creators & Everyday PCs)

You do not need to write code or leave a terminal window open.

1. **Dormant Standby When Idle (`0.0 MB` VRAM)**:
   When your computer is idle or you are doing light work, Cortivra stays in Dormant Standby using **`0.0 MB` of GPU memory** and `0%` GPU load.
2. **Automatic Workload Detection**:
   When a GPU-heavy application runs (or when you click **Auto-Detect Game / AI & Tune** in the Control Center), Cortivra detects the active workload type and applies the matching profile:
   - **Gaming Mode**: Locks on-chip GPU L2 cache (`27.2 MB` on RTX 5080), enables a `1.0 ms` high-precision frame timer to prevent `1% Low` frame stutters, keeps the GPU clock ready so it doesn't down-clock between frames, and trims idle background process working sets when system RAM exceeds 80%.
   - **Local AI & LLMs**: Enables 4-bit (`7.10x` smaller) and 8-bit (`3.75x` smaller) weight compression, single-pass `RMSNorm`, online `Softmax`, and fast long-context attention.
   - **3D & Video Creation**: Enables single-pass HDR frame synthesis and 3D spatial convolution.
   - **Manual Pause Anytime**: Click **Pause (0 MB Standby)** in the Control Center at any time to immediately release all GPU memory (`0.0 MB`).

---

## 3. How Developers Use Cortivra-Apex in Python & PyTorch

Because first launch registers `cortivra_apex` in your Python `site-packages`, you can import `cortivra_apex` from any project directory, virtual environment, or Jupyter Notebook.

### Way A: 1-Line Model & Weight Memory Compression (`7.1x` Smaller in 4-Bit, `3.75x` in 8-Bit)

If a model is too large to fit on your GPU, load it on the CPU first (`device_map="cpu"` or default PyTorch CPU loading) and call `apex.boost(...)`. Cortivra streams the model onto the GPU one layer at a time, compresses each layer in VRAM, and frees the uncompressed weights as it goes:

```python
import torch
import cortivra_apex as apex
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)

# 1. Load on CPU first so GPU VRAM never spikes during loading
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)

# 2. Compress and move to GPU in 1 line:
#    bits=8 -> 3.75x smaller vs FP32 (1.88x vs FP16), word-for-word identical output
#    bits=4 -> 7.10x smaller vs FP32 (3.55x vs FP16), maximum VRAM savings
model = apex.boost(model, compress_weights=True, bits=4)

inputs = tok(["Write a Python function to reverse a linked list:"], return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))
```

If you are training or fine-tuning a model with an optimizer, pass both `model` and `optimizer`—a single `apex.boost()` call automatically upgrades all 11 GPU operator pillars across your `nn.Module` tree (`ApexLinear`, `ApexRMSNorm`, `ApexLayerNorm`, `ApexSoftmax`, `ApexWaveAttention`, `ApexPersistentGraph`, `ApexFrameSynthesizer`, `ApexSpatialConv2d`, `ApexScaleOutInterconnect`, `ApexCompensatedLinear`, and `ApexOptimizer`), binds `model.apex_allreduce_gradients(compress_wire_4bit=True)` and PyTorch DDP communication hooks, and records `model._apex_boost_summary.active_pillars == 11`:

```python
import cortivra_apex as apex

# Mode 1: No arguments -> Boosts the entire process & hardware session (locks L2 cache, intercepts all raw tensor math, returns 11-pillar engine)
engine = apex.boost()

# Mode 2: Function / Simulation Loop / Render Pass Decorator -> Accelerates any raw Python function
@apex.boost
def run_simulation_or_render_step(x, w):
    return torch.softmax(torch.nn.functional.rms_norm(x, (x.shape[-1],), w, 1e-6), dim=-1)

# Mode 3: Raw Tensor / Scientific Matrix -> Streams to GPU and compresses 7.11x in VRAM (bits=4) or 3.77x (bits=8)
compressed_tensor = apex.boost(raw_matrix_tensor, bits=4)
reconstructed = compressed_tensor.decompress()

# Mode 4: Multi-GPU / Multi-Shard Tensor List -> Synchronizes across ranks with 7.11x 4-bit wire compression
synced_tensor, wire_telemetry = apex.boost([shard_gpu0, shard_gpu1], bits=4)

# Mode 5: PyTorch Model & Optimizer -> Upgrades all 11 GPU pillars across the entire nn.Module tree
model, optimizer = apex.boost(model, optimizer, compress_weights=True, bits=4)
```

### Way B: 0-Line Automatic Acceleration for Existing Scripts

To accelerate an existing PyTorch script without changing its model code:

```bash
Cortivra-Apex.exe run train_or_infer.py
```

Or enable runtime auto-interception at the top of your script:

```python
import cortivra_apex as apex

# Exact standard softmax attention (recommended for frozen pretrained checkpoints):
apex.enable_auto_intercept(wave_attention=False)

# Or enable fast linear-time wave attention for 8K+ long-context workloads:
# apex.enable_auto_intercept(wave_attention=True)
```

### Way C: Calling Individual GPU Kernels Directly

```python
import torch
import cortivra_apex as apex

# 1. Single-pass RMSNorm & Softmax (Cos = 1.000000 vs float64)
y_norm = apex.rmsnorm(x, weight=w, eps=1e-6)
y_prob = apex.softmax(x)

# 2. Fast 8K+ Long-Context Attention
attn_out = apex.wave_attention(q, k, v)

# 3. Direct 4-Bit & 8-Bit Tensor Compression
q4 = apex.vram_compress_4bit(weight_tensor)
w_reconstructed = apex.vram_decompress_4bit(q4)

# 4. Extended-Precision Compensated Matrix-Vector Multiply (float64 accuracy on FP32 cores)
y_fp32, y_fp64 = apex.compensated_gemv(A_matrix, x_vector)

# 5. Multi-Stage Persistent Graph Runner
out = apex.run_persistent_graph(input_tensor, num_stages=100)

# 6. Multi-GPU Scale-Out AllReduce (with 7.11x 4-bit wire compression)
synced_tensor, telemetry = apex.distributed_scaleout_allreduce(local_tensor, compress_wire_4bit=True)
```

---

## 4. How License Activation & 1–4 GPU Hardware Slots Work

1. **Free Evaluation ($0)**:
   Runs the 15-second hardware verifier (`Cortivra-Apex.exe verify`), the local Control Center dashboard, and the open-source verification scripts (`verify_independently.py` and `verify_real_llm_vram.py`).
2. **Personal Lifetime Pass ($50 / GPU One-Time Forever, 1 to 4 Consumer GPUs)**:
   - Works on 1 to 4 consumer GPUs (NVIDIA GeForce GTX/RTX, AMD Radeon RX, Apple M1–M4, Intel Arc) in a single computer or across up to 4 separate personal computers.
   - When you complete Stripe Checkout on the website while your local Control Center (`http://127.0.0.1:7860`) is running, your browser automatically sends your `APEX-PERS-...` license key to your PC and unlocks it in 1 click.
   - To unlock a 2nd, 3rd, or 4th GPU on another computer, open the Control Center (`http://127.0.0.1:7860`) on that computer, paste your Stripe Order ID (`cs_...`) or `APEX-PERS-...` key into Section 4, and click **Activate / Claim GPU Slot**.
3. **Enterprise & Multi-GPU Clusters (5 to 100,000+ GPUs or Datacenter GPUs)**:
   - Required for datacenter GPUs (`H100`, `H200`, `B200`, `A100`, `L40S`, `RTX 6000 Ada`, `MI300X`, `Gaudi`) or clusters with 5+ GPUs.
   - Roll out across an entire cluster with zero external license servers by setting one environment variable:
     ```bash
     export CORTIVRA_APEX_LICENSE_KEY="APEX-ENT-..."
     ```

---

## 5. Lifecycle Management (Check Order Slots, Deactivate & Clean Uninstall)

1. **Check Remaining GPU Slots on an Order**:
   Paste your Stripe Order Session ID (`cs_...`) into Section 4 of the Local Control Center (`http://127.0.0.1:7860`) or the public website (`https://cortivra-apex.com#pricing`) and click **Check Order Slots** to view `slots_claimed`, `slots_remaining`, and privacy-masked bound GPU UUIDs (`GPU-123456****-****7788`).
2. **Deactivate Local Workstation**:
   To remove an active license key from a machine and revert it to Free Evaluation Mode (`EVALUATION`), click **Deactivate Local PC** in Section 4 of the Control Center or run:
   ```bash
   Cortivra-Apex.exe deactivate
   # or on macOS / Linux:
   python3 cortivra-apex.pyz deactivate
   ```
3. **1-Click Clean Uninstallation (`100%` Reversible)**:
   To remove all OS login startup entries (`HKCU\...\Run`, macOS `LaunchAgents`, Linux `systemd --user`) and remove `cortivra_apex_autoload.pth` from Python `site-packages`, double-click `Uninstall-Windows.bat` (or run `bash uninstall_unix.sh`) or run:
   ```bash
   Cortivra-Apex.exe uninstall
   ```

---

## 6. Operational Environment Variables Reference

| Environment Variable | Values | Purpose |
| :--- | :---: | :--- |
| `CORTIVRA_APEX_LICENSE_KEY` | `APEX-PERS-...` / `APEX-ENT-...` | Activates a Personal or Enterprise Fleet license key in memory with zero disk config or external license servers. Automatically reloads if updated at runtime. |
| `CORTIVRA_SAFE_BYPASS` | `1` / `true` | Emergency incident kill-switch: immediately bypasses all custom GPU kernels and delegates 100% of operations to unmodified native PyTorch. |
| `CORTIVRA_REQUIRE_PAID_LICENSE` | `1` / `true` | Strict production enforcement mode: raises `LicenseVerificationError` if only a Free Evaluation key (`TIER_EVALUATION`) is active. |
| `CORTIVRA_NO_BROWSER` | `1` | Prevents `cortivra-apex dashboard` from automatically opening a desktop browser window on headless servers. |

---

## 7. Self-Service Troubleshooting & Support

- **Using a New Virtual Environment (`venv` or `conda`)**:
  If you create a new Python virtual environment after installing Cortivra-Apex, register the library inside that active environment by running `python /path/to/cortivra-apex.pyz verify` once inside the environment (or add the release folder to `PYTHONPATH`).
- **OS Reinstall or Defective GPU RMA Replacement**:
  Cortivra-Apex hardware fingerprints depend only on physical silicon identity and are invariant to OS hostnames, DHCP/VPN changes, or Windows/Linux OS reinstalls. If a physical GPU fails under manufacturer warranty (RMA), email **`hello@cortivra-apex.com`** with your Stripe Order Session ID (`cs_...`) to transfer your lifetime slot to your replacement GPU.
- **Direct Engineering & Customer Support**:
  For technical support, enterprise cluster onboarding (`5 to 100,000+ GPUs`), or billing questions, contact Omni Intelligence LLC at **`hello@cortivra-apex.com`**.
