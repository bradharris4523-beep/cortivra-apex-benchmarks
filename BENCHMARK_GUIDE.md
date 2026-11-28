# Benchmark Methodology, Fairness Guarantees & Operating Thresholds

This document specifies the exact experimental methodology, baseline construction rules, categorized numerical acceptance gates, and automatic hardware dispatch thresholds used across the Cortivra-Apex verification suite.

---

## 1. Eight Rules of Benchmark Integrity

Every benchmark in [verify_independently.py](./verify_independently.py) and [verify_real_llm_vram.py](./verify_real_llm_vram.py) enforces the following rules so independent auditors can verify that comparisons are strictly fair:

1. **Unmodified Native Baselines (`apex.disable_auto_intercept()`)**:
   - Before any baseline is timed or evaluated (including inside `verify_real_llm_vram.py`), Cortivra-Apex's runtime dispatcher hook is disabled via `apex.disable_auto_intercept()`.
   - Every baseline executes 100% unmodified native PyTorch C++ (`F.rms_norm`, `torch.softmax`, `F.scaled_dot_product_attention`, `torch.addcmul`, `F.grid_sample`, `F.conv2d`, `torch.optim.AdamW`), NVIDIA cuBLAS (`nn.Linear` / `torch.mv`), NVIDIA cuDNN, FlashAttention-2, or `bitsandbytes` 4-Bit `NF4` (`bnb.nn.Linear4bit`).
2. **Pre-Allocated Setup Buffers, Pre-Allocated Optimizers & Independently Validated `out=None` Outputs**:
   - No baseline is ever penalized by avoidable inside-the-loop setup allocations.
   - In `RMSNorm`, non-unit learned scaling weights (`torch.linspace(0.5, 1.5, 4096)`) are used so nontrivial scaling cannot be skipped, and the main table reports **both** pre-allocated `out=buf` (`Row 2a`, `2.11x` faster on `RTX 5080`) and dynamic `out=None` (`Row 2b`, `1.47x` faster on `RTX 5080`), with `Max |d| = 1.0e-06` independently validated on the `out=None` return tensor.
   - In `Softmax` (`Row 4`), both `torch.softmax(x, dim=-1, out=out_pt_soft)` and `apex.softmax(x, out=out_soft)` write into pre-allocated output tensors (`1.64x` faster, `Max |d| = 2.3e-09`).
   - In `100-Stage Affine Recurrence` (`Row 7`), Cortivra-Apex is compared against **both** the 100-launch eager loop (`176.66x` faster) and PyTorch's single-launch fused `torch.addcmul(b_100, v0, a_100, out=fold_out)` C++ kernel (`0.91x`, within `< 1 us` of a single native C++ FMA launch).
   - In `50-Step MLP Training` (`Row 14`), the baseline `torch.optim.AdamW` instance and parameter tensors are pre-allocated outside the timed loop (`12.76x` faster over 50 steps).
3. **Dual Hardware Stream Timing + Exported Raw Per-Launch Sample Arrays (`samples_ms`)**:
   - All timings are measured directly on the hardware CUDA stream via `torch.cuda.Event(enable_timing=True)` after warmup iterations and explicit `torch.cuda.synchronize()` barriers.
   - `time_cuda_stats()` records both the batched multi-iteration stream mean (`stream_mean_ms`) and 25 individual per-launch `cuEventElapsedTime` measurements (`p50_ms`, `p95_ms`, `std_ms`, `min_ms`, `max_ms`, plus the complete `samples_ms` array), exported in [rtx_5080_verified_evidence.json](./rtx_5080_verified_evidence.json).
4. **Both Warm-L2 and Cold-L2 (`128 MiB` Cache Flush) Regimes**:
   - **Warm-L2 Regime (`Row 2a` / `2b`)**: Models sequential layer execution in deep neural networks where activations produced by Layer $k$ remain resident in the GPU's on-chip L2 cache (`40 MiB` on A100, `64 MiB` on RTX 5080, `72 MiB` on RTX 4090, `96 MiB` on L40S/5090, `128 MiB` on RTX PRO 6000) when consumed by Layer $k+1$.
   - **Cold-L2 Regime (`Row 3`)**: Explicitly flushes a separate **`128 MiB` GPU buffer** (`33,554,432` `float32` elements via `l2_flush_buf.zero_()`) and synchronizes the stream before **every single timed iteration**, forcing 100% of input data to be fetched cold from off-chip HBM/GDDR DRAM.
5. **64-Bit IEEE `float64` Ground-Truth Reference (Including `Wave Attention`), Triple Error Metrics & Categorized Acceptance Gates**:
   - Exact operator checks (`RMSNorm`, `Softmax`, `Extended-Precision Solver`) and `Wave Attention` (`ref_wave_attention_f64` on distinct `Q, K, V` tensors) promote input tensors to 64-bit IEEE `torch.float64`, compute the 64-bit mathematical reference, and verify **Cosine Similarity (`Cos`)**, **Maximum Element-Wise Absolute Error (`Max |d|`)**, and **Relative $L_2$ Error (`RelL2`)**.
   - Every check in [verify_independently.py](./verify_independently.py) registers a categorized pass/fail gate (`EXACTNESS+PERF`, `ALGO_REF_EXACTNESS`, `COMPRESSION+PERF+QUALITY`, `INTEGRATION+PERF+QUALITY`, `UNIFIED_11_PILLAR_ORCHESTRATION`, `TRAINING_QUALITY+MEMORY`, `RESOURCE+EXACTNESS`, `NUMERICAL_PRECISION`, `ECONOMIC_TCO+ENERGY`, `24/24` total); the script exits with non-zero status (`sys.exit(1)`) if any gate fails.
6. **Binary Memory Units (`KiB`, `MiB`, `GiB`), Active Peak Scratch Tracking & OS Process RSS Profiling**:
   - All memory measurements in [verify_independently.py](./verify_independently.py) and [verify_real_llm_vram.py](./verify_real_llm_vram.py) use strict binary IEC units (`1 MiB = 1024^2 bytes`, `1 GiB = 1024^3 bytes`) and report steady-state allocated VRAM (`torch.cuda.memory_allocated()`), active peak VRAM (`torch.cuda.max_memory_allocated()`), and OS-level process working-set / RSS memory (`GetProcessMemoryInfo` / `/proc/self/statm`).
7. **True Sequential Multi-Layer QLoRA Backpropagation, Head-to-Head `bitsandbytes` `NF4` Comparison & Non-Double-Counted Fleet TCO**:
   - `Scientific Regime [A2]` (and `verify_real_llm_vram.py`) benchmarks `ApexLinear` (`bits=4`) directly against `bitsandbytes` 4-Bit `NF4` (`bnb.nn.Linear4bit`) on both `5120 x 13824` matrices (`M = 1, 2, 4, 8, 64, 256`) and full 36-layer `Qwen2.5-Coder-3B-Instruct` generation (`36.1 tok/s` vs `26.8 tok/s` on `RTX 5080`; `48.8 tok/s` vs `37.2 tok/s` on `NVIDIA L40S`, with higher Logit Cosine `0.995508` vs `0.993138`).
   - `Scientific Regime [C]` trains a **true 8-layer sequential residual network** ($h_{k+1} = h_k + \text{Layer}_k(h_k) + \text{LoRA}_k(h_k)$) where `loss.backward()` propagates activation gradients through all 8 frozen 4-bit layers down to Layer 0 (`||grad_L0|| = 4.45e-02`), gating that the 4-bit final loss matches the `FP16` final loss within `10%` while cutting full training peak VRAM by `3.12x`. Furthermore, `verify_real_llm_vram.py` executes real 36-layer pretrained LLM QLoRA backpropagation (`Layer 35 -> Layer 0`) in `3.31–3.43 GiB` peak VRAM.
   - `Scientific Regime [D]` (`18_non_double_counted_tco_verification`) measures live NVML power draw (Watts) and energy efficiency (`mJ/token`, `1.74x–2.77x` more efficient) and computes non-double-counted Fleet TCO via the step-function saturation inequality $G = \max(\lceil W \cdot M_{\text{model}} / V_{\text{usable}} \rceil, \lceil D_{\text{SLA}} / (T_{\text{model}} \cdot \lfloor V_{\text{usable}} / M_{\text{model}} \rfloor) \rceil)$, including explicit neutral/negative control regimes (`0a` Micro-Tensor Bypass: `1 -> 1 GPU`, `0.0%` TCO delta; `0b` Unconstrained-VRAM `M=256`: `10 -> 10–11 GPUs`, `<= 0%` TCO delta) alongside University Lab (`32 -> 12 GPUs`, `-62.5%`), AI SaaS Startup (`64 -> 13 GPUs`, `-79.7%`), and Hyperscale Datacenter (`10,000 -> 1,909–2,028 H100 GPUs`, `-79.7%` to `-80.9%`).
8. **Hardened Side-by-Side Pretrained Model Evaluation & Committed JSON Evidence ([rtx_5080_llm_verified_evidence.json](./rtx_5080_llm_verified_evidence.json))**:
   - Real pretrained HuggingFace checkpoints (`Qwen2.5-Coder-3B/14B/32B`, `Qwen2.5-72B-Instruct`, `Flux.1-Lite-8B`) are compressed in-place with zero fine-tuning, zero calibration prompts, and exact softmax attention (`wave_attention=False`).
   - Models are graded on multi-domain teacher-forced Perplexity (Python, Systems Prose, Math Theorem, SQL), full-vocabulary (`151,936`-token) 64-bit Logit Cosine Similarity, Top-1 / Top-5 token agreement, and side-by-side (`FP16` Baseline vs `4-Bit`) `OpenAI HumanEval` (`10` problems executed inside a `python -I` isolated subprocess guarded by `sys.addaudithook` and an allowlisted `_safe_import` for standard pure-algorithmic modules `typing`, `math`, `functools`, `itertools`, `collections`, `re`), `GSM8K` (`5` problems), and `MMLU` (`5` problems).

---

## 2. Two Attention Execution Modes (`Exact Softmax` vs. `Linear O(N) Wave`)

Cortivra-Apex provides two distinct attention paths depending on whether the user requires strict token-for-token pretrained checkpoint equivalence or linear-complexity long-context scaling:

| Mode | API / Flag | Complexity & Aux Memory | When It Is Used |
| :--- | :--- | :---: | :--- |
| **Exact Softmax Attention (Default for Pretrained LLMs)** | `apex.enable_auto_intercept(wave_attention=False)` (and default inside `apex.boost(model)`) | $O(N^2)$ exact causal softmax | **Default for all frozen pretrained HuggingFace LLMs** (`Qwen`, `Llama`, `Mistral`, `DeepSeek`) and **Row `12c` (`4-Layer 14B-Dim, Growing KV=64->68`)**. Preserves exact attention probabilities while accelerating `4-Bit/8-Bit Linear Projections` (`2.86x–4.40x` GEMV + `3.56x–7.11x` VRAM), `RMSNorm`, and `Softmax`. |
| **Linear $O(N)$ Wave-Projection Attention (Opt-In)** | `apex.enable_auto_intercept(wave_attention=True)` or `apex.wave_attention(q, k, v)` | $O(N)$ linear spectral projection (`0.00 MiB` measured aux matrix) | **Opt-in (`[ALGO-O(N)]`, Rows 5 & 6)** for custom long-context models (`8K–128K+` tokens), real-time streaming inference, or models trained/fine-tuned with linear wave attention (`54.3x–124.5x` faster at `8,192` tokens, verified against `ref_wave_attention_f64` at `Max \|d\| = 5.7e-07`). |

---

## 3. Automatic Hardware Dispatch & Small-Tensor Bypass Thresholds

Custom GPU kernels are not beneficial on arbitrarily small tensors where launch latency dominates arithmetic time. On first launch, Cortivra-Apex executes a 250ms hardware self-calibration pass (`~/.cortivra_apex/hardware_profile.json`) and automatically routes sub-threshold shapes directly to native PyTorch C++ via a `< 100 ns` inline guard clause:

- **Matrix Multiplication (`M < 32`, `N < 32`, or `K < 16`)**: Delegated directly to native PyTorch / cuBLAS.
- **Normalization (`Cols < 64` or `Total Elements <= 1024`)**: Delegated directly to native PyTorch `F.rms_norm` in `< 100 ns` (verified across 4 micro-tensor shapes `1x32`, `4x64`, `8x32`, `16x64` at `0.92x–1.03x` native C++ parity in `Regime [B]` of [verify_independently.py](./verify_independently.py)).
- **Attention (`SeqLen < 32` or `HeadDim < 16`)**: Delegated directly to native PyTorch `F.scaled_dot_product_attention`.
- **Batch-Size Routing in 4-Bit & 8-Bit Compressed Linear Layers (`ApexLinear`)**:
  - **Decode & Small-Batch Regime (`M = Batch * SeqLen <= 8`)**: Dispatches the fused 128-bit in-register transform-domain GEMV kernel, computing inner products directly on compressed 4-bit blocks in registers without materializing uncompressed weights (`2.86x–4.40x` faster than `FP16` cuBLAS).
  - **Large-Batch Prefill Regime (`M = Batch * SeqLen > 8`)**: Decompresses the active layer's weights into a single reusable pre-allocated scratch buffer (`_PREFILL_SCRATCH_CACHE`) shared across all layers (and cached across consecutive prefill micro-batches on the active layer) and invokes native Tensor Core `FP16`/`BF16` cuBLAS GEMM (`1.53x–1.58x` faster than `bitsandbytes` `NF4` at `M = 64..256` and `0.94x–0.99x` parity with uncompressed `FP16` cuBLAS).
- **Zero-Crash Transactional Fallback**:
  If any custom kernel raises a driver exception or produces non-finite values, the runtime dispatcher catches it and falls back to native PyTorch C++ for that call.
