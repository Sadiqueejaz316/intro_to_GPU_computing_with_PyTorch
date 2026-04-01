
import torch
import numpy as np
import time
from typing import Optional, List, Tuple

def check_gpu():
    """Check GPU availability and print device info."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"Device name:     {props.name}")
        print(f"CUDA version:    {torch.version.cuda}")
        print(f"Total VRAM:      {props.total_mem / 1e9:.1f} GB")
        print(f"SMs:             {props.multi_processor_count}")
        print(f"Compute cap.:    {props.major}.{props.minor}")
    else:
        raise RuntimeError("GPU not available! Change runtime to GPU.")

_memory_baseline_allocated = 0
_memory_baseline_reserved = 0

def init_gpu_memory_baseline():
    """
    Initialize CUDA context and capture memory baseline.
    
    Must be called once at the start of the notebook, BEFORE any user allocations.
    Forces CUDA context + cuBLAS workspace initialization, then records the
    resulting memory as baseline. All subsequent print_gpu_memory() calls
    report usage relative to this baseline, hiding PyTorch internals.
    """
    global _memory_baseline_allocated, _memory_baseline_reserved
    _ = torch.randn(100, 100, device='cuda') @ torch.randn(100, 100, device='cuda')
    torch.cuda.synchronize()
    _memory_baseline_allocated = torch.cuda.memory_allocated()
    _memory_baseline_reserved  = torch.cuda.memory_reserved()
    print(f"Memory baseline set: "
          f"allocated={_memory_baseline_allocated / 1e6:.1f} MB, "
          f"reserved={_memory_baseline_reserved / 1e6:.1f} MB "
          f"(CUDA context + internal buffers)")


def print_gpu_memory(label: str = ""):
    """Print GPU memory usage relative to baseline (user tensors only)."""
    allocated = (torch.cuda.memory_allocated() - _memory_baseline_allocated) / 1e6
    reserved  = (torch.cuda.memory_reserved() - _memory_baseline_reserved) / 1e6
    print(f"[{label:>25s}]  allocated: {allocated:8.1f} MB  |  reserved: {reserved:8.1f} MB")



def print_peak_memory(label: str = ""):
    """Print peak GPU memory allocated since last reset_peak_memory_stats(), relative to baseline."""
    peak = (torch.cuda.max_memory_allocated() - _memory_baseline_allocated) / 1e6
    print(f"[{label:>25s}]  peak allocated: {peak:8.1f} MB")


def reset_gpu_memory():
    """Free all cached GPU memory including internal workspaces."""
    import gc
    gc.collect()
    with torch.no_grad():
        try:
            torch._C._cuda_clearCublasWorkspaces()
        except:
            pass
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

def gpu_memory_report(top_n: int = 10):
    """
    List the largest GPU tensors currently in memory.
    Scans all Python objects via gc - may be slow with many objects.
    """
    import gc
    
    allocated = torch.cuda.memory_allocated() / 1e6
    reserved  = torch.cuda.memory_reserved() / 1e6
    print(f"Total allocated: {allocated:.1f} MB  |  reserved: {reserved:.1f} MB\n")
    
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                size_mb = obj.nelement() * obj.element_size() / 1e6
                tensors.append({
                    'shape': tuple(obj.shape),
                    'dtype': obj.dtype,
                    'size_mb': size_mb,
                    'data_ptr': obj.data_ptr(),
                    'requires_grad': obj.requires_grad,
                })
        except:
            pass
    
    # Deduplicate views (same data_ptr)
    seen_ptrs = {}
    for t in tensors:
        ptr = t['data_ptr']
        if ptr not in seen_ptrs or t['size_mb'] > seen_ptrs[ptr]['size_mb']:
            seen_ptrs[ptr] = t
    
    unique = sorted(seen_ptrs.values(), key=lambda x: -x['size_mb'])
    
    print(f"{'#':<4s} {'Shape':<25s} {'Dtype':<12s} {'Size MB':>10s} {'Grad?':>6s}")
    print("-" * 60)
    for i, t in enumerate(unique[:top_n]):
        print(f"{i+1:<4d} {str(t['shape']):<25s} {str(t['dtype']):<12s} "
              f"{t['size_mb']:>10.2f} {str(t['requires_grad']):>6s}")
    
    total_tracked = sum(t['size_mb'] for t in unique)
    print(f"\nTracked: {total_tracked:.1f} MB across {len(unique)} unique tensors")
    if allocated - total_tracked > 1.0:
        print(f"Untracked: ~{allocated - total_tracked:.1f} MB "
              f"(internal buffers, cached computations)")
        

def measure_transfer_time(
    size: int,
    direction: str = 'cpu_to_gpu',
    num_repeats: int = 20,
    pinned: bool = False
) -> float:
    """
    Measure median transfer time for a (size x size) float32 tensor.

    Args:
        size: Matrix dimension (size x size).
        direction: 'cpu_to_gpu' or 'gpu_to_cpu'.
        num_repeats: Number of repetitions for stable measurement.
        pinned: If True, use pinned (page-locked) memory for CPU tensor.

    Returns:
        Median transfer time in milliseconds.
    """
    times = []

    for _ in range(num_repeats):
        if direction == 'cpu_to_gpu':
            data = torch.randn(size, size)
            if pinned:
                data = data.pin_memory()

            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)

            start.record()
            data_gpu = data.to('cuda', non_blocking=pinned)
            end.record()
            torch.cuda.synchronize()

        else:  # gpu_to_cpu
            data_gpu = torch.randn(size, size, device='cuda')
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)

            start.record()
            data_cpu = data_gpu.to('cpu')
            end.record()
            torch.cuda.synchronize()

        times.append(start.elapsed_time(end))  # ms

    return np.median(times)


def benchmark_transfers(
    sizes: Optional[List[int]] = None,
    num_repeats: int = 20
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Benchmark CPU↔GPU transfers for multiple tensor sizes.

    Returns:
        (sizes_mb, times_to_gpu, times_from_gpu, bandwidths_gb_s)
    """
    if sizes is None:
        sizes = [100, 500, 1000, 2000, 3000, 5000, 7000, 10000]

    sizes_mb = [(s * s * 4) / 1e6 for s in sizes]  # float32 = 4 bytes
    times_to_gpu   = [measure_transfer_time(s, 'cpu_to_gpu', num_repeats) for s in sizes]
    times_from_gpu = [measure_transfer_time(s, 'gpu_to_cpu', num_repeats) for s in sizes]
    bandwidths = [(mb / (t / 1000)) / 1000 for mb, t in zip(sizes_mb, times_to_gpu)]

    return sizes_mb, times_to_gpu, times_from_gpu, bandwidths


def compare_pinned_vs_pageable(size: int, num_repeats: int = 20) -> Tuple[float, float]:
    """
    Compare transfer time: pageable vs pinned memory.

    Returns:
        (time_pageable_ms, time_pinned_ms)
    """
    t_pageable = measure_transfer_time(size, 'cpu_to_gpu', num_repeats, pinned=False)
    t_pinned   = measure_transfer_time(size, 'cpu_to_gpu', num_repeats, pinned=True)
    return t_pageable, t_pinned



def plot_transfer_benchmarks(sizes_mb, times_to_gpu, times_from_gpu):
    """Plot transfer time vs data size (linear and log-log scales)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    axes[0].plot(sizes_mb, times_to_gpu, 'o-', label='CPU → GPU', color='#e74c3c')
    axes[0].plot(sizes_mb, times_from_gpu, 's-', label='GPU → CPU', color='#3498db')
    axes[0].set_xlabel('Data size [MB]')
    axes[0].set_ylabel('Transfer time [ms]')
    axes[0].set_title('Transfer cost CPU ↔ GPU (linear scale)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Log-log scale
    axes[1].loglog(sizes_mb, times_to_gpu, 'o-', label='CPU → GPU', color='#e74c3c')
    axes[1].loglog(sizes_mb, times_from_gpu, 's-', label='GPU → CPU', color='#3498db')
    axes[1].set_xlabel('Data size [MB]')
    axes[1].set_ylabel('Transfer time [ms]')
    axes[1].set_title('Transfer cost CPU ↔ GPU (log-log scale)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.show()


def plot_pinned_comparison(sizes, num_repeats=10):
    """Plot pageable vs pinned memory transfer times for multiple sizes."""
    import matplotlib.pyplot as plt

    results_pageable = []
    results_pinned = []
    sizes_mb = []

    for s in sizes:
        mb = (s * s * 4) / 1e6
        sizes_mb.append(mb)
        t_p, t_pin = compare_pinned_vs_pageable(s, num_repeats)
        results_pageable.append(t_p)
        results_pinned.append(t_pin)

    x = np.arange(len(sizes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, results_pageable, width, label='Pageable', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, results_pinned,   width, label='Pinned',   color='#2ecc71', alpha=0.8)

    ax.set_xlabel('Tensor size')
    ax.set_ylabel('Transfer time [ms]')
    ax.set_title('Pageable vs Pinned Memory Transfer')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}x{s}\n({mb:.0f} MB)' for s, mb in zip(sizes, sizes_mb)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()




def cuda_timer(func, *args, num_repeats=10, warmup=3, **kwargs) -> float:
    """
    Time a GPU function using CUDA events. Returns median time in ms.

    Args:
        func: Callable to time.
        num_repeats: Number of timed repetitions.
        warmup: Number of warmup runs (not timed).

    Returns:
        Median execution time in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_repeats):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)

        start.record()
        func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    return np.median(times)


def profiler_to_df(key_averages, sort_by: str = 'cuda_time_total', row_limit: int = 15):
    """
    Convert torch.profiler key_averages() to a pandas DataFrame.

    Renders as an HTML table in Jupyter - much more readable than .table()
    which produces a plain-text table that wraps badly in notebooks.

    Compatible with PyTorch < 2.0 (cuda_time_total) and >= 2.0 (device_time_total).

    Args:
        key_averages: Result of prof.key_averages()
        sort_by:      Attribute to sort by ('cuda_time_total' works on all versions).
        row_limit:    Maximum number of rows shown.

    Returns:
        pandas.DataFrame with columns:
            Name, Calls, CUDA ms, Self CUDA ms, CPU ms, Self CPU ms
    """
    import pandas as pd

    def _t(evt, *attrs):
        """Return first available attribute value (µs), or 0."""
        for a in attrs:
            v = getattr(evt, a, None)
            if v is not None:
                return v
        return 0

    # Normalise sort_by so 'cuda_time_total' also works on PyTorch 2.x
    def _sort_key(e):
        return _t(e, sort_by, sort_by.replace('cuda', 'device'))

    rows = []
    for evt in sorted(key_averages, key=_sort_key, reverse=True)[:row_limit]:
        rows.append({
            'Name':         evt.key,
            'Calls':        evt.count,
            'CUDA ms':      round(_t(evt, 'cuda_time_total',      'device_time_total')      / 1000, 3),
            'Self CUDA ms': round(_t(evt, 'self_cuda_time_total', 'self_device_time_total') / 1000, 3),
            'CPU ms':       round(_t(evt, 'cpu_time_total')                                 / 1000, 3),
            'Self CPU ms':  round(_t(evt, 'self_cpu_time_total')                            / 1000, 3),
        })

    return pd.DataFrame(rows)

import sqlite3
import json

def nsys_sqlite_to_chrome_trace(sqlite_path, output_json):
    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()
    
    # Load string lookup table
    c.execute("SELECT id, value FROM StringIds")
    strings = dict(c.fetchall())
    
    events = []
    
    c.execute("""
        SELECT start, end, demangledName, shortName, streamId, 
               registersPerThread, blockX, blockY, blockZ, gridX, gridY, gridZ
        FROM CUPTI_ACTIVITY_KIND_KERNEL
    """)
    for row in c.fetchall():
        start, end, nameId, shortId, stream, regs, bx, by, bz, gx, gy, gz = row
        name = strings.get(nameId, strings.get(shortId, f"kernel_{nameId}"))
        events.append({
            "name": name,
            "cat": "gpu_kernel",
            "ph": "X",
            "ts": start / 1000,       # ns → µs
            "dur": (end - start) / 1000,
            "pid": "GPU",
            "tid": f"Stream {stream}",
            "args": {
                "registers": regs,
                "block": f"{bx}x{by}x{bz}",
                "grid": f"{gx}x{gy}x{gz}",
            }
        })
    
    c.execute("""
        SELECT start, end, nameId, globalTid, correlationId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
    """)
    for start, end, nameId, gtid, corrId in c.fetchall():
        name = strings.get(nameId, f"cuda_api_{nameId}")
        events.append({
            "name": name,
            "cat": "cuda_api",
            "ph": "X",
            "ts": start / 1000,
            "dur": (end - start) / 1000,
            "pid": "CPU",
            "tid": f"Thread {gtid & 0xFFFF}",  # extract thread from globalTid
        })
    
    c.execute("""
        SELECT start, end, syncType, streamId
        FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
    """)
    for start, end, syncType, stream in c.fetchall():
        sync_names = {0: "unknown", 1: "cudaDeviceSynchronize", 
                      2: "cudaStreamSync", 3: "cudaEventSync"}
        events.append({
            "name": sync_names.get(syncType, f"sync_{syncType}"),
            "cat": "sync",
            "ph": "X",
            "ts": start / 1000,
            "dur": (end - start) / 1000,
            "pid": "CPU",
            "tid": "Synchronization",
        })
    
    c.execute("SELECT start, end, text, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL")
    for start, end, text, gtid in c.fetchall():
        if text:
            events.append({
                "name": text,
                "cat": "nvtx",
                "ph": "X",
                "ts": start / 1000,
                "dur": (end - start) / 1000,
                "pid": "CPU",
                "tid": "NVTX",
            })
    
    try:
        c.execute("SELECT name, smCount, totalMemory FROM TARGET_INFO_GPU LIMIT 1")
        gpu = c.fetchone()
        if gpu:
            events.append({
                "name": "process_name", "ph": "M", "pid": "GPU",
                "args": {"name": f"GPU ({gpu[0]}, {gpu[1]} SMs, {gpu[2]/1e9:.0f} GB)"}
            })
    except:
        pass
    
    events.append({"name": "process_name", "ph": "M", "pid": "CPU",
                    "args": {"name": "CPU (CUDA API)"}})
    
    conn.close()
    
    with open(output_json, 'w') as f:
        json.dump({"traceEvents": events}, f)
    
    n_kernels = sum(1 for e in events if e.get('cat') == 'gpu_kernel')
    n_api = sum(1 for e in events if e.get('cat') == 'cuda_api')
    n_sync = sum(1 for e in events if e.get('cat') == 'sync')
    print(f"Converted: {n_kernels} GPU kernels, {n_api} CUDA API calls, {n_sync} syncs")
    print(f"Output: {output_json}")

import sqlite3
import json
import os

def nsys_sqlite_to_chrome_trace(sqlite_path, output_json):
    if not os.path.exists(sqlite_path):
        print(f"Error: Database {sqlite_path} does not exist.")
        return

    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()
    
    events = []

    # Helper function to check if a table exists
    def table_exists(table_name):
        c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        return c.fetchone()[0] == 1

    # Load string lookup table
    if table_exists("StringIds"):
        c.execute("SELECT id, value FROM StringIds")
        strings = dict(c.fetchall())
    else:
        strings = {}
    
    if table_exists("CUPTI_ACTIVITY_KIND_KERNEL"):
        c.execute("""
            SELECT start, end, demangledName, shortName, streamId, 
                   registersPerThread, blockX, blockY, blockZ, gridX, gridY, gridZ
            FROM CUPTI_ACTIVITY_KIND_KERNEL
        """)
        for row in c.fetchall():
            start, end, nameId, shortId, stream, regs, bx, by, bz, gx, gy, gz = row
            name = strings.get(nameId, strings.get(shortId, f"kernel_{nameId}"))
            events.append({
                "name": name,
                "cat": "gpu_kernel",
                "ph": "X",
                "ts": start / 1000,       # ns → µs
                "dur": (end - start) / 1000,
                "pid": "GPU",
                "tid": f"Stream {stream}",
                "args": {
                    "registers": regs,
                    "block": f"{bx}x{by}x{bz}",
                    "grid": f"{gx}x{gy}x{gz}",
                }
            })
    
    if table_exists("CUPTI_ACTIVITY_KIND_RUNTIME"):
        c.execute("""
            SELECT start, end, nameId, globalTid, correlationId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
        """)
        for start, end, nameId, gtid, corrId in c.fetchall():
            name = strings.get(nameId, f"cuda_api_{nameId}")
            events.append({
                "name": name,
                "cat": "cuda_api",
                "ph": "X",
                "ts": start / 1000,
                "dur": (end - start) / 1000,
                "pid": "CPU",
                "tid": f"Thread {gtid & 0xFFFF}",  # extract thread from globalTid
            })
    
    if table_exists("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"):
        c.execute("""
            SELECT start, end, syncType, streamId
            FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
        """)
        for start, end, syncType, stream in c.fetchall():
            sync_names = {0: "unknown", 1: "cudaDeviceSynchronize", 
                          2: "cudaStreamSync", 3: "cudaEventSync"}
            events.append({
                "name": sync_names.get(syncType, f"sync_{syncType}"),
                "cat": "sync",
                "ph": "X",
                "ts": start / 1000,
                "dur": (end - start) / 1000,
                "pid": "CPU",
                "tid": "Synchronization",
            })
    
    if table_exists("NVTX_EVENTS"):
        c.execute("SELECT start, end, text, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL")
        for start, end, text, gtid in c.fetchall():
            if text:
                events.append({
                    "name": text,
                    "cat": "nvtx",
                    "ph": "X",
                    "ts": start / 1000,
                    "dur": (end - start) / 1000,
                    "pid": "CPU",
                    "tid": "NVTX",
                })
    
    if table_exists("TARGET_INFO_GPU"):
        try:
            c.execute("SELECT name, smCount, totalMemory FROM TARGET_INFO_GPU LIMIT 1")
            gpu = c.fetchone()
            if gpu:
                events.append({
                    "name": "process_name", "ph": "M", "pid": "GPU",
                    "args": {"name": f"GPU ({gpu[0]}, {gpu[1]} SMs, {gpu[2]/1e9:.0f} GB)"}
                })
        except Exception:
            pass
    
    events.append({"name": "process_name", "ph": "M", "pid": "CPU",
                    "args": {"name": "CPU (CUDA API)"}})
    
    conn.close()
    
    with open(output_json, 'w') as f:
        json.dump({"traceEvents": events}, f)
    
    n_kernels = sum(1 for e in events if e.get('cat') == 'gpu_kernel')
    n_api = sum(1 for e in events if e.get('cat') == 'cuda_api')
    n_sync = sum(1 for e in events if e.get('cat') == 'sync')
    print(f"Converted: {n_kernels} GPU kernels, {n_api} CUDA API calls, {n_sync} syncs")
    print(f"Output: {output_json}")


import sqlite3
import json
import os

def nsys_sqlite_to_chrome_trace(sqlite_path, output_json):
    if not os.path.exists(sqlite_path):
        print(f"Error: Database {sqlite_path} does not exist.")
        return

    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()
    
    events = []

    # Helper function to check if a table exists
    def table_exists(table_name):
        c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        return c.fetchone()[0] == 1

    # Load string lookup table
    if table_exists("StringIds"):
        c.execute("SELECT id, value FROM StringIds")
        strings = dict(c.fetchall())
    else:
        strings = {}
    
    if table_exists("CUPTI_ACTIVITY_KIND_KERNEL"):
        c.execute("""
            SELECT start, end, demangledName, shortName, streamId, 
                   registersPerThread, blockX, blockY, blockZ, gridX, gridY, gridZ
            FROM CUPTI_ACTIVITY_KIND_KERNEL
        """)
        for row in c.fetchall():
            start, end, nameId, shortId, stream, regs, bx, by, bz, gx, gy, gz = row
            name = strings.get(nameId, strings.get(shortId, f"kernel_{nameId}"))
            events.append({
                "name": name,
                "cat": "gpu_kernel",
                "ph": "X",
                "ts": start / 1000,       # ns → µs
                "dur": (end - start) / 1000,
                "pid": "GPU",
                "tid": f"Stream {stream}",
                "args": {
                    "registers": regs,
                    "block": f"{bx}x{by}x{bz}",
                    "grid": f"{gx}x{gy}x{gz}",
                }
            })

    if table_exists("CUPTI_ACTIVITY_KIND_MEMCPY"):
        c.execute("SELECT start, end, streamId, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY")
        # 1: HtoD, 2: DtoH, 8: DtoD, 10: PtoP
        copy_map = {1: "Memcpy HtoD", 2: "Memcpy DtoH", 8: "Memcpy DtoD", 10: "Memcpy PtoP"}
        for start, end, stream, copyKind, bytes_transferred in c.fetchall():
            kind = copy_map.get(copyKind, f"Memcpy_{copyKind}")
            events.append({
                "name": kind,
                "cat": "memcpy",
                "ph": "X",
                "ts": start / 1000,
                "dur": (end - start) / 1000,
                "pid": "GPU",
                "tid": f"Stream {stream}",
                "args": {
                    "bytes": bytes_transferred
                }
            })
    
    if table_exists("CUPTI_ACTIVITY_KIND_RUNTIME"):
        c.execute("""
            SELECT start, end, nameId, globalTid, correlationId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
        """)
        for start, end, nameId, gtid, corrId in c.fetchall():
            name = strings.get(nameId, f"cuda_api_{nameId}")
            events.append({
                "name": name,
                "cat": "cuda_api",
                "ph": "X",
                "ts": start / 1000,
                "dur": (end - start) / 1000,
                "pid": "CPU",
                "tid": f"Thread {gtid & 0xFFFF}",
            })
    
    if table_exists("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"):
        c.execute("""
            SELECT start, end, syncType, streamId
            FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
        """)
        for start, end, syncType, stream in c.fetchall():
            sync_names = {0: "unknown", 1: "cudaDeviceSynchronize", 
                          2: "cudaStreamSync", 3: "cudaEventSync"}
            events.append({
                "name": sync_names.get(syncType, f"sync_{syncType}"),
                "cat": "sync",
                "ph": "X",
                "ts": start / 1000,
                "dur": (end - start) / 1000,
                "pid": "CPU",
                "tid": "Synchronization",
            })
    
    if table_exists("NVTX_EVENTS"):
        c.execute("SELECT start, end, text, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL")
        for start, end, text, gtid in c.fetchall():
            if text:
                events.append({
                    "name": text,
                    "cat": "nvtx",
                    "ph": "X",
                    "ts": start / 1000,
                    "dur": (end - start) / 1000,
                    "pid": "CPU",
                    "tid": "NVTX",
                })
    
    if table_exists("TARGET_INFO_GPU"):
        try:
            c.execute("SELECT name, smCount, totalMemory FROM TARGET_INFO_GPU LIMIT 1")
            gpu = c.fetchone()
            if gpu:
                events.append({
                    "name": "process_name", "ph": "M", "pid": "GPU",
                    "args": {"name": f"GPU ({gpu[0]}, {gpu[1]} SMs, {gpu[2]/1e9:.0f} GB)"}
                })
        except Exception:
            pass
    
    events.append({"name": "process_name", "ph": "M", "pid": "CPU",
                    "args": {"name": "CPU (CUDA API)"}})
    
    conn.close()
    
    with open(output_json, 'w') as f:
        json.dump({"traceEvents": events}, f)
    
    n_kernels = sum(1 for e in events if e.get('cat') == 'gpu_kernel')
    n_memcpy = sum(1 for e in events if e.get('cat') == 'memcpy')
    n_api = sum(1 for e in events if e.get('cat') == 'cuda_api')
    n_sync = sum(1 for e in events if e.get('cat') == 'sync')
    
    print(f"Converted: {n_kernels} GPU kernels, {n_memcpy} Memcpy transfers, {n_api} CUDA API calls, {n_sync} syncs")
    print(f"Output: {output_json}")