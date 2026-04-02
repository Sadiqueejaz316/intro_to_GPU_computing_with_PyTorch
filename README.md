# 🚀 PyTorch GPU Acceleration & Triton Kernels

## 📌 Overview
This project demonstrates practical implementation of **GPGPU (General-Purpose GPU Computing)** using **PyTorch** and **Triton**. It focuses on accelerating numerical computations, optimizing performance, and developing custom GPU kernels.

The project is designed as a **modular Python codebase (no notebooks)** to simulate real-world development workflows.

---

## 🎯 Objectives
- Implement GPU-accelerated numerical algorithms using PyTorch  
- Perform efficient linear algebra computations on GPU  
- Analyze CPU vs GPU performance  
- Profile and optimize GPU workloads  
- Develop custom GPU kernels using Triton  

---

## 🧠 Key Concepts

### 🔹 GPGPU Computing
- GPU multiprocessing paradigm  
- CUDA fundamentals and hardware-agnostic approaches  

### 🔹 PyTorch Fundamentals
- Tensor operations (creation, reshaping, broadcasting)  
- Indexing, slicing, masking  
- Matrix multiplication and element-wise operations  
- Linear algebra  

### 🔹 GPU Acceleration
- CPU vs GPU benchmarking  
- Memory management (CPU ↔ GPU transfers)  
- Bottleneck analysis  

### 🔹 Profiling & Optimization
- PyTorch profiler  
- Execution tracing  
- Performance bottleneck detection  

### 🔹 Triton Kernel Development
- Custom GPU kernel design  
- Vector operations and reductions  
- Integration with PyTorch  

---

## ⚙️ Tech Stack
- Python 3.x  
- PyTorch  
- Triton  
- CUDA (GPU backend)  

---

## 📁 Project Structure

```bash
.
├── src/
│   ├── core/
│   │   ├── tensor_ops.py
│   │   ├── linear_algebra.py
│   │   └── gpu_utils.py
│   │
│   ├── profiling/
│   │   ├── profiler.py
│   │   └── benchmarks.py
│   │
│   ├── kernels/
│   │   ├── triton_kernels.py
│   │   └── custom_ops.py
│   │
│   └── main.py
│
├── tests/
│   ├── test_tensor_ops.py
│   ├── test_kernels.py
│
├── results/
│   └── logs/
│
├── requirements.txt
├── setup.py
└── README.md

## Structure

* `notebooks/`: Contains interactive Jupyter notebooks covering lessons and exercises.
* `src/`: Python source code containing helper functions and visualization utilities.
* `requirements.txt`: Project dependencies.

## Setup

To set up the project locally:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On MacOS/Linux: `source venv/bin/activate`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
