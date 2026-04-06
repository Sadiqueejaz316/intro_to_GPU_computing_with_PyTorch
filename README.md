# 🚀 PyTorch GPU Acceleration & Triton Kernels

## 📌 Overview
This project demonstrates practical implementation of **GPGPU (General-Purpose GPU Computing)** using **PyTorch** and **Triton**. It focuses on accelerating numerical computations, optimizing performance, and developing custom GPU kernels.

The project is designed as a **modular Python codebase (no notebooks)** to simulate real-world development workflows. *(Note: Jupyter notebooks are still available in the `notebooks/` directory for historical reference, but all functional examples are being migrated to `src/`.)*

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

```text
.
├── notebooks/                     # Original instructional Jupyter notebooks
├── src/                           # Refactored standalone Python scripts
│   ├── gpu_helpers.py             # Reusable GPU profiling and memory utilities
│   ├── part_1_introduction_to_pytorch.py
│   ├── part_2_linear_regression_and_pca.py
│   ├── regression_viz.py          # Visualization routines
│   └── tensor_viz.py              # Tensor visualization utilities
├── requirements.txt               # Dependencies
└── README.md
```

---

## 🚀 Setup & Execution

To set up the project locally:

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```
2. **Activate the virtual environment:**
   - **Windows:** `venv\Scripts\activate`
   - **MacOS/Linux:** `source venv/bin/activate`
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run scripts:**
   ```bash
   python src/part_1_introduction_to_pytorch.py
   python src/part_2_linear_regression_and_pca.py
   ```
