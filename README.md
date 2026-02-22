# ⚛️ PEE886 - Quantum Machine Learning

[![Python Module Import Check](https://github.com/jodafons/pee-886-2026-01/actions/workflows/import_check.yml/badge.svg)](https://github.com/jodafons/pee-886-2026-01/actions/workflows/import_check.yml)

Welcome to the official repository of the **PEE886 - Quantum Machine Learning** course. This space is designed to organize implementations, experiments, and discoveries from students throughout the course.

---

## 🏗️ Repository Organization

The project follows a modular structure where each student has their own "workspace" identified by a unique letter (e.g., `a`, `b`, `c`).

### 📂 Folder Structure

```text
.
├── 📁 notebooks/          # 📓 Jupyter Notebooks with demonstrations and lessons
│   ├── 📁 a/              # 👤 Student A Notebooks
│   ├── 📁 b/              # 👤 Student B Notebooks
│   └── ...
├── 📁 data/               # 💾 Raw data, configurations, and exports
│   ├── 📁 a/              # 👤 Student A data and configs
│   ├── 📁 b/              # 👤 Student B data and configs
│   └── ...
├── 📁 qml/                # 🧠 Main module for Quantum Machine Learning
│   ├── 📁 a/              # 👤 Student A Implementation
│   ├── 📁 b/              # 👤 Student B Implementation
│   │   ├── 📁 loaders/    # 📥 Data loading and processing
│   │   ├── 📁 models/     # 🤖 Quantum model architectures
│   │   ├── 📁 trainer/    # ⚡ Training and optimization loops
│   │   ├── 📁 evaluation/ # 📊 Metrics and performance evaluation
│   │   └── 📁 visualization/ # 📈 Charts and circuit visualization
│   └── ...             # 👤 Other students (c, d, e...)
├── 📁 scripts/            # 🛠️ Utility and automation scripts
│   ├── 📁 a/              # 👤 Student A Scripts
│   ├── 📁 b/              # 👤 Student B Scripts
│   └── ...
├── 📄 Makefile            # ⚙️ Shortcut commands (build, jupyter, etc.)
├── 📄 requirements.txt    # 📦 Global project dependencies
└── 📄 activate.sh         # 🚀 Virtual environment activation script
```

---

## 👤 Students and Implementations

Each student has a dedicated space for their implementation. Click on the corresponding letter to access each one's specific documentation:

- [👤 Student A](./qml/a/README.md)
- [👤 Student B](./qml/b/README.md)
<!-- Add new links here as new students are included -->

---

## 📜 Contribution Rules

To maintain code harmony and readability, all students must follow these guidelines:

### 👤 Reserved Areas and Documentation
Each student has an identifying letter (e.g., `a`, `b`, `c`). You must work **exclusively** within the folders corresponding to your letter in:
- 📁 `qml/<letter>/`
- 📁 `notebooks/<letter>/`
- 📁 `scripts/<letter>/`
- 📁 `data/<letter>/`

📌 **Important**: Each student module (`qml/<letter>/`) contains its own **README.md**. It is mandatory for students to fill out this file explaining:
- Architecture and technologies used.
- Usage instructions for your implementation.
- Bibliographic references.

### 🐍 Naming Convention
- **Files**: Always use lowercase letters separated by underscores (snake_case).
  - ✅ `my_model.py`, `quantum_trainer.py`
  - ❌ `MyModel.py`, `QuantumTrainer.py`
- **Classes**: Use CamelCase (e.g., `QuantumLayer`).
- **Functions**: Use snake_case (e.g., `run_simulation`).

### 📦 Dependencies
- **NEVER** install local packages or create `requirements.txt` files inside student subfolders.
- All necessary libraries (Qiskit, PennyLane, Scikit-Learn, etc.) must be added to the `requirements.txt` file at the **root** of the repository.

### 🛠️ Implementation Standard
Each student must implement all their experiment requirements within their respective lettered folder, respecting the recommended internal sub-organization (`loaders`, `models`, `trainer`, etc.).

---

## 🔄 Collaboration Workflow

To contribute to this repository, follow these steps:

1. **Fork the Repository**: All students must create a fork of this repository to their own GitHub accounts.
2. **Implement Changes**: Work on your implementation within your assigned lettered folders.
3. **Submit a Pull Request (PR)**: Once your implementation is ready and tested, submit a PR to the main repository.
4. **Detailed Description**: All PRs **must** include a comprehensive description of the changes made, the logic implemented, and any results obtained.
5. **Automated Checks (CI)**: Every Pull Request and commit will trigger an automated check to verify that the modules can be imported without errors. Ensure your implementation doesn't break the main `qml` package.
6. **Review Process**: A selected student will be responsible for analyzing and reviewing all submitted Pull Requests to ensure they meet the course standards and follow the contribution rules.

---

## 🚀 How to Start

### 🛠️ Installation via Makefile
The repository uses a `Makefile` to simplify environment configuration. To install all dependencies and set up the virtual environment (`.qml-env`), use the command:

```bash
make install
```

This command will:
1. Create the virtual environment if it doesn't exist.
2. Install dependencies listed in `requirements.txt`.
3. Install the `qml` package in editable mode.

### 🐍 Manual Environment Activation
Whenever you work on the project, activate the virtual environment:

```bash
source activate.sh
```

### 📓 Jupyter Lab
To start Jupyter Lab with the configured environment, use:

```bash
make jupyter
```

---

## ⚠️ Troubleshooting: Import Errors

Since this is a collaborative repository, an error in one student's module might prevent the entire `qml` package from being imported. 

If you encounter an `ImportError` or any execution error caused by another student's code (e.g., student **B** has a bug and you are student **A**), you can temporarily disable their module:

1. Open `qml/__init__.py`.
2. Locate the import section for the problematic student.
3. Comment out the lines related to that student.

**Example:**
If Student **B**'s code is broken:
```python
# In qml/__init__.py

from . import a
__all__.extend( a.__all__ )
from .a import *

# Comment these if B is broken:
# from . import b
# __all__.extend( b.__all__ )
# from .b import *
```

This will allow you to continue working on your own implementation without being blocked by external bugs. **Do not forget to uncomment it once the issue is resolved!**

---

## 🎓 About the Course

The [**PEE-886 Quantum Machine Learning**](https://sites.google.com/lps.ufrj.br/jodafons/ensino/ppe886-quantum-machine-learning) course is offered by the COPPE electrical engineering program and proposes a technical immersion in the convergence between quantum computing and artificial intelligence, starting with a rigorous foundation of the pillars that support this new paradigm. In the first stage, we will explore the architecture and functioning of the main quantum computer models, consolidating the mathematical framework through Dirac notation applied to linear algebra. Additionally, we will examine simulation methodologies and the programming library ecosystem — such as Qiskit and PennyLane — that allow for the practical implementation of algorithms and noise analysis in contemporary quantum systems. The second phase of the course will be entirely dedicated to Quantum Machine Learning (QML), focusing on the main strategies for developing models that seek quantum advantage. Variational hybrid algorithms, quantum neural networks, and quantum kernel-based methods will be analyzed, emphasizing the integration of parameterized circuits with classical optimization techniques. The goal is to enable researchers to understand how properties of superposition, entanglement, and interference can be exploited to enhance processing capacity and generalization in problems of high computational complexity.

---
Developed with ❤️ for the **Quantum Lab**.

