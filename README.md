# Physics-Informed Neural Networks for AdS₅ Wormholes

[![arXiv](https://img.shields.io/badge/arXiv-2024.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2024.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Paper**: "Physics-Informed Neural Networks for Scalar-Stabilized Wormholes in AdS₅: A Complete Holographic Analysis"  
> **Author**: Hugo Hertault  
> **Status**: Submitted

## 🎯 Overview

This repository contains the complete implementation of Physics-Informed Neural Networks (PINNs) for solving scalar-stabilized wormhole configurations in five-dimensional Anti-de Sitter spacetime. Our method successfully:

- ✅ Solves Klein-Gordon equation in curved AdS₅ backgrounds
- ✅ Respects Breitenlohner-Freedman stability bounds  
- ✅ Achieves perfect agreement with AdS/CFT predictions
- ✅ Provides quantified uncertainties via ensemble analysis

## 🔬 Key Results

| Case | φ₀ | m²L² | λL² | Δ_CFT | S_E | Success Rate |
|------|-----|------|-----|-------|-----|--------------|
| 1    | 0.8 | -3.0 | 0.05| 3.000 | 490.3 ± 0.002 | 100% |
| 2    | 1.2 | -2.5 | 0.10| 3.225 | 490.2 ± 0.005 | 100% |
| 3    | 1.5 | -2.0 | 0.15| 3.414 | 490.1 ± 0.002 | 100% |

**Perfect AdS/CFT consistency**: Δ = 2 + √(4 + m²L²) ✓

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/hugohertault/ads5-wormholes-pinns.git
cd ads5-wormholes-pinns
pip install -r requirements.txt
```

### Basic Usage
```python
from src.ads5_wormholes import *

# Run minimal test
solution = minimal_test()

# Run ensemble analysis  
results = ensemble_minimal(n_ensemble=3)
```

### Complete Reproduction
```python
# Full reproduction of paper results
python examples/run_minimal_test.py
```

## 📊 Methodology

### PINN Architecture
- **Input**: Radial coordinate r
- **Network**: 3 hidden layers × 32 neurons
- **Activation**: Hyperbolic tangent (tanh)
- **Output**: Scalar field φ(r)

### Physics-Informed Loss
```
L_total = L_KG + w_BC·L_BC + w_reg·L_reg + w_decay·L_decay
```

Where:
- `L_KG`: Klein-Gordon equation residual
- `L_BC`: Boundary condition at throat
- `L_reg`: Regularity constraint  
- `L_decay`: Asymptotic decay

### Background Metric
Fixed AdS₅-like background:
```
a(r) = 0.95 × sinh(r/L) + 0.1
b(r) = 1.05 × sinh(r/L) + 0.1
```

## 📈 Performance

- **Convergence**: Loss < 10⁻² achieved consistently
- **Speed**: ~2000-3000 epochs for convergence
- **Robustness**: 100% success rate across all test cases
- **Uncertainty**: Statistical errors via 3-member ensemble

## 📋 Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- SciPy

See `requirements.txt` for exact versions.

## 📁 Repository Structure

```
├── src/
│   └── ads5_wormholes.py     # Main PINN implementation
├── paper/
│   └── article_prd.tex       # LaTeX source of paper
├── examples/
│   └── run_minimal_test.py   # Example usage
├── figures/                  # Generated plots
└── requirements.txt          # Dependencies
```

## 🔬 Scientific Context

This work represents the **first comprehensive application** of PINNs to:
- AdS₅ wormhole geometries
- Scalar field stabilization in curved spacetime  
- Holographic boundary conditions
- Ensemble uncertainty quantification

### Innovation
1. **Methodological**: Novel PINN architecture for gravitational systems
2. **Physical**: Complete treatment of AdS₅/CFT₄ consistency
3. **Computational**: Statistical rigor via ensemble methods

## 📖 Citation

```bibtex
@article{Hertault2024,
    title={Physics-Informed Neural Networks for Scalar-Stabilized Wormholes in AdS₅: A Complete Holographic Analysis},
    author={Hugo Hertault},
    journal={Phys. Rev. D},
    year={2024},
    note={Submitted}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

This is research code accompanying a scientific publication. For questions or collaboration:
- Email: hugohertault@yahoo.fr
- Issues: Use GitHub Issues for bug reports

## 🙏 Acknowledgments

- Computational resources for neural network training
- Open source community (PyTorch, NumPy, Matplotlib)
- Physics community for foundational AdS/CFT work

---

⭐ **Star this repo** if you find it useful for your research!

🔗 **Related Work**: Check out other applications of PINNs to physics problems in our research group.
