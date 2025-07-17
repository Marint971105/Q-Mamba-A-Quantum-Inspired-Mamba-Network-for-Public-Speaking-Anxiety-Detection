# Quantum Progressive Ablation Experiments


This project implements progressive ablation experiments for quantum mechanisms, validating the contribution of each component by gradually adding quantum components.

## Experimental Design

### Progressive Model Architecture
1.**BaselineMamba** - Baseline Model

Pure Mamba + traditional L2Norm fusion
No quantum mechanisms used
Serves as performance baseline


2.**QuantumSuperposition** - Quantum Superposition Model

Baseline + quantum superposition state modeling
Uses phase and amplitude to construct quantum states
Computes outer products to represent quantum superposition


3.**QuantumEntanglement** - Quantum Entanglement Model

Quantum superposition + quantum entanglement
Uses Hadamard+CNOT to implement quantum entanglement
Applies CNOT operations to adjacent modalities


4.**QMamba** - Complete Quantum Model

Quantum entanglement + multi-head attention fusion
Supports both attention and magnitude weight calculation methods
Includes all quantum components

### Experimental Objectives

Validate progressive contributions of quantum components
Analyze performance improvements of each quantum mechanism
Provide experimental evidence for quantum mechanism effectiveness

## File Structure

```
models/
├── BaselineMamba.py              
├── QuantumSuperposition.py      
├── QuantumEntanglement.py        
└── QMamba.py                     

run_quantum_progressive_ablation.py  
README_quantum_progressive.md        
```

## Usage


### 1. Comparison Experiments (Recommended)

```bash
python run_quantum_progressive_ablation.py --mode comparison
```

This will sequentially run:
- BaselineMamba
- QuantumSuperposition  
- QuantumEntanglement
- QMamba

### 2. Ablation Experiments
Run ablation experiments:

```bash
python run_quantum_progressive_ablation.py --mode ablation
```

### 3. Individual Experiments

Run experiments for specific models:

```bash
# # Baseline model
python run_quantum_progressive_ablation.py --mode single --model baseline

# Quantum superposition model
python run_quantum_progressive_ablation.py --mode single --model superposition

# Quantum entanglement model
python run_quantum_progressive_ablation.py --mode single --model entanglement

#  Complete Q-Mamba
python run_quantum_progressive_ablation.py --mode single --model qmamba
```

## Experimental Configuration

### Default Parameters

**Dataset**: CN dataset (features4quantum)
**Sequence Model**: CNN
**Embedding Dimension**: 50
**Number of Layers**: 1
**Batch Size**: 64
**Learning Rate**: 0.0005
**Training Epochs**: 50
### Supported Sequence Models

All models support the following sequence models:

- `mamba` 
- `transformer` 
- `cnn` 
- `rnn`
- `rwkv` 

## Output Results

### Result Files

After experiment completion, the following files will be generated:

1. **Comparison Results:**: `results/quantum_progressive/quantum_progressive_comparison.json`
2. **Ablation Results: **: `results/quantum_progressive/quantum_ablation_results.json`
3. **Incremental Contributions:**: `results/quantum_progressive/incremental_contributions.json`

### Result Format

```json
{
  "BaselineMamba": {
    "description": "基线模型：纯Mamba + 传统L2Norm融合",
    "performance": {
      "accuracy": 0.8234,
      "precision": 0.8156,
      "recall": 0.8234,
      "f1": 0.8194
    }
  },
  "QuantumSuperposition": {
    "description": "量子叠加：基线 + 量子叠加状态建模",
    "performance": {
      "accuracy": 0.8456,
      "precision": 0.8389,
      "recall": 0.8456,
      "f1": 0.8422
    }
  }
}
```

### Incremental Contribution Analysis (Not necessarily useful)

The script will automatically calculate and display incremental contributions of each quantum component:

### Debug Mode
Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Extended Experiments

### Custom Parameters


Modify the setup_params() function to adjust experimental parameters:


```python
def setup_params():
    opt = SimpleNamespace(
       
        embed_dim=100,       
        num_layers=2,         
        batch_size=32,        
        lr=0.001,           
        sequence_model='transformer', 
        # ... 
    )
    return opt
```

### Adding New Models

Create new model class
Inherit the same interface
Add model mapping in experiment script
Update documentation

## Citation
If using this experimental code, please cite the relevant paper:
bibtexwaiting
Contact
For questions or suggestions, please contact via:

## Submit Issues
Send emails
Participate in discussions
---
 
