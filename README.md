# FunctionFitter

A neural network-based function approximation tool that visualizes the training process through animated GIF generation. This project implements a Multi-Layer Perceptron (MLP) using PyTorch to fit arbitrary mathematical functions.

## Features

- **Custom Function Support** - Parse and fit user-defined mathematical expressions
- **MLP Architecture** - 3-layer neural network with Tanh activation functions
- **Training Visualization** - Automatic GIF generation showing the fitting process
- **Configurable Parameters** - Adjustable learning rate, hidden layer size, and training epochs

## Installation environment

```bash
pip install torch numpy matplotlib
```

## Instructions for use

### Base（random function）

```bash
python main.py
```

This will generate a random objective function and fit it.

### Fit custom function

```bash
#For example:
python main.py --func "sin(x)"

python main.py --func "sin(2*x) + cos(x) * exp(-0.1*x**2)"

python main.py --func "x**3 - 2*x**2 + x - 1"
```

### Adjust training parameters

```bash
python main.py --func "tan(x)" --hidden 256 --lr 0.001 --epochs 5000
```

## Parameter description

| parameter | default value | illustrate |
|------|--------|------|
| `--func` | `None` | random function |
| `--hidden` | `128` | Number of neurons in hidden layer |
| `--lr` | `0.002` | learning rate |
| `--epochs` | `3000` | Number of training rounds |

## Supported mathematical functions

- **Basic operations**: `+`, `-`, `*`, `/`, `**`
- **Trigonometric functions**: `sin`, `cos`, `tan`
- **Exponential logarithm**: `exp`, `log`
- **other**: `sqrt`, `abs`, `tanh`
- **constant**: `pi`, `e`

## Project Structure

```
functionfitter/
├── main.py          # Main entry point and training loop
├── data_gen.py      # Function parser and data generator
├── mod.py           # MLP model definition
├── visualizer.py    # Animation generator
└── README.md        # Documentation
```

## Implementation Details

### Model Architecture
```
Input(1) → Tanh(hidden) → Tanh(hidden) → Output(1)
```

### Training Configuration
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Training Points**: 80 randomly sampled points
- **Visualization Grid**: 300 points for smooth curve rendering

### Output Format
The generated `Data Fitting.gif` displays:
- **Black dashed line**: True target function
- **Red scatter points**: Training data samples
- **Blue curve**: MLP prediction (evolving through training)

---

*Built with PyTorch, NumPy, and Matplotlib*
