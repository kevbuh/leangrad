# Leangrad

Leangrad is a [Lean 4](https://lean-lang.org/) neural network library inspired by Karpathy's [Micrograd](https://github.com/karpathy/micrograd).

## Features

- **Automatic Differentiation Engine**: Implementation of reverse-mode autodiff with a computation graph
- **Basic Operations**: Support for addition, multiplication, power operations, and ReLU activation
- **Neural Network Components**: 
  - Single neurons with customizable activation
  - Neural network layers
  - Multi-layer perceptrons (MLPs)
- **Module System**: Type class-based module system for parameter management and gradient zeroing

## Core Components

### Leangrad.Engine

The engine provides the fundamental autodiff functionality:

- `Value`: Core data structure representing a node in the computation graph
- Basic operations: `add`, `mul`, `pow`, `relu`
- Automatic gradient computation via `backward`
- Operator overloads for natural mathematical syntax

### Leangrad.NN

The neural network module provides:

- `Neuron`: Single neuron implementation with weights, bias, and optional nonlinearity
- `Layer`: Collection of neurons
- `MLP`: Multi-layer perceptron implementation
- `Module` typeclass for parameter management

## Example

This creates a network with 3 input neurons and two hidden layers and computes gradients via backpropagation:

```lean
import Leangrad.Engine
import Leangrad.NN
import Std

open Std Leangrad.Engine Leangrad.Engine.Ops Leangrad.NN

def main : IO Unit := do
  -- Create an MLP with 3 inputs and [4,4,1] architecture
  let mlp ← mkMLP 3 [4, 4, 1]
  
  -- Create input values
  let x1 ← mkValue 2
  let x2 ← mkValue (-3)
  let x3 ← mkValue 10
  let input := #[x1, x2, x3]
  
  -- Forward pass
  let out ← mlpForward mlp input
  
  -- Backward pass (on first output)
  match out[0]? with
  | some firstOut => backward firstOut
  | none => pure ()
```