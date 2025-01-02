import Leangrad.Engine
import Std

open Std Leangrad.Engine
open Leangrad.Engine.Ops

namespace Leangrad.NN

/-
  1) Define a typeclass `Module α` with the methods `parameters` and `zeroGrad`.
     In Lean, "classes" with `def parameters ...` won't let you do
     `instance : Module Neuron` the same way.
     Instead, do `class Module (α : Type) where ...`.
-/
class Module (α : Type) where
  parameters : α → IO (Array Value)
  zeroGrad   : α → IO Unit

--------------------------------------------------------------------------------
-- NEURON
--------------------------------------------------------------------------------

/-- A single Neuron has:
    - `w`: array of weights (Values)
    - `b`: bias (Value)
    - `nonlin`: use ReLU if true, else linear
-/
structure Neuron where
  w      : Array Value
  b      : Value
  nonlin : Bool
  -- no `deriving Inhabited`

/-- Create a random Value in [-1,1]. -/
def randVal : IO Value := do
  -- `IO.rand` wants natural (≥0) numbers. We'll do 0..999999, then shift/scale.
  let r ← IO.rand 0 1000000
  let floatVal := (r.toFloat / 500000.0) - 1.0  -- in approximately [-1,1]
  mkValue floatVal

/-- Initialize a Neuron with random weights in [-1,1] and bias=0. -/
def mkNeuron (nin : Nat) (nonlin : Bool := true) : IO Neuron := do
  let mut wArr := #[]
  for _ in [0: nin] do
    wArr := wArr.push (← randVal)
  let b ← mkValue 0
  pure { w := wArr, b := b, nonlin := nonlin }

/-- Implement the Module interface for Neuron. -/
instance : Module Neuron where
  parameters n := do
    -- Combine all weights + bias
    pure (n.w.push n.b)
  zeroGrad n := do
    for w in n.w do
      w.gradRef.set 0
    n.b.gradRef.set 0

/-- Forward pass of a single Neuron:
    act = sum(wᵢ*xᵢ) + b, then ReLU if nonlin else identity.
-/
def neuronForward (n : Neuron) (x : Array Value) : IO Value := do
  -- First, confirm the sizes match
  if x.size != n.w.size then
    throw <| IO.userError s!"Mismatched sizes: x.size={x.size} vs w.size={n.w.size}"

  -- Zip the arrays so each pair is guaranteed to exist
  let pairs := n.w.zip x
  -- pairs : Array (Value × Value)

  let mut sumVal ← mkValue 0
  for (wi, xi) in pairs do
    let tmp ← mul wi xi
    sumVal ← add sumVal tmp

  let out ← add sumVal n.b
  if n.nonlin then relu out else pure out

--------------------------------------------------------------------------------
-- LAYER
--------------------------------------------------------------------------------

/-- A Layer is an array of Neurons. -/
structure Layer where
  neurons : Array Neuron
  -- no `deriving Inhabited`

/-- Create a Layer of `nout` Neurons, each with `nin` inputs. -/
def mkLayer (nin nout : Nat) (nonlin : Bool) : IO Layer := do
  let mut ns := #[]
  for _ in [0:nout] do
    let n ← mkNeuron nin nonlin
    ns := ns.push n
  pure { neurons := ns }

/-- Implement `Module` for `Layer`. -/
instance : Module Layer where
  parameters L := do
    let mut ps := #[]
    for neuron in L.neurons do
      let arr ← Module.parameters neuron
      ps := ps.append arr
    pure ps
  zeroGrad L := do
    for neuron in L.neurons do
      Module.zeroGrad neuron

/-- Forward pass of a Layer: apply each Neuron to input x, gather results. -/
def layerForward (L : Layer) (x : Array Value) : IO (Array Value) := do
  let mut outs := #[]
  for n in L.neurons do
    outs := outs.push (← neuronForward n x)
  pure outs

--------------------------------------------------------------------------------
-- MLP
--------------------------------------------------------------------------------

/-- An MLP is an array of Layers. -/
structure MLP where
  layers : Array Layer
  -- no `deriving Inhabited`

/-- Create an MLP.
    Example usage: mkMLP 3 [4,4,1] creates:
      - Layer 1: input=3, output=4
      - Layer 2: input=4, output=4
      - Layer 3: input=4, output=1
    The boolean `nonlin` is true for all but the last layer.
-/
def mkMLP (nin : Nat) (nouts : List Nat) : IO MLP := do
  let mut arr := #[]
  -- e.g. if nin=3, nouts=[4,4,1], the pairs are (3->4), (4->4), (4->1)
  let sizes := nin :: nouts
  for i in [0:nouts.length] do
    let nIn  := sizes[i]!
    let nOut := sizes[i+1]!
    let isNonlin := i != nouts.length - 1
    let lay ← mkLayer nIn nOut isNonlin
    arr := arr.push lay
  pure { layers := arr }

/-- Implement `Module` for MLP. -/
instance : Module MLP where
  parameters m := do
    let mut ps := #[]
    for lay in m.layers do
      let arr ← Module.parameters lay
      ps := ps.append arr
    pure ps
  zeroGrad m := do
    for lay in m.layers do
      Module.zeroGrad lay

/-- Forward pass of an MLP, sequentially through each Layer. -/
def mlpForward (m : MLP) (x : Array Value) : IO (Array Value) := do
  let mut cur := x
  for lay in m.layers do
    let tmp ← layerForward lay cur
    cur := tmp
  pure cur

end Leangrad.NN
