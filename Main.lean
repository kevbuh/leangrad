import Leangrad.Engine
import Leangrad.NN
import Std

open Std
open Leangrad.Engine
open Leangrad.Engine.Ops
open Leangrad.NN

/--
A helper that prints an Array of Values by fetching their data/grad
so we don't need a "ToString (Array Value)" instance.
-/
def printValues (msg : String) (arr : Array Value) : IO Unit := do
  IO.println msg
  for v in arr do
    let d ← v.dataRef.get
    let g ← v.gradRef.get
    IO.println s!"  Value(id={v.id}, data={d}, grad={g}, op={v.op})"

def main : IO Unit := do

  --------------------------------------------------------------------------------
  -- 1) Simple demonstration of Value ops
  --------------------------------------------------------------------------------

  let x1 ← mkValue 2
  let x2 ← mkValue (-3)
  let x3 ← mkValue 10

  -- y1 = x1 + x2
  let y1 ← add x1 x2
  -- y2 = y1 * x3
  let y2 ← mul y1 x3

  -- Backprop on y2
  backward y2

  -- Print out the data/grad of x1, x2, x3, y1, y2
  -- (Using "printValues" for arrays, or just individually)
  IO.println "==== After backprop on y2 ===="
  for v in #[x1, x2, x3, y1, y2] do
    let d ← v.dataRef.get
    let g ← v.gradRef.get
    IO.println s!"Value(id={v.id}, data={d}, grad={g}, op={v.op})"

  --------------------------------------------------------------------------------
  -- 2) Simple MLP usage
  --------------------------------------------------------------------------------

  -- Build an MLP with 3 inputs -> [4,4,1] hidden/out
  let mlp ← mkMLP 3 [4, 4, 1]

  -- Create input array [x1, x2, x3]
  let input : Array Value := #[x1, x2, x3]

  -- Forward pass
  let out ← mlpForward mlp input
  printValues "MLP output (before backprop):" out

  -- Backward pass on first output if it exists
  match out[0]? with
  | some firstOut =>
    backward firstOut
    IO.println "Backprop done on the first MLP output."
  | none =>
    IO.println "No MLP outputs, skipping backprop."

  -- Print how many parameters in MLP
  let ps ← Module.parameters mlp
  IO.println s!"Number of parameters in MLP = {ps.size}"

  -- If you want, show the first parameter's grad
  match ps[0]? with
  | some p0 =>
    let g ← p0.gradRef.get
    IO.println s!"First param grad = {g}"
  | none =>
    IO.println "No parameters in MLP."
