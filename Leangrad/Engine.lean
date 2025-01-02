import Std

namespace Leangrad.Engine

open Std

--------------------------------------------------------------------------------
-- 1) A global counter for unique IDs, so each `Value` can be given an integer id
--------------------------------------------------------------------------------

initialize nextValueId : IO.Ref Nat ← IO.mkRef 0

--------------------------------------------------------------------------------
-- 2) Definition of the `Value` record
--------------------------------------------------------------------------------

/--
A `Value` node that stores:
- `id`: a unique integer ID
- `dataRef`: IO.Ref Float
- `gradRef`: IO.Ref Float
- `prev`: the list of parent nodes
- `op`: a string for debugging (e.g. "+", "*", etc.)
- `backwardRef`: holds an IO action that updates the parents' gradients
-/
structure Value where
  id          : Nat
  dataRef     : IO.Ref Float
  gradRef     : IO.Ref Float
  prev        : List Value
  op          : String
  backwardRef : IO.Ref (IO Unit)
  -- no `deriving Inhabited` because references are not automatically inhabited.

--------------------------------------------------------------------------------
-- 3) Instances so `Value` can be hashed and compared (needed for HashSet)
--------------------------------------------------------------------------------

instance : BEq Value where
  -- Two Values are considered equal iff their IDs are equal
  beq v1 v2 := v1.id == v2.id

instance : Hashable Value where
  -- We just hash the `id` field
  hash v := hash v.id

--------------------------------------------------------------------------------
-- 4) Constructors and helper functions
--------------------------------------------------------------------------------

/-- Make a fresh `Value` with a unique ID. -/
def mkValue (data : Float) (prev : List Value := []) (op : String := "") : IO Value := do
  let dataRef ← IO.mkRef data
  let gradRef ← IO.mkRef 0.0
  let backwardRef ← IO.mkRef (pure ())
  let currentId ← nextValueId.get
  nextValueId.set (currentId + 1)
  pure {
    id          := currentId
    dataRef     := dataRef
    gradRef     := gradRef
    prev        := prev
    op          := op
    backwardRef := backwardRef
  }

def setBackwardFn (v : Value) (fn : IO Unit) : IO Unit :=
  v.backwardRef.set fn

def getData (v : Value) : IO Float :=
  v.dataRef.get

def getGrad (v : Value) : IO Float :=
  v.gradRef.get

def setGrad (v : Value) (g : Float) : IO Unit :=
  v.gradRef.set g

--------------------------------------------------------------------------------
-- 5) The core ops (add, mul, pow, relu)
--------------------------------------------------------------------------------

def add (a b : Value) : IO Value := do
  let aVal ← a.dataRef.get
  let bVal ← b.dataRef.get
  let out ← mkValue (aVal + bVal) [a, b] "+"
  setBackwardFn out do
    let og ← out.gradRef.get
    let aGrad ← a.gradRef.get
    let bGrad ← b.gradRef.get
    a.gradRef.set (aGrad + og)
    b.gradRef.set (bGrad + og)
  pure out

def mul (a b : Value) : IO Value := do
  let aVal ← a.dataRef.get
  let bVal ← b.dataRef.get
  let out ← mkValue (aVal * bVal) [a, b] "*"
  setBackwardFn out do
    let og ← out.gradRef.get
    let aGrad ← a.gradRef.get
    let bGrad ← b.gradRef.get
    a.gradRef.set (aGrad + bVal * og)
    b.gradRef.set (bGrad + aVal * og)
  pure out

def pow (a : Value) (ex : Float) : IO Value := do
  let aVal ← a.dataRef.get
  let out ← mkValue (Float.pow aVal ex) [a] s!"**{ex}"
  setBackwardFn out do
    let og ← out.gradRef.get
    let aGrad ← a.gradRef.get
    let d := ex * Float.pow aVal (ex - 1.0) * og
    a.gradRef.set (aGrad + d)
  pure out

def relu (a : Value) : IO Value := do
  let aVal ← a.dataRef.get
  let out ← mkValue (if aVal < 0 then 0 else aVal) [a] "ReLU"
  setBackwardFn out do
    let og ← out.gradRef.get
    let aGrad ← a.gradRef.get
    let outVal ← out.dataRef.get
    let d := if outVal > 0 then og else 0
    a.gradRef.set (aGrad + d)
  pure out

--------------------------------------------------------------------------------
-- 6) Topological sort & backward pass
--------------------------------------------------------------------------------

/--
A **functional** DFS that returns an updated `(visited, acc)`.
- `visited` is a `HashSet Value`
- `acc` is an `Array Value` for the topological ordering
-/
partial def buildTopo (v : Value) (visited : HashSet Value) (acc : Array Value)
  : (HashSet Value) × (Array Value) :=
  if visited.contains v then
    (visited, acc)
  else
    let visited := visited.insert v
    let (visited, acc) := v.prev.foldl
      (fun (s : HashSet Value × Array Value) p =>
        let (vis, arr) := s
        buildTopo p vis arr
      )
      (visited, acc)
    -- now push `v` after all its parents
    (visited, acc.push v)

/--
Set `v`'s grad to 1, then walk in reverse topological order calling each node’s
backward function.
-/
def backward (v : Value) : IO Unit := do
  let (visited, topo) := buildTopo v {} #[]
  -- set out node's grad = 1
  v.gradRef.set 1
  for node in topo.reverse do
    let fn ← node.backwardRef.get
    fn

--------------------------------------------------------------------------------
-- 7) Operator overloads
--------------------------------------------------------------------------------

namespace Ops

instance : HAdd Value Value (IO Value) where
  hAdd := add

instance : HMul Value Value (IO Value) where
  hMul := mul

def neg (a : Value) : IO Value := do
  let minus1 ← mkValue (-1)
  mul a minus1

def sub (a b : Value) : IO Value := do
  let nb ← neg b
  add a nb

def div (a b : Value) : IO Value := do
  let bPow ← pow b (-1.0)
  mul a bPow

def fpow (a : Value) (ex : Float) : IO Value :=
  pow a ex

end Ops

end Leangrad.Engine
