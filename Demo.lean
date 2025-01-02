-- import NN

-- open NN Engine

-- def main : IO Unit :=
--   let model := MLP.init 3 [4, 2]
--   let input := [Value.init 1.0, Value.init 2.0, Value.init 3.0]
--   let output := MLP.call model input
--   IO.println output
