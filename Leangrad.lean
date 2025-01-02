-- This module serves as the root of the `Leangrad` library.
-- Import modules here that should be built as part of the library.
-- import Leangrad.Basic

import Leangrad.Engine
import Leangrad.NN

-- Re-export symbols if desired
namespace Leangrad

abbrev Value := Engine.Value
abbrev Neuron := NN.Neuron
abbrev Layer := NN.Layer
abbrev MLP := NN.MLP

-- You can also add convenience functions, etc.

end Leangrad
