# Activation Shape Gallery

Read these first:

- `activation_output_shapes.png`: direct comparison of the most interesting discovered shapes
- `activation_frankenstein_family.png`: the parametric family we are now testing

Skip if you want:

- `activation_gate_shapes.png`: just the gate terms by themselves

What to look for:

- good candidates stay exactly zero on the negative side
- good candidates grow faster than linear on the positive side
- good gates mostly rescale the positive branch instead of inventing a whole new smooth path
