from z3 import *

# Define the universe of elements
elements = ['a', 'b', 'c', 'd', 'e']

# Create a Z3 solver
solver = Solver()
elem_map = {elem: Int(elem) for elem in elements}

# Define Z3 sets
set1 = SetSort(IntSort())
set2 = SetSort(IntSort())

# Define the sets by adding elements
set1_expr = SetAdd(SetAdd(EmptySet(IntSort()), elem_map['a']), elem_map['b'])
set2_expr = SetAdd(SetAdd(SetAdd(EmptySet(IntSort()), elem_map['c']), elem_map['d']), elem_map['e'])

# Add the sets to the solver
solver.add(set1 == set1_expr)
solver.add(set2 == set2_expr)

# Ensure that the sets have no common elements
solver.add(Distinct(set1, set2))

# Check if the constraints are satisfiable
if solver.check() == sat:
    model = solver.model()
    print("Model found:")
    print("Set1:", [elem for elem in elements if model.evaluate(IsMember(elem_map[elem], set1))])
    print("Set2:", [elem for elem in elements if model.evaluate(IsMember(elem_map[elem], set2))])
else:
    print("No solution found.")
