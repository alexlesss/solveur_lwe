#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#        x, y, z binary

from gurobipy import * 

# create a model 
m = Model("mip_simple_eg")

# set parameters
m.setParam('presolve', 0)
m.setParam('heuristics', 0)

# create variables
x = m.addVar(vtype = GRB.BINARY, name='x')
y = m.addVar(vtype = GRB.BINARY, name='y')
z = m.addVar(vtype = GRB.BINARY, name='z')

# set objective 
m.setObjective(x + y + 2*z, GRB.MAXIMIZE)

# add constraint x + 2 y + 3 z <= 4
m.addConstr(x + 2*y + 3*z <= 4, 'c1')

# add constraint x +   y       >= 1
m.addConstr(x + y >= 1, 'c2')

# optimize model 
m.optimize()

print('obj: %g' % m.objVal)

for v in m.getVars():
    print('%s = %g' % (v.varName, v.x))

print('# branch-and-cut node', m.NodeCount)