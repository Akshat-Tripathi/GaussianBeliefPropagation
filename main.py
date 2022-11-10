from graph import FactorNode, VariableNode
import numpy as np

v1 = VariableNode()
v2 = VariableNode()

f1 = FactorNode([v1], np.array([5]), np.array([[1]]))
f2 = FactorNode([v1, v2], np.array([1, 2]), np.array([[5, 4], [4, 6]]))
# f2.set_eta_lambda(np.array([1, 2]), np.array([[5, -4], [-4, 6]]))

f1.send_messages()
v1.send_messages()
f2.send_messages()
v2.send_messages()
f2.send_messages()
v1.send_messages()

# print(v2.messages)

v1.belief_update()
v2.belief_update()

# print(v1.get_dist())
print(v1.eta, v1.Lambda)
print(v2.eta, v2.Lambda)