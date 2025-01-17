#tequila importations
import tequila as tq
# import primitives
from tequila.circuit.gates import H, X,Y,Z
# import controlled 
from tequila.circuit.gates import CX,CY,CZ
# import rotations 
from tequila.circuit.gates import Rx, Ry, Rz
from tequila.hamiltonian.qubit_hamiltonian import QubitHamiltonian

#extra importations
import numpy as np
import matplotlib.pyplot as plt
from cudaq import spin
import cudaq



""" 
in this file test the expectation value computation using cudaq 
with primitive gates only 

"""




""" 
single qubit ex. vals

test gates primitive not controlled gates 
    - basic case is <0|Z|0> = 1 and with the state 1 then -1 
    - test for states 0 & 1 and paulis 

"""

print(" \n \n @@@@@ experiment with primitives without controlls ")

# define backends 
backend_cudaq = 'cudaq'
backend_qulacs = 'qulacs'

state_0 = X(0) + X(0) # prepare state 0 
state_1 = X(target=0) # prepare state 1 

# hamiltonians 
x = QubitHamiltonian.from_string("X(0)")
y = QubitHamiltonian.from_string("Y(0)")
z = QubitHamiltonian.from_string("Z(0)")
# h = QubitHamiltonian.from_string("H(0)")


# list with combinations 
states = [state_0, state_1]
hamiltonians = [x,y,z]

# hamiltonians = [y]

count = 0

for state in states:
    for hermitian_operator in hamiltonians:    
        
        if state == state_0:
            pr = "0"
        elif state == state_1:
            pr = "1"

        matrix = None

        if hermitian_operator == x: 
            matrix = 'X'
        elif hermitian_operator == y: 
            matrix = 'Y'
        elif hermitian_operator == z: 
            matrix = 'Z'
        # elif hermitian_operator == h: 
        #    matrix = 'H'



        # print(f" \n \n experiment for <{pr}| {hermitian_operator.paulistrings} | {pr}> ")
        #Hamiltonian definition

        # convert states to objectives 
        br_exp_value_real, br_exp_value_im = tq.braket(ket=state, operator=hermitian_operator)
        br_exp_value_tmp = br_exp_value_real + 1.0j*br_exp_value_im

        # simulate with both backends 
        ex_val_cuda = tq.simulate(objective=br_exp_value_tmp, backend=backend_cudaq)
        ex_val_qulacs = tq.simulate(objective=br_exp_value_tmp, backend=backend_qulacs)


        # assert ex_val_cuda == ex_val_qulacs

        print(f" \n \n experiment for <{pr}| {matrix} | {pr}> ")
        print(f"ex val qulacs: {ex_val_qulacs} cudaq: {ex_val_cuda}")

        count += 1

print(f"{count} tests gelaufen \n ")








""" test on multiple qubits """

























""" compute with all backends 0y0 """

SUPPORTED_BACKENDS = ["qulacs", "cirq", "symbolic", "cudaq"]

for backend in SUPPORTED_BACKENDS:
    br_real, br_im = tq.braket(ket=state_0, operator=y)
    br_temp = br_real + 1.0j*br_im
    exval = tq.simulate(objective=br_temp, backend=backend)
    # print(f"exval of 0Y0 with {backend} = {exval}")







""" try compute 0y0 alone with cudaq 

--> produces also 10^-15 but not 0, so isnt the fault of the sim_cudaq.py 
moreover after internet research -> seems like each backend computes differently and 
in cudaq's case it can be that bc Y gate has imaginary parts that the multiplication 
yields a very small number but not exactly zero 

    -> ask if i should consider such numbers zero and just not handle such 
    expectation values differently 

"""


import cudaq
from cudaq import spin

op_y = spin.y(0)

@cudaq.kernel
def ker():
    # v = cudaq.qvector(1)
    v  =cudaq.qubit()
    x(v)
    x(v)

energy = cudaq.observe(ker,op_y).expectation()
print(f"Energy is {energy}")





