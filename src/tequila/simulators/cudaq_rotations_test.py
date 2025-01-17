#tequila importations
import tequila as tq
# import primitives
from tequila.circuit.gates import H, X,Y,Z
# import controlled 
from tequila.circuit.gates import CX,CY,CZ
# import rotations 
from tequila.circuit.gates import Rx, Ry, Rz
from numpy import pi

from tequila.hamiltonian.qubit_hamiltonian import QubitHamiltonian

#extra importations
import numpy as np
import matplotlib.pyplot as plt
from cudaq import spin
import cudaq
import math

""" modify state with rotations and compute exp. value thereafter """

# define backends 
backend_cudaq = 'cudaq'
backend_qulacs = 'qulacs'

state_minus_i = Rx(pi/2, 0) # prepare superposition state 1/sqrt(2) * (|0> - i* |1>)
state_minus_minus_i = X(0) + Rx(pi/2, 0) # prepare superposition state 1/sqrt(2) * (|1> - i* |0>)


# hamiltonians 
x = QubitHamiltonian.from_string("X(0)")
y = QubitHamiltonian.from_string("Y(0)")
z = QubitHamiltonian.from_string("Z(0)")
h = QubitHamiltonian.from_string("X(0)Z(0)X(0)")

# list with combinations 
states = [state_minus_i, state_minus_minus_i]
hamiltonians = [x,y,z]

# hamiltonians = [y]


print(f"pi {math.pi} {type(math.pi)}")



for state in states:
    for hermitian_operator in hamiltonians:    
        
        if state == state_minus_i:
            pr = "-i"
        elif state == state_minus_minus_i:
            pr = "-(-i)"

        hamil_op = None

        if hermitian_operator == x: 
            hamil_op = 'X'
        elif hermitian_operator == y: 
            hamil_op = 'Y'
        elif hermitian_operator == z: 
            hamil_op = 'Z'
        # elif hermitian_operator == h: 
            # hamil_op = 'H'


        # print(f" \n \n experiment for <{pr}| {hermitian_operator.paulistrings} | {pr}> ")
        #Hamiltonian definition

        # convert states to objectives 
        br_exp_value_real, br_exp_value_im = tq.braket(ket=state, operator=hermitian_operator)
        br_exp_value_tmp = br_exp_value_real + 1.0j*br_exp_value_im

        # simulate with both backends 
        print("before sim p")

        ex_val_cuda = tq.simulate(objective=br_exp_value_tmp, backend=backend_cudaq)

        # ex_val_cuda = None

        ex_val_qulacs = tq.simulate(objective=br_exp_value_tmp, backend=backend_qulacs)


        # assert ex_val_cuda == ex_val_qulacs

        print(f"\n experiment for <{pr} | {hamil_op} | {pr}> ")
        print(f"ex val qulacs: {ex_val_qulacs} cudaq: {ex_val_cuda}")
