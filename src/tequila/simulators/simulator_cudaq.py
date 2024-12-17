import cudaq
from cudaq import spin

import qulacs
import numbers, numpy
import warnings

from tequila import TequilaException, TequilaWarning
from tequila.utils.bitstrings import BitNumbering, BitString, BitStringLSB
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.simulators.simulator_base import BackendCircuit, BackendExpectationValue, QCircuit, change_basis
from tequila.utils.keymap import KeyMapRegisterToSubregister


"""
Developer Note:
    Qulacs uses different Rotational Gate conventions: Rx(angle) = exp(i angle/2 X) instead of exp(-i angle/2 X)
    And the same for MultiPauli rotational gates
    The angles are scaled with -1.0 to keep things consistent with the rest of tequila
"""

class TequilaCudaqException(TequilaException):
    def __str__(self):
        return "Error in cudaq (cuda-quantum) backend:" + self.message

class BackendCircuitCudaq(BackendCircuit):
    """
    Class representing circuits compiled to cudaq (cuda-quantum of NVIDIA).
    See BackendCircuit for documentation of features and methods inherited therefrom

    Attributes
    ----------
    counter:
        counts how many distinct sympy.Symbol objects are employed in the circuit.
    has_noise:
        whether or not the circuit is noisy. needed by the expectationvalue to do sampling properly.
    noise_lookup: dict:
        dict mapping strings to lists of constructors for cirq noise channel objects.
    op_lookup: dict:
        dictionary mapping strings (tequila gate names) to cirq.ops objects.
    variables: list:
        a list of the qulacs variables of the circuit.

    Methods
    -------
    add_noise_to_circuit:
        apply a tequila NoiseModel to a qulacs circuit, by translating the NoiseModel's instructions into noise gates.
    """

    compiler_arguments = {
        "trotterized": False,
        "swap": True,
        "multitarget": False,
        "controlled_rotation": False, # needed for gates depending on variables
        "generalized_rotation": False,
        "exponential_pauli": False,
        "controlled_exponential_pauli": True,
        "phase": True,
        "power": False,
        "hadamard_power": False,
        "controlled_power": False,
        "controlled_phase": False,
        "toffoli": False,
        "phase_to_z": False,
        "cc_max": False
    }
    # try MSB, if doesnt work try LSB 
    numbering = BitNumbering.LSB

    def __init__(self, abstract_circuit, noise=None, *args, **kwargs):
        """

        Parameters
        ----------
        abstract_circuit: QCircuit:
            the circuit to compile to qulacs
        noise: optional:
            noise to apply to the circuit.
        args
        kwargs
        """


        # rotations are instantiated like this: 
        # doesnt work 
        self.op_lookup = {
            'I': None,
            'X': 1,
            'Y': 2,
            'Z': 3,
            'H': 4,
            'Rx': 5,
            'Ry': 6,
            'Rz': 7,
            'SWAP': 8,
            'Measure': 9,
            'Exp-Pauli': None
        }

        # instantiate a cudaq circuit as a list
        self.circuit = self.initialize_circuit()

        self.measurements = None
        self.variables = []
        super().__init__(abstract_circuit=abstract_circuit, noise=noise, *args, **kwargs)
        self.has_noise=False


        # noise part is still not implemented for cudaq 
        if noise is not None:

            warnings.warn("Warning: noise in qulacs module will be dropped. Currently only works for qulacs version 0.5 or lower", TequilaWarning)

            self.has_noise=True
            self.noise_lookup = {
                'bit flip': [qulacs.gate.BitFlipNoise],
                'phase flip': [lambda target, prob: qulacs.gate.Probabilistic([prob],[qulacs.gate.Z(target)])],
                'phase damp': [lambda target, prob: qulacs.gate.DephasingNoise(target,(1/2)*(1-numpy.sqrt(1-prob)))],
                'amplitude damp': [qulacs.gate.AmplitudeDampingNoise],
                'phase-amplitude damp': [qulacs.gate.AmplitudeDampingNoise,
                                         lambda target, prob: qulacs.gate.DephasingNoise(target,(1/2)*(1-numpy.sqrt(1-prob)))
                                         ],
                'depolarizing': [lambda target,prob: qulacs.gate.DepolarizingNoise(target,3*prob/4)]
            }

            self.circuit=self.add_noise_to_circuit(noise)


    def do_simulate(self, variables, initial_state, *args, **kwargs):
        """
        Helper function to perform simulation.

        Parameters
        ----------
        variables: dict:
            variables to supply to the circuit.
        initial_state:
            information indicating the initial state on which the circuit should act.
        args
        kwargs

        Returns
        -------
        QubitWaveFunction:
            QubitWaveFunction representing result of the simulation.
        """
        pass

    def convert_measurements(self, backend_result, target_qubits=None) -> QubitWaveFunction:
        """
        Transform backend evaluation results into QubitWaveFunction
        Parameters
        ----------
        backend_result:
            the return value of backend simulation.

        Returns
        -------
        QubitWaveFunction
            results transformed to tequila native QubitWaveFunction
        """

        result = QubitWaveFunction()
        # todo there are faster ways


        for k in backend_result:
            converted_key = BitString.from_binary(BitStringLSB.from_int(integer=k, nbits=self.n_qubits).binary)
            if converted_key in result._state:
                result._state[converted_key] += 1
            else:
                result._state[converted_key] = 1

        if target_qubits is not None:
            mapped_target = [self.qubit_map[q].number for q in target_qubits]
            mapped_full = [self.qubit_map[q].number for q in self.abstract_qubits]
            keymap = KeyMapRegisterToSubregister(subregister=mapped_target, register=mapped_full)
            result = result.apply_keymap(keymap=keymap)

        return result

    def do_sample(self, samples, circuit, noise_model=None, initial_state=0, *args, **kwargs) -> QubitWaveFunction:
        """
        Helper function for performing sampling.

        Parameters
        ----------
        samples: int:
            the number of samples to be taken.
        circuit:
            the circuit to sample from.
        noise_model: optional:
            noise model to be applied to the circuit.
        initial_state:
            sampling supports initial states for qulacs. Indicates the initial state to which circuit is applied.
        args
        kwargs

        Returns
        -------
        QubitWaveFunction:
            the results of sampling, as a Qubit Wave Function.
        """
        state = self.initialize_state(self.n_qubits)
        lsb = BitStringLSB.from_int(initial_state, nbits=self.n_qubits)
        state.set_computational_basis(BitString.from_binary(lsb.binary).integer)
        circuit.update_quantum_state(state)
        sampled = state.sampling(samples)
        return self.convert_measurements(backend_result=sampled, target_qubits=self.measurements)


    def initialize_circuit(self, *args, **kwargs):
        """
        return an empty circuit.
        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        an empty list, since the circuits in cudaq has to be created 
        based on a list of gates within a kernel 
        """
        return []


    def add_parametrized_gate(self, gate, circuit, variables, *args, **kwargs):
        """
        add a parametrized gate.
        Parameters
        ----------
        gate: QGateImpl:
            the gate to add to the circuit.
        circuit:
            the circuit to which the gate is to be added
        variables:
            dict that tells values of variables; needed IFF the gate is an ExpPauli gate.
        args
        kwargs

        Returns
        -------
        None
        """

        # has to be implemented individually for cudaq
        pass

    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        """
        add an unparametrized gate to the circuit.
        Parameters
        ----------
        gate: QGateImpl:
            the gate to be added to the circuit.
        circuit:
            the circuit, to which a gate is to be added.
        args
        kwargs

        Returns
        -------
        None
        """
        gate_encoding = self.op_lookup[gate.name]


        # target qubits as a list 
        target_qubits = [self.qubit(t) for t in gate.target]

        print("gate enc ", gate_encoding, " gate name", gate.name)
        print("target qub from add basic gate ", target_qubits)
        
        conrtol_qubits = []

        if gate.is_controlled():
            for control in gate.control:
                conrtol_qubits.append(self.qubit(control))


        # print("target qubits ", target_qubits)
        # print("control qubits ", conrtol_qubits)

        # print(f"op is {gate.name} = {gate_encoding} with target {target_qubits} and conrtol {conrtol_qubits}")

        # a tupel: (gate_code, target, control)

        gate_to_apply = (gate_encoding, target_qubits, conrtol_qubits)
        # print("gate to apply", gate_to_apply)
        # print("intermediate circuit ", circuit)

        circuit.append(gate_to_apply)



    def add_measurement(self, circuit, target_qubits, *args, **kwargs):
        """
        Add a measurement operation to a circuit.
        Parameters
        ----------
        circuit:
            a circuit, to which the measurement is to be added.
        target_qubits: List[int]
            abstract target qubits
        args
        kwargs

        Returns
        -------
        None
        """
        pass


class BackendExpectationValueCudaq(BackendExpectationValue):
    """
    Class representing Expectation Values compiled for Cudaq.

    Overrides some methods of BackendExpectationValue, which should be seen for details.
    """
    use_mapping = True
    BackendCircuitType = BackendCircuitCudaq

    def simulate(self, variables, *args, **kwargs) -> numpy.array:
        """
        Perform simulation of this expectationvalue.
        Parameters
        ----------
        variables:
            variables, to be supplied to the underlying circuit.
        args
        kwargs

        Returns
        -------
        numpy.array:
            the result of simulation as an array.
        """
        # fast return if possible
        if self.H is None:
            return numpy.asarray([0.0])
        elif len(self.H) == 0:
            return numpy.asarray([0.0])
        elif isinstance(self.H, numbers.Number):
            return numpy.asarray[self.H]

        # a tupel: (gate_code, target, control)
        # gate_to_apply = (gate_encoding, target_qubits, conrtol_qubits)

        print("self cir ", self.U.circuit)

        for g, t, c in self.U.circuit:
            print(g,t,c)

        print("amt of qbits ", self.U.n_qubits)

        # prepare parameters for usage in kernal since "self. " access doesnt work within kernels  
        number_of_qubits = self.n_qubits

        # decompose gate_encodings, targets and controls from circuit
        gate_encodings = []
        target_qubits = []
        control_qubits = []

        for tuple_in_circuit in self.U.circuit:
            gate_encodings.append(tuple_in_circuit[0])
            target_qubits.append(tuple_in_circuit[1])
            control_qubits.append(tuple_in_circuit[2])


        # ensure there is only one target per gate encoding 
        for target in target_qubits:
            if len(target) != 1:
                TequilaCudaqException(" each gate should have exactly one target ")

        # convert elements of target qubits into the indices, to which they should apply, and not a list of lists
        target_qubits = [x[0] if len(x) == 1 else x for x in target_qubits]

        # constant - indicates kernel if circuit has conrtol qubits 
        is_controlled_parent_scope = 0

        print("g " , gate_encodings)
        print("tr ", target_qubits)
        print("cq ", control_qubits)

        """ as for now without control qubits """
        @cudaq.kernel
        def state_modifier():

            is_controlled = is_controlled_parent_scope

            # create an empty state with given number of qubits
            state = cudaq.qvector(number_of_qubits)

            # if no controlled qubits - apply basic gates with their targets 
            if is_controlled == 0:
                # ensure each gate has exactly one control 
                if len(gate_encodings) == len(target_qubits):
                    # iterate over both lists with their indices (since zip is forbidden in cudaq-kernels)
                    for index in range(len(gate_encodings)):
                        if gate_encodings[index] == 1:
                            x(state[target_qubits[index]])
                        elif gate_encodings[index] == 2:
                            y(state[target_qubits[index]])
                        elif gate_encodings[index] == 3:
                            z(state[target_qubits[index]])

            # if there are some controlls             
            else:
                # TODO implement behaviour when controlled - trickier than initially assumed 
                pass


        # array containing results of simulation 
        resulting_expectation_values = []

        # go over all given hamiltonians 
        for hamiltonian in self.H:
            # compute expectation value between hamiltonian and state 
            expectation_value = cudaq.observe(state_modifier, hamiltonian).expectation()
            print("one before")
            print(expectation_value)
            resulting_expectation_values.append(expectation_value)

        print("res exvals ",resulting_expectation_values)

        return numpy.asarray(resulting_expectation_values)
    


    def initialize_hamiltonian(self, hamiltonians):
        """
        Convert reduced hamiltonians to native Qulacs types for efficient expectation value evaluation.
        Parameters
        ----------
        hamiltonians:
            an interable set of hamiltonian objects.

        Returns
        -------
        list:
            initialized hamiltonian objects.

        """


        """ 
        # get information for spins 
        print(" \n \n")
        for h in hamiltonians:
            print(h, h.paulistrings)
            print(type(h), type(h.paulistrings))
            for pauli in h.paulistrings:
                print(pauli, type(pauli))
                for a, b in pauli.items():
                    print(f"{a} bit {b} gate {pauli._coeff} coeff")
        """ 

        list_of_initialized_hamiltonians = []
        hamiltonian_as_spin = 0
        # assemble hamiltonian with cudaq
        for h in hamiltonians:
            for paulistring in h.paulistrings:
                # get qubit on which the gate operate, gate and coefficient
                for qubit, gate in paulistring.items():
                    if gate == 'X':
                        hamiltonian_as_spin += paulistring._coeff * spin.x(qubit)
                    elif gate == 'Y':
                        hamiltonian_as_spin += paulistring._coeff * spin.y(qubit)
                    elif gate == 'Z':
                        hamiltonian_as_spin += paulistring._coeff * spin.z(qubit)
            list_of_initialized_hamiltonians.append(hamiltonian_as_spin)
        
        # show hamils 
        for index, h in enumerate(list_of_initialized_hamiltonians):
            print(f"hamil number {index + 1} is {h}")

        return list_of_initialized_hamiltonians

    def sample(self, variables, samples, *args, **kwargs) -> numpy.array:
        """
        Sample this Expectation Value.
        Parameters
        ----------
        variables:
            variables, to supply to the underlying circuit.
        samples: int:
            the number of samples to take.
        args
        kwargs

        Returns
        -------
        numpy.ndarray:
            the result of sampling as a number.
        """
        self.update_variables(variables)
        state = self.U.initialize_state(self.n_qubits)
        self.U.circuit.update_quantum_state(state)
        result = []
        for H in self._reduced_hamiltonians: # those are the hamiltonians which where non-used qubits are already traced out
            E = 0.0
            if H.is_all_z() and not self.U.has_noise:
                E = super().sample(samples=samples, variables=variables, *args, **kwargs)
            else:
                for ps in H.paulistrings:
                    # change basis, measurement is destructive so the state will be copied
                    # to avoid recomputation (except when noise was required)
                    bc = QCircuit()
                    for idx, p in ps.items():
                        bc += change_basis(target=idx, axis=p)
                    qbc = self.U.create_circuit(abstract_circuit=bc, variables=None)
                    Esamples = []
                    for sample in range(samples):
                        if self.U.has_noise and sample>0:
                            state = self.U.initialize_state(self.n_qubits)
                            self.U.circuit.update_quantum_state(state)
                            state_tmp = state
                        else:
                            state_tmp = state.copy()
                        if len(bc.gates) > 0:  # otherwise there is no basis change (empty qulacs circuit does not work out)
                            qbc.update_quantum_state(state_tmp)
                        ps_measure = 1.0
                        for idx in ps.keys():
                            assert idx in self.U.abstract_qubits # assert that the hamiltonian was really reduced
                            M = qulacs.gate.Measurement(self.U.qubit(idx), self.U.qubit(idx))
                            M.update_quantum_state(state_tmp)
                            measured = state_tmp.get_classical_value(self.U.qubit(idx))
                            ps_measure *= (-2.0 * measured + 1.0)  # 0 becomes 1 and 1 becomes -1
                        Esamples.append(ps_measure)
                    E += ps.coeff * sum(Esamples) / len(Esamples)
            result.append(E)
        return numpy.asarray(result)
