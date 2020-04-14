from tequila.circuit import gates
from tequila.objective import ExpectationValue
from tequila.hamiltonian import paulis
from tequila.circuit.noise import BitFlip
import numpy
import pytest
import tequila as tq


@pytest.mark.parametrize("simulator", [['qiskit','cirq'][numpy.random.randint(0,2,1)[0]]])
@pytest.mark.parametrize("p", numpy.random.uniform(0.1,.4,1))
@pytest.mark.parametrize('method',[['NELDER-MEAD', 'COBYLA'][numpy.random.randint(0,2)]])
def test_bit_flip_scipy_gradient_free(simulator, p,method):

    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.Rx(target=qubit,angle=tq.Variable('a'))
    O = ExpectationValue(U=U, H=H)
    NM=BitFlip(p,1)
    result = tq.optimizer_scipy.minimize(objective=O,samples=10000,backend=simulator, method=method,noise=NM, tol=1.e-4,silent=False)
    assert(numpy.isclose(result.energy, p, atol=3.e-2))

@pytest.mark.parametrize("simulator", [['qiskit','cirq','pyquil'][numpy.random.randint(0,3,1)[0]]])
@pytest.mark.parametrize("p", numpy.random.uniform(0.1,.4,1))
@pytest.mark.parametrize('method',[tq.optimizer_scipy.OptimizerSciPy.gradient_based_methods[numpy.random.randint(0,4,1)[0]]])
def test_bit_flip_scipy_gradient(simulator, p,method):

    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.Rx(target=qubit,angle=tq.Variable('a'))
    O = ExpectationValue(U=U, H=H)
    NM=BitFlip(p,1)
    result = tq.optimizer_scipy.minimize(objective=O,samples=10000,backend=simulator, method=method,noise=NM, tol=1.e-4,silent=False)
    assert(numpy.isclose(result.energy, p, atol=1.e-2))

@pytest.mark.parametrize("simulator", ['qiskit'])
@pytest.mark.parametrize("p", numpy.random.uniform(0.1,.4,1))
@pytest.mark.parametrize('method',[["TRUST-KRYLOV", "NEWTON-CG", "TRUST-NCG", "TRUST-CONSTR"][numpy.random.randint(0,4,1)[0]]])
def test_bit_flip_scipy_hessian(simulator, p,method):

    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.Rx(target=qubit,angle=tq.Variable('a'))
    O = ExpectationValue(U=U, H=H)
    NM=BitFlip(p,1)
    result = tq.optimizer_scipy.minimize(objective=O,samples=10000,backend=simulator, method=method,noise=NM, tol=1.e-4,silent=False)
    assert(numpy.isclose(result.energy, p, atol=3.e-2))

@pytest.mark.parametrize("simulator", ['qiskit'])
@pytest.mark.parametrize("p", numpy.random.uniform(0.1,.4,1))
def test_bit_flip_phoenics(simulator, p):

    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.Rx(target=qubit,angle=tq.Variable('a'))
    O = ExpectationValue(U=U, H=H)
    NM=BitFlip(p,1)
    result = tq.optimizer_phoenics.minimize(objective=O,maxiter=3,samples=1000,backend=simulator,noise=NM)
    assert(numpy.isclose(result.energy, p, atol=3.e-2))


@pytest.mark.parametrize("simulator", ['cirq'])
@pytest.mark.parametrize("p", numpy.random.uniform(0.1,.4,1))
@pytest.mark.parametrize('method',['lbfgs','DIRECT','CMA'])
def test_bit_flip_gpyopt(simulator, p,method):

    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.Rx(target=qubit,angle=tq.Variable('a'))
    O = ExpectationValue(U=U, H=H)
    NM=BitFlip(p,1)
    result = tq.optimizer_gpyopt.minimize(objective=O,maxiter=10,samples=10000,backend=simulator, method=method,noise=NM)
    assert(numpy.isclose(result.energy, p, atol=3.e-2))