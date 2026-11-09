"""Variation of MaxCut.py from https://github.com/Timo59/VQAPySim"""

import numpy as np
import networkx as nx
from PauliString import PauliString
from PureState import PureState
from problems.Problem import Problem
from typing import Any, Union


class MaxCut(nx.Graph, Problem):
    """
    Class that defines the MaxCut optimization problem
    Attributes:
    Methods:
    """
    def __init__(self, Graph: nx.Graph, fixed_bits: dict = {}, opt: bool = False):
        """
        Takes in an adjacency matrix and initializes the Graph class

        Parameters
        ----------
        adjacency_matrix: np.ndarray
            Adjacency matrix as a numpy array
        fixed_bits: dict
            fixed bits and their values as dictionary
        opt: bool
            If False (default) the value is minimized, otherwise it is maximized
        """
        nx.Graph.__init__(self)
        self.add_edges_from(Graph.edges(data=True))
        self.add_nodes_from(Graph.nodes(data=True))
        self._num_var: int = len(self.nodes) - len(fixed_bits)
        self._fixed_bits: dict = fixed_bits
        self._free_qubits = [i for i in self.nodes if i not in self._fixed_bits]
        self._opt: int = 2 * (1 * opt) - 1
        self._objective_vector: np.ndarray = np.array([self.objective_value(z) for z in range(2 ** self._num_var)])
        self._num_pqgs: int = 2
        self._hamiltonian: list[PauliString] = []
        total_weight = 0
        for i, j in self.edges:
            if i in self._fixed_bits or j in self._fixed_bits:
                #One or both qubits are fixed
                if i in self._fixed_bits and j in self._fixed_bits:
                    #Both qubits are fixed; update the constant term
                    if self._fixed_bits[i] != self._fixed_bits[j]:
                        total_weight += self.edges[i, j]['weight']
                else:
                    #Exactly one qubit is fixed; create and append the correct Pauli string
                    if i in self._fixed_bits:
                        fixed_value_contribution = (-1) ** self._fixed_bits[i]/2.0
                        free_qubit = self._free_qubits.index(j)
                    else:
                        fixed_value_contribution = (-1) ** self._fixed_bits[j]/2.0
                        free_qubit = self._free_qubits.index(i)
            
                    #Create the Pauli string with the correct operator
                    pauli_list = ['id'] * self._num_var
                    pauli_list[free_qubit] = 'z'
                    prefactor=-self._opt * self.edges[i, j]['weight'] * fixed_value_contribution
                    self._hamiltonian.append(PauliString(pauli_list, float(prefactor)))
                    total_weight += self.edges[i, j]['weight']
            else:
                #Neither qubit is fixed; append the standard interaction term
                free_qubit_i = self._free_qubits.index(i)
                free_qubit_j = self._free_qubits.index(j)
                pauli_list = ['id'] * self._num_var
                pauli_list[free_qubit_i] = 'z'
                pauli_list[free_qubit_j] = 'z'
                prefactor=-self._opt * self.edges[i, j]['weight'] / 2
                self._hamiltonian.append(PauliString(pauli_list, float(prefactor)))
                total_weight += self.edges[i, j]['weight']
        #Add the constant term to the Hamiltonian
        self._hamiltonian.append(PauliString(['id'] * self._num_var, np.cdouble(self._opt * total_weight / 2.0)))
        
    @property
    def num_var(self):
        return self._num_var

    @num_var.setter
    def num_var(self, arg: Any):
        raise ValueError('Number of variables cannot be changed externally')

    @property
    def fixed_bits(self):
        return self._fixed_bits

    @fixed_bits.setter
    def fixed_bits(self, arg: Any):
        raise ValueError('Fixed bits cannot be changed externally')
    
    @property
    def free_qubits(self):
        return self._free_qubits

    @free_qubits.setter
    def free_qubits(self, arg: Any):
        raise ValueError('Free bits cannot be changed externally')    
        
    @property
    def opt(self):
        return self._opt

    @opt.setter
    def opt(self, new_opt: bool):
        self._opt = new_opt
        for pauli_string in self._hamiltonian:
            pauli_string.prefactor *= 1 - 2 * (1 * self._opt ^ new_opt)

    @property
    def objective_vector(self) -> np.ndarray:
        return self._objective_vector

    @objective_vector.setter
    def objective_vector(self, arg: Any):
        raise ValueError('Objective vector cannot be changed externally')

    @property
    def num_pqgs(self) -> int:
        return self._num_pqgs

    @num_pqgs.setter
    def num_pqgs(self, arg: Any):
        raise ValueError('Number of parameterized quantum gates cannot be changed externally')

    @property
    def hamiltonian(self) -> list[PauliString]:
        return self._hamiltonian

    def objective_value(self, z: Union[str, int]) -> float:
        """
        Returns the objective value of the bipartition specified by the bitstring z.
        Integers are considered as bitstrings as well.

        Parameters
        ---------
        z: string or integer

        """
        #print('z start',z)
        if isinstance(z, int):
            z = bin(z)[2:]
        # Check if length of z matches number of nodes, otherwise append/cut '0's from the left until lengths match
        a = len(self.free_qubits) - len(z)        
        if a >= 0:
            z = a * '0' + z
        else:
            z = z[-a:]
     
        # Integrate fixed_bits into string z if fixed_bits is not empty.
        # Left to right.
        if len(self.fixed_bits) > 0:
            b = []
            j = 0
            for i in self.nodes:
                if i in self.fixed_bits:
                    b.append(str(self.fixed_bits[i]))
                else:
                    b.append(z[j])
                    j += 1                
            z = ''.join(b)
            #z = z[::-1] #reverse order of z
        nodes_list = list(self.nodes())

        obj = 0
        for i, j in self.edges:
            if z[nodes_list.index(i)] != z[nodes_list.index(j)]:
                obj += self.edges[i, j]['weight'] * self.opt
        return obj

    def apply_pqg(self, index: int, state: PureState, angle: float, dagger: bool = False):
        """
        Applies the parametrized QAOA's phase separator and mixer for indices 0 and 1, respectively.

        Parameters:
        -----------
        index: int
            The index according to which either the phase separator or the mixer is applied
        state: PureState
            The state the parametrized quantum gate is applied to
        angle: float
            The angle of the parametrized quantum gate
        dagger: bool
            Whether to apply the parametrized quantum gate or its inverse
        """
        if index == 0:
            np.multiply(np.exp(-1j * angle * self.objective_vector), state, out=state)
        elif index == 1:
            for qubit in range(self._num_var):
                state.rx(qubit=qubit, angle=2 * angle, dagger=dagger)

    def init_state(self) -> PureState:
        return PureState([1 / np.sqrt(2**self._num_var) for _ in range(2**self._num_var)])
