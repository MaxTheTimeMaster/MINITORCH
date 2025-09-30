from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    vals = list(vals)
    vals[arg] += epsilon

    if isinstance(vals, float):
        f_h = f(vals)[0]
    else:
        f_h = f(*vals)

    vals[arg] -= 2 * epsilon

    if isinstance(vals, float):
        f_hh = f(vals)[0]
    else:
        f_hh = f(*vals)


    v1 = f_h *  (1 / epsilon)
    v2 = f_hh * (1 / epsilon)

    # print( (v2 - v1) / 2)
    
    return (v1 - v2) / 2

variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    visited = dict()
    order = []
    def dfs(var: Variable) -> None:
        visited[var.unique_id] = 1

        if var.is_constant():
            return

        if var.is_leaf():
            order.append(var)
            return
        
        for parent in var.parents: 
            if parent.unique_id in visited.keys():
                continue

            dfs(parent)
        
        order.append(var)

    dfs(variable)

    # print(order)

    return list(reversed(order))

    

    

    

            

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    sorted_vars = list(topological_sort(variable))
    sorted_vars_index = list(map(lambda x: x.unique_id, sorted_vars))

    # print(sorted_vars) 

    if variable.is_leaf():
        variable.accumulate_derivative(0.0)
        return 

    if variable.is_constant():
        return      
    
    """vars = variable.chain_rule(deriv)

    for var, d in vars:
        if var.is_leaf():
            var.accumulate_derivative(d)
            continue
        
        var.chain_rule(d)"""
    
    vars = variable.chain_rule(deriv)

    vars = list(filter(lambda x: not x[0].is_constant(), vars))

    vars_index = list(zip(vars, list(map(lambda x: x[0].unique_id, vars))))

    # print(sorted_vars_index)
    # print(vars_index)
    vars_index.sort(key = lambda x: -sorted_vars_index.index(x[1]))


    for var, d in vars: 
        if var.is_leaf():
            var.accumulate_derivative(d)
            # continue

        backpropagate(var, d)
    
    """for var, d in vars: 
        if var.is_leaf()

    for var in sorted_vars:
        if var.is_leaf():
            continue

        vars = variable.chain_rule(deriv)
        for var2, d in vars:
            if var2.is_leaf():
                var2.accumulate_derivative(d)
            
    backpropagate(sorted_vars[1], d)"""
    
    return


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
