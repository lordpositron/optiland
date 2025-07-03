"""
Module for the Dual Annealing optimizer class.
"""
import warnings
from scipy import optimize
import optiland.backend as be
from ..optimization import OptimizationProblem # Assuming OptimizationProblem stays here
from .optimizer_generic import OptimizerGeneric

class DualAnnealing(OptimizerGeneric):
    """DualAnnealing is an optimizer that uses the dual annealing algorithm
    to find the minimum of an optimization problem.

    Args:
        problem (OptimizationProblem): The optimization problem to be solved.

    Methods:
        optimize(maxiter=1000, disp=True): Runs the dual annealing algorithm
            to optimize the problem and returns the result.

    """

    def __init__(self, problem: OptimizationProblem):
        super().__init__(problem)

    def optimize(self, maxiter=1000, disp=True, callback=None):
        """Runs the dual annealing algorithm to optimize the problem.

        Args:
            maxiter (int): Maximum number of iterations.
            disp (bool): Whether to display the optimization process.
            callback (callable): A callable called after each iteration.
                            It is called after each iteration of the global
                            search phase.

        Returns:
            scipy.optimize.OptimizeResult: The optimization result.

        """
        # Get initial values in backend format
        x0_backend = [var.value for var in self.problem.variables]
        self._x.append(x0_backend)  # Store backend values for undo
        # Convert x0 to NumPy for SciPy
        x0_numpy = be.to_numpy(x0_backend)

        bounds = tuple([var.bounds for var in self.problem.variables])
        if any(None in bound_pair or len(bound_pair) != 2 for bound_pair in bounds):
            raise ValueError("Dual annealing requires all variables have valid (min, max) bounds.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = optimize.dual_annealing(
                self._fun, # Inherited from OptimizerGeneric
                bounds=bounds,
                maxiter=maxiter,
                x0=x0_numpy, # Optional initial guess
                callback=callback,
                # local_search_options can be passed here if needed, e.g.,
                # local_search_options={"method": "L-BFGS-B"}
            )

        # Update Optiland variables with the solution found by SciPy
        for idvar, var in enumerate(self.problem.variables):
            var.update(result.x[idvar])

        self.problem.update_optics() # Final update to the optical system

        return result
