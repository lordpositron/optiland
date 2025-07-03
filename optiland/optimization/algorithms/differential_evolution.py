"""
Module for the Differential Evolution optimizer class.
"""
import warnings
from scipy import optimize
import optiland.backend as be
from ..optimization import OptimizationProblem # Assuming OptimizationProblem stays here
from .optimizer_generic import OptimizerGeneric

class DifferentialEvolution(OptimizerGeneric):
    """Differential Evolution optimizer for solving optimization problems.

    Args:
        problem (OptimizationProblem): The optimization problem to be solved.

    Methods:
        optimize(maxiter=1000, disp=True, workers=-1): Runs the differential
            evolution optimization algorithm.

    """

    def __init__(self, problem: OptimizationProblem):
        """Initializes a new instance of the DifferentialEvolution class.

        Args:
            problem (OptimizationProblem): The optimization problem to be
                solved.

        """
        super().__init__(problem)

    def optimize(self, maxiter=1000, disp=True, workers=-1, callback=None, **kwargs):
        """Runs the differential evolution optimization algorithm.

        Args:
            maxiter (int): Maximum number of algorithm generations.
            disp (bool): Set to True to print progress messages.
            workers (int or map-like callable): If -1, all available CPU cores are used.
                                     If a map-like callable, it is used for parallel evaluation.
            callback (callable): A callable called after each iteration.
                                 `callback(xk, convergence=val)`
            **kwargs: Additional keyword arguments to pass to `scipy.optimize.differential_evolution`.
                      e.g., `strategy`, `popsize`, `tol`, `mutation`, `recombination`, `seed`, etc.

        Returns:
            scipy.optimize.OptimizeResult: The optimization result.

        Raises:
            ValueError: If any variable in the problem does not have valid (min, max) bounds.
        """
        # Get initial values in backend format
        x0_backend = [var.value for var in self.problem.variables]
        self._x.append(x0_backend)  # Store backend values for undo
        # Convert x0 to NumPy for SciPy, can be used as `x0` initial population hint
        x0_numpy = be.to_numpy(x0_backend)

        bounds = tuple([var.bounds for var in self.problem.variables])
        if any(None in bound_pair or len(bound_pair) != 2 for bound_pair in bounds):
            raise ValueError(
                "Differential evolution requires all variables have valid (min, max) bounds.",
            )

        # SciPy's differential_evolution uses 'updating' and 'workers' for parallelism.
        # 'workers' can be an int (number of cores) or a map-like callable.
        # 'updating' can be 'immediate' or 'deferred'. 'deferred' is often better with 'workers=-1'.
        current_updating_strategy = kwargs.pop('updating', 'deferred' if workers == -1 else 'immediate')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            result = optimize.differential_evolution(
                self._fun, # Inherited from OptimizerGeneric
                bounds=bounds,
                maxiter=maxiter,
                x0=x0_numpy, # Hint for initial population
                disp=disp,
                updating=current_updating_strategy,
                workers=workers,
                callback=callback,
                **kwargs
            )

        # Update Optiland variables with the solution found by SciPy
        for idvar, var in enumerate(self.problem.variables):
            var.update(result.x[idvar])

        self.problem.update_optics() # Final update to the optical system

        return result
