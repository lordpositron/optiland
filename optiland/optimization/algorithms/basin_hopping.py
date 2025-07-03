"""
Module for the Basin-hopping optimizer class.
"""

import warnings

from scipy import optimize

import optiland.backend as be

from ..optimization import OptimizationProblem
from .optimizer_generic import OptimizerGeneric


class BasinHopping(OptimizerGeneric):
    """Basin-hopping optimizer for solving optimization problems.

    This algorithm is designed for global optimization. It uses a local
    minimization algorithm for each basin-hopping iteration and randomly
    perturbs the solution to explore new basins.

    Args:
        problem (OptimizationProblem): The optimization problem to be solved.

    Methods:
        optimize(niter=100, callback=None, *args, **kwargs): Runs the basin-hopping
            optimization algorithm.
    """

    def __init__(self, problem: OptimizationProblem):
        """Initializes a new instance of the BasinHopping class.

        Args:
            problem (OptimizationProblem): The optimization problem to be
                solved.
        """
        super().__init__(problem)

    def optimize(self, niter=100, callback=None, minimizer_kwargs=None, **kwargs):
        """Runs the basin-hopping algorithm.

        Accepts arguments similar to `scipy.optimize.basinhopping`.

        Args:
            niter (int): Number of basin-hopping iterations. Default is 100.
            callback (callable): A function called after each iteration of
                                 basin-hopping. `callback(xk, f, accept)`
            minimizer_kwargs (dict, optional): Keyword arguments for the local minimizer
                                               (passed to `scipy.optimize.minimize`).
                                               For example, `minimizer_kwargs=
                                               {'method':'L-BFGS-B', 'bounds': bounds}`.
                                               If bounds are provided here, they will be
                                               used by the local minimizer.
                                               Basin-hopping itself does not directly
                                               take a `bounds` argument for the global
                                               search.
            **kwargs: Additional keyword arguments to pass to
                      `scipy.optimize.basinhopping`. e.g., `T` (temperature),
                      `stepsize`, `take_step`, `accept_test`, `interval`, `disp`.

        Returns:
            scipy.optimize.OptimizeResult: The optimization result. The solution `x`
                                           is the best minimum found.

        Raises:
            ValueError: If bounds are specified directly in `problem.variables` and
                        `minimizer_kwargs` does not handle them appropriately, as
                        BasinHopping itself does not use top-level bounds.
        """
        # Get initial values in backend format
        x0_backend = [var.value for var in self.problem.variables]
        self._x.append(x0_backend)  # Store backend values for undo
        # Convert x0 to NumPy for SciPy
        x0_numpy = be.to_numpy(x0_backend)

        # Basin-hopping itself doesn't take a 'bounds' argument.
        # Bounds must be handled by the local minimizer via 'minimizer_kwargs'.
        # Check if problem has bounds defined, and if so, warn if not handled by
        # minimizer_kwargs
        problem_bounds = tuple([var.bounds for var in self.problem.variables])
        has_problem_bounds = not all(
            b[0] is None and b[1] is None for b in problem_bounds
        )

        if minimizer_kwargs is None:
            minimizer_kwargs = {}

        if has_problem_bounds and "bounds" not in minimizer_kwargs:
            print(
                "Warning: BasinHopping is used and variables have bounds, "
                "but 'bounds' are not specified in 'minimizer_kwargs'. "
                "The local minimizer might not respect these bounds unless "
                "its default behavior or other settings in minimizer_kwargs "
                "handle them."
            )
        elif not has_problem_bounds and "bounds" in minimizer_kwargs:
            # If bounds are only in minimizer_kwargs, that's fine.
            pass

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            result = optimize.basinhopping(
                self._fun,  # Inherited from OptimizerGeneric
                x0=x0_numpy,
                niter=niter,
                callback=callback,
                minimizer_kwargs=minimizer_kwargs,
                **kwargs,
            )

        # Update Optiland variables with the solution found by SciPy
        if result.x is not None:
            for idvar, var in enumerate(self.problem.variables):
                var.update(result.x[idvar])

        self.problem.update_optics()  # Final update to the optical system

        return result
