"""
Module for the generic optimizer class.
"""
import warnings
from scipy import optimize
import optiland.backend as be
# Assuming OptimizationProblem will be accessible from ..problem or similar
# For now, let's use a placeholder or expect it to be in ..optimization
from ..optimization import OptimizationProblem

class OptimizerGeneric:
    """Generic optimizer class for solving optimization problems.

    Args:
        problem (OptimizationProblem): The optimization problem to be solved.

    Attributes:
        problem (OptimizationProblem): The optimization problem to be solved.
        _x (list): List to store the values of the variables during
            optimization.

    Methods:
        optimize(maxiter=1000, disp=True, tol=1e-3): Optimize the problem
            using the specified parameters.
        undo(): Undo the last optimization step.
        _fun(x): Internal function to evaluate the objective function.

    """

    def __init__(self, problem: OptimizationProblem):
        self.problem = problem
        self._x = []

        if self.problem.initial_value == 0.0:
            self.problem.initial_value = self.problem.sum_squared()

    def optimize(self, method=None, maxiter=1000, disp=True, tol=1e-3, callback=None):
        """Optimize the problem using the specified parameters.

        Args:
            method (str, optional): The optimization method to use. Default is
                chosen to be one of BFGS, L-BFGS-B, SLSQP, depending on
                whether contraints or bounds given. Follows
                scipy.optimize.minimize method.
            maxiter (int, optional): Maximum number of iterations.
                Default is 1000.
            disp (bool, optional): Whether to display optimization information.
                Default is True.
            tol (float, optional): Tolerance for convergence. Default is 1e-3.
            callback (callable): A callable called after each iteration.

        Returns:
            result (OptimizeResult): The optimization result.

        """
        x0 = [var.value for var in self.problem.variables]
        self._x.append(x0)
        x0 = be.to_numpy(x0)
        bounds = tuple([var.bounds for var in self.problem.variables])

        options = {"maxiter": maxiter, "disp": disp}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = optimize.minimize(
                self._fun,
                x0,
                method=method,
                bounds=bounds,
                options=options,
                tol=tol,
                callback=callback,
            )

        # The last function evaluation is not necessarily the lowest.
        # Update all lens variables to their optimized values
        for idvar, var in enumerate(self.problem.variables):
            var.update(result.x[idvar])
        self.problem.update_optics()

        return result

    def undo(self):
        """Undo the last optimization step."""
        if len(self._x) > 0:
            x0 = self._x[-1]
            for idvar, var in enumerate(self.problem.variables):
                var.update(x0[idvar])
            self._x.pop(-1)

    def _fun(self, x) -> float:
        """Internal function to evaluate the objective function.

        Args:
            x (array-like): The values of the variables.

        Returns:
            rss (float): The residual sum of squares.

        """
        # Update all variables to their new values
        for idvar, var in enumerate(self.problem.variables):
            var.update(be.array(x[idvar]))

        # Update optics (e.g., pickups and solves)
        self.problem.update_optics()

        # Compute merit function value
        try:
            rss = self.problem.sum_squared()
            if be.isnan(rss):
                return 1e10
            # --- Convert result back to float for SciPy ---
            return float(be.to_numpy(rss))
        except ValueError:
            return 1e10
