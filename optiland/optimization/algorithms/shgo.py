"""
Module for the Simplicial Homology Global Optimization (SHGO) class.
"""
import warnings
from scipy import optimize
import optiland.backend as be
from ..optimization import OptimizationProblem # Assuming OptimizationProblem stays here
from .optimizer_generic import OptimizerGeneric

class SHGO(OptimizerGeneric):
    """Simplicity Homology Global Optimization (SHGO).

    This optimizer is suitable for global optimization of functions with
    multiple local minima. It systematically explores the search space.

    Args:
        problem (OptimizationProblem): The optimization problem to be solved.

    Methods:
        optimize(workers=-1, *args, **kwargs): Runs the SHGO algorithm.

    """

    def __init__(self, problem: OptimizationProblem):
        """Initializes a new instance of the SHGO class.

        Args:
            problem (OptimizationProblem): The optimization problem to be
                solved.
        """
        super().__init__(problem)

    def optimize(self, workers=-1, callback=None, options=None, **kwargs):
        """Runs the SHGO algorithm.

        Note that the SHGO algorithm accepts many arguments similar to
        `scipy.optimize.shgo`. Consult SciPy documentation for details on
        `sampling_method`, `n`, `iters`, `minimizer_kwargs`, etc.

        Args:
            workers (int or map-like callable): If -1 (default), all available CPU cores are used
                                     for parallel evaluation of certain internal steps if supported
                                     by the local minimizers used by SHGO.
                                     If a map-like callable, it's used for parallelization.
            callback (callable): A callable called after each iteration of the global search.
                                 `callback(xk)` where `xk` is the current best guess.
            options (dict, optional): A dictionary of solver options. All methods accept the following
                                      generic options:
                                        maxiter (int): Maximum number of iterations to perform.
                                        maxfev (int): Maximum number of function evaluations.
                                        disp (bool): Set to True to print convergence messages.
            **kwargs: Arbitrary keyword arguments passed directly to `scipy.optimize.shgo`.

        Returns:
            scipy.optimize.OptimizeResult: The optimization result.

        Raises:
            ValueError: If any variable in the problem does not have valid (min, max) bounds.
        """
        # x0 is not directly used by shgo in the same way as some other optimizers,
        # but we can store the initial state for potential 'undo' functionality.
        x0_backend = [var.value for var in self.problem.variables]
        self._x.append(x0_backend)

        bounds = tuple([var.bounds for var in self.problem.variables])
        if any(None in bound_pair or len(bound_pair) != 2 for bound_pair in bounds):
            raise ValueError("SHGO requires all variables have valid (min, max) bounds.")

        # Prepare options for SHGO, combining explicitly passed options with kwargs
        shgo_options = {}
        if options:
            shgo_options.update(options)

        # disp is a common option, ensure it's passed correctly if present in kwargs or options
        if 'disp' in kwargs:
            shgo_options.setdefault('disp', kwargs.pop('disp'))
        elif 'disp' in shgo_options: # Already set via options dict
            pass
        else: # Default if not specified
             shgo_options.setdefault('disp', False)


        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Note: `workers` in `shgo` is for parallelizing sampling point generation if sampling method supports it.
            # It does not parallelize the local searches directly in the same way as `differential_evolution`.
            # Local search parallelization depends on the `minimizer_kwargs` and the chosen local method.
            result = optimize.shgo(
                self._fun, # Inherited from OptimizerGeneric
                bounds=bounds,
                workers=workers,
                callback=callback,
                options=shgo_options,
                **kwargs # Pass through other shgo specific args like n, iters, sampling_method
            )

        # SHGO returns solution in result.x. If successful, result.x is the global minimum.
        # If not successful, it might be the best point found or from the minimizer_pool.
        # It's generally good practice to update to result.x
        if result.x is not None:
            for idvar, var in enumerate(self.problem.variables):
                var.update(result.x[idvar])

        self.problem.update_optics() # Final update to the optical system

        return result
