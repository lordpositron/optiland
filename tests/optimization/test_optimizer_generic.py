"""
Tests for the OptimizerGeneric class.
"""
import warnings
import optiland.backend as be
import pytest

# Assuming OptimizationProblem remains in optiland.optimization.optimization
# and OptimizerGeneric is in the new 'algorithms' submodule
from optiland.optimization.optimization import OptimizationProblem
from optiland.optimization.algorithms.optimizer_generic import OptimizerGeneric
from optiland.samples.microscopes import Microscope20x, UVReflectingMicroscope


class TestOptimizerGeneric:
    def test_optimize(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = OptimizerGeneric(problem)
        result = optimizer.optimize(maxiter=10, disp=False, tol=1e-3)
        assert result.success

    def test_undo(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = OptimizerGeneric(problem)
        optimizer.optimize(maxiter=10, disp=False, tol=1e-3)
        optimizer.undo()
        assert len(optimizer._x) == 0

    def test_fun_nan_rss(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            lens = UVReflectingMicroscope()
            # this will "break" the lens, resulting in NaN (for testing)
            lens.set_radius(0.2, 3)
            problem = OptimizationProblem()
            input_data = {
                "optic": lens,
                "Hx": 0.0,
                "Hy": 0.1,
                "wavelength": 0.5,
                "num_rays": 100,
                "surface_number": -1,
            }
            problem.add_operand(
                operand_type="rms_spot_size",
                target=0.0,
                weight=1.0,
                input_data=input_data,
            )
            # Initialize problem's initial_value if sum_squared() might be NaN initially
            # This can happen if the lens is broken from the start.
            # OptimizerGeneric's __init__ calls problem.sum_squared().
            # If it's NaN, initial_value becomes NaN, which might not be desired.
            # For this specific test, we want to test _fun, so ensure problem setup is valid before _fun.
            try:
                problem.initial_value = problem.sum_squared()
                if be.isnan(problem.initial_value):
                    problem.initial_value = 1e10 # Fallback if already NaN
            except ValueError: # Or whatever specific error might occur
                problem.initial_value = 1e10


            optimizer = OptimizerGeneric(problem)
            # The value passed to _fun (0.2) might not be used if variables are not correctly linked,
            # but the key is that problem.sum_squared() inside _fun should return NaN
            # due to the broken lens state, which then _fun should convert to 1e10.

            # We need to ensure the variable being changed by _fun is the one causing NaN
            # The current OptimizerGeneric._fun updates variables based on the input `x`.
            # Let's assume the variable list corresponds to the `x` array.
            # If there are no variables, x would be empty.
            # For this test, let's add a variable that matches the change.
            problem.clear_variables() # Clear previous variables
            # Add a variable corresponding to the radius we changed.
            # This setup is a bit contrived for testing _fun directly with a NaN scenario.
            # Typically, the optimizer would set the variable that causes the NaN.

            # Simpler: ensure the problem state itself will lead to NaN, then call _fun.
            # The _fun method takes an array `x` which corresponds to `problem.variables`.
            # If `problem.variables` is empty, `x` will be empty.
            # Let's ensure the lens is broken, and then call _fun with some dummy `x`
            # if there are variables, or test the sum_squared directly if easier.

            # The original test implies that _fun is called with a value that, when applied
            # to a variable, makes the lens produce NaN.
            # Here, lens is already broken. If there are no variables, _fun(be.array([]))
            # would just evaluate the already broken lens.

            # If we ensure there's one variable, and its update doesn't "fix" the lens:
            problem.add_variable(lens, "radius", surface_number=2) # some other radius

            # Call _fun with a value for this variable. The lens state for surface 3 (radius 0.2)
            # should still cause a NaN in sum_squared().
            assert optimizer._fun(be.array([lens.surface_group.surfaces[2].geometry.radius])) == 1e10
