"""
Tests for the SHGO (Simplicial Homology Global Optimization) optimizer class.
"""
import pytest
from optiland.optimization.optimization import OptimizationProblem
from optiland.optimization.algorithms.shgo import SHGO
from optiland.samples.microscopes import Microscope20x

class TestSHGO:
    def test_optimize_basic_run(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        # SHGO requires bounds for all variables.
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2", # Effective focal length
            target=90,       # Target EFL
            weight=1.0,
            input_data=input_data,
        )
        optimizer = SHGO(problem)
        # SHGO can be slow; use minimal options for a quick test run.
        # `iters=1` performs one iteration of the SHGO algorithm's main loop.
        # `options={'maxfev': 20}` limits function evaluations for local searches.
        result = optimizer.optimize(iters=1, options={'maxfev': 20, 'disp': False}, workers=1)

        # SHGO success flag might depend on finding the true global optimum,
        # which is unlikely with such limited iterations.
        # Instead, check if the optimization ran and produced a result.
        assert result is not None
        assert hasattr(result, 'x') # Check if a solution vector is present
        assert hasattr(result, 'fun') # Check if a function value is present
        # result.success might be False if global minimum not certified, which is fine for this basic test.

    def test_raise_error_no_bounds(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        # Add a variable without bounds
        problem.add_variable(lens, "radius", surface_number=1)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = SHGO(problem)
        # SHGO should raise ValueError if bounds are not fully provided.
        with pytest.raises(ValueError, match="SHGO requires all variables have valid .* bounds."):
            optimizer.optimize(iters=1)

    def test_raise_error_incomplete_bounds(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1, min_val=10) # Missing max_val
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = SHGO(problem)
        with pytest.raises(ValueError, match="SHGO requires all variables have valid .* bounds."):
            optimizer.optimize(iters=1)

        problem.clear_variables()
        problem.add_variable(lens, "radius", surface_number=1, max_val=100) # Missing min_val
        optimizer2 = SHGO(problem)
        with pytest.raises(ValueError, match="SHGO requires all variables have valid .* bounds."):
            optimizer2.optimize(iters=1)

    def test_options_passthrough(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)
        input_data = {"optic": lens}
        problem.add_operand(operand_type="f2", target=90, weight=1.0, input_data=input_data)

        optimizer = SHGO(problem)
        # Test passing some SHGO specific options
        # These options are for quick execution rather than finding an actual optimum
        custom_options = {
            'maxiter': 5, # Max global iterations (different from iters for SHGO)
            'maxfev': 15,
            'disp': False,
        }
        # SHGO's iters is a direct argument, not in options dict for the main loop.
        result = optimizer.optimize(iters=1, options=custom_options, sampling_method='sobol', workers=1)
        assert result is not None
        assert hasattr(result, 'x')
        # We can't easily verify `sampling_method` was used without deeper inspection,
        # but this confirms the call doesn't fail with these parameters.
        # result.nit might be the number of local minimizations or global iterations,
        # depending on SHGO's internals and success. For a run with iters=1,
        # result.nit (if it means global iterations) should be low.
        # If result.message indicates limited iterations, that's also a good sign.
        # For SHGO, result.nit refers to the number of successful minimizations.
        # Check result.message or result.success for more info if needed.
        # For now, just ensuring it runs is the primary goal.
