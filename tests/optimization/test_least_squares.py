"""
Tests for the LeastSquares optimizer class.
"""
import warnings
import optiland.backend as be
import pytest

from optiland.optimization.optimization import OptimizationProblem
from optiland.optimization.algorithms.least_squares import LeastSquares
from optiland.samples.microscopes import Microscope20x

# Mock classes for error handling tests
class MockOperandNaN:
    def __init__(self, target=0, weight=1):
        self.target = target
        self.weight = weight
        self.operand_type = "mock_nan"
        self.input_data = {}
        # Required attributes by OptimizationProblem.operand_info (if called)
        self.min_val = None
        self.max_val = None


    def fun(self): # This is what LeastSquares._compute_residuals_vector calls
        return be.nan

    def delta(self): # For completeness, though not directly used by _compute_residuals_vector
        return be.nan

    @property # Changed to property to match typical operand structure
    def value(self):
        return be.nan

class MockOperandException:
    def __init__(self, target=0, weight=1):
        self.target = target
        self.weight = weight
        self.operand_type = "mock_exception"
        self.input_data = {}
        self.min_val = None
        self.max_val = None

    def fun(self):
        raise RuntimeError("Test Exception from mock operand")

    def delta(self):
        raise RuntimeError("Test Exception from mock operand")

    @property
    def value(self):
        raise RuntimeError("Test Exception from mock operand")


class TestLeastSquares:
    def test_optimize(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        problem.add_variable(lens, "conic", surface_number=1, min_val=-1, max_val=1)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = LeastSquares(problem)
        result = optimizer.optimize(maxiter=10, disp=False, tol=1e-3)
        assert result.success

    def test_no_bounds(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        problem.add_variable(lens, "conic", surface_number=1)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = LeastSquares(problem)
        result = optimizer.optimize(maxiter=10, disp=False, tol=1e-3)
        assert result.success

    def test_verbose(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        problem.add_variable(
            lens,
            "radius",
            surface_number=1,
            min_val=-1000,
            max_val=None,
        )
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = LeastSquares(problem)
        result = optimizer.optimize(maxiter=100, disp=True, tol=1e-3) # disp=True
        assert result.success

    def test_method_trf_with_bounds(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        min_b, max_b = 10.0, 100.0 # Ensure float for precise comparison later
        # Ensure initial value is within bounds or test will be problematic
        # For Microscope20x, surface 1 radius is 40.0, which is within [10, 100]
        problem.add_variable(
            lens, "radius", surface_number=1, min_val=min_b, max_val=max_b
        )
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2", # Effective focal length
            target=90, # Target EFL
            weight=1.0,
            input_data=input_data,
        )
        optimizer = LeastSquares(problem)
        result = optimizer.optimize(method_choice="trf", maxiter=10, tol=1e-3)
        assert result.success

        optimized_radius = lens.surface_group.surfaces[1].geometry.radius
        # Add a small tolerance for floating point comparisons if needed
        assert min_b <= optimized_radius <= max_b

    def test_method_dogbox_with_bounds(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        min_b, max_b = 10.0, 100.0
        problem.add_variable(
            lens, "radius", surface_number=1, min_val=min_b, max_val=max_b
        )
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = LeastSquares(problem)
        result = optimizer.optimize(method_choice="dogbox", maxiter=10, tol=1e-3)
        assert result.success
        optimized_radius = lens.surface_group.surfaces[1].geometry.radius
        assert min_b <= optimized_radius <= max_b

    def test_method_lm_with_bounds_warning(self, capsys):
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
        optimizer = LeastSquares(problem)
        optimizer.optimize(method_choice="lm", maxiter=5) # lm ignores bounds
        captured = capsys.readouterr()
        expected_warning = (
            "Warning: Method 'lm' (Levenberg-Marquardt) chosen, "
            "but variable bounds are set. SciPy's 'lm' method does not "
            "support bounds; bounds will be ignored."
        )
        assert expected_warning in captured.out

    def test_unknown_method_choice_warning(self, capsys):
        lens = Microscope20x()
        problem = OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = LeastSquares(problem)
        optimizer.optimize(method_choice="unknown_method", maxiter=5)
        captured = capsys.readouterr()
        # The implementation defaults to 'trf' upon unknown method.
        expected_warning = (
            "Warning: Unknown method_choice 'unknown_method'. Defaulting to "
            "'trf' method."
        )
        assert expected_warning in captured.out


class TestLeastSquaresErrorHandling:
    def test_nan_residual_handling(self):
        lens = Microscope20x() # Dummy optic
        problem = OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)

        mock_op = MockOperandNaN()
        # Manually add mock operand to the problem's list
        # This bypasses problem.add_operand which might do type checking or expect certain structures
        problem.operands.operands.append(mock_op)
        problem.initial_value = 1.0 # Avoid OptimizerGeneric init call to sum_squared if it could be NaN

        optimizer = LeastSquares(problem)
        result = optimizer.optimize(maxiter=5)

        # Expected residual from _compute_residuals_vector is be.sqrt(1e10 / 1)
        # cost = 0.5 * sum(residuals**2) = 0.5 * (sqrt(1e10))^2 = 0.5 * 1e10
        assert be.isclose(result.cost, 0.5 * 1e10)
        assert result.status is not None

    def test_exception_in_residual_handling(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)

        mock_op = MockOperandException()
        problem.operands.operands.append(mock_op)
        problem.initial_value = 1.0

        optimizer = LeastSquares(problem)
        result = optimizer.optimize(maxiter=5)
        assert be.isclose(result.cost, 0.5 * 1e10)
        assert result.status is not None

    def test_optimize_no_operands(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)
        # No operands are added

        optimizer = LeastSquares(problem)
        result = optimizer.optimize(maxiter=5)

        assert result.success # Should complete successfully with 0 cost
        assert be.isclose(result.cost, 0.0)
        # result.fun for least_squares is the vector of residuals, should be empty
        assert len(result.fun) == 0
