"""
Tests for the DualAnnealing optimizer class.
"""
import pytest
from optiland.optimization.optimization import OptimizationProblem
from optiland.optimization.algorithms.dual_annealing import DualAnnealing
from optiland.samples.microscopes import Microscope20x

class TestDualAnnealing:
    def test_optimize(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        problem.add_variable(
            lens,
            "thickness", # Variable type
            surface_number=1,
            min_val=10, # Bounds are required for DualAnnealing
            max_val=100,
        )
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2", # Effective focal length
            target=95, # Target EFL
            weight=1.0,
            input_data=input_data,
        )
        optimizer = DualAnnealing(problem)
        # Use a low number of maxiter for testing speed
        result = optimizer.optimize(maxiter=10, disp=False)
        assert result.success # Check if optimization reported success

    def test_raise_error_no_bounds(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        # Add a variable without bounds
        problem.add_variable(lens, "thickness", surface_number=1)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=95,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = DualAnnealing(problem)
        # DualAnnealing should raise ValueError if bounds are not provided
        with pytest.raises(ValueError, match="Dual annealing requires all variables have valid .* bounds."):
            optimizer.optimize(maxiter=10, disp=False)

    def test_raise_error_incomplete_bounds(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        # Add a variable with one bound missing
        problem.add_variable(lens, "thickness", surface_number=1, min_val=10)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=95,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = DualAnnealing(problem)
        with pytest.raises(ValueError, match="Dual annealing requires all variables have valid .* bounds."):
            optimizer.optimize(maxiter=10, disp=False)

        problem.clear_variables()
        problem.add_variable(lens, "thickness", surface_number=1, max_val=100)
        optimizer2 = DualAnnealing(problem)
        with pytest.raises(ValueError, match="Dual annealing requires all variables have valid .* bounds."):
            optimizer2.optimize(maxiter=10, disp=False)
