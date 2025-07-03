"""
Tests for the DifferentialEvolution optimizer class.
"""
import pytest
from optiland.optimization.optimization import OptimizationProblem
from optiland.optimization.algorithms.differential_evolution import DifferentialEvolution
from optiland.samples.microscopes import Microscope20x

class TestDifferentialEvolution:
    def test_optimize(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        problem.add_variable(
            lens,
            "index", # Variable type: refractive index
            surface_number=1,
            min_val=1.2, # Bounds are required
            max_val=1.8,
            wavelength=0.5, # Wavelength for index variable
        )
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2", # Effective focal length
            target=90,       # Target EFL
            weight=1.0,
            input_data=input_data,
        )
        optimizer = DifferentialEvolution(problem)
        # Use low maxiter for speed, workers=1 to avoid issues in some CI environments
        result = optimizer.optimize(maxiter=10, disp=False, workers=1)
        assert result.success

    def test_raise_error_no_bounds(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        # Add variable without bounds
        problem.add_variable(lens, "index", surface_number=1, wavelength=0.5)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=95,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = DifferentialEvolution(problem)
        # DifferentialEvolution should raise ValueError if bounds are not fully provided
        with pytest.raises(ValueError, match="Differential evolution requires all variables have valid .* bounds."):
            optimizer.optimize(maxiter=10, disp=False)

    def test_raise_error_incomplete_bounds(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        problem.add_variable(lens, "index", surface_number=1, min_val=1.2, wavelength=0.5)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=95,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = DifferentialEvolution(problem)
        with pytest.raises(ValueError, match="Differential evolution requires all variables have valid .* bounds."):
            optimizer.optimize(maxiter=10, disp=False)

        problem.clear_variables()
        problem.add_variable(lens, "index", surface_number=1, max_val=1.8, wavelength=0.5)
        optimizer2 = DifferentialEvolution(problem)
        with pytest.raises(ValueError, match="Differential evolution requires all variables have valid .* bounds."):
            optimizer2.optimize(maxiter=10, disp=False)


    def test_workers_parallel_execution(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        problem.add_variable(
            lens,
            "index",
            surface_number=1,
            min_val=1.2,
            max_val=1.8,
            wavelength=0.5,
        )
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = DifferentialEvolution(problem)
        # Test with workers = -1 (use all available cores)
        # This mainly checks if it runs without error with parallelization enabled.
        # Actual speedup depends on the problem and hardware.
        result = optimizer.optimize(maxiter=10, disp=False, workers=-1)
        assert result.success

    def test_custom_kwargs_passthrough(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        problem.add_variable(
            lens, "index", surface_number=1, min_val=1.2, max_val=1.8, wavelength=0.5
        )
        input_data = {"optic": lens}
        problem.add_operand(operand_type="f2", target=90, weight=1.0, input_data=input_data)

        optimizer = DifferentialEvolution(problem)
        # Example of passing a custom strategy and popsize
        # Ensure these are valid SciPy options for differential_evolution
        result = optimizer.optimize(
            maxiter=5,
            disp=False,
            workers=1,
            strategy='best1bin',
            popsize=5 # Small popsize for faster test
        )
        assert result.success
        # We can't easily assert that the strategy was *used* without introspection
        # or a more complex setup, but this checks it doesn't error out.
        assert result.nit <= 5 # Check if maxiter was respected roughly
                               # (nit is number of generations)
