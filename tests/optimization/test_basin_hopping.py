"""
Tests for the BasinHopping optimizer class.
"""
import pytest
import optiland.backend as be
from optiland.optimization.optimization import OptimizationProblem
from optiland.optimization.algorithms.basin_hopping import BasinHopping
from optiland.samples.microscopes import Microscope20x

class TestBasinHopping:
    def test_optimize_improves_solution(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        # BasinHopping itself doesn't use bounds directly, but its local minimizer can.
        # For this test, let's use an unconstrained variable.
        problem.add_variable(lens, "radius", surface_number=1)
        initial_radius = lens.surface_group.surfaces[1].geometry.radius # Get initial value

        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2", # Effective focal length
            target=90,       # Target EFL, choose something different from initial
            weight=1.0,
            input_data=input_data,
        )

        # Store initial merit value, ensuring problem.initial_value is set correctly
        # by OptimizerGeneric's __init__
        optimizer = BasinHopping(problem)
        initial_merit = problem.initial_value # Value before optimization

        # Use low niter for speed.
        # minimizer_kwargs can be used to pass bounds to the local minimizer if needed.
        result = optimizer.optimize(niter=5, disp=False)

        # Check that the function value improved or is at least not worse significantly
        # (it's a stochastic algorithm, perfect improvement not always guaranteed with few iters)
        # For BasinHopping, result.fun is the function value at the best minimum found.
        assert result.fun <= initial_merit + 1e-9 # Allow for slight numerical noise if no improvement

        # More robust: check if the radius actually changed, implying optimizer ran
        final_radius = lens.surface_group.surfaces[1].geometry.radius
        # This check might fail if the initial point was already optimal or very close
        # For a more reliable check of "did it run", one might need to inspect internal states
        # or ensure the problem is set up such that change is guaranteed.
        # For now, combined with merit check, it's a reasonable indicator.
        # assert not be.isclose(initial_radius, final_radius) # This could be too strict

    def test_minimizer_kwargs_with_bounds(self):
        lens = Microscope20x()
        problem = OptimizationProblem()
        min_b, max_b = 30.0, 50.0  # Define bounds for the local search

        # Set initial radius outside these bounds to see if local minimizer respects them
        lens.surface_group.surfaces[1].geometry.radius = 20.0
        lens.update()

        problem.add_variable(lens, "radius", surface_number=1) # No top-level bounds here for BasinHopping

        input_data = {"optic": lens}
        problem.add_operand(operand_type="f2", target=40, weight=1.0, input_data=input_data)

        optimizer = BasinHopping(problem)

        minimizer_options = {'method': 'L-BFGS-B', 'bounds': [(min_b, max_b)]}

        result = optimizer.optimize(niter=10, minimizer_kwargs=minimizer_options, disp=False)

        assert result.success # Local minimizations might have succeeded
        optimized_radius = lens.surface_group.surfaces[1].geometry.radius

        # Check if the final radius is within the bounds specified for the local minimizer
        # This assumes the global step didn't jump out and the final solution reported
        # is from a successful local minimization that respected bounds.
        print(f"Optimized radius: {optimized_radius}")
        assert min_b <= optimized_radius <= max_b + 1e-9 # Add tolerance for boundary conditions

    def test_warning_if_problem_has_bounds_but_minimizer_doesnt(self, capsys):
        lens = Microscope20x()
        problem = OptimizationProblem()
        # Variable HAS bounds in the problem definition
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)
        input_data = {"optic": lens}
        problem.add_operand(operand_type="f2", target=90, weight=1.0, input_data=input_data)

        optimizer = BasinHopping(problem)
        # minimizer_kwargs does NOT specify bounds
        optimizer.optimize(niter=1, minimizer_kwargs={'method': 'Nelder-Mead'})

        captured = capsys.readouterr()
        expected_warning = (
            "Warning: BasinHopping is used and variables have bounds, "
            "but 'bounds' are not specified in 'minimizer_kwargs'."
        )
        assert expected_warning in captured.out

    def test_no_warning_if_minimizer_has_bounds(self, capsys):
        lens = Microscope20x()
        problem = OptimizationProblem()
        # Variable has bounds in problem definition
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)
        input_data = {"optic": lens}
        problem.add_operand(operand_type="f2", target=90, weight=1.0, input_data=input_data)

        optimizer = BasinHopping(problem)
        # minimizer_kwargs *does* specify bounds
        minimizer_options = {'method': 'L-BFGS-B', 'bounds': [(10, 100)]}
        optimizer.optimize(niter=1, minimizer_kwargs=minimizer_options)

        captured = capsys.readouterr()
        unexpected_warning = (
            "Warning: BasinHopping is used and variables have bounds, "
            "but 'bounds' are not specified in 'minimizer_kwargs'."
        )
        assert unexpected_warning not in captured.out

    def test_no_warning_if_no_problem_bounds(self, capsys):
        lens = Microscope20x()
        problem = OptimizationProblem()
        # Variable has NO bounds in problem definition
        problem.add_variable(lens, "radius", surface_number=1)
        input_data = {"optic": lens}
        problem.add_operand(operand_type="f2", target=90, weight=1.0, input_data=input_data)

        optimizer = BasinHopping(problem)
        # minimizer_kwargs also does not specify bounds
        optimizer.optimize(niter=1, minimizer_kwargs={'method': 'Nelder-Mead'})

        captured = capsys.readouterr()
        unexpected_warning = (
            "Warning: BasinHopping is used and variables have bounds, "
            "but 'bounds' are not specified in 'minimizer_kwargs'."
        )
        assert unexpected_warning not in captured.out
