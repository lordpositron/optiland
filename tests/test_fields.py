import optiland.backend as be
import pytest

from optiland.fields.field import Field
from optiland.fields.field_group import FieldGroup
from optiland.fields import field_modes
from .utils import assert_allclose


@pytest.mark.parametrize(
    "x, y",
    [(0, 0), (5.3, 8.5), (0, 4.2)],
)
def test_field(set_test_backend, x, y):
    f = Field(x, y)

    assert f.x == x
    assert f.y == y


def test_field_group_inputs(set_test_backend):
    input_data = [(0, 0), (5, 0), (0, 6), (7, 9.2)]
    f = FieldGroup(mode=field_modes.AngleFieldMode())
    for field_data in input_data:
        f.add_field(y=field_data[1], x=field_data[0])

    assert_allclose(f.x_fields, be.array([0, 5, 0, 7]))
    assert_allclose(f.y_fields, be.array([0, 0, 6, 9.2]))

    assert f.max_x_field == 7
    assert f.max_y_field == 9.2
    assert f.max_field == be.sqrt(be.array(7**2 + 9.2**2))


def test_field_group_getters(set_test_backend):
    input_data = [(0, 0), (2.5, 0), (0, 2), (4, 3)]
    f = FieldGroup(mode=field_modes.AngleFieldMode())
    for field_data in input_data:
        f.add_field(y=field_data[1], x=field_data[0])

    assert f.get_field_coords() == [(0.0, 0.0), (0.5, 0.0), (0.0, 0.4), (0.8, 0.6)]

    assert f.get_field(0).x == 0
    assert f.get_field(0).y == 0
    assert f.get_field(3).x == 4
    assert f.get_field(3).y == 3

    # test case when max field is zero
    f = FieldGroup(mode=field_modes.AngleFieldMode())
    f.add_field(y=0, x=0)
    assert f.get_field_coords() == [(0, 0)]


def test_field_group_get_vig_factor(set_test_backend):
    f = FieldGroup(mode=field_modes.AngleFieldMode())
    f.add_field(y=0, x=0)

    vx, ny = f.get_vig_factor(1, 1)
    assert vx == 0.0
    assert vx == 0.0

    f = FieldGroup(mode=field_modes.AngleFieldMode())
    f.add_field(y=0, x=0, vx=0.2, vy=0.2)
    f.add_field(y=7, x=0, vx=0.2, vy=0.2)
    f.add_field(y=10, x=0, vx=0.2, vy=0.2)

    vx, ny = f.get_vig_factor(0.5, 0.7)
    assert vx == 0.2
    assert vx == 0.2

    vx, ny = f.get_vig_factor(1, 1)
    assert vx == 0.2
    assert vx == 0.2


def test_field_group_telecentric(set_test_backend):
    f = FieldGroup(mode=field_modes.AngleFieldMode())
    assert f.telecentric is False

    f.set_telecentric(True)
    assert f.telecentric is True


def test_field_to_dict(set_test_backend):
    f = Field(x=0, y=0)
    assert f.to_dict() == {"x": 0, "y": 0, "vx": 0.0, "vy": 0.0}


def test_field_group_to_dict(set_test_backend):
    input_data = [(0, 0), (2.5, 0), (0, 2), (4, 3)]
    f = FieldGroup(mode=field_modes.AngleFieldMode())
    for field_data in input_data:
        f.add_field(y=field_data[1], x=field_data[0])

    assert f.to_dict() == {
        "fields": [
            {"x": 0, "y": 0, "vx": 0.0, "vy": 0.0},
            {"x": 2.5, "y": 0, "vx": 0.0, "vy": 0.0},
            {"x": 0, "y": 2, "vx": 0.0, "vy": 0.0},
            {"x": 4, "y": 3, "vx": 0.0, "vy": 0.0},
        ],
        "mode": {"type": "AngleFieldMode"},
        "telecentric": False,
    }


def test_field_from_dict(set_test_backend):
    f = Field.from_dict(
        {"x": 0, "y": 0, "vx": 0, "vy": 0},
    )
    assert f.x == 0
    assert f.y == 0
    assert f.vx == 0
    assert f.vy == 0


def test_field_group_from_dict(set_test_backend):
    f = FieldGroup.from_dict(
        {
            "fields": [
                {"x": 0, "y": 0, "vx": 0, "vy": 0},
                {"x": 2.5, "y": 0, "vx": 0, "vy": 0},
                {"x": 0, "y": 2, "vx": 0, "vy": 0},
                {"x": 4, "y": 3, "vx": 0, "vy": 0},
            ],
            "mode": {"type": "AngleFieldMode"},
            "telecentric": False,
        },
    )
    assert f.get_field(0).x == 0
    assert f.get_field(0).y == 0
    assert f.get_field(3).x == 4
    assert f.get_field(3).y == 3
    assert f.telecentric is False
