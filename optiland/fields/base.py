from abc import ABC, abstractmethod

class BaseFieldStrategy(ABC):
    """
    Abstract base class for defining field strategies.

    This class serves as a foundation for the Strategy Pattern implementation
    to decouple field-type logic from the core ray tracing classes.
    """

    @abstractmethod
    def get_ray_origins(self, optic, Hx, Hy, Px, Py, vx, vy):
        """
        Calculates the origin points for rays based on field type.
        """
        pass

    @abstractmethod
    def get_paraxial_object_position(self, optic, Hy, y1, EPL):
        """
        Calculates the paraxial object position based on field type.
        """
        pass

    @abstractmethod
    def get_chief_ray_start_params(self, optic, chief_ray_y_at_stop, chief_ray_u_at_stop):
        """
        Calculates the starting parameters for the chief ray based on field type.
        """
        pass

    @abstractmethod
    def validate_optic_state(self, optic):
        """
        Validates the optic state based on the field type.
        """
        pass
