# Placeholder for AbbeMaterial
# In a real scenario, this would have a proper definition.


class AbbeMaterial:
    def __init__(self, index, abbe_number, name="AbbeMaterial"):
        self.index = index
        self.abbe = abbe_number
        self.name = name

    def n(self, wavelength_um):
        # Dummy implementation for refractive index
        # This would typically use Sellmeier or other dispersion formula
        # For simplicity, returning a constant index here.
        return self.index

    def to_dict(self):
        return {
            "type": "AbbeMaterial",
            "index": self.index,
            "abbe": self.abbe,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data["index"], data["abbe"], data.get("name", "AbbeMaterial"))
