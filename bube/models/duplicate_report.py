from pydantic import BaseModel

from .image_embedding import ImageEmbeddingNeighbour


class SuspiciousFile(BaseModel):
    """A suspicious file that is a potential duplicate of another file."""

    filename: str
    distance: float
    duplicate_chance_in_percent: int

    @classmethod
    def from_neighbour_embedding(cls, neighbour: ImageEmbeddingNeighbour) -> "SuspiciousFile":
        """Create a SuspiciousFile object from an ImageEmbeddingNeighbour object."""
        return cls(
            filename=neighbour.filename,
            distance=neighbour.distance,
            duplicate_chance_in_percent=int((1 - neighbour.distance) * 100),
        )


class DuplicateReportPart(BaseModel):
    """Contains multiple suspicious files and the number of files in total."""

    num_of_files: int
    filenames: list[SuspiciousFile]


class DuplicateReport(BaseModel):
    """Report containing the information about the duplicates and suspicious files for a given image."""

    original_filename: str
    duplicates: DuplicateReportPart
    suspicious: DuplicateReportPart
