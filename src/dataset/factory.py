"""Factory for creating datasets given the library and model name."""

from .datasets import OtherDataset, TransfomerDataset


class DatasetFactory:
    """Class for creating datasets given the library and model name."""

    @staticmethod
    def dataset_from_lib(
        lib_name: str, model_name: str, dataset_name: str, n_classes: int, class_labels: list, images: list
    ):
        """Given the library and model name, return the corresponding dataset."""
        if lib_name == "transformers":
            return TransfomerDataset(dataset_name, n_classes, class_labels, model_name, images)
        if lib_name in ("timm", "local"):
            return OtherDataset(dataset_name, n_classes, class_labels, images)
        return ValueError(f"Library {lib_name} not implemented.")
