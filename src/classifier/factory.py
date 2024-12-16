"""Factory for creating classifiers given the library and model name."""

from .classifiers import LocalClassifier, PretrainedOther, PretrainedTransformer


class ClassifierFactory:
    """_summary_ Factory for creating classifiers given the library and model name.

    Returns:
        _type_: _description_
    """

    @staticmethod
    def model_from_lib(lib_name: str, model_name: str, device: str):
        """Create a pretrained model from a library."""
        if lib_name == "transformers":
            return PretrainedTransformer(model_name, device)
        if lib_name == "timm":
            return PretrainedOther(model_name, device)
        if lib_name == "local":
            return LocalClassifier(model_name, device)
        return ValueError(f"Library {lib_name} not implemented.")
