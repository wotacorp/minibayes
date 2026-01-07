"""Export and import utilities for inference results."""

from minibayes.results import InferenceResult


def save_npz(result: InferenceResult, path: str) -> None:
    """
    Save inference result to NumPy compressed archive.

    Parameters
    ----------
    result : InferenceResult
        Results to save.
    path : str
        Output file path.
    """
    raise NotImplementedError()


def load_npz(path: str) -> InferenceResult:
    """
    Load inference result from NumPy compressed archive.

    Parameters
    ----------
    path : str
        Input file path.

    Returns
    -------
    InferenceResult
        Loaded results.
    """
    raise NotImplementedError()


def to_json(result: InferenceResult) -> dict[str, object]:
    """
    Convert inference result to JSON-serializable dict.

    Parameters
    ----------
    result : InferenceResult
        Results to convert.

    Returns
    -------
    dict
        JSON-serializable dictionary.
    """
    raise NotImplementedError()
