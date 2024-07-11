import argparse

from benchmark_models.utils import SUPPORTED_DATASETS, SUPPORTED_MODELS


def parse_args():
    """
    Parses command line arguments for the converter program

    Returns
    ----
    argparse Namespace object containg the arguments supplied by the user
    """

    parser = argparse.ArgumentParser(
        description="Run Inferences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--forbid-cuda",
        action="store_true",
        help="Completely disable the usage of CUDA. This command overrides any other gpu options.",
    )
    parser.add_argument(
        "--use-cuda", action="store_true", help="Use the gpu if available."
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=64, help="Test set batch size"
    )
    parser.add_argument(
        "--network-name",
        "-n",
        type=str,
        required=True,
        help="Target network",
        choices=SUPPORTED_MODELS,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Dataset to use",
        choices=SUPPORTED_DATASETS,
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="Allow overwriting the existing network.",
    )

    parser.add_argument(
        "--output-path",
        "-p",
        type=str,
        help="Override output .keras file path. Default is models/converted-tf/{model_name}.keras.",
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Entierly skip compared validation between PyTorch and converted TensorFlow model.",
    )

    parser.add_argument(
        "--skip-pt-validation",
        action="store_true",
        help="Validate only the Keras model, skipping the PyTorch validation.",
    )

    parser.add_argument(
        "--skip-manipulation",
        action="store_true",
        help="Skip Manipulation test",
    )

    parsed_args = parser.parse_args()

    return parsed_args
