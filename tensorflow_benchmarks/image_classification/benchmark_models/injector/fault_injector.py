from abc import ABC

from benchmark_models.inference_tools.inference_manager import InferenceManager


class FaultInjector(InferenceManager, ABC):

    def __init__(self, wrapped_inf_manager: InferenceManager):
        self.wrapped_inf_manager = wrapped_inf_manager
