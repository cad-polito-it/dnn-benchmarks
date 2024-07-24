from abc import ABC, abstractmethod
import numpy as np

# TODO Implement metrics


class MetricEvaluator(ABC):
    @abstractmethod
    def __call__(self, inferences_count: int, labels, outputs):
        pass


class TopKAccuracy(MetricEvaluator):
    def __init__(self, k):
        if k < 1:
            raise ValueError("k must be greather than 0")
        self.k = k

    def __call__(self, inferences_count, labels, outputs):
        flat_labels = np.expand_dims(
            np.array(labels), axis=1
        )  # SHAPE (inferences_count) CONTENT (correct_label)
        flat_outputs = np.array(
            outputs
        )  # SHAPE (inferences_count, n_classes) CONTENT (image, class_score)
        top_k_indexes = flat_outputs.argpartition(-self.k, axis=-1)[
            :, -self.k :
        ]  # SHAPE (inferences_count, self.k) CONTENT (image, top_k_idxs)
        correct_elements = flat_labels == top_k_indexes
        correct = correct_elements.any(axis=1).sum()
        return correct, correct / inferences_count
