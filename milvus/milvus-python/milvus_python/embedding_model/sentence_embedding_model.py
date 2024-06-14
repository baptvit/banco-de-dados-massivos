from abc import ABC, abstractmethod
from typing import List


class SentenceEmbeddingModel(ABC):
    """Abstract class representing a sentence embedding model."""

    @abstractmethod
    def get_sentence_embedding(self, _: str) -> List[float]:
        """Embeds the preprocessed sentence into a vector."""
        raise NotImplementedError
