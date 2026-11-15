"""FashionCLIP and Marqo FashionSigLIP encoders for clothing descriptions."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol, cast

from torch import Tensor, nn
from torch.nn import functional as F
from ..config import DEFAULT_MODEL_CONFIG, FASHIONCLIP_MODEL_NAME

from ._marqo import DEFAULT_MODEL_NAME, _MarqoFeatures, _load_backbone, _positive_dimension


class TextEncoder(nn.Module, ABC):
    """Contract required by the multimodal outfit encoder."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Number of features returned for each description."""

    @abstractmethod
    def forward(self, descriptions: Sequence[str]) -> Tensor:
        """Return one L2-normalized embedding per description."""


class FashionCLIPTextEncoder(TextEncoder):
    """FashionCLIP text tower returning L2-normalized CLIP embeddings."""

    DEFAULT_MODEL_NAME = FASHIONCLIP_MODEL_NAME

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        trainable: bool = DEFAULT_MODEL_CONFIG.encoder_trainable,
    ) -> None:
        super().__init__()
        try:
            from transformers.models.auto.tokenization_auto import AutoTokenizer
            from transformers.models.clip.modeling_clip import CLIPTextModelWithProjection
        except ImportError as error:
            raise ImportError(
                "FashionCLIPTextEncoder requires the 'transformers' package"
            ) from error

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = CLIPTextModelWithProjection.from_pretrained(model_name)
        projection_dim = self.backbone.config.projection_dim
        if projection_dim is None or projection_dim <= 0:
            raise ValueError(
                "FashionCLIP text config must define a positive projection_dim"
            )
        self._max_length = _max_position_embeddings(self.backbone)
        self._output_dim = projection_dim
        self._trainable = trainable

        if not trainable:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def train(self, mode: bool = True) -> "FashionCLIPTextEncoder":
        super().train(mode)
        if not self._trainable:
            self.backbone.eval()
        return self

    def forward(self, descriptions: Sequence[str]) -> Tensor:
        _validate_descriptions(descriptions)
        device = next(self.backbone.parameters()).device
        tokens = self.tokenizer(
            list(descriptions),
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        ).to(device)
        embeddings = self.backbone(**tokens).text_embeds
        return F.normalize(embeddings, p=2, dim=-1)


class _TextProcessor(Protocol):
    def __call__(
        self,
        *,
        text: list[str],
        padding: Literal["max_length"],
        truncation: bool,
        max_length: int,
        return_tensors: Literal["pt"],
    ) -> Mapping[str, Tensor]: ...


class MarqoFashionSigLIPTextEncoder(TextEncoder):
    """Encode descriptions using Marqo's text cleaning and fixed padding."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        from transformers.models.auto.processing_auto import AutoProcessor

        self.backbone = backbone if backbone is not None else _load_backbone(model_name)
        self.processor = cast(
            _TextProcessor,
            AutoProcessor.from_pretrained(model_name, trust_remote_code=True),
        )
        features = cast(_MarqoFeatures, self.backbone)
        text_tower = features.model.text
        self._output_dim = _positive_dimension(text_tower.output_dim, "text output_dim")
        self._max_length = _positive_dimension(text_tower.context_length, "context_length")
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def train(self, mode: bool = True) -> "MarqoFashionSigLIPTextEncoder":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, descriptions: Sequence[str]) -> Tensor:
        _validate_descriptions(descriptions)
        device = next(self.backbone.parameters()).device
        processed = self.processor(
            text=list(descriptions),
            padding="max_length",
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        features = cast(_MarqoFeatures, self.backbone)
        embeddings = features.get_text_features(
            processed["input_ids"].to(device), normalize=False
        )
        return F.normalize(embeddings, p=2, dim=-1)


def _validate_descriptions(descriptions: Sequence[str]) -> None:
    if not descriptions:
        raise ValueError("descriptions cannot be empty")
    if any(not isinstance(text, str) or not text.strip() for text in descriptions):
        raise ValueError("descriptions must contain non-empty strings")


def _max_position_embeddings(backbone: nn.Module) -> int:
    config = getattr(backbone, "config", None)
    max_length = getattr(config, "max_position_embeddings", None)
    if (
        not isinstance(max_length, int)
        or isinstance(max_length, bool)
        or max_length <= 0
    ):
        raise ValueError(
            "FashionCLIP text config must define positive "
            "max_position_embeddings"
        )
    return max_length
