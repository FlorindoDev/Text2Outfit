"""A configurable feed-forward network for item embeddings."""

from collections.abc import Sequence

from torch import Tensor, nn


class MLP(nn.Module):
    """Apply Linear/ReLU hidden layers followed by a linear output layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self.hidden_dims = tuple(hidden_dims)
        dimensions = (input_dim, *self.hidden_dims, output_dim)
        if any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
            for dim in dimensions
        ):
            raise ValueError("all layer dimensions must be positive integers")
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers: list[nn.Module] = []
        for index, (in_features, out_features) in enumerate(
            zip(dimensions, dimensions[1:])
        ):
            layers.append(nn.Linear(in_features, out_features))
            if index < len(self.hidden_dims):
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """Map ``[..., input_dim]`` to ``[..., output_dim]`` without normalization."""
        if inputs.ndim == 0 or inputs.size(-1) != self.input_dim:
            raise ValueError(f"inputs must have shape [..., {self.input_dim}]")
        return self.layers(inputs)
