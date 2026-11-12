# MLP configurabile

`MLP` riceve un embedding e restituisce un vettore della dimensione scelta.
`hidden_dims` specifica la larghezza di ciascun layer nascosto: il numero di
elementi determina la profondita. Ogni layer nascosto usa ReLU; l'ultimo
layer e lineare, senza attivazione o normalizzazione.

```python
import torch
from torch import nn

from model import MLP

# Quattro reti con pesi indipendenti e architetture diverse.
networks = nn.ModuleDict({
    "small": MLP(1024, 128, hidden_dims=[256]),
    "deep": MLP(1024, 64, hidden_dims=[512, 256, 128]),
    "wide": MLP(1024, 256, hidden_dims=[2048, 512]),
    "linear": MLP(1024, 1),
})

embeddings = torch.randn(8, 1024)
outputs = {name: network(embeddings) for name, network in networks.items()}
# outputs["deep"].shape == [8, 64]
```

Per Marqo usare `input_dim=1536`; per FashionCLIP `input_dim=1024`.
L'input puo avere forma `[D]`, `[batch, D]` o `[..., D]`: cambia solo l'ultima
dimensione. Input, output e layer nascosti devono avere dimensioni intere
positive; `hidden_dims=[]` crea una sola proiezione lineare.

`ModuleDict` registra i parametri di tutte le reti, quindi si puo passare
`networks.parameters()` a un optimizer. Ogni MLP puo anche essere usato,
salvato con `state_dict()` e addestrato separatamente. Per leggere gli
embedding su disco vedere la [guida degli script](../../scripts/README.md#precomputare-gli-embedding).
