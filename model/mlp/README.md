# MLP configurabile

`MLP` riceve un embedding e restituisce un vettore della dimensione scelta.
`hidden_dims` specifica la larghezza di ciascun layer nascosto: il numero di
elementi determina la profondita. Ogni layer nascosto usa ReLU; l'ultimo
layer e lineare, senza attivazione o normalizzazione.

## Funzione di attivazione

L'attivazione usata è **ReLU**, implementata con `torch.nn.ReLU()` in
[`mlp.py`](mlp.py): `ReLU(x) = max(0, x)`. Viene applicata dopo ogni layer
lineare nascosto. Il layer di uscita è **lineare, senza attivazione**.

| Parte della MLP | Operazione |
|---|---|
| Ogni layer nascosto | `Linear(in, out) → ReLU` |
| Layer finale | `Linear(in, out)` |
| Caso `hidden_dims=()` | Un solo layer `Linear`, senza ReLU. |

Per esempio, con `hidden_dims=(512,)` il percorso è
`Linear(input_dim, 512) → ReLU → Linear(512, output_dim)`.
Tutti i layer lineari hanno bias abilitato. La MLP non contiene dropout,
BatchNorm, LayerNorm o normalizzazione L2.

Questa scelta vale per tutte le MLP del modello: seed, item, retrieval,
presenza e sottocategoria. Le teste di classificazione restituiscono logits
grezzi: le trasformazioni richieste per BCE e cross-entropy sono gestite
dalle funzioni di loss. Gli embedding retrieval e item vengono normalizzati
nel calcolo della loss retrieval, senza modificare l'uscita della MLP.

## Configurazione e utilizzo

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

`mlp.dimensions` restituisce la tupla completa `(input, *hidden_dims, output)`.
Per esempio `MLP(1024, 256, [512]).dimensions` è `(1024, 512, 256)`.
Nel modello completo le cinque famiglie di MLP sono configurate separatamente
in [`model/config.py`](../config.py); il builder e la tabella delle dimensioni
sono descritti nella [guida del modello](../README.md#modello-completo-e-dimensioni-delle-mlp).

`ModuleDict` registra i parametri di tutte le reti, quindi si puo passare
`networks.parameters()` a un optimizer. Ogni MLP puo anche essere usato,
salvato con `state_dict()` e addestrato separatamente. Per leggere gli
embedding su disco vedere la [guida degli script](../../scripts/README.md#precomputare-gli-embedding).
