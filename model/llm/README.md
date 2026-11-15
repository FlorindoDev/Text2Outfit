# LLM per le query di categoria

Questo modulo trasforma le 11 query iniziali di `CategoryQueryEmbeddings`
in rappresentazioni contestualizzate dal prompt e dalle altre categorie.
Il risultato contiene esclusivamente gli hidden state finali delle query.

| File | Responsabilità |
|---|---|
| `base.py` | Interfaccia astratta `LLM`, comune ai diversi backbone. |
| `download.py` | Riusa un checkpoint locale o nella cache; scarica da Hugging Face solo quando necessario. |
| `qwen3.py` | Implementazione `Qwen3LLM`, con checkpoint predefinito `Qwen/Qwen3-Embedding-8B`. |
| `__init__.py` | Esporta `LLM` e `Qwen3LLM`, disponibili anche da `model`. |

## Input e output

```python
hidden_states = llm(prompts, category_embeddings, seed_embeddings=None)
```

| Argomento | Formato |
|---|---|
| `prompts` | Una stringa non vuota oppure una sequenza di `B` stringhe. |
| `category_embeddings` | Query floating point `[B, 11, H]`; per un solo prompt anche `[11, H]`. |
| `seed_embeddings` | Opzionale: embedding seed floating point `[B, H]`; per un solo prompt anche `[H]`. |
| Valore restituito | Un solo `Tensor [B, 11, H]`, anche con un solo prompt. |

`H` è `llm.output_dim`: **4096** per il checkpoint predefinito. Nel modello
completo `ModelConfig.category_embedding_dim` deve coincidere con `H`.
Il numero di
prompt deve coincidere con il batch delle query. Il limite di contesto include
il prompt, l'eventuale token seed e le 11 query; un input troppo lungo viene rifiutato senza
troncare il testo. Query con dimensioni errate o valori non finiti sono rifiutate.

Le posizioni seguono `model.OUTFIT_CATEGORIES`: `tops`, `bottoms`, `all-body`,
`outerwear`, `shoes`, `bags`, `jewelry`, `accessories`, `sunglasses`, `hats`,
`scarves`. La posizione di ciascuna categoria resta la stessa nell'output.

## Come vengono usate le query

`Qwen3LLM` tokenizza il prompt, ricava gli embedding dei suoi token e aggiunge
le 11 query alla fine tramite `inputs_embeds`. Le query sono vettori esterni
al tokenizer: non diventano nuove parole del vocabolario. Se presente,
`seed_embeddings` viene inserito come un token dopo il prompt e prima delle
query. La sequenza è quindi `prompt -> seed opzionale -> 11 query`; lo stato
del seed non viene restituito. Per collegare anche encoder e teste, usa
[`SeedToOutfitModel`](../README.md#forward-seed-to-outfit).

L'attenzione segue queste regole:

- I token del testo vedono il testo precedente e se stessi.
- Il seed vede il prompt e se stesso.
- Ogni query vede tutto il prompt, l'eventuale seed e tutte le 11 query, incluse quelle successive.
- Il padding viene escluso e le posizioni tengono conto della lunghezza reale
  di ogni prompt nel batch.

L'attenzione bidirezionale tra categorie consente, per esempio, alla query
delle scarpe di dipendere anche dalle altre query. È una modifica della
maschera causale originale di Qwen3; le query e le teste vanno addestrate per
renderla utile al compito. Non viene applicata automaticamente la attention
mask loss del paper.

Il backbone è `Qwen3Model`: non vengono calcolati logits del vocabolario.
Il modulo seleziona gli stati dell'ultimo layer corrispondenti alle query,
senza pooling, normalizzazione L2 aggiuntiva o selezione di articoli.
Le norm interne del backbone restano quelle del modello originale.

## Checkpoint locale e download da Hugging Face

`Qwen3LLM` usa `resolve_model_path()` di `download.py` prima di caricare
tokenizer e backbone. L'ordine è:

1. Se `model_name` indica una cartella locale esistente, usa quella.
2. Altrimenti cerca il modello nella cache Hugging Face, senza richieste di rete.
3. Se manca o la cache è incompleta, scarica il repository da Hugging Face.

Il controllo verifica configurazione, file del tokenizer e pesi. Per i
checkpoint suddivisi in shard devono essere presenti tutti i file elencati
nell'indice. Una cartella locale esplicita incompleta viene rifiutata; una
cache incompleta viene completata tramite il download. Gli errori di rete
o di accesso al repository vengono propagati al chiamante.

Il percorso risolto è disponibile in `llm.model_path`. Qwen carica da quel
percorso usando `local_files_only=True`: l'esecuzione del modello avviene
sulla macchina che esegue Python, senza chiamate a OpenRouter o a un servizio
remoto di inferenza. Il download è gestito soltanto da `download.py`.

Per scaricare i file prima di caricare il modello in memoria, dalla root del
progetto puoi eseguire:

```bash
python -m model.llm.download Qwen/Qwen3-Embedding-8B --cache-dir .cache/llm
```

Lo script stampa il percorso locale del checkpoint e non carica i pesi in RAM
o GPU. Ripetendo il comando, un checkpoint completo viene riusato offline.
Senza `--cache-dir` usa la cache standard di Hugging Face, configurabile anche
con le variabili d'ambiente della libreria. Per repository privati o gated
usa l'autenticazione Hugging Face già configurata, per esempio con `hf auth login`.

La stessa cache si sceglie nel costruttore; in alternativa puoi passare una
cartella che contiene già modello e tokenizer:

```python
llm = Qwen3LLM(cache_dir=".cache/llm")
# Oppure, con un checkpoint già salvato:
llm = Qwen3LLM(model_name="/percorso/al/checkpoint")
```

## Esempio su GPU, anche su AWS

Il test automatico usa un modello minuscolo inizializzato localmente e simula
i download: non scarica il checkpoint completo 8B.

```python
import torch

from model import CategoryQueryEmbeddings, Qwen3LLM

llm = Qwen3LLM(
    cache_dir=".cache/llm", trainable=False, dtype=torch.bfloat16,
).to("cuda")
queries = CategoryQueryEmbeddings(embedding_dim=llm.output_dim).to("cuda")

prompts = [
    "Outfit casual: maglietta rossa, jeans blu e sneakers bianche.",
    "Outfit elegante per una cena invernale.",
]
category_embeddings = queries(batch_size=len(prompts))
hidden_states = llm(prompts, category_embeddings)  # [2, 11, 4096]

shoes_index = llm.categories.index("shoes")
shoes_hidden_state = hidden_states[:, shoes_index, :]  # [2, 4096]
```

Usa questi hidden state come input delle teste MLP di retrieval, presenza e
sottocategoria. Ogni MLP deve avere `input_dim=llm.output_dim`.

## Gradienti e dipendenze

Per default `trainable=False` congela il backbone e lo mantiene in `eval()`
anche quando il contenitore entra in training. Il forward conserva comunque
il gradiente verso le query esterne e il seed: puoi addestrarli insieme alle tue MLP.
Il modulo adatta query e seed a dispositivo e dtype del backbone con operazioni
differenziabili. L'output usa dispositivo e precisione del backbone.

`trainable=True` rende allenabili tutti i pesi del backbone. Per il fine-tuning
LoRA, costruire invece Qwen congelato e chiamare:

```python
llm.enable_lora(rank=64, alpha=128, dropout=0.05, gradient_checkpointing=True)
```

Il metodo congela il backbone e inserisce adattatori PEFT sulle proiezioni
Q/K/V, mantenendo il forward e la maschera delle categorie. Gli adattatori
usano FP32; il gradient checkpointing è non rientrante. `train()` abilita il
training degli adattatori senza sbloccare i pesi base. Servono le dipendenze
PEFT del progetto. La [guida training](../../training/README.md) descrive
optimizer, loss, checkpoint e ripresa; i checkpoint contengono tutti i pesi
LoRA insieme alle query e alle MLP.
Per la sola inferenza puoi racchiudere il forward in `torch.inference_mode()`;
evitalo quando vuoi addestrare le query.

Servono `transformers>=4.51.3,<4.52`, come indicato in `requirements.txt`.
La versione viene mantenuta nell'intervallo verificato dal progetto.
`Qwen3LLM` usa SDPA e una maschera esplicita `[B, 1, L, L]`, dove `L` include
prompt, seed opzionale e query: il costo della maschera cresce quadraticamente e va considerato
prima di usare prompt lunghi o batch grandi.

Per aggiungere un'altra famiglia LLM, crea un file che implementi `LLM` e
mantenga lo stesso contratto di input, ordine delle categorie e output.
