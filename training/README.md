# Training seed-to-outfit

Il modello impara a completare un outfit partendo da **un capo seed e un
prompt**. Il target è l'outfit originale di Polyvore, seed compreso. Per ogni
categoria il modello predice se includerla, quale sottocategoria scegliere e
quale embedding usare per recuperare un capo dal catalogo.

Con la configurazione predefinita si allenano gli adattatori LoRA di Qwen,
le 11 query di categoria e le MLP. I pesi di Qwen, degli encoder e del modello
di compatibilità (CP) restano congelati.

## Indice

- [Avvio rapido](#avvio-rapido)
- [Iperparametri e configurazione](#iperparametri-e-configurazione)
  - [Optimizer AdamW](#optimizer-adamw) · [Scheduler OneCycleLR](#scheduler-onecyclelr) · [LoRA](#lora) · [Loss](#loss)
  - [Batch e durata](#batch-e-durata) · [Dataset, seed e prompt](#dataset-seed-e-prompt) · [Qwen e query di categoria](#qwen-e-query-di-categoria)
  - [Encoder ed embedding](#encoder-ed-embedding) · [MLP](#mlp) · [Modello di compatibilità CP](#modello-di-compatibilità-cp) · [Dispositivo e output](#dispositivo-e-output)
- [Flag della CLI](#flag-della-cli)
- [Preparazione dei dati](#preparazione-dei-dati)
- [Log, grafici e checkpoint](#log-grafici-e-checkpoint)
- [Riprendere un training](#riprendere-un-training)
- [Memoria GPU](#memoria-gpu)
- [Esempi di comando](#esempi-di-comando)
- [File del training e test](#file-del-training-e-test)

## Avvio rapido

Dalla root di Text2Outfit, nell'ambiente Python del progetto, eseguire questi
comandi in Bash o PowerShell.

```text
python -m pip install -r requirements.txt
python -m training.train_seed_to_outfit
```

L'avvio predefinito usa Qwen3 Embedding 8B, FashionCLIP, il subset
`nondisjoint` e le immagini e descrizioni originali. Per usare una cache,
passare `--embedding-root`. Con FashionSigLIP serve un checkpoint CP
compatibile. Gli esempi includono [FashionCLIP con cache](#training-fashionclip-con-cache)
e [FashionSigLIP con cache e CP locale](#training-fashionsiglip-con-cache-e-cp-locale).

Per scegliere le opzioni, consultare la [tabella dei flag](#flag-della-cli).
La CLI mostra lo stesso elenco con `python -m training.train_seed_to_outfit --help`.


## Iperparametri e configurazione

I valori predefiniti provengono da [`TrainingConfig`](config.py) e
[`ModelConfig`](../model/config.py). I flag valgono per il singolo run. Le
tabelle indicano quando un valore si imposta nel config Python o deriva dal
codice o dalle librerie. I percorsi relativi partono dalla cartella da cui
si lancia il comando.

### Optimizer AdamW

AdamW usa un unico gruppo per tutti i parametri allenabili, bias inclusi.
Dopo aver mediato i gradienti sul numero effettivo di outfit, il trainer
applica il clipping, aggiorna i pesi e avanza lo scheduler.

| Parametro | Default | Dove si imposta |
|---|---|---|
| Picco del learning rate (`learning_rate`) | `1e-4` | `--learning-rate`, passato a OneCycleLR come `max_lr` |
| Weight decay (`weight_decay`) | `0.01` | `--weight-decay` |
| Clipping L2 (`max_grad_norm`) | `1.0` | `TrainingConfig`. Il trainer genera un errore se la norma dei gradienti è infinita o NaN. |
| `betas` | β₁ tra `0.85` e `0.95`, β₂ `0.999` | OneCycleLR varia β₁ in senso inverso al LR. β₂ usa il default di AdamW. |
| `eps` | `1e-8` | Default di AdamW |
| `amsgrad`, `maximize` | `False` | Default di AdamW |
| `foreach`, `fused` | `None` | Selezione lasciata a PyTorch |
| `capturable`, `differentiable` | `False` | Default di AdamW |
| Azzeramento dei gradienti | `set_to_none=True` | Nel trainer, a inizio epoca e dopo ogni aggiornamento |

### Scheduler OneCycleLR

Lo scheduler avanza a ogni aggiornamento di AdamW e resta fermo in validation.
Con `--learning-rate 1e-4`, il LR parte da `4e-6`, sale a `1e-4` e scende
verso `4e-10`.

| Parametro | Default | Origine o calcolo |
|---|---|---|
| `max_lr` | `1e-4` | `--learning-rate`, picco del ciclo |
| `pct_start` | `0.3` | `ONE_CYCLE_PCT_START` in `config.py`. Il primo 30% del ciclo è la salita al picco. |
| `anneal_strategy` | `"cos"` | Interpolazione coseno impostata nel trainer |
| `div_factor` | `25.0` | `ONE_CYCLE_DIV_FACTOR` in `config.py`. LR iniziale = picco / fattore. |
| `final_div_factor` | `10000.0` | `ONE_CYCLE_FINAL_DIV_FACTOR` in `config.py`. LR minimo = LR iniziale / fattore. |
| `cycle_momentum` | `True` | Default di PyTorch |
| `base_momentum`, `max_momentum` | `0.85`, `0.95` | Limiti del ciclo di β₁, default di PyTorch |
| `three_phase` | `False` | Due fasi, salita e discesa. Default di PyTorch. |
| `total_steps` | `ceil(batch di train / accumulo) × epoche` | Calcolato una volta all'avvio |

### LoRA

| Parametro | Default | Dove si imposta |
|---|---|---|
| `lora_enabled` | `True` | `ModelConfig` |
| `lora_rank` | `64` | `--lora-rank` |
| `lora_alpha` | `128` | `--lora-alpha` |
| `lora_dropout` | `0.05` | `--lora-dropout` |
| `lora_target_modules` | `("q_proj", "k_proj", "v_proj")` | `ModelConfig` |
| Scala degli adattatori | `2` | `lora_alpha / lora_rank` |
| Bias del backbone | `bias="none"` | Impostato nell'adattatore LoRA |

### Loss

| Parametro | Default | Dove si imposta |
|---|---|---|
| `loss_weights` | `(1.0, 1.0, 1.0)` | `--loss-weights RET CP COMP`. I pesi devono essere finiti e ≥ 0, con almeno un peso positivo. |
| `temperature` | `0.07` | `--temperature`. Temperatura positiva dell'InfoNCE. |

```text
total = weight_ret * retrieval + weight_cp * cp + weight_comp * composition
composition = category + fine_grained
```

- **Retrieval** avvicina gli embedding predetti ai capi target e li distingue
  dai negativi.
- **CP** è la media degli score di compatibilità con segno invertito. Una
  loss che scende verso `-1` indica una compatibilità più alta.
- **Composizione** combina la presenza delle categorie (`category`) e la
  classificazione delle sottocategorie (`fine_grained`).

Le loss sommano i contributi per categoria o capo e li mediano sugli outfit.
Se più capi condividono una categoria, tutti contribuiscono a retrieval e
fine-grained usando lo stesso slot predetto. I pesi predefiniti delle loss
sono valori di partenza per il tuning in validation.

### Batch e durata

| Parametro | Default | Configurazione |
|---|---|---|
| Epoche pianificate (`epochs`) | `10` | `--epochs` |
| Outfit per batch (`batch_size`) | `2` | `--batch-size` |
| Batch di training per aggiornamento (`gradient_accumulation_steps`) | `4` | `--accumulation-steps` |
| Outfit per aggiornamento | `8` | Prodotto di batch size e accumulo per un gruppo completo |

I loader mantengono anche l'ultimo batch incompleto (`drop_last=False`).
L'ultimo gruppo può quindi contenere meno outfit. Il trainer divide sempre
i gradienti per il numero di outfit effettivamente elaborati nel gruppo.

### Dataset, seed e prompt

| Parametro | Default | Flag CLI |
|---|---|---|
| Cartella locale Polyvore (`dataset_root`) | `datasets/polyvore-outfits` | `--dataset-root` |
| Cache dei download (`dataset_cache`) | `.cache/datasets` | `--dataset-cache` |
| Variante Polyvore (`subset`) | `nondisjoint` | `--subset nondisjoint` o `--subset disjoint` |
| Cache delle feature (`embedding_root`) | `None`, usa immagini e descrizioni | `--embedding-root` |
| Negativi per capo (`negatives_per_item`) | `4` | `--negatives-per-item` |
| Modalità del capo seed (`seed_mode`) | `image` | `--seed-mode image` o `--seed-mode text` |
| Prompt comune (`prompt`) | `Complete a compatible outfit using this seed item.` | `--prompt` |
| Prompt per outfit (`prompts`) | `None`, usa il prompt comune | `--prompts`, file JSON con precedenza sul prompt comune |

### Qwen e query di categoria

| Parametro | Default | Dove si imposta |
|---|---|---|
| Checkpoint LLM (`llm_model_name`) | `Qwen/Qwen3-Embedding-8B` | `--llm` |
| Cache LLM (`llm_cache`) | `.cache/llm` | `--llm-cache` |
| Precisione del backbone (`llm_dtype`) | `"auto"` | `ModelConfig`. Usa BF16 su CUDA se supportato, altrimenti FP32. Accetta anche `"float32"` e `"bfloat16"`. |
| Aggiornamento del backbone (`llm_trainable`) | `False` | `ModelConfig`. Il fine-tuning completo richiede `lora_enabled=False`. |
| Dimensione delle query (`category_embedding_dim`) | `4096` | `--category-embedding-dim`. Deve coincidere con `hidden_size` del LLM. |
| `gradient_checkpointing` | `True` | `--gradient-checkpointing` o `--no-gradient-checkpointing`. Quando attivo usa `use_reentrant=False`. |

Le 11 query seguono l'ordine `tops`, `bottoms`, `all-body`, `outerwear`,
`shoes`, `bags`, `jewelry`, `accessories`, `sunglasses`, `hats`, `scarves`.
Qwen usa SDPA con `use_cache=False`. Le query vedono il prompt, il seed
e le altre query. LoRA e MLP usano FP32.

### Encoder ed embedding

| Parametro | Default | Dove si imposta |
|---|---|---|
| Famiglia (`encoder`) | `fashionclip` | `--encoder fashionclip` o `--encoder siglip` |
| Checkpoint (`encoder_model_name`) | `None`, risolto in `patrickjohncyh/fashion-clip` o `Marqo/marqo-fashionSigLIP` | `--encoder-model`. Deve coincidere con l'encoder della cache. |
| Aggiornamento encoder (`encoder_trainable`) | `False` | `ModelConfig`. Il modello completo richiede encoder congelati. |
| Feature visuali / testuali (`V` / `T`) | `512 / 512` con FashionCLIP, `768 / 768` con FashionSigLIP | Determinate dagli encoder o dal manifest della cache |
| Dimensione item e retrieval (`retrieval_dim`, `D`) | `None`, risolto in `1024` con FashionCLIP o `1536` con FashionSigLIP | `--retrieval-dim`. Deve coincidere con `model_dim` del CP. |

### MLP

| Famiglia | Campo in `ModelConfig` | Strati nascosti | Ingresso → nascosto → uscita |
|---|---|---|---|
| Seed immagine | `seed_hidden_dims` | `(512,)` | `V → 512 → H` |
| Item del catalogo | `item_hidden_dims` | `(512,)` | `(V + T) → 512 → D` |
| Retrieval | `retrieval_hidden_dims` | `(512,)` | `H → 512 → D` |
| Presenza categoria | `category_hidden_dims` | `(512,)` | `H → 512 → 1` |
| Sottocategoria | `fine_grained_hidden_dims` | `(512,)` | `H → 512 → Cᵢ` |

`H` è la larghezza del LLM e `Cᵢ` è il numero di sottocategorie della
categoria `i`, ricavato dai metadati di train e validation e salvato nel
checkpoint. Con seed immagine ci sono 35 MLP indipendenti, una per il seed,
una per gli item e tre per ciascuna delle 11 categorie. Con seed testuale la
descrizione entra nel prompt.

`--hidden-dim N` imposta un singolo strato nascosto da `N` unità in tutte
e cinque le famiglie. Nel config Python una tupla vuota `()` produce una
sola proiezione lineare. Gli strati nascosti usano `Linear → ReLU`, l'uscita
è lineare e i bias sono attivi. Le MLP operano senza dropout o normalizzazione interna.

### Modello di compatibilità CP

| Parametro | Default | Flag CLI |
|---|---|---|
| Codice OutfitTransformer (`cp_repo`) | `.cache/outfit_transformer` | `--cp-repo` |
| Checkpoint locale (`cp_checkpoint`) | `None`, risolto dal file dei pesi | `--cp-checkpoint` |
| Sorgente del codice (`cp_source_url`) | [ZIP OutfitTransformer, ramo `main-branch`](https://codeload.github.com/FlorindoDev/OutfitTransformer/zip/refs/heads/main-branch) | `--cp-source-url` |
| Repository dei pesi (`cp_weights_repo`) | [`FlorindoDev/OutfitTransformer-weights`](https://huggingface.co/FlorindoDev/OutfitTransformer-weights) | `--cp-weights-repo` |
| File dei pesi (`cp_weights_file`) | `checkpoints/nondisjoint/cp_precomputed-flashionclip/best.pt` | `--cp-weights-file` |
| Cache dei pesi (`cp_cache`) | `.cache/huggingface-cp` | `--cp-cache` |

Il CP riceve gli embedding retrieval predetti e la maschera delle categorie
target. I gradienti lo attraversano e allenano il generatore. I suoi pesi
restano congelati e gli embedding entrano direttamente nel CP. La dimensione
richiesta dal checkpoint è `1024` per FashionCLIP o `1536` per FashionSigLIP.

Il codice CP viene recuperato dallo ZIP se manca il package `model/`. Per
i pesi si cerca prima il file locale sotto `--cp-repo`, poi nella cache Hugging
Face, infine si scarica il solo checkpoint selezionato. `--cp-weights-file`
richiede un percorso relativo che termina con `best.pt`. Un
`--cp-checkpoint` esplicito deve esistere. Il loader verifica i pesi e le
dimensioni. I formati supportati sono nella [guida del modello](../model/README.md#caricamento-automatico-del-cp).

Con `--encoder siglip` indicare il CP da `1536` con `--cp-checkpoint` o
`--cp-weights-file`. `--subset disjoint` seleziona dati e cache. Per usare
anche il CP disjoint, indicarne esplicitamente il checkpoint o il file remoto.

### Dispositivo e output

| Parametro | Default | Flag CLI |
|---|---|---|
| Dispositivo (`device`) | `auto`, CUDA se disponibile, altrimenti CPU | `--device`, ad esempio `cpu` o `cuda:0` |
| Seed casuale (`seed`) | `42` | `--seed` |
| Worker dei DataLoader (`num_workers`) | `0`, caricamento nel processo principale | `--num-workers` |
| Cartella del run (`output_dir`) | `checkpoints/seed_to_outfit` | `--output-dir` |
| Checkpoint da riprendere (`resume`) | `None`, nuovo run | `--resume` |

Il seed inizializza Python, PyTorch e il loader di training e determina i
negativi della validation. Il loader mescola solo i dati di training.
Entrambi i loader usano `pin_memory=False`.

## Flag della CLI

La tabella riporta tutti i flag di `python -m training.train_seed_to_outfit`.
I valori `None` indicano che il flag è omesso e si applica il comportamento
descritto nella riga. La definizione dei flag è in [`train_seed_to_outfit.py`](train_seed_to_outfit.py).

| Flag | Default | Cosa fa |
|---|---|---|
| `-h`, `--help` | — | Mostra l'aiuto della CLI ed esce. |
| `--dataset-root` | `datasets/polyvore-outfits` | Indica la cartella locale di Polyvore. Cerca le risorse mancanti in cache e poi le scarica. |
| `--dataset-cache` | `.cache/datasets` | Indica la cartella di cache per i download del dataset. |
| `--subset` | `nondisjoint` | Seleziona la variante `nondisjoint` o `disjoint` per dati e cache. |
| `--embedding-root` | `None` | Usa feature precomputate dalla cartella indicata. Se omesso, parte da immagini e descrizioni originali. |
| `--encoder` | `fashionclip` | Seleziona gli encoder immagine e testo, `fashionclip` o `siglip`. |
| `--encoder-model` | `None` | Imposta il checkpoint dell'encoder. Se omesso, usa `patrickjohncyh/fashion-clip` o `Marqo/marqo-fashionSigLIP` in base a `--encoder`. |
| `--llm` | `Qwen/Qwen3-Embedding-8B` | Seleziona il checkpoint LLM tramite ID o percorso locale. |
| `--llm-cache` | `.cache/llm` | Indica la cache del checkpoint LLM. |
| `--category-embedding-dim` | `4096` | Imposta la dimensione delle query di categoria, pari a `hidden_size` del LLM. |
| `--cp-repo` | `.cache/outfit_transformer` | Indica la cartella del codice OutfitTransformer, che contiene `model/`. |
| `--cp-checkpoint` | `None` | Usa un checkpoint CP locale esplicito. Se omesso, cerca `--cp-weights-file` sotto `--cp-repo`, poi in cache e nel repository dei pesi. |
| `--cp-source-url` | [`https://codeload.github.com/FlorindoDev/OutfitTransformer/zip/refs/heads/main-branch`](https://codeload.github.com/FlorindoDev/OutfitTransformer/zip/refs/heads/main-branch) | Seleziona l'archivio ZIP da cui recuperare il codice CP. |
| `--cp-weights-repo` | `FlorindoDev/OutfitTransformer-weights` | Seleziona il repository Hugging Face dei pesi CP. |
| `--cp-weights-file` | `checkpoints/nondisjoint/cp_precomputed-flashionclip/best.pt` | Seleziona il file dei pesi CP tramite un percorso relativo che termina con `best.pt`. |
| `--cp-cache` | `.cache/huggingface-cp` | Indica la cache Hugging Face dei pesi CP. |
| `--output-dir` | `checkpoints/seed_to_outfit` | Indica dove salvare configurazione, metriche, checkpoint e grafici. |
| `--resume` | `None` | Riparte da un checkpoint salvato, ripristinando pesi allenabili, optimizer, scheduler, storico e stati casuali. Se omesso, avvia un nuovo run. |
| `--epochs` | `10` | Imposta il totale di epoche per OneCycleLR. Deve coincidere con il run originale alla ripresa. |
| `--batch-size` | `2` | Imposta il numero di outfit per batch in train e validation. |
| `--accumulation-steps` | `4` | Imposta il numero di batch di training per aggiornamento dei pesi. |
| `--learning-rate` | `1e-4` | Imposta il picco del learning rate nel ciclo OneCycleLR. |
| `--weight-decay` | `0.01` | Imposta il weight decay di AdamW. |
| `--negatives-per-item` | `4` | Imposta quanti capi negativi distinti campionare per ogni capo target, dalla stessa categoria e dallo stesso split. |
| `--retrieval-dim` | `None` | Imposta la dimensione condivisa da retrieval, item e ingresso CP. Se omesso, usa `1024` con FashionCLIP o `1536` con FashionSigLIP. |
| `--hidden-dim` | `None` | Imposta uno strato nascosto della stessa larghezza in tutte le MLP. Se omesso, usa `ModelConfig`, con uno strato da `512` per famiglia. |
| `--seed-mode` | `image` | Usa l'immagine o le feature visuali con `image`. Con `text`, aggiunge la descrizione del seed al prompt. |
| `--prompt` | `Complete a compatible outfit using this seed item.` | Imposta il prompt comune agli outfit. |
| `--prompts` | `None` | Legge i prompt per outfit da un JSON che copre train e validation. Ha precedenza su `--prompt`. |
| `--loss-weights RET CP COMP` | `1.0 1.0 1.0` | Imposta i pesi di retrieval, compatibilità e composizione, in quest'ordine. |
| `--temperature` | `0.07` | Imposta la temperatura della loss InfoNCE di retrieval. |
| `--lora-rank` | `64` | Imposta il rango degli adattatori LoRA. |
| `--lora-alpha` | `128` | Imposta il fattore di scala LoRA, applicato come `alpha / rank`. |
| `--lora-dropout` | `0.05` | Imposta la probabilità di dropout negli adattatori LoRA. |
| `--gradient-checkpointing`, `--no-gradient-checkpointing` | `True` | Attiva o disattiva il ricalcolo delle attivazioni del LLM durante il backward, che riduce l'uso di memoria. |
| `--seed` | `42` | Inizializza la casualità di Python, PyTorch e del loader di training. Determina anche i negativi di validation. |
| `--num-workers` | `0` | Imposta i processi worker dei DataLoader. Con `0`, il processo principale carica i dati. |
| `--device` | `auto` | Seleziona il dispositivo, ad esempio `cpu` o `cuda:0`. Con `auto`, usa CUDA se disponibile, altrimenti CPU. |

## Preparazione dei dati

### Outfit, seed e prompt

Il training usa solo outfit positivi (`label=1`). I negativi sono capi
distinti della stessa categoria, esterni all'outfit e presenti nello stesso
split. Il capo seed viene ricampionato quando si legge un outfit in training.
In validation la scelta del seed e dei negativi è deterministica.

`--seed-mode image` usa l'immagine del seed o la sua feature visuale in cache.
`--seed-mode text` aggiunge al prompt solo la descrizione del seed.

Per richieste su stagione o occasione, preparare prompt coerenti con gli
outfit target. `--prompts` legge un oggetto JSON che deve coprire gli outfit
positivi caricati in train e validation. Gli ID includono il numero di riga
nei file ufficiali.

```json
{
  "compatibility_train:1": "Create a casual summer outfit.",
  "compatibility_valid:1": "Create a formal winter outfit."
}
```

### Feature precomputate

`python -m scripts.precompute_embeddings` produce feature prima della MLP
item, concatenando l'embedding visuale e quello testuale, normalizzati
separatamente. `--embedding-root` punta alla cartella che contiene il subset.
Esempio per FashionCLIP con `--subset nondisjoint`.

```text
precomputed_embeddings/patrickjohncyh-fashion-clip/
└── nondisjoint/
    ├── train/manifest.json
    └── validation/manifest.json
```

Il training verifica subset, split, dimensioni, encoder, corrispondenza
del fingerprint tra gli split e copertura degli ID. Legge gli shard tramite
memory mapping e salta il caricamento degli encoder e la decodifica delle
immagini. Gli item passano comunque per la MLP item e il seed immagine usa
la metà visuale delle feature prima della sua MLP.

La cache resta utilizzabile se si cambiano le dimensioni di uscita della MLP.
Il wrapper del modello accetta item originali e precomputati nello stesso
batch, mentre tutti i seed immagine di un batch devono usare la stessa
modalità.

## Log, grafici e checkpoint

I log annunciano l'inizio di ogni fase e riportano l'avanzamento ogni 50
microbatch e all'ultimo. `microbatch` conta i batch del loader. `loss` indica
la loss totale del batch e `lr` è il learning rate corrente. Il riepilogo a
fine epoca riporta le medie di train e validation.

```text
2026-11-16 12:20:37,325 | INFO | epoch=1/10 phase=train batches=1250
2026-11-16 12:21:15,100 | INFO | epoch=1 microbatch=50/1250 loss=12.345678 lr=4.038e-06 phase=train
```

| File nella cartella del run | Contenuto |
|---|---|
| `config.json` | Configurazione del run e del modello, dimensioni delle MLP e parametri OneCycleLR |
| `architecture.txt` | Dimensioni delle MLP e conteggi dei parametri totali e allenabili, riportati anche all'avvio |
| `history.json` | Valori medi di tutte le loss per train e validation |
| `epochs/seed_to_outfit_epoch_001.pt` | Un checkpoint per epoca |
| `best.pt` | Checkpoint con la loss totale di validation più bassa |
| `plots/losses_epoch_001.png` | Pannello cumulativo delle loss di train e validation |
| `plots/<loss>_loss_epoch_001.png` | Sei grafici individuali per `total`, `retrieval`, `cp`, `composition`, `category` e `fine_grained` |

I salvataggi avvengono a fine epoca. Il trainer esegue tutte le epoche
pianificate, senza early stopping. I checkpoint Text2Outfit
(`checkpoint_schema_version=2`) contengono i pesi allenabili
(`model_state_dict`), gli stati di optimizer, scheduler e generatori casuali,
l'epoca, il global step, lo storico, le metriche e le configurazioni.
Si caricano con `weights_only=True`. Per ricostruire il modello servono anche
i checkpoint base del run, inclusi Qwen, CP e l'encoder quando richiesto.

## Riprendere un training

Mantenere gli argomenti del run originale e aggiungere `--resume` con il
percorso dell'ultimo checkpoint completo. `--epochs` indica il totale
pianificato da OneCycleLR. Un run da 10 epoche ripreso dal checkpoint
dell'epoca 5 esegue le epoche dalla 6 alla 10. [Esempio di comando](#ripresa-del-training).

| Impostazioni da mantenere | Motivo |
|---|---|
| Epoche pianificate e numero di batch per epoca | Determinano la durata del ciclo OneCycleLR. |
| Batch size, accumulo, picco LR, weight decay e clipping | Mantengono la configurazione degli aggiornamenti. |
| Modello, CP, tassonomia, dataset, subset, cache e prompt | Devono corrispondere al run salvato. Il checksum del CP viene verificato. |

Se la cartella contiene già epoche successive al checkpoint scelto, usare
un nuovo `--output-dir`. Per riprodurre il run mantenere anche il numero di
worker, l'ambiente e i file di dati e checkpoint. I checkpoint senza stato
OneCycleLR (schema 1) e quelli con embedding da 256 e adattatore CP
richiedono un nuovo run.

## Memoria GPU

Qwen 8B deve elaborare ogni batch anche con LoRA e feature in cache. In caso
di `CUDA out of memory`, ridurre `--batch-size` e tenere attivo il gradient
checkpointing. Aumentare `--accumulation-steps` per mantenere il batch
effettivo desiderato. Con OneCycleLR un cambio di batch size richiede un
nuovo run. Il training completo di Qwen 8B su CPU resta da verificare.

All'avvio il trainer verifica le cache e carica dati e pesi. La tabella delle
MLP segnala che il modello è stato costruito. La prima riga `phase=train`
annuncia l'inizio del training.

## Esempi di comando

Eseguire dalla root del progetto e sostituire i percorsi con quelli del
proprio ambiente.

### Preparazione separata del CP

Prepara il CP FashionCLIP predefinito senza avviare il training. Il comando
è lo stesso in Bash e PowerShell.

```text
python -m model.compatibility
```

### Training con CP locale

**Bash**

```bash
python -m training.train_seed_to_outfit \
  --cp-repo "percorso/OutfitTransformer" \
  --cp-checkpoint "percorso/checkpoint-cp/best.pt" \
  --epochs 10
```

**PowerShell**

```powershell
python -m training.train_seed_to_outfit `
  --cp-repo "percorso/OutfitTransformer" `
  --cp-checkpoint "percorso/checkpoint-cp/best.pt" `
  --epochs 10
```

### Training FashionCLIP con cache

**Bash**

```bash
python -m training.train_seed_to_outfit \
  --embedding-root precomputed_embeddings/patrickjohncyh-fashion-clip \
  --output-dir checkpoints/seed_cached --epochs 10
```

**PowerShell**

```powershell
python -m training.train_seed_to_outfit `
  --embedding-root precomputed_embeddings/patrickjohncyh-fashion-clip `
  --output-dir checkpoints/seed_cached --epochs 10
```

### Training FashionCLIP disjoint con cache

Usa `precomputed/disjoint/` con i manifest degli split `train` e
`validation` e seleziona esplicitamente il CP FashionCLIP disjoint.

**Bash**

```bash
python -m training.train_seed_to_outfit \
  --subset disjoint \
  --embedding-root precomputed \
  --cp-weights-file checkpoints/disjoint/cp_precomputed-flashionclip/best.pt \
  --output-dir checkpoints/seed_disjoint_cached --epochs 10
```

**PowerShell**

```powershell
python -m training.train_seed_to_outfit `
  --subset disjoint `
  --embedding-root precomputed `
  --cp-weights-file checkpoints/disjoint/cp_precomputed-flashionclip/best.pt `
  --output-dir checkpoints/seed_disjoint_cached --epochs 10
```

### Training FashionSigLIP con cache e CP locale

L'esempio usa OutfitTransformer accanto a Text2Outfit e il checkpoint CP
locale dell'epoca 118. Gli embedding item e retrieval hanno dimensione `1536`.

**Bash**

```bash
python -m training.train_seed_to_outfit \
  --encoder siglip \
  --embedding-root ../OutfitTransformer/huggingface/precomputed_embeddings/fashionSigLIP \
  --cp-repo ../OutfitTransformer \
  --cp-checkpoint ../OutfitTransformer/checkpoints/cp_precomputed_fashionSigLIP/epochs/cp_epoch_118.pt \
  --output-dir checkpoints/seed_siglip --epochs 10
```

**PowerShell**

```powershell
python -m training.train_seed_to_outfit `
  --encoder siglip `
  --embedding-root ../OutfitTransformer/huggingface/precomputed_embeddings/fashionSigLIP `
  --cp-repo ../OutfitTransformer `
  --cp-checkpoint ../OutfitTransformer/checkpoints/cp_precomputed_fashionSigLIP/epochs/cp_epoch_118.pt `
  --output-dir checkpoints/seed_siglip --epochs 10
```

### Ripresa del training

Riprende FashionCLIP con cache dal checkpoint dell'epoca 5. Se sono già
presenti checkpoint delle epoche 6–10, cambiare `--output-dir`.

**Bash**

```bash
python -m training.train_seed_to_outfit \
  --embedding-root precomputed_embeddings/patrickjohncyh-fashion-clip \
  --output-dir checkpoints/seed_cached --epochs 10 \
  --resume checkpoints/seed_cached/epochs/seed_to_outfit_epoch_005.pt
```

**PowerShell**

```powershell
python -m training.train_seed_to_outfit `
  --embedding-root precomputed_embeddings/patrickjohncyh-fashion-clip `
  --output-dir checkpoints/seed_cached --epochs 10 `
  --resume checkpoints/seed_cached/epochs/seed_to_outfit_epoch_005.pt
```

## File del training e test

- [`config.py`](config.py) contiene i default del training e di OneCycleLR.
- [`train_seed_to_outfit.py`](train_seed_to_outfit.py) definisce la CLI e prepara il run.
- [`data.py`](data.py) gestisce cache, prompt, seed e vocabolario.
- [`trainer.py`](trainer.py) gestisce aggiornamenti, validation, salvataggio e ripresa.
- [`plots.py`](plots.py) genera i grafici cumulativi.

La suite locale usa un Qwen piccolo e copre LoRA, passaggio dei gradienti
attraverso il CP congelato, aggiornamento delle MLP, equivalenza degli input
originali e in cache, categorie ripetute, ripresa dai checkpoint e grafici.

```text
python -m unittest discover -s tests
```
