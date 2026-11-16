# Training seed-to-outfit

Il modulo allena insieme LoRA, le 11 query, la MLP del seed, le tre teste
per categoria e la MLP degli item. Retrieval e item producono embedding da
1024 componenti con FashionCLIP e 1536 con FashionSigLIP, passati direttamente
al CP corrispondente senza adattatore. I backbone Qwen,
FashionCLIP/FashionSigLIP e il modello CP restano congelati; su Qwen si
aggiornano soltanto gli adattatori LoRA nelle proiezioni Q/K/V.

| File | Responsabilità |
|---|---|
| `config.py` | `TrainingConfig` e default di optimizer, batch, epoche, dati e output. |
| `train_seed_to_outfit.py` | CLI: dataset, cache, modello, loader e avvio del training. |
| `data.py` | Lettura delle cache esistenti, prompt, seed e vocabolario delle sottocategorie. |
| `trainer.py` | AdamW, accumulo dei gradienti, validation, checkpoint e ripresa. |
| `plots.py` | Grafici cumulativi con curve train/validation nello stile di OutfitTransformer. |

Il modello completo è in [`model/outfit/text2outfit.py`](../model/outfit/text2outfit.py);
la costruzione di tutti i moduli e delle MLP è in
[`model/outfit/build.py`](../model/outfit/build.py). Il caricamento CP è in
[`model/compatibility/scorer.py`](../model/compatibility/scorer.py). Il training non
istanzia reti: usa `build_model()` e `model.compute_losses(batch)`.

## Avvio

[`config.py`](config.py) contiene `TrainingConfig` e `DEFAULT_TRAINING_CONFIG`
per optimizer, batch, epoche, dati, runtime e output. `ModelConfig` resta in
[`model/config.py`](../model/config.py), con architettura, dimensioni delle
MLP, backbone, LoRA e loss. I flag della CLI sovrascrivono i config per una
singola esecuzione. La [tabella delle MLP](../model/README.md#modello-completo-e-dimensioni-delle-mlp)
mostra input, layer nascosti, output e conteggio dei parametri.

`--category-embedding-dim` sovrascrive la larghezza delle query di categoria
(default `4096`). Deve coincidere con `hidden_size` del modello scelto con
`--llm`; viene verificato prima di costruire LoRA e MLP.

I comandi di installazione e training sono raccolti negli
[esempi in fondo alla pagina](#esempi-di-comando), ciascuno nelle versioni
Bash e PowerShell. Vanno eseguiti dalla root del progetto, nell'ambiente Python.

Il default usa Qwen3 Embedding 8B, FashionCLIP, seed immagine, batch size 2,
4 microbatch per aggiornamento e gradient checkpointing. Usa CUDA se
disponibile, altrimenti CPU. Su CUDA con supporto BF16, solo il backbone
Qwen usa BF16; adattatori e MLP restano FP32. Negli altri casi usa FP32.
Il checkpoint 8B richiede una macchina con memoria adeguata: CPU è utile
per i test con modelli piccoli, non è una configurazione di training 8B verificata.

Il CP viene preparato automaticamente: il codice modello mancante viene
scaricato da [GitHub](https://github.com/FlorindoDev/OutfitTransformer), i pesi
da [Hugging Face](https://huggingface.co/FlorindoDev/OutfitTransformer-weights).
Viene scaricato soltanto `checkpoints/nondisjoint/cp_precomputed-flashionclip/best.pt`;
i file locali e la cache vengono riutilizzati prima di accedere alla rete.
È possibile preparare il CP separatamente con il relativo
[esempio di comando](#preparazione-separata-del-cp).
Questo checkpoint predefinito è per FashionCLIP. Con `--encoder siglip`
specificare il CP FashionSigLIP da 1536 componenti tramite `--cp-checkpoint`
oppure `--cp-weights-file`; il codice rifiuta il default FashionCLIP.

Per usare un CP già presente, specificare i due percorsi con questi flag:

| Flag | Percorso da indicare |
|---|---|
| `--cp-repo` | Cartella del repository OutfitTransformer, contenente il package `model/`. |
| `--cp-checkpoint` | File `.pt` del modello CP già addestrato, per esempio `best.pt`. |

Sostituire i percorsi dell'[esempio con CP locale](#training-con-cp-locale)
con quelli del proprio ambiente. Possono
essere assoluti oppure relativi alla cartella da cui si avvia il comando.

I flag `--cp-source-url`, `--cp-weights-repo`, `--cp-weights-file` e `--cp-cache`
sovrascrivono le sorgenti e la cache del config. Il file remoto deve essere
un `best.pt`; un `--cp-checkpoint` locale esplicito deve esistere.
I [default CP](../model/README.md#caricamento-automatico-del-cp) sono in `ModelConfig`.

Il loader supporta anche i checkpoint con Transformer comune, LayerNorm
finali e token CP non normalizzato, caricando tutti i pesi con controlli
stretti. Supporta il formato CP corrente senza Transformer comune.
I vecchi checkpoint ResNet con chiavi `encoder.image_encoder` non sono
compatibili e vengono rifiutati con un errore esplicito.

Il CP valuta direttamente gli embedding retrieval predetti, usando la maschera
di presenza target. I suoi pesi restano in `eval()`, ma i gradienti attraversano
lo scorer fino al generatore: questo ramo non usa `no_grad()`.
`--retrieval-dim` deve coincidere con `model_dim` del checkpoint CP. Se omesso,
il default è `1024` con FashionCLIP e `1536` con FashionSigLIP, sia per la MLP
item sia per tutte le teste retrieval. Un checkpoint con una dimensione diversa
viene rifiutato.

## Flag della CLI

Tutti i flag della CLI di training sono riportati qui
sotto, inclusi help e disattivazione del gradient checkpointing. I default
provengono da [`TrainingConfig`](config.py), [`ModelConfig`](../model/config.py)
e dal [parser della CLI](train_seed_to_outfit.py). I percorsi relativi partono
dalla cartella da cui si esegue il comando; `None` indica che non è stato
specificato un valore, con il comportamento descritto nella tabella.

| Flag | Cosa fa | Valore di default |
|---|---|---|
| `-h`, `--help` | Mostra i flag disponibili ed esce senza avviare il training. | — |
| `--dataset-root` | Cartella locale di Polyvore; le risorse mancanti vengono cercate in cache e poi scaricate. | `datasets/polyvore-outfits` |
| `--dataset-cache` | Cartella di cache dei download del dataset. | `.cache/datasets` |
| `--subset` | Variante Polyvore: `nondisjoint` oppure `disjoint`. | `nondisjoint` |
| `--embedding-root` | Cartella delle feature precomputate prima della MLP item, contenente `SUBSET/train/manifest.json` e `SUBSET/validation/manifest.json`. Se specificata, evita il caricamento degli encoder e la decodifica delle immagini. | `None` — usa immagini e descrizioni originali. |
| `--encoder` | Famiglia degli encoder visuale e testuale: `fashionclip` oppure `siglip`. | `fashionclip` |
| `--encoder-model` | Nome del checkpoint degli encoder; deve corrispondere a quello delle eventuali feature precomputate. | `None` — `patrickjohncyh/fashion-clip` con `fashionclip`, `Marqo/marqo-fashionSigLIP` con `siglip`. |
| `--llm` | Nome o percorso locale del checkpoint LLM. | `Qwen/Qwen3-Embedding-8B` |
| `--llm-cache` | Cartella di cache del checkpoint LLM. | `.cache/llm` |
| `--category-embedding-dim` | Dimensione delle 11 query di categoria; deve coincidere con `hidden_size` del LLM selezionato. | `4096` |
| `--cp-repo` | Cartella del codice OutfitTransformer, contenente `model/`; se assente, viene preparata dalla sorgente ZIP. | `.cache/outfit_transformer` |
| `--cp-checkpoint` | Percorso di un checkpoint CP locale già addestrato; se esplicito, il file deve esistere. | `None` — cerca `--cp-repo/--cp-weights-file`, poi usa la cache o scarica i pesi. |
| `--cp-source-url` | URL dell'archivio ZIP da cui recuperare il codice CP mancante. | `https://codeload.github.com/FlorindoDev/OutfitTransformer/zip/refs/heads/main-branch` |
| `--cp-weights-repo` | Repository Hugging Face dei pesi CP. | `FlorindoDev/OutfitTransformer-weights` |
| `--cp-weights-file` | Percorso relativo del checkpoint nel repository dei pesi CP; deve terminare con `best.pt`. | `checkpoints/nondisjoint/cp_precomputed-flashionclip/best.pt` |
| `--cp-cache` | Cartella di cache Hugging Face dei pesi CP. | `.cache/huggingface-cp` |
| `--output-dir` | Cartella del run: configurazione, architettura, metriche, checkpoint e grafici. | `checkpoints/seed_to_outfit` |
| `--resume` | Checkpoint di training da cui ripristinare parametri allenabili, optimizer, storico e stati RNG. | `None` — avvia un nuovo training. |
| `--epochs` | Numero totale di epoche da raggiungere, anche quando si riprende un training. | `10` |
| `--batch-size` | Numero di outfit per batch nei loader di training e validation. | `2` |
| `--accumulation-steps` | Numero di microbatch su cui accumulare i gradienti prima di un aggiornamento dell'optimizer. | `4` |
| `--learning-rate` | Learning rate di AdamW per i parametri allenabili. | `0.0001` |
| `--weight-decay` | Weight decay di AdamW. | `0.01` |
| `--negatives-per-item` | Numero di articoli negativi distinti per item, della stessa categoria e dello stesso split, esterni all'outfit positivo. | `4` |
| `--retrieval-dim` | Dimensione comune degli embedding retrieval e item; deve coincidere con `model_dim` del CP. | `None` — automatico: `1024` FashionCLIP / `1536` FashionSigLIP. |
| `--hidden-dim` | Sostituisce gli strati nascosti di tutte le famiglie MLP con un solo layer della larghezza indicata. | `None` — conserva le dimensioni di `ModelConfig`, inizialmente un layer nascosto da `512` per ogni famiglia. |
| `--seed-mode` | Modalità del seed: `image` usa l'immagine o le feature visuali cached; `text` aggiunge la descrizione al prompt. | `image` |
| `--prompt` | Prompt comune a tutti gli outfit quando non si specifica `--prompts`. | `Complete a compatible outfit using this seed item.` |
| `--prompts` | File JSON che associa gli ID degli outfit ai prompt; deve coprire gli outfit positivi caricati di train e validation e ha precedenza su `--prompt`. | `None` — usa il prompt comune. |
| `--loss-weights RET CP COMP` | Tre pesi, nell'ordine retrieval, compatibilità e composizione. Devono essere finiti, non negativi e almeno uno deve essere positivo. | `1.0 1.0 1.0` |
| `--temperature` | Temperatura positiva della loss InfoNCE di retrieval. | `0.07` |
| `--lora-rank` | Rango degli adattatori LoRA nelle proiezioni Q/K/V del LLM. | `64` |
| `--lora-alpha` | Fattore alpha di LoRA; insieme al rango determina la scala degli adattatori. | `128` |
| `--lora-dropout` | Probabilità di dropout nei rami LoRA. | `0.05` |
| `--gradient-checkpointing`, `--no-gradient-checkpointing` | Abilita o disabilita il gradient checkpointing del LLM: riduce la memoria ricalcolando attivazioni durante il backward. Non richiede un valore dopo il flag. | `True` — attivo; `--no-gradient-checkpointing` lo disattiva. |
| `--seed` | Seed casuale di Python, PyTorch e del loader; determina anche i negativi riproducibili della validation. | `42` |
| `--num-workers` | Numero di processi worker dei DataLoader; `0` carica i dati nel processo principale. | `0` |
| `--device` | Dispositivo PyTorch, per esempio `cpu`, `cuda` o `cuda:0`; `auto` sceglie CUDA se disponibile, altrimenti CPU. | `auto` |

Per consultare l'help della versione corrente, usare il relativo
[esempio di comando](#help-della-cli).

## Item già precomputati

La cache deve contenere le feature **prima della MLP item**:

```text
[embedding visuale normalizzato, embedding testuale normalizzato]
```

È il formato prodotto da `scripts.precompute_embeddings`. La cartella passata
a `--embedding-root` è quella del modello, con entrambi gli split:

```text
precomputed_embeddings/patrickjohncyh-fashion-clip/
└── nondisjoint/
    ├── train/manifest.json
    └── validation/manifest.json
```

Il training verifica split, dimensioni, modello, fingerprint e copertura degli
ID. Gli shard vengono indicizzati e letti tramite memory mapping; non si
caricano in GPU tutti gli embedding del catalogo. In questa modalità non
vengono caricati gli encoder né decodificate le immagini. Il seed immagine
usa la metà visuale della cache e attraversa comunque la propria MLP.

Per Marqo usare `--encoder siglip`, la cache generata dal relativo modello e
un CP FashionSigLIP da 1536 componenti. Gli esempi in fondo includono
[FashionCLIP con cache](#training-fashionclip-con-cache) e
[FashionSigLIP con cache e CP locale](#training-fashionsiglip-con-cache-e-cp-locale).
Non serve impostare `--retrieval-dim 1536`: è già automatico con `siglip`.
La MLP seed diventa `768 → 512 → 4096`, la MLP item `1536 → 512 → 1536`
e le teste retrieval `4096 → 512 → 1536` con gli altri default invariati.
Le cache restano feature prima della MLP: cambiare l'uscita della MLP
non richiede di rigenerarle. Senza `W_cp`, le teste retrieval devono imparare
embedding adatti anche al CP congelato; verificare questo allineamento in validation.

La stessa possibilità è disponibile direttamente nell'API:

```python
encoder = MultimodalEncoder(None, None, MLP(input_dim=1024, output_dim=1024, hidden_dims=(512,)))
items = encoder(precomputed=features)  # [N, 1024] -> [N, 1024]
```

Con gli encoder presenti resta valida `encoder(images, descriptions)`.
Le feature precomputate non saltano mai la MLP, che resta allenabile.
Il wrapper di training gestisce anche batch con item raw e cached misti;
i seed immagine di uno stesso batch devono invece usare tutti immagini
oppure tutti feature cached.

## Dati, prompt e loss

Si usano solo outfit con `label=1` come target della generazione. I negativi
sono articoli distinti della stessa categoria, esterni all'outfit e appartenenti
allo split richiesto. Non vengono usati outfit con label 0 come positivi.

Il seed viene campionato dall'outfit durante il training. In validation seed
e negativi sono deterministici. `--seed-mode text` usa la descrizione del seed;
il default `image` usa l'immagine o le sue feature visuali precomputate.

Il prompt predefinito è neutro: `Complete a compatible outfit using this seed item.`
Per il controllo di stagione, occasione e stile servono prompt annotati coerenti
con gli outfit; non vengono inventati dai metadati né generate annotazioni con LLM.
`--prompt` imposta un prompt comune. `--prompts prompts.json` accetta un oggetto
JSON con un prompt per outfit, per entrambi gli split:

```json
{
  "compatibility_train:1": "Create a casual summer outfit.",
  "compatibility_valid:1": "Create a formal winter outfit."
}
```

Gli ID includono il numero di riga nei file ufficiali; fornire tutte le righe
positive che verranno caricate. Il testo dei capi target non viene aggiunto
automaticamente al prompt. Il vocabolario delle sottocategorie è la tassonomia
fissa ricavata dai metadati dei cataloghi, ordinata e salvata nel checkpoint;
la validation non introduce una codifica diversa.

La loss totale segue il ramo seed-to-outfit:

```text
total = weight_ret * retrieval + weight_cp * cp + weight_comp * composition
composition = category + fine_grained
```

`--loss-weights RET CP COMP` cambia i tre pesi, tutti `1` per default.
`--temperature` imposta la temperatura InfoNCE, default `0.07`.
La CP loss è negativa: scendere verso `-1` significa aumentare la compatibilità.
Le loss sono medie sul batch di somme per categoria/item. L'accumulo divide i
gradienti per il numero reale di esempi, anche nell'ultimo gruppo incompleto.
Se un outfit contiene più item della stessa categoria, tutti contribuiscono
a retrieval e fine-grained; il generatore conserva un solo slot per categoria.

## Log di training e validation

I log usano il formato di OutfitTransformer: `timestamp | INFO | messaggio`,
con i campi `epoch`, `microbatch`, `loss` e `lr` nel messaggio. Il campo `phase`
distingue training e validation. Ogni fase annuncia subito il numero di batch
e stampa l'avanzamento ogni **50 batch**, più l'ultimo batch anche se la fase
è più corta o non termina su un multiplo di 50.

```text
2026-11-16 12:20:37,325 | INFO | epoch=1/10 phase=train batches=1250
2026-11-16 12:21:15,100 | INFO | epoch=1 microbatch=50/1250 loss=12.345678 lr=0.0001 phase=train
2026-11-16 12:40:00,000 | INFO | epoch=1/10 phase=validation batches=125
2026-11-16 12:40:20,000 | INFO | epoch=1 microbatch=50/125 loss=11.234567 lr=0.0001 phase=validation
2026-11-16 12:41:00,000 | INFO | epoch=1 microbatch=125/125 loss=10.987654 lr=0.0001 phase=validation
2026-11-16 12:41:05,000 | INFO | epoch=1/10 train_loss=12.000000 val_loss=11.000000 best_val_loss=11.000000
```

L'esempio omette i batch intermedi. `microbatch` conta i batch del loader,
indipendentemente dall'accumulo dei gradienti; `loss` è la loss totale del
batch appena elaborato. `lr` è il learning rate corrente dell'optimizer,
riportato anche in validation senza aggiornare i pesi. Il riepilogo di fine
epoca mostra le loss medie; tutte le componenti restano in `history.json`
e nei checkpoint.

## Grafici e checkpoint a ogni epoca

```text
checkpoints/seed_cached/
├── config.json
├── architecture.txt
├── history.json
├── best.pt
├── epochs/
│   ├── seed_to_outfit_epoch_001.pt
│   └── seed_to_outfit_epoch_002.pt
└── plots/
    ├── losses_epoch_001.png
    ├── total_loss_epoch_001.png
    ├── retrieval_loss_epoch_001.png
    ├── cp_loss_epoch_001.png
    ├── composition_loss_epoch_001.png
    ├── category_loss_epoch_001.png
    └── fine_grained_loss_epoch_001.png
```

Ogni epoca produce un pannello con tutte le loss e sei grafici separati,
ciascuno con train e validation dall'epoca 1 a quella corrente. Titoli,
marker, griglia e figure individuali 10×6 seguono lo stile del training CP.
`best.pt` è selezionato dalla loss totale di validation più bassa.

Ogni `.pt` è un dizionario PyTorch con `model_state_dict`, `optimizer_state_dict`,
epoca, global step, configurazione LoRA/modello/tassonomia, metriche, storico e
stati RNG. **`model_state_dict` contiene i parametri allenabili**: LoRA, query e
tutte le MLP. I backbone congelati non vengono ricopiati a ogni
epoca: per ricostruire il modello servono gli stessi checkpoint base Qwen/encoder/CP.
Non sono salvati oggetti modello Python serializzati. I checkpoint si caricano
con `torch.load(..., weights_only=True)`.
Il checkpoint e `config.json` includono anche il config completo del modello
e le dimensioni effettive delle MLP, per controllare l'architettura alla ripresa.
I vecchi checkpoint con embedding da 256 e adattatore CP non sono compatibili
con questa architettura: avviare un nuovo training.

Per riprendere, mantenere gli stessi argomenti di modello/dati e aggiungere
`--resume` con il percorso del checkpoint, come nell'[esempio di ripresa](#ripresa-del-training).

`--epochs` è il numero totale di epoche, non quelle aggiuntive. La ripresa
ricrea i backbone congelati, valida tassonomia e configurazione e ripristina
parametri allenabili, optimizer e RNG. LR, weight decay e accumulo devono
restare uguali. Non vengono sovrascritti run esistenti o epoche successive
al checkpoint scelto: per ripartire da un'epoca precedente scegliere un nuovo
`--output-dir`. Per riprendere in modo deterministico, mantenere anche il
numero di worker, l'ambiente e i file di dati/checkpoint.
Il checksum del checkpoint CP viene salvato e verificato alla ripresa:
un nuovo `best.pt` con pesi diversi non viene accettato come lo stesso scorer.

## Verifiche locali

La suite di test copre LoRA con un Qwen piccolo reale,
gradienti attraverso CP congelato, aggiornamento delle MLP, equivalenza
raw/cache, categorie ripetute, checkpoint riprendibili e PNG a ogni epoca.
Non scarica Qwen 8B e non misura la qualità finale degli outfit.
Il comando è nella sezione [Esecuzione dei test](#esecuzione-dei-test).

## Esempi di comando

Eseguire i comandi dalla root di Text2Outfit, nell'ambiente Python del progetto.
Ogni esempio ha una versione Bash e una PowerShell con gli stessi argomenti.
Sostituire i percorsi con quelli del proprio ambiente.

### Installazione e avvio predefinito

**Bash**

```bash
python -m pip install -r requirements.txt
python -m training.train_seed_to_outfit --epochs 10
```

**PowerShell**

```powershell
python -m pip install -r requirements.txt
python -m training.train_seed_to_outfit --epochs 10
```

### Help della CLI

**Bash**

```bash
python -m training.train_seed_to_outfit --help
```

**PowerShell**

```powershell
python -m training.train_seed_to_outfit --help
```

### Preparazione separata del CP

Prepara il CP FashionCLIP predefinito senza avviare il training.

**Bash**

```bash
python -m model.compatibility
```

**PowerShell**

```powershell
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

### Training FashionSigLIP con cache e CP locale

Questo esempio usa il repository OutfitTransformer accanto a Text2Outfit e il
CP locale dell'epoca 118. Aggiornare il percorso se si sceglie un'altra epoca.
L'output item e retrieval è automaticamente da 1536 componenti.

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

Riprende l'esempio FashionCLIP con cache dall'epoca 10 fino a 20 epoche totali.

**Bash**

```bash
python -m training.train_seed_to_outfit \
  --embedding-root precomputed_embeddings/patrickjohncyh-fashion-clip \
  --output-dir checkpoints/seed_cached --epochs 20 \
  --resume checkpoints/seed_cached/epochs/seed_to_outfit_epoch_010.pt
```

**PowerShell**

```powershell
python -m training.train_seed_to_outfit `
  --embedding-root precomputed_embeddings/patrickjohncyh-fashion-clip `
  --output-dir checkpoints/seed_cached --epochs 20 `
  --resume checkpoints/seed_cached/epochs/seed_to_outfit_epoch_010.pt
```

### Esecuzione dei test

**Bash**

```bash
python -m unittest discover -s tests
```

**PowerShell**

```powershell
python -m unittest discover -s tests
```
