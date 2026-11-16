# Training seed-to-outfit

Il training insegna al modello a completare un outfit partendo da **un capo
seed e un prompt**. Il target è l'outfit originale di Polyvore, seed compreso.
Il modello impara quali categorie includere e gli embedding con cui recuperare
i capi dal catalogo.

| Componente | Cosa succede durante il training |
|---|---|
| Qwen | Pesi originali congelati; si allenano gli adattatori LoRA nelle proiezioni Q/K/V. |
| Query di categoria | Si allenano 11 vettori, uno per categoria. |
| MLP | Si allenano seed, item e le teste di retrieval, presenza e sottocategoria. |
| Encoder FashionCLIP / FashionSigLIP | Pesi congelati; con le feature precomputate non vengono caricati. |
| Modello di compatibilità CP | Pesi congelati; il suo score contribuisce ad allenare il generatore. |

## Indice

- [Prima di iniziare](#prima-di-iniziare)
- [Scegliere il modello CP](#scegliere-il-modello-cp)
- [Item già precomputati](#item-già-precomputati)
- [Dati, prompt e loss](#dati-prompt-e-loss)
  - [Seed e prompt](#seed-e-prompt)
  - [Significato delle loss](#significato-delle-loss)
- [Optimizer e aggiornamento dei pesi](#optimizer-e-aggiornamento-dei-pesi)
  - [Batch e accumulo dei gradienti](#batch-e-accumulo-dei-gradienti)
  - [AdamW](#adamw)
  - [OneCycleLR](#onecyclelr)
- [Parametri del modello e del training](#parametri-del-modello-e-del-training)
  - [Valori di partenza](#valori-di-partenza)
  - [Parametri senza un flag CLI dedicato](#parametri-senza-un-flag-cli-dedicato)
  - [Dimensioni e impostazioni determinate dal codice o dai checkpoint](#dimensioni-e-impostazioni-determinate-dal-codice-o-dai-checkpoint)
- [Flag della CLI](#flag-della-cli)
- [Log di training e validation](#log-di-training-e-validation)
- [Grafici e checkpoint a ogni epoca](#grafici-e-checkpoint-a-ogni-epoca)
- [Riprendere un training](#riprendere-un-training)
- [Memoria GPU e tempi di avvio](#memoria-gpu-e-tempi-di-avvio)
- [Struttura del modulo](#struttura-del-modulo)
- [Verifiche locali](#verifiche-locali)
- [Esempi di comando](#esempi-di-comando)
  - [Installazione e avvio predefinito](#installazione-e-avvio-predefinito)
  - [Help della CLI](#help-della-cli)
  - [Preparazione separata del CP](#preparazione-separata-del-cp)
  - [Training con CP locale](#training-con-cp-locale)
  - [Training FashionCLIP con cache](#training-fashionclip-con-cache)
  - [Training FashionCLIP disjoint con cache](#training-fashionclip-disjoint-con-cache)
  - [Training FashionSigLIP con cache e CP locale](#training-fashionsiglip-con-cache-e-cp-locale)
  - [Ripresa del training](#ripresa-del-training)
  - [Esecuzione dei test](#esecuzione-dei-test)

## Prima di iniziare

Per preparare un run, scegliere:

1. **Subset del dataset:** `nondisjoint` oppure `disjoint`, tramite `--subset`.
2. **Origine delle feature:** immagini e descrizioni originali, oppure una
   cache già generata indicata con `--embedding-root`.
3. **Encoder e CP:** FashionCLIP oppure FashionSigLIP, con un checkpoint CP
   compatibile con la dimensione degli embedding.
4. **Durata e memoria:** epoche totali, batch size, accumulo e picco del
   learning rate. Le epoche totali definiscono la durata del ciclo OneCycleLR.

Questa è la configurazione predefinita:

| Impostazione | Default |
|---|---|
| LLM | Qwen3 Embedding 8B |
| Encoder e seed | FashionCLIP, seed immagine |
| Subset | `nondisjoint` |
| Feature | Da immagini e descrizioni; cache opzionale |
| Batch / accumulo / batch effettivo | `2` / `4` / normalmente `8` outfit per aggiornamento |
| Epoche totali | `10` |
| Optimizer e scheduler | AdamW + OneCycleLR |
| Picco LR | `1e-4` |
| Gradient checkpointing | Attivo |

I default di training sono in [`config.py`](config.py); quelli del modello,
di LoRA e delle loss sono in [`model/config.py`](../model/config.py).
I flag della CLI sovrascrivono questi valori solo per il singolo comando.
Le [tabelle dei parametri](#parametri-del-modello-e-del-training) e dei
[flag CLI](#flag-della-cli) riportano tutti i valori e come modificarli.

I comandi di installazione e training sono raccolti negli
[esempi in fondo alla pagina](#esempi-di-comando), ciascuno nelle versioni
Bash e PowerShell. Vanno eseguiti dalla root del progetto, nell'ambiente Python.

Il dispositivo viene scelto automaticamente: CUDA se disponibile, altrimenti
CPU. Su CUDA con supporto BF16, il backbone Qwen usa BF16; adattatori e MLP
restano FP32. Negli altri casi Qwen usa FP32. La disponibilità di memoria
determina il batch utilizzabile: il default `2` va verificato sulla propria GPU.

Se si cambia LLM, `--category-embedding-dim` deve coincidere con la sua
larghezza interna `hidden_size`: `4096` per il modello predefinito.

## Scegliere il modello CP

Il CP assegna uno score di compatibilità all'outfit. Durante il training
riceve gli embedding retrieval predetti e la maschera delle categorie target.
I suoi pesi restano congelati, ma i gradienti attraversano il CP per allenare
il generatore.

| Encoder | Feature per immagine / testo | Uscita item e retrieval | Ingresso richiesto al CP |
|---|---|---|---|
| FashionCLIP | `512` / `512` | `1024` | `1024` |
| FashionSigLIP | `768` / `768` | `1536` | `1536` |

Le dimensioni di uscita vengono selezionate automaticamente. Un valore
esplicito di `--retrieval-dim` deve coincidere con `model_dim` del CP scelto.

Il CP FashionCLIP predefinito viene preparato automaticamente: il codice
mancante viene scaricato da [GitHub](https://github.com/FlorindoDev/OutfitTransformer), i pesi
da [Hugging Face](https://huggingface.co/FlorindoDev/OutfitTransformer-weights).
Viene scaricato soltanto `checkpoints/nondisjoint/cp_precomputed-flashionclip/best.pt`;
i file locali e la cache vengono riutilizzati prima di accedere alla rete.
È possibile preparare il CP separatamente con il relativo
[esempio di comando](#preparazione-separata-del-cp).
Con `--encoder siglip`, specificare il CP FashionSigLIP da 1536 componenti tramite `--cp-checkpoint`
oppure `--cp-weights-file`; il codice rifiuta il default FashionCLIP.

Il flag `--subset` sceglie i dati e le cache; **non cambia automaticamente
il checkpoint CP**. Quello predefinito proviene da `nondisjoint`. Per usare
un CP addestrato su `disjoint`, indicare il relativo checkpoint o file remoto.

Per usare un CP già presente, specificare:

| Flag | Percorso da indicare |
|---|---|
| `--cp-repo` | Cartella del repository OutfitTransformer, contenente il package `model/`. |
| `--cp-checkpoint` | File `.pt` del modello CP già addestrato, per esempio `best.pt`. |

I percorsi possono essere assoluti o relativi alla cartella da cui si avvia
il comando. L'[esempio con CP locale](#training-con-cp-locale) mostra entrambi
i flag insieme.

I flag `--cp-source-url`, `--cp-weights-repo`, `--cp-weights-file` e `--cp-cache`
sovrascrivono le sorgenti e la cache del config. Il file remoto deve essere
un `best.pt`; un `--cp-checkpoint` locale esplicito deve esistere.
I [default CP](../model/README.md#caricamento-automatico-del-cp) sono in `ModelConfig`.

Il caricamento controlla tutti i pesi e rifiuta dimensioni incompatibili e
vecchi checkpoint ResNet. I dettagli sui formati supportati sono nella
[guida del modello](../model/README.md#caricamento-automatico-del-cp).

## Item già precomputati

Senza `--embedding-root`, il training carica immagini e descrizioni e ne
estrae le feature con l'encoder scelto. Con `--embedding-root`, legge invece
le feature già salvate: evita la decodifica delle immagini e il caricamento
degli encoder, ma continua ad allenare gli adattatori LoRA, le query e le MLP.

La cache deve contenere le feature **prima della MLP item**:

```text
[embedding visuale normalizzato, embedding testuale normalizzato]
```

È il formato prodotto da `scripts.precompute_embeddings`. La cartella passata
a `--embedding-root` contiene il subset e, al suo interno, entrambi gli split.
Per esempio, con `--subset nondisjoint`:

```text
precomputed_embeddings/patrickjohncyh-fashion-clip/
└── nondisjoint/
    ├── train/manifest.json
    └── validation/manifest.json
```

Con `--subset disjoint` deve esistere la stessa struttura sotto `disjoint/`.
Il percorso di `--embedding-root` si ferma **prima** di `disjoint/` o
`nondisjoint/`. La cache deve appartenere allo stesso encoder selezionato
per il modello.

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

Con i dati originali vengono elaborati immagini e descrizioni; con la cache
si usano le feature già estratte. In entrambi i casi si applica la MLP item,
che resta allenabile.
Il wrapper di training gestisce anche batch con item raw e cached misti;
i seed immagine di uno stesso batch devono invece usare tutti immagini
oppure tutti feature cached.

## Dati, prompt e loss

Si usano solo outfit con `label=1` come target della generazione. I negativi
sono articoli distinti della stessa categoria, esterni all'outfit e appartenenti
allo split richiesto. Non vengono usati outfit con label 0 come positivi.

### Seed e prompt

Il seed viene campionato dall'outfit durante il training. In validation seed
e negativi sono deterministici. `--seed-mode text` usa la descrizione del seed;
il default `image` usa l'immagine o le sue feature visuali precomputate.

Il prompt predefinito è uguale per tutti gli outfit:

```text
Complete a compatible outfit using this seed item.
```

| Opzione | Prompt usato |
|---|---|
| Nessuna opzione | Il prompt predefinito riportato sopra. |
| `--prompt` | Il testo indicato, comune a tutti gli outfit. |
| `--prompts` | Il prompt associato all'ID dell'outfit nel file JSON; ha precedenza sul prompt comune. |

Per controllare stagione e occasione servono prompt specifici, coerenti con
gli outfit target. Il progetto carica annotazioni già preparate e non le genera
automaticamente. Le feature precomputate non sostituiscono queste annotazioni.

Il file di `--prompts` è un oggetto JSON con un prompt per outfit e deve
coprire entrambi gli split. Esempio di struttura, da completare con tutti gli ID:

```json
{
  "compatibility_train:1": "Create a casual summer outfit.",
  "compatibility_valid:1": "Create a formal winter outfit."
}
```

Gli ID includono il numero di riga nei file ufficiali; fornire tutte le righe
positive che verranno caricate. Le descrizioni degli altri capi target non
vengono aggiunte al prompt; in modalità seed testuale si aggiunge solo la
descrizione del seed. Il vocabolario delle sottocategorie viene ricavato dai
metadati di train e validation, ordinato e salvato nel checkpoint.

### Significato delle loss

| Componente | Obiettivo |
|---|---|
| `retrieval` | Avvicinare gli embedding predetti ai capi target, distinguendoli dai negativi. |
| `cp` | Aumentare lo score di compatibilità dell'outfit predetto. |
| `category` | Predire quali categorie sono presenti nell'outfit. |
| `fine_grained` | Predire la sottocategoria dei capi target. |
| `composition` | Somma di `category` e `fine_grained`. |

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

## Optimizer e aggiornamento dei pesi

### Batch e accumulo dei gradienti

**Batch size** è il numero di outfit elaborati insieme. **Accumulo** è il
numero di batch elaborati prima di aggiornare i pesi.

Con i default, `2 outfit × 4 batch = 8 outfit` per aggiornamento. Il batch
effettivo è più piccolo nell'ultimo gruppo incompleto; i gradienti vengono
sempre mediati sul numero reale di esempi. Un **microbatch** nei log corrisponde
a un batch del loader, non a un aggiornamento dell'optimizer.

### AdamW

Il training usa **AdamW** su un unico gruppo contenente tutti i parametri
allenabili: LoRA, query di categoria e MLP. Lo stesso weight
decay si applica a tutto il gruppo, inclusi i bias allenabili.

| Parametro | Valore di default | Configurazione |
|---|---|---|
| Tipo di optimizer | `AdamW` | Fissato in `trainer.py`. |
| Picco del learning rate (`max_lr`) | `0.0001` | `learning_rate` / `--learning-rate`. |
| Weight decay | `0.01` | `weight_decay` / `--weight-decay`. |
| Momenti (`betas`) | Beta2 `0.999`; beta1 tra `0.85` e `0.95` | OneCycleLR varia beta1 in senso inverso al learning rate. |
| Stabilità numerica (`eps`) | `1e-8` | Default di PyTorch. |
| `amsgrad` | `False` | Default di PyTorch. |
| `maximize` | `False` | Default di PyTorch: minimizza la loss. |
| `foreach`, `fused` | `None`, `None` | Scelta dell'implementazione lasciata a PyTorch. |
| `capturable`, `differentiable` | `False`, `False` | Default di PyTorch. |
| Clipping dei gradienti | Norma L2 massima `1.0` | `max_grad_norm`; errore se la norma non è finita (`error_if_nonfinite=True`). |
| Azzeramento dei gradienti | `set_to_none=True` | All'inizio dell'epoca e dopo ogni aggiornamento. |

I default dell'optimizer non passati esplicitamente dipendono dalla libreria
PyTorch installata; la tabella riporta quelli verificati nell'ambiente del
progetto.

### OneCycleLR

Il learning rate **varia durante il run**. Nella prima parte del ciclo sale
verso il picco; nella seconda scende verso il minimo. La configurazione del
ciclo è la stessa del training CP/CIR del repository OutfitTransformer locale.

**`--learning-rate` imposta il picco del ciclo, non il valore iniziale.**
Con il default `1e-4`, i valori di riferimento sono:

```text
iniziale 4e-6  →  picco 1e-4  →  minimo 4e-10
```

Lo scheduler avanza una volta per aggiornamento dell'optimizer, anche per
l'ultimo gruppo incompleto di batch. In validation resta fermo.

| Parametro | Default | Significato |
|---|---|---|
| Scheduler del learning rate | `OneCycleLR` | Avanza dopo ogni aggiornamento dell'optimizer, senza avanzare in validation. |
| Warmup (`pct_start`) | `0.3` | Primo 30% degli aggiornamenti del ciclo. |
| Curva (`anneal_strategy`) | `"cos"` | Salita e discesa con interpolazione coseno. |
| Fattore iniziale (`div_factor`) | `25.0` | LR iniziale = picco / 25; con i default `4e-6`. |
| Fattore finale (`final_div_factor`) | `10000.0` | LR minimo = LR iniziale / 10000; con i default `4e-10`. |
| Ciclo del momento (`cycle_momentum`) | `True` | `base_momentum=0.85`, `max_momentum=0.95`. |
| Numero di fasi (`three_phase`) | `False` | Due fasi: salita al picco e discesa al minimo. |
| Durata (`total_steps`) | Aggiornamenti per epoca × epoche totali | Gli aggiornamenti per epoca sono i batch divisi per l'accumulo, arrotondando per eccesso. |

I fattori del ciclo sono definiti in `training/config.py` come
`ONE_CYCLE_PCT_START`, `ONE_CYCLE_DIV_FACTOR` e `ONE_CYCLE_FINAL_DIV_FACTOR`.
`--learning-rate` imposta il picco del ciclo: il valore predefinito resta
`1e-4`, mentre OutfitTransformer locale usa `2e-5`; per usare anche quel
picco, impostare `--learning-rate 2e-5`.

La durata del ciclo viene fissata all'avvio. **Alla ripresa non si possono
cambiare le epoche totali pianificate**, il batch size o l'accumulo.
Vedere [Riprendere un training](#riprendere-un-training).

## Parametri del modello e del training

Le tabelle di questa sezione e dei [flag CLI](#flag-della-cli) elencano tutti
i campi di `TrainingConfig` e `ModelConfig`, con i valori predefiniti del
progetto. I flag sovrascrivono i campi indicati per il singolo run; i campi
senza un flag dedicato si modificano nel relativo config Python.

### Valori di partenza

I default definiscono una configurazione iniziale modificabile in base al
dataset e alle risorse disponibili.

| Gruppo | Impostazioni |
|---|---|
| Architettura | LoRA rank `64` su Q/K/V, 11 query di categoria, encoder e CP congelati, loss retrieval/compatibilità/composizione per seed-to-outfit. |
| Ottimizzazione | AdamW e OneCycleLR. |
| Modello e iperparametri | Qwen3 Embedding 8B, picco LR `1e-4`, 10 epoche, layer nascosti MLP da 512, temperatura `0.07`, LoRA alpha `128` e dropout `0.05`, pesi delle loss `(1, 1, 1)`. |
| Memoria | Batch size `2`; con accumulo `4`, batch effettivo normalmente di 8 outfit. |

Questi valori non sono il risultato di un tuning su validation. In particolare,
i pesi delle loss vanno verificati rispetto alle loro scale effettive.

### Parametri senza un flag CLI dedicato

| Campo | Default | Significato e sorgente |
|---|---|---|
| `max_grad_norm` | `1.0` | Soglia del clipping L2; `TrainingConfig` in `training/config.py`. |
| `llm_dtype` | `"auto"` | `ModelConfig`: Qwen usa BF16 su CUDA con supporto BF16, altrimenti FP32. Accetta anche `"float32"` e `"bfloat16"`. LoRA e MLP restano FP32. |
| `llm_trainable` | `False` | Pesi originali di Qwen congelati. Con `lora_enabled=True` si aggiornano solo gli adattatori; il fine-tuning completo richiede `lora_enabled=False`. |
| `encoder_trainable` | `False` | Encoder visuale e testuale congelati; il modello completo rifiuta `True`. |
| `seed_hidden_dims` | `(512,)` | Layer nascosti della MLP seed: `V → 512 → H`. |
| `item_hidden_dims` | `(512,)` | Layer nascosti della MLP item: `(V + T) → 512 → D`. |
| `retrieval_hidden_dims` | `(512,)` | Layer nascosti di ciascuna delle 11 MLP retrieval: `H → 512 → D`. |
| `category_hidden_dims` | `(512,)` | Layer nascosti di ciascuna delle 11 MLP presenza: `H → 512 → 1`. |
| `fine_grained_hidden_dims` | `(512,)` | Layer nascosti di ciascuna delle 11 MLP sottocategoria: `H → 512 → C_i`. |
| `lora_enabled` | `True` | Abilita gli adattatori LoRA su Qwen. |
| `lora_target_modules` | `("q_proj", "k_proj", "v_proj")` | Proiezioni di attenzione a cui applicare LoRA; i bias del backbone restano congelati (`bias="none"`). |

Tranne `max_grad_norm`, questi campi appartengono a
[`ModelConfig`](../model/config.py). Le tuple `*_hidden_dims` elencano solo
i layer nascosti: `()` crea una singola proiezione lineare. Il flag
`--hidden-dim` sostituisce tutte e cinque le tuple con `(N,)` per quel run.

### Dimensioni e impostazioni determinate dal codice o dai checkpoint

| Parametro | Valore / comportamento |
|---|---|
| Categorie e query | 11: `tops`, `bottoms`, `all-body`, `outerwear`, `shoes`, `bags`, `jewelry`, `accessories`, `sunglasses`, `hats`, `scarves`. |
| Larghezza LLM `H` | Letta dal checkpoint Qwen; `4096` con il default. Deve coincidere con `category_embedding_dim`. |
| Feature visuali/testuali `V`, `T` | FashionCLIP: `512`, `512`; FashionSigLIP: `768`, `768`. Lette dagli encoder o dal manifest della cache. |
| Dimensione condivisa `D` | `1024` con FashionCLIP o `1536` con SigLIP se `retrieval_dim=None`; un valore esplicito ha precedenza. |
| Sottocategorie `C_i` | Numero di etichette della categoria, ricavato dai metadati di train e validation e salvato nel checkpoint. |
| Ingresso del CP | `model_dim` del checkpoint CP; deve coincidere con `D`. Pesi sempre congelati, senza adattatore. |
| MLP | 35 reti indipendenti: seed, item e 3 teste per ciascuna delle 11 categorie. Layer nascosti `Linear → ReLU`; uscita `Linear`. Bias abilitati, senza dropout o normalizzazione interni. |
| Attenzione Qwen | `attn_implementation="sdpa"`; cache di generazione disabilitata (`use_cache=False`). Le query di categoria si vedono reciprocamente e vedono il prompt e il seed. |
| Gradient checkpointing | Quando attivo, `use_reentrant=False`. |
| LoRA | Scala standard `lora_alpha / lora_rank = 2` con i default; dropout `0.05`. |
| Loader | Split `train` e `validation`; shuffle solo sul training. `drop_last=False`, `pin_memory=False`, default del DataLoader. |
| Target e negativi | Solo outfit positivi (`label=1`); 4 negativi per articolo, della stessa categoria, esterni all'outfit e nello stesso split. |
| Seed dell'outfit | Un articolo casuale a ogni lettura in training; scelta deterministica in validation. |
| Logging | Ogni 50 microbatch e all'ultimo batch di entrambe le fasi; anche un messaggio a inizio fase. |
| Selezione del checkpoint | Minima loss totale di validation; salvataggio a ogni epoca. Nessun early stopping. |

L'architettura interna dei backbone (numero di layer, attention head e altre
impostazioni) proviene dai checkpoint selezionati. I conteggi effettivi dei
parametri totali e allenabili vengono stampati all'avvio insieme alle
dimensioni di ogni MLP e salvati in `architecture.txt`; la configurazione
del run è salvata in `config.json` e nei checkpoint.

## Flag della CLI

Tutti i flag della CLI di training sono riportati qui
sotto, inclusi help e disattivazione del gradient checkpointing. I default
provengono da [`TrainingConfig`](config.py), [`ModelConfig`](../model/config.py)
e dal [parser della CLI](train_seed_to_outfit.py). I percorsi relativi partono
dalla cartella da cui si esegue il comando; `None` indica che non è stato
specificato un valore, con il comportamento descritto nella tabella.

| Flag | Campo nel config | Cosa fa | Valore di default |
|---|---|---|---|
| `-h`, `--help` | — | Mostra i flag disponibili ed esce senza avviare il training. | — |
| `--dataset-root` | `dataset_root` | Cartella locale di Polyvore; le risorse mancanti vengono cercate in cache e poi scaricate. | `datasets/polyvore-outfits` |
| `--dataset-cache` | `dataset_cache` | Cartella di cache dei download del dataset. | `.cache/datasets` |
| `--subset` | `subset` | Variante Polyvore: `nondisjoint` oppure `disjoint`. | `nondisjoint` |
| `--embedding-root` | `embedding_root` | Cartella delle feature precomputate prima della MLP item, contenente `SUBSET/train/manifest.json` e `SUBSET/validation/manifest.json`. Se specificata, evita il caricamento degli encoder e la decodifica delle immagini. | `None` — usa immagini e descrizioni originali. |
| `--encoder` | `encoder` | Famiglia degli encoder visuale e testuale: `fashionclip` oppure `siglip`. | `fashionclip` |
| `--encoder-model` | `encoder_model_name` | Nome del checkpoint degli encoder; deve corrispondere a quello delle eventuali feature precomputate. | `None` — `patrickjohncyh/fashion-clip` con `fashionclip`, `Marqo/marqo-fashionSigLIP` con `siglip`. |
| `--llm` | `llm_model_name` | Nome o percorso locale del checkpoint LLM. | `Qwen/Qwen3-Embedding-8B` |
| `--llm-cache` | `llm_cache` | Cartella di cache del checkpoint LLM. | `.cache/llm` |
| `--category-embedding-dim` | `category_embedding_dim` | Dimensione delle 11 query di categoria; deve coincidere con `hidden_size` del LLM selezionato. | `4096` |
| `--cp-repo` | `cp_repo` | Cartella del codice OutfitTransformer, contenente `model/`; se assente, viene preparata dalla sorgente ZIP. | `.cache/outfit_transformer` |
| `--cp-checkpoint` | `cp_checkpoint` | Percorso di un checkpoint CP locale già addestrato; se esplicito, il file deve esistere. | `None` — cerca `--cp-repo/--cp-weights-file`, poi usa la cache o scarica i pesi. |
| `--cp-source-url` | `cp_source_url` | URL dell'archivio ZIP da cui recuperare il codice CP mancante. | `https://codeload.github.com/FlorindoDev/OutfitTransformer/zip/refs/heads/main-branch` |
| `--cp-weights-repo` | `cp_weights_repo` | Repository Hugging Face dei pesi CP. | `FlorindoDev/OutfitTransformer-weights` |
| `--cp-weights-file` | `cp_weights_file` | Percorso relativo del checkpoint nel repository dei pesi CP; deve terminare con `best.pt`. | `checkpoints/nondisjoint/cp_precomputed-flashionclip/best.pt` |
| `--cp-cache` | `cp_cache` | Cartella di cache Hugging Face dei pesi CP. | `.cache/huggingface-cp` |
| `--output-dir` | `output_dir` | Cartella del run: configurazione, architettura, metriche, checkpoint e grafici. | `checkpoints/seed_to_outfit` |
| `--resume` | `resume` | Checkpoint di training da cui ripristinare parametri allenabili, optimizer, storico e stati RNG. | `None` — avvia un nuovo training. |
| `--epochs` | `epochs` | Numero totale di epoche pianificate per OneCycleLR; deve restare uguale alla ripresa. | `10` |
| `--batch-size` | `batch_size` | Numero di outfit per batch nei loader di training e validation. | `2` |
| `--accumulation-steps` | `gradient_accumulation_steps` | Numero di microbatch su cui accumulare i gradienti prima di un aggiornamento dell'optimizer. | `4` |
| `--learning-rate` | `learning_rate` | Learning rate massimo del ciclo OneCycleLR di AdamW. | `0.0001` |
| `--weight-decay` | `weight_decay` | Weight decay di AdamW. | `0.01` |
| `--negatives-per-item` | `negatives_per_item` | Numero di articoli negativi distinti per item, della stessa categoria e dello stesso split, esterni all'outfit positivo. | `4` |
| `--retrieval-dim` | `retrieval_dim` | Dimensione comune degli embedding retrieval e item; deve coincidere con `model_dim` del CP. | `None` — automatico: `1024` FashionCLIP / `1536` FashionSigLIP. |
| `--hidden-dim` | Tutte le tuple `*_hidden_dims` | Sostituisce gli strati nascosti di tutte le famiglie MLP con un solo layer della larghezza indicata. | `None` — conserva le dimensioni di `ModelConfig`, inizialmente un layer nascosto da `512` per ogni famiglia. |
| `--seed-mode` | `seed_mode` | Modalità del seed: `image` usa l'immagine o le feature visuali cached; `text` aggiunge la descrizione al prompt. | `image` |
| `--prompt` | `prompt` | Prompt comune a tutti gli outfit quando non si specifica `--prompts`. | `Complete a compatible outfit using this seed item.` |
| `--prompts` | `prompts` | File JSON che associa gli ID degli outfit ai prompt; deve coprire gli outfit positivi caricati di train e validation e ha precedenza su `--prompt`. | `None` — usa il prompt comune. |
| `--loss-weights RET CP COMP` | `loss_weights` | Tre pesi, nell'ordine retrieval, compatibilità e composizione. Devono essere finiti, non negativi e almeno uno deve essere positivo. | `1.0 1.0 1.0` |
| `--temperature` | `temperature` | Temperatura positiva della loss InfoNCE di retrieval. | `0.07` |
| `--lora-rank` | `lora_rank` | Rango degli adattatori LoRA nelle proiezioni Q/K/V del LLM. | `64` |
| `--lora-alpha` | `lora_alpha` | Fattore alpha di LoRA; insieme al rango determina la scala degli adattatori. | `128` |
| `--lora-dropout` | `lora_dropout` | Probabilità di dropout nei rami LoRA. | `0.05` |
| `--gradient-checkpointing`, `--no-gradient-checkpointing` | `gradient_checkpointing` | Abilita o disabilita il gradient checkpointing del LLM: riduce la memoria ricalcolando attivazioni durante il backward. Non richiede un valore dopo il flag. | `True` — attivo; `--no-gradient-checkpointing` lo disattiva. |
| `--seed` | `seed` | Seed casuale di Python, PyTorch e del loader; determina anche i negativi riproducibili della validation. | `42` |
| `--num-workers` | `num_workers` | Numero di processi worker dei DataLoader; `0` carica i dati nel processo principale. | `0` |
| `--device` | `device` | Dispositivo PyTorch, per esempio `cpu`, `cuda` o `cuda:0`; `auto` sceglie CUDA se disponibile, altrimenti CPU. | `auto` |

Per consultare l'help della versione corrente, usare il relativo
[esempio di comando](#help-della-cli).

## Log di training e validation

I log usano il formato di OutfitTransformer: `timestamp | INFO | messaggio`,
con i campi `epoch`, `microbatch`, `loss` e `lr` nel messaggio. Il campo `phase`
distingue training e validation. Ogni fase annuncia subito il numero di batch
e stampa l'avanzamento ogni **50 batch**, più l'ultimo batch anche se la fase
è più corta o non termina su un multiplo di 50.

```text
2026-11-16 12:20:37,325 | INFO | epoch=1/10 phase=train batches=1250
2026-11-16 12:21:15,100 | INFO | epoch=1 microbatch=50/1250 loss=12.345678 lr=4.038e-06 phase=train
2026-11-16 12:40:00,000 | INFO | epoch=1/10 phase=validation batches=125
2026-11-16 12:40:20,000 | INFO | epoch=1 microbatch=50/125 loss=11.234567 lr=2.8e-05 phase=validation
2026-11-16 12:41:00,000 | INFO | epoch=1 microbatch=125/125 loss=10.987654 lr=2.8e-05 phase=validation
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

Ogni `.pt` contiene i pesi allenabili (`model_state_dict`), gli stati di
optimizer e scheduler (`optimizer_state_dict`, `scheduler_state_dict`),
epoca, global step, configurazione LoRA/modello/tassonomia, metriche, storico e
stati dei generatori casuali. Il formato corrente è `checkpoint_schema_version=2`.
**`model_state_dict` contiene i parametri allenabili**: LoRA, query e
tutte le MLP. I backbone congelati non vengono ricopiati a ogni
epoca: per ricostruire il modello servono gli stessi checkpoint base Qwen/encoder/CP.
I checkpoint contengono pesi e configurazioni, senza oggetti modello Python
serializzati, e supportano il caricamento con `weights_only=True`.
Il checkpoint e `config.json` includono anche il config completo del modello
e le dimensioni effettive delle MLP, per controllare l'architettura alla ripresa.
`config.json` riporta inoltre i parametri del ciclo e il numero totale di
aggiornamenti. Lo stato dello scheduler viene salvato dopo gli aggiornamenti
dell'epoca, insieme a quello dell'optimizer.

I salvataggi avvengono **a fine epoca**, non ogni 50 batch. Se un run si
interrompe durante un'epoca, si può ripartire dall'ultimo checkpoint completo.
Il `best.pt` di questa cartella è il checkpoint di training Text2Outfit;
il checkpoint CP indicato con `--cp-checkpoint` appartiene a un altro modello.

## Riprendere un training

Per riprendere, mantenere gli stessi argomenti di modello/dati e aggiungere
`--resume` con il percorso del checkpoint, come nell'[esempio di ripresa](#ripresa-del-training).

`--epochs` è il numero totale di epoche pianificate, non quelle aggiuntive.
Con OneCycleLR deve coincidere con il valore del run iniziale: per esempio,
un run pianificato su 10 epoche può riprendere dall'epoca 5 fino alla 10.
Non si può trasformare quel run in un ciclo da 20 epoche tramite `--resume`.

| Impostazioni da mantenere | Motivo |
|---|---|
| Epoche totali e numero di batch del loader | Determinano la durata del ciclo OneCycleLR. |
| Batch size, accumulo, picco LR, weight decay e clipping | Mantengono gli stessi aggiornamenti dei pesi. |
| Modello, CP, tassonomia, dataset, subset, cache e prompt | Devono corrispondere al run salvato. |

La ripresa ripristina pesi allenabili, optimizer, scheduler, storico e stati
dei generatori casuali. Non vengono sovrascritti run esistenti o epoche successive
al checkpoint scelto: per ripartire da un'epoca precedente scegliere un nuovo
`--output-dir`. Per riprendere in modo deterministico, mantenere anche il
numero di worker, l'ambiente e i file di dati/checkpoint.
Il checksum del checkpoint CP viene salvato e verificato alla ripresa:
un nuovo `best.pt` con pesi diversi non viene accettato come lo stesso scorer.

**Serve un nuovo run** per i vecchi checkpoint senza stato OneCycleLR
(schema 1) e per quelli della precedente architettura con embedding da 256
e adattatore CP. La ripresa corrente richiede lo schema 2.

## Memoria GPU e tempi di avvio

Il modello 8B richiede memoria per i pesi, le attivazioni, i gradienti e lo
stato dell'optimizer. Anche con LoRA e feature precomputate, Qwen deve essere
eseguito per ogni batch. La CPU è utile per i test con modelli piccoli;
il training completo 8B su CPU non è stato verificato.

In caso di `CUDA out of memory`, ridurre `--batch-size` e mantenere il
gradient checkpointing attivo. Si può aumentare `--accumulation-steps` per
mantenere il batch effettivo desiderato. Per esempio, un batch da 2 con
accumulo 4 corrisponde normalmente a 8 outfit per aggiornamento.
Con OneCycleLR, un batch size diverso richiede un nuovo run.

All'avvio vengono controllate le cache, letti i dati, caricati i pesi e
preparato il modello sul dispositivo. La tabella delle MLP indica che il
modello è stato costruito; il messaggio `phase=train` annuncia l'inizio
dell'epoca. Da quel momento l'avanzamento compare ogni 50 batch e all'ultimo.
I file modificati durante un processo già avviato saranno letti solo al
successivo avvio.

## Struttura del modulo

| File | Responsabilità |
|---|---|
| `config.py` | `TrainingConfig`, default del training e fattori OneCycleLR. |
| `train_seed_to_outfit.py` | CLI, dataset, cache e avvio del training. |
| `data.py` | Cache, prompt, seed e vocabolario delle sottocategorie. |
| `trainer.py` | AdamW, OneCycleLR, accumulo, validation, checkpoint e ripresa. |
| `plots.py` | Grafici cumulativi di train e validation. |

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

### Training FashionCLIP disjoint con cache

Usa `precomputed/disjoint/train/manifest.json` e
`precomputed/disjoint/validation/manifest.json`. Seleziona esplicitamente
anche il CP FashionCLIP disjoint; sorgenti e pesi vengono riutilizzati se
già presenti, altrimenti scaricati.

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

Riprende l'esempio FashionCLIP con cache dall'epoca 5 fino alle 10 epoche
totali pianificate, mantenendo la stessa curva OneCycleLR.

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

### Esecuzione dei test

**Bash**

```bash
python -m unittest discover -s tests
```

**PowerShell**

```powershell
python -m unittest discover -s tests
```
