# Project Workflow

## Overview

The project has two entry paths:

1. Run `train.py` directly with command-line arguments.
2. Use `tui.py` to select scripts, arguments, model settings, previous runs, or
   checkpoints.

Both paths eventually execute the same training workflow in `train.main()`.

```text
CLI or TUI
  -> TrainingConfig
  -> model configuration
  -> dataset and DataLoaders
  -> dynamic model factory
  -> optimizer, scheduler, and loss
  -> Trainer
  -> history, checkpoints, diagnostics, and test metrics
```

## Naming Rules

Model names must be lowercase Python identifiers using underscores.

```text
mlp
simple_cnn
resnet
mlp_mixer
```

Each name maps directly to a Python module.

```text
mlp        -> models/mlp.py
simple_cnn -> models/simple_cnn.py
resnet     -> models/resnet.py
mlp_mixer  -> models/mlp_mixer.py
```

The model names in `models/__init__.py` and the top-level keys in
`models/config.json` must match.

## Direct CLI Entry

A typical command is:

```bash
python train.py \
  --dataset fashion \
  --model_name mlp \
  --epochs 20 \
  --batch_size 256
```

`parse_config()` converts the arguments into `TrainingConfig`.

`TrainingConfig` contains settings shared by all models:

- Dataset name
- Model name
- Epochs and batch size
- Learning rate and weight decay
- Device and random seed
- Optimizer and scheduler
- Validation monitor
- Storage and checkpoint options
- Logging and diagnostic options

Model-specific parameters such as hidden sizes, channels, dropout, activation,
or input flattening are not CLI fields.

## Model Configuration

### Default Configuration

The default parameters are stored in `models/config.json`.

```json
{
  "mlp": {
    "flatten": true,
    "normalize": true,
    "hidden_sizes": [512, 256, 128],
    "activation": "relu",
    "use_batch_norm": true,
    "dropout_p": 0.3
  }
}
```

`resolve_model_configuration()` calls
`models.config.load_model_configuration()`.

The configuration source priority is:

```text
--resume checkpoint
  -> --model-config-path
  -> models/config.json
```

Supported configuration sources are:

- Default registry: `models/config.json`
- Another model registry JSON
- A previous run's `storage/.../config.json`
- A `.pt` or `.pth` checkpoint
- A temporary TUI override JSON

The loader returns a `ModelConfiguration` containing:

```python
ModelConfiguration(
    name="mlp",
    parameters={...},
    source_type="default",
    source_path=".../models/config.json",
)
```

The selected parameters and their source are copied into `TrainingConfig`.

## Dataset Loading

`load_training_data()` calls `dataset.build()`.

The model configuration controls preprocessing fields such as:

```text
flatten
normalize
```

`dataset.build()` then:

1. Loads the preprocessed dataset through `data.load.load()`.
2. Shuffles indices reproducibly using the configured seed.
3. Splits the original training data into train and validation subsets.
4. Keeps the original test subset separate.
5. Applies image conversion, optional normalization, and optional flattening.
6. Keeps labels one-hot encoded.
7. Creates train, validation, and test `DataLoader` instances.

After loading one training sample, `train.py` derives:

```text
input_shape
num_classes
```

`num_classes` comes from the one-hot target size. It is not read from the model
configuration.

## Model Loading

`train.py` calls:

```python
model = build_model(
    model_name=config.model_name,
    input_shape=config.input_shape,
    num_classes=config.num_classes,
    model_config=config.model_parameters,
)
```

`models.build_model()` dynamically imports the matching module:

```python
module = importlib.import_module(f"models.{model_name}")
```

The module must provide:

```python
def build_model(input_shape, num_classes, model_config):
    return Model(...)
```

The returned object must be a `torch.nn.Module`.

The model module should not reload `models/config.json`. The resolved
configuration is already passed through `model_config`. This guarantees that
default settings, TUI overrides, previous-run settings, and checkpoint settings
all construct the same model architecture.

Every model must follow this output contract:

```python
logits = model(inputs)
```

`logits` must have shape:

```text
(batch_size, num_classes)
```

Softmax and log-softmax should not be included in the model output because the
trainer uses `CrossEntropyLoss`.

## Run Storage

`prepare_storage()` creates a new run directory or reuses the checkpoint's run
directory when resuming.

```text
storage/{timestamp}_{dataset}_{model}/
├── config.json
├── train.log
├── history.json
├── history.csv
├── test_metrics.json
├── checkpoints/
└── diagnostics/
```

The run's `config.json` is an immutable snapshot of the effective settings:

```json
{
  "training": {
    "dataset_path": "fashion",
    "epochs": 20
  },
  "model": {
    "name": "mlp",
    "parameters": {
      "flatten": true,
      "hidden_sizes": [512, 256, 128]
    }
  },
  "source": {
    "type": "default",
    "path": "models/config.json"
  }
}
```

Changing `models/config.json` later does not alter this saved run snapshot.

## Training Components

After the dataset and model are ready, `train.py` builds:

- Optimizer: Adam, AdamW, SGD, or RMSprop
- Scheduler: none, step, cosine, or plateau
- Loss: `torch.nn.CrossEntropyLoss`
- `Trainer`

The one-hot target is converted inside the trainer:

```python
target_indices = one_hot_targets.argmax(dim=1)
```

## Epoch Workflow

For each epoch, `Trainer.fit()` performs:

```text
train epoch
  -> validation epoch
  -> scheduler update
  -> history.json and history.csv update
  -> best-model comparison
  -> last/best/periodic checkpoint save
  -> optional weight diagnostics
```

Training mode:

- `model.train()`
- Gradient reset
- Forward pass
- Finite-logit and finite-loss checks
- Backward pass
- Optimizer step

Validation and test mode:

- `model.eval()`
- Gradients disabled
- No optimizer update

## Metrics

`ClassificationMetricTracker` accumulates metrics over all samples:

- Mean loss
- Top-1 accuracy
- Top-5 accuracy
- Macro precision
- Macro recall
- Macro F1

The confusion matrix is accumulated across the full epoch rather than averaging
batch-level F1 values.

## Best Model Selection

The default monitored metric is:

```text
valid_f1, mode=max
```

Supported monitors are:

- `valid_loss`
- `valid_accuracy`
- `valid_f1`

Tie-breaking order:

1. Better monitored metric
2. Lower validation loss
3. Earlier epoch

The test set is not used for model selection.

## Checkpoints and Resume

Checkpoint format version is currently `2`.

```text
checkpoints/
├── last.pt
├── best.pt
└── epoch_NNN.pt
```

A checkpoint includes:

- Model name and model parameters
- Model state
- Optimizer state
- Scheduler state
- Best-model state
- Training configuration
- History
- Input shape and class count
- Configuration source

When `--resume` is used:

1. The model configuration is loaded from the checkpoint.
2. The same run directory is reused.
3. The model, optimizer, scheduler, history, and best state are restored.
4. Model name, parameters, input shape, class count, and monitor settings are
   checked for consistency.
5. Training continues from `checkpoint_epoch + 1`.

## Diagnostics and Final Evaluation

Weight diagnostics are saved:

- Before the first epoch
- At the middle epoch
- At the final epoch

Each diagnostic point can contain:

- Weight histograms
- Conv2d kernel or Linear heatmap visualization
- Weight statistics JSON

After training:

1. `best.pt` is restored.
2. The test set is evaluated once.
3. The result is saved to `test_metrics.json`.

## TUI Workflow

Run:

```bash
python tui.py
```

Main menu:

```text
train
eval
print_inference
model_config
quit
```

### Model Config Menu

Configuration sources:

- Default model registry
- Previous run
- Checkpoint

Parameters are edited as JSON values so their original types are preserved.

Available actions depend on the source:

- `use_model_parameters`: use only the selected model parameters
- `clone_configuration`: apply previous training and model settings to a new run
- `resume_training`: restore the previous checkpoint state
- `evaluate`: apply the selected checkpoint to an evaluation script
- Save as model default: update `models/config.json` and create
  `models/config.backup.json`

Edited parameters are written to a temporary override JSON and passed through
`--model-config-path`. The file records the original configuration source.

### Script Form

When a script path is selected, the TUI parses its argparse declarations and
builds an input form. It supports:

- Choice menus
- Boolean flags
- Path completion
- Dataset selection
- Extra arguments

The TUI then runs the script as a subprocess, streams combined stdout/stderr,
and displays CPU, memory, and GPU utilization when available.

## Current Model Implementation Contract

Adding a model requires three matching pieces:

1. Add the model name to `MODEL_NAMES`.
2. Add the same key and default parameters to `models/config.json`.
3. Implement `models/{model_name}.py` with:

```python
def build_model(input_shape, num_classes, model_config):
    ...
```

The existing `models/mlp.py` is currently outside Git and incomplete. It must
be brought into this factory contract before `--model_name mlp` can train
successfully.
