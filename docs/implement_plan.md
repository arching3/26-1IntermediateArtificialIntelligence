# Training Workflow Implementation Plan

## Goal

Build a reusable image-classification training workflow that supports the
project datasets and future MLP, CNN, ResNet, and MLP-Mixer implementations.
All outputs from one run are grouped under a single storage directory.

## Scope

- Keep dataset labels one-hot encoded.
- Accept any model that implements `logits = model(inputs)`.
- Track train and validation loss, top-1 accuracy, top-5 accuracy, macro
  precision, macro recall, and macro F1.
- Write detailed console and file logs.
- Persist history as JSON and CSV without TensorBoard.
- Save structured checkpoints and select a best model from validation metrics.
- Save model weight diagnostics before training, halfway through training, and
  after the final epoch.
- Evaluate the test set once using the selected best model.
- Support resuming a run from a checkpoint.

Early stopping, gradient diagnostics, and TensorBoard are excluded.

## Module Layout

```text
train.py          CLI configuration and workflow assembly
trainer.py        train, validation, and test loops
metrics.py        classification metric accumulation
history.py        JSON and CSV history persistence
checkpoint.py     atomic checkpoint save and load
diagnostics.py    weight histograms and weight visualizations
models/           model factory and model implementations
```

## Model Contract

Every model must:

- Accept a batched tensor.
- Return raw logits with shape `(batch_size, num_classes)`.
- Exclude softmax and log-softmax from its output layer.

The dataset keeps one-hot targets. The trainer converts them to class indices
with `targets.argmax(dim=1)` for loss and metric calculations.

## Run Storage

Each run uses:

```text
storage/{yyyymmdd_HHMMSS}_{dataset}_{model_name}/
├── config.json
├── train.log
├── history.json
├── history.csv
├── test_metrics.json
├── checkpoints/
│   ├── best.pt
│   ├── last.pt
│   └── epoch_NNN.pt
└── diagnostics/
    ├── initial/
    ├── middle_epoch_NNN/
    └── final_epoch_NNN/
```

`--storage-root` changes the parent directory. `--run-name` overrides the
automatically generated directory name. Existing run directories are not
overwritten unless training is explicitly resumed.

## Metrics

For train and validation, record:

- Mean sample-weighted loss
- Top-1 accuracy
- Top-5 accuracy, capped by the number of classes
- Macro precision
- Macro recall
- Macro F1
- Learning rate
- Epoch duration

Metric values are accumulated across all samples rather than averaged from
batch-level percentages.

## Logging and History

Python `logging` writes to both stdout and `train.log`. Logs include:

- Configuration and environment
- Dataset sizes and input shape
- Model structure and parameter count
- Epoch metrics and elapsed time
- Best-model updates
- Checkpoint save and restore events
- Final test metrics
- Exception tracebacks

`history.json` and `history.csv` are rewritten atomically after every epoch so
completed epochs remain available after interruption.

## Checkpoints

Checkpoint payloads contain only dictionaries and primitive configuration
values where possible:

- Format version and epoch
- Model name and model state
- Optimizer and optional scheduler state
- Best metric, best validation loss, and best epoch
- Configuration, input shape, and class count
- Complete history

Checkpoint policy:

- `last.pt` after every epoch
- `best.pt` when the monitored validation metric improves
- `epoch_NNN.pt` at the configured checkpoint interval
- Temporary-file save followed by atomic replacement

Resume restores model, optimizer, scheduler, history, and best-model tracking.

## Best Model Selection

Default:

```text
monitor = valid_f1
mode = max
```

Supported monitors are `valid_loss`, `valid_accuracy`, and `valid_f1`.
The monitor mode can be `min` or `max`.

Tie-breaking order:

1. Better monitored metric
2. Lower validation loss
3. Earlier epoch

The test set never participates in model selection.

## Weight Diagnostics

Diagnostics are saved at no more than three unique points:

1. `initial`: immediately after model construction
2. `middle_epoch_NNN`: after `ceil(total_epochs / 2)`
3. `final_epoch_NNN`: after the final completed epoch

Each point contains:

- `histograms.png`: weight distributions for representative layers
- `weights.png`: Conv2d kernels or Linear weight heatmaps
- `statistics.json`: mean, standard deviation, min, max, norm, zero ratio,
  and non-finite counts

At most six representative Conv2d or Linear layers are selected from the
beginning, middle, and end of the model. Duplicate diagnostic points are
skipped for short runs.

## Implementation Units

1. Initialize Git and capture the current project as the initial commit.
2. Add this implementation plan.
3. Implement metric accumulation, history persistence, and logging.
4. Implement run storage and checkpoint management.
5. Implement weight diagnostics.
6. Implement the trainer and connect the complete workflow in `train.py`.
7. Perform a static review only.

Each completed unit is committed locally. Per request, runtime validation,
training runs, and tests are skipped.
