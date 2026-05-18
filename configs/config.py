import os

BASE_DIR    = "/teamspace/studios/this_studio"
DATA_DIR    = os.path.join(BASE_DIR, "split_dataset")
SAVE_DIR    = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR     = os.path.join(BASE_DIR, "logs")

DATA_CONFIG = {
    "train_dir"   : os.path.join(DATA_DIR, "train"),
    "val_dir"     : os.path.join(DATA_DIR, "val"),
    "test_dir"    : os.path.join(DATA_DIR, "test"),
    "img_size"    : 224,
    "batch_size"  : 32,
    "num_workers" : 4,
    "pin_memory"  : True
}

MODEL_CONFIG = {
    "name"        : "resnet18",
    "pretrained"  : True,
    "dropout"     : 0.3,
}

TRAIN_CONFIG = {
    "epochs"          : 50,
    "lr"              : 1e-4,
    "weight_decay"    : 1e-4,
    "label_smoothing" : 0.1,
    "save_dir"        : SAVE_DIR,
    "log_dir"         : LOG_DIR,
}


SCHEDULER_CONFIG = {
    "name"      : "cosine",     # cosine, step, plateau
    "T_max"     : 50,           # for cosine
    "step_size" : 10,           # for step scheduler
    "gamma"     : 0.1,          # for step scheduler
    "patience"  : 5,            # for plateau scheduler
}

CONFIG = {
    "data"       : DATA_CONFIG,
    "model"      : MODEL_CONFIG,
    "training"   : TRAIN_CONFIG,
    "scheduler"  : SCHEDULER_CONFIG,
}