"""Train basic network."""
from typing import Final
import torch
from .data import get_datasets
from .network import TrPredictor, SharedFanTuner
from .tools import train_tunable_model

torch.manual_seed(1337)
# tensor cores on
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"


def main() -> None:
    """Train model and evaluate."""
    N_TR: Final = 4
    # these two seem to be locked together
    EMB_SIZE: Final = 32
    BLOCK_SIZE: Final = 32
    NUM_HEADS: Final = 4
    BATCH_SIZE: Final = 32
    N_EPOCHS: Final = 10
    GRAD_CLIP: Final = 300
    EVAL_BATCH_SIZE: Final = int(2**11)
    REPORT_STRIDE: Final = 5
    COMPILE: Final = True

    train_dataset, valid_dataset = get_datasets(device=DEVICE)

    underlying_model = TrPredictor(
        alphabet_size=256,
        n_transformers=N_TR,
        emb_size=EMB_SIZE,
        block_size=BLOCK_SIZE,
        num_heads=NUM_HEADS,
    )
    model = SharedFanTuner(underlying_model, n_heads=8).to(DEVICE)
    opter = torch.optim.AdamW(
        model.parameters(), lr=3e-4, fused=True, weight_decay=0.005
    )

    train_tunable_model(
        model=model,
        optimizer=opter,
        device=DEVICE,
        n_epochs=N_EPOCHS,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        train_batch_size=BATCH_SIZE,
        reporting_batch_size=EVAL_BATCH_SIZE,
        compile=COMPILE,
        grad_clip=GRAD_CLIP,
        report_stride=REPORT_STRIDE
    )


if __name__ == "__main__":
    main()
