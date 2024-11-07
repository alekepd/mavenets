"""Train basic network."""
from typing import Final
import torch
import pandas as pd  # type: ignore
from .data import get_datasets
from .network import TrPredictor, SharedFanTuner
from .tools import train_tunable_model

torch.manual_seed(1337)
# tensor cores on
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"
REPORT_STRIDE: Final = 5


def test_transformer(
    n_blocks: int,
    n_heads: int,
    embedding_size: int,
    compile: bool = True,
    batch_size: int = 32,
    eval_batch_size: int = int(2**11),
    learning_rate: float = 3e-4,
    weight_decay: float = 0.005,
    n_epochs: int = 500,
    grad_clip: int = 300,
    fan_size: int = 16,
) -> pd.DataFrame:
    """Train model and evaluate."""
    # these two seem to be locked together

    train_dataset, valid_dataset = get_datasets(device=DEVICE)

    underlying_model = TrPredictor(
        alphabet_size=256,
        n_transformers=n_blocks,
        emb_size=embedding_size,
        block_size=embedding_size,
        num_heads=n_heads,
    )
    model = SharedFanTuner(underlying_model, n_heads=8, fan_size=fan_size).to(DEVICE)
    opter = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        fused=True,
        weight_decay=weight_decay,
    )

    results = train_tunable_model(
        model=model,
        optimizer=opter,
        device=DEVICE,
        n_epochs=n_epochs,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        train_batch_size=batch_size,
        reporting_batch_size=eval_batch_size,
        compile=compile,
        grad_clip=grad_clip,
        report_stride=REPORT_STRIDE,
        progress_bar=True,
    )

    return results


def main() -> None:
    WEIGHT_DECAYS: Final = [0.01, 0.002, 0.0001, 0.0, 0.005]
    N_BLOCKS: Final = 4
    N_HEADS: Final = 8
    EMBEDDING_SIZE: Final = 32
    for w in WEIGHT_DECAYS:
        epoch, val, table = test_transformer(
            n_blocks=N_BLOCKS,
            n_heads=N_HEADS,
            embedding_size=EMBEDDING_SIZE,
            weight_decay=w,
        )
        print(w, epoch, round(val,3))
        table.to_csv("b{}_h{}_e{}_wdecay{}.csv".format(N_BLOCKS,N_HEADS,EMBEDDING_SIZE,w))


if __name__ == "__main__":
    main()
