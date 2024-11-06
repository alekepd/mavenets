"""Train basic network."""
from typing import Final, Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import data
import network
from tqdm import tqdm  # type: ignore
import pandas as pd  # type: ignore

torch.manual_seed(1337)
# tensor cores on
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"


def main() -> None:
    """Train model and evaluate."""
    BATCH_SIZE: Final = 32
    N_EPOCHS: Final = 5000
    N_TR: Final = 4
    # these two seem to be locked together
    EMB_SIZE: Final = 32
    BLOCK_SIZE: Final = 32
    NUM_HEADS: Final = 4
    GRAD_CLIP: Final = 300
    RAW_LOSS_MIXIN: Final = [0.9] * 0 + list(np.linspace(0.9, 0.00001, 10))
    EVAL_BATCH_SIZE: Final = int(2**11)
    REPORT_STRIDE: Final = 5

    train_dataset, valid_dataset = data.get_data(device=DEVICE)

    model = network.TrPredictor(
        n_calibrators=8,
        alphabet_size=256,
        n_transformers=N_TR,
        emb_size=EMB_SIZE,
        block_size=BLOCK_SIZE,
        num_heads=NUM_HEADS,
    ).to(DEVICE)
    opter = torch.optim.AdamW(
        model.parameters(), lr=3e-4, fused=True, weight_decay=0.005
    )
    loss_f = nn.MSELoss()
    raw_loss_f = nn.MSELoss()

    def _train_eval(
        inp: torch.Tensor,
        out: torch.Tensor,
        experiment: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            cal_pred, raw_pred = model(inp, experiment, include_raw=True)
            cal_loss = loss_f(cal_pred, out)
            raw_loss = raw_loss_f(raw_pred, out)
        return cal_loss, raw_loss

    def _train_step(
        inp: torch.Tensor,
        out: torch.Tensor,
        experiment: torch.Tensor,
        mix_raw: torch.Tensor,
        mix_cal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model.train()
        cal_loss, raw_loss = _train_eval(inp, out, experiment)
        # was this: loss = cal_loss + mix * raw_loss
        loss = mix_cal * cal_loss + mix_raw * raw_loss
        opter.zero_grad()
        loss.backward()
        opter.step()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        return cal_loss, raw_loss, gnorm

    _compiled_train_step = torch.compile(_train_step, mode="reduce-overhead")

    e_pbar = tqdm(range(N_EPOCHS))
    for e in e_pbar:
        try:
            mix_raw = torch.as_tensor(RAW_LOSS_MIXIN.pop(0), device=DEVICE)
            mix_cal = 1.0 - mix_raw
        except IndexError:
            pass
        for seq, signal, experiment in DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        ):
            loss, raw_loss, gnorm = _compiled_train_step(
                seq, signal, experiment, mix_raw=mix_raw, mix_cal=mix_cal
            )
        if e % REPORT_STRIDE == 0:
            with torch.no_grad():
                model.eval()

                val_losses = []
                raw_val_losses = []
                pre_table = []
                for inp, out, experiment in DataLoader(
                    valid_dataset, batch_size=EVAL_BATCH_SIZE
                ):
                    pred, raw_pred = model(inp, experiment, include_raw=True)
                    val_losses.append(loss_f(pred, out).item())
                    raw_val_losses.append(loss_f(raw_pred, out).item())
                    pre_table.append(
                        (
                            experiment.cpu().numpy(),
                            out.cpu().numpy(),
                            pred.cpu().numpy(),
                            raw_pred.cpu().numpy(),
                        )
                    )
                catted = {
                    ind: np.concatenate(x) for ind, x in enumerate(zip(*pre_table))
                }
                pd.DataFrame(catted).to_csv("live_output.csv")

                train_losses: List[float] = []
                raw_train_losses: List[float] = []
                for inp, out, experiment in DataLoader(
                    train_dataset, batch_size=EVAL_BATCH_SIZE
                ):
                    pred, raw_pred = model(inp, experiment, include_raw=True)
                    train_losses.append(loss_f(pred, out).item())
                    raw_train_losses.append(loss_f(raw_pred, out).item())

                val_l = round(sum(val_losses) / len(val_losses), 3)
                val_rl = round(sum(raw_val_losses) / len(raw_val_losses), 3)
                train_l = round(sum(train_losses) / len(train_losses), 3)
                train_rl = round(sum(raw_train_losses) / len(raw_train_losses), 3)

                e_pbar.set_description(str((val_l, val_rl, mix_raw.item())))

                print(e, ":", val_l, val_rl, train_l, train_rl, mix_raw.item())


if __name__ == "__main__":
    main()
