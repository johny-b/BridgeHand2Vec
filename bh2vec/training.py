import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend; avoids opening a GUI window on macOS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import bridge_helpers

TRUMPS = bridge_helpers.TRUMPS  # ['C', 'D', 'H', 'S', 'NT']


def train(
    net,
    train_fname,
    val_fname,
    run_dir,
    epochs,
    batch_size=10000,
    lr=5e-3,
    weight_decay=0.0,
    lr_gamma=0.999,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

    device = next(net.parameters()).device

    df = pd.read_csv(train_fname, sep=';', index_col=0)
    X, Y = inputs_to_numpy(df)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    df_val = pd.read_csv(val_fname, sep=';', index_col=0)
    X_val, Y_val = inputs_to_numpy(df_val)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).to(device)

    train_ds = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    config = {
        'train_fname': str(train_fname),
        'val_fname': str(val_fname),
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'lr_gamma': lr_gamma,
        'n_train': int(X_tensor.shape[0]),
        'n_val': int(X_val_tensor.shape[0]),
        'net_repr': repr(net),
        'n_params': int(sum(p.numel() for p in net.parameters())),
    }
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'[{run_dir}] params={config["n_params"]} train={config["n_train"]} val={config["n_val"]}')

    history = {
        'epoch': [],
        'lr': [],
        'train_mse': [],
        'train_mae': [],
        'val_mse': [],
        'val_mae': [],
        'val_within_0.5': [],
        'val_within_1': [],
        'val_within_2': [],
        'val_round_eq_0': [],
        'val_round_le_1': [],
        'val_round_le_2': [],
        # per-trump val MAE, one list per trump
        **{f'val_mae_{t}': [] for t in TRUMPS},
    }

    t0 = time.time()
    best_val_mse = float('inf')
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_mse_sum = 0.0
        epoch_mae_sum = 0.0
        epoch_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            out = net(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            bs = batch_y.shape[0]
            epoch_mse_sum += loss.item() * bs
            epoch_mae_sum += (out - batch_y).abs().mean().item() * bs
            epoch_count += bs

        scheduler.step()

        train_mse = epoch_mse_sum / epoch_count
        train_mae = epoch_mae_sum / epoch_count

        val_metrics = _evaluate(net, X_val_tensor, Y_val_tensor)

        history['epoch'].append(epoch)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['train_mse'].append(train_mse)
        history['train_mae'].append(train_mae)
        history['val_mse'].append(val_metrics['mse'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_within_0.5'].append(val_metrics['within_0.5'])
        history['val_within_1'].append(val_metrics['within_1'])
        history['val_within_2'].append(val_metrics['within_2'])
        history['val_round_eq_0'].append(val_metrics['round_eq_0'])
        history['val_round_le_1'].append(val_metrics['round_le_1'])
        history['val_round_le_2'].append(val_metrics['round_le_2'])
        for i, t in enumerate(TRUMPS):
            history[f'val_mae_{t}'].append(val_metrics['mae_per_trump'][i])

        torch.save(net.state_dict(), run_dir / 'last.pth')
        is_best = val_metrics['mse'] < best_val_mse
        if is_best:
            best_val_mse = val_metrics['mse']
            best_epoch = epoch
            torch.save(net.state_dict(), run_dir / 'best.pth')

        elapsed = time.time() - t0
        print(
            f'epoch {epoch:4d}/{epochs} | '
            f'train_mse {train_mse:.4f} val_mse {val_metrics["mse"]:.4f} | '
            f'val_mae {val_metrics["mae"]:.4f} | '
            f'val<=1 {val_metrics["within_1"]:.3f} val_round=0 {val_metrics["round_eq_0"]:.3f} | '
            f'lr {optimizer.param_groups[0]["lr"]:.2e} | '
            f'{elapsed:.0f}s'
            f'{"  <- new best" if is_best else ""}'
        )

        _save_plot(history, run_dir / 'plot.png')
        with (run_dir / 'metrics.json').open('w') as f:
            json.dump(history, f, indent=2)

    print(f'Best val_mse {best_val_mse:.4f} at epoch {best_epoch} -> {run_dir / "best.pth"}')
    return history


@torch.no_grad()
def _evaluate(net, X_val, Y_val):
    net.eval()
    out = net(X_val)
    diff = out - Y_val
    mae_per_sample_per_trump = diff.abs()  # (N, 5)
    mae_per_trump = mae_per_sample_per_trump.mean(dim=0).cpu().numpy()
    mae = mae_per_sample_per_trump.mean().item()
    mse = (diff ** 2).mean().item()

    # within-tolerance (MAE-style: any element within k)
    within_0_5 = (mae_per_sample_per_trump <= 0.5).float().mean().item()
    within_1 = (mae_per_sample_per_trump <= 1.0).float().mean().item()
    within_2 = (mae_per_sample_per_trump <= 2.0).float().mean().item()

    # rounded accuracy
    out_round = torch.round(out)
    y_round = torch.round(Y_val)
    err_round = (out_round - y_round).abs()
    round_eq_0 = (err_round == 0).float().mean().item()
    round_le_1 = (err_round <= 1).float().mean().item()
    round_le_2 = (err_round <= 2).float().mean().item()

    return {
        'mse': mse,
        'mae': mae,
        'mae_per_trump': mae_per_trump.tolist(),
        'within_0.5': within_0_5,
        'within_1': within_1,
        'within_2': within_2,
        'round_eq_0': round_eq_0,
        'round_le_1': round_le_1,
        'round_le_2': round_le_2,
    }


def _save_plot(history, path):
    epochs = history['epoch']
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    ax = axes[0, 0]
    ax.plot(epochs, history['train_mse'], label='train')
    ax.plot(epochs, history['val_mse'], label='val')
    ax.set_title('MSE')
    ax.set_xlabel('epoch')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, history['train_mae'], label='train')
    ax.plot(epochs, history['val_mae'], label='val')
    ax.set_title('MAE')
    ax.set_xlabel('epoch')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 2]
    ax.plot(epochs, history['val_within_0.5'], label='|err| <= 0.5')
    ax.plot(epochs, history['val_within_1'], label='|err| <= 1')
    ax.plot(epochs, history['val_within_2'], label='|err| <= 2')
    ax.set_title('Val accuracy (within tolerance)')
    ax.set_xlabel('epoch')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    for t in TRUMPS:
        ax.plot(epochs, history[f'val_mae_{t}'], label=t)
    ax.set_title('Val MAE per trump')
    ax.set_xlabel('epoch')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(epochs, history['lr'])
    ax.set_title('Learning rate')
    ax.set_xlabel('epoch')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(epochs, history['val_round_eq_0'], label='rounded == truth')
    ax.plot(epochs, history['val_round_le_1'], label='|round err| <= 1')
    ax.plot(epochs, history['val_round_le_2'], label='|round err| <= 2')
    ax.set_title('Val accuracy (rounded predictions)')
    ax.set_xlabel('epoch')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def inputs_to_numpy(data, swap=True):
    rowsX = []
    rowsY = []

    for _, row in data.iterrows():
        binaryN = bridge_helpers.pbn_to_binary(row['N']).reshape(1, -1)
        binaryS = bridge_helpers.pbn_to_binary(row['S']).reshape(1, -1)
        rowsX.append(np.concatenate((binaryN, binaryS), axis=1))
        if swap:
            rowsX.append(np.concatenate((binaryS, binaryN), axis=1))
        resN = ['N' + t for t in bridge_helpers.TRUMPS]
        resS = ['S' + t for t in bridge_helpers.TRUMPS]
        rowsY.append(np.array(row[resN]).reshape(1, -1))
        if swap:
            rowsY.append(np.array(row[resS]).reshape(1, -1))

    X = np.concatenate(rowsX, axis=0)
    Y = np.concatenate(rowsY, axis=0).astype('float32')
    return X, Y
