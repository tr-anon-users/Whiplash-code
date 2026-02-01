import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data(data_path, link_path, seq_len=36, pred_len=1, batch_size=32):
    df = pd.read_csv(data_path)
    link = pd.read_csv(link_path)

    df['LINK_ID'] = df['LINK_ID'].astype(int)
    link['s_link'] = link['s_link'].astype(int)

    link_list = link['s_link'].unique().tolist()
    df = df[df['LINK_ID'].isin(link_list)].copy()

    df['datetime'] = pd.to_datetime(
        df[['PRCS_YEAR', 'PRCS_MON', 'PRCS_DAY', 'PRCS_HH', 'PRCS_MIN']].rename(
            columns={'PRCS_YEAR': 'year', 'PRCS_MON': 'month', 'PRCS_DAY': 'day',
                     'PRCS_HH': 'hour', 'PRCS_MIN': 'minute'}
        )
    )

    df = df.sort_values(by=['datetime', 'LINK_ID'])

    X, Y, scaler = create_sequences(df, link_list, seq_len, pred_len)

    train_size = int(len(X) * 0.80)
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

    print(f"[INFO] Loaded data: {len(X_train)} train, {len(X_val)} validation samples")

    return train_loader, val_loader, scaler

def create_sequences(df, link_ids, seq_len, pred_len):
    df = df[df['LINK_ID'].isin(link_ids)].copy()
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['weekday'] = df['datetime'].dt.weekday

    spd_df = df.pivot(index='datetime', columns='LINK_ID', values='PRCS_SPD')
    delta_df = spd_df.diff().fillna(0)
    lag_df = spd_df.shift(1).fillna(0)

    scaler = MinMaxScaler()
    spd_scaled = scaler.fit_transform(spd_df)
    delta_scaled = MinMaxScaler().fit_transform(delta_df)
    lag_scaled = MinMaxScaler().fit_transform(lag_df)

    data = np.stack([spd_scaled, delta_scaled, lag_scaled], axis=-1)
    spd_target = np.expand_dims(spd_scaled, axis=-1)  # (T, L, 1)

    inputs, targets = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        input_seq = data[i:i + seq_len]
        target_seq = spd_target[i + seq_len : i + seq_len + pred_len]
        inputs.append(input_seq)
        targets.append(target_seq)

    return (
        torch.tensor(np.array(inputs), dtype=torch.float32),     # (B, seq_len, L, F)
        torch.tensor(np.array(targets), dtype=torch.float32),    # (B, pred_len, L, 1)
        scaler
    )
