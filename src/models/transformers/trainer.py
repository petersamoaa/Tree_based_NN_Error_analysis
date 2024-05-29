import json
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# from metrics import pearson_corr_v2 as pearson_corr
from metrics import report_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TBCCDataset(Dataset):
    def __init__(self, data, scaler):
        self.data = data
        self.scaler = scaler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        y = self.scaler.transform([[self.data[idx]["y"]]]) if self.scaler else [[self.data[idx]["y"]]]

        return {
            "code_input_ids": torch.tensor(self.data[idx]["code_input_ids"]).long(),
            "code_attention_mask": torch.tensor(self.data[idx]["code_attention_mask"]).long(),
            "past_input_ids": torch.tensor(self.data[idx]["past_input_ids"]).long(),
            "past_attention_mask": torch.tensor(self.data[idx]["past_attention_mask"]).long(),
            "y": torch.tensor(y).float()[0],
            "row": self.data[idx]
        }


def padding(x_list, max_seq_length=None, pad_token=0):
    # Truncate if necessary and pad each sequence to max_seq_length
    if max_seq_length is not None:
        x_padded = []
        for x in x_list:
            x = x[:max_seq_length]  # Truncate
            pad_size = max_seq_length - len(x)  # Calculate padding size
            if pad_size > 0:
                # Pad sequence to the desired length
                x = F.pad(x, (0, pad_size), 'constant', pad_token)
            x_padded.append(x)
        x_padded = torch.stack(x_padded)
    else:
        x_padded = pad_sequence(x_list, batch_first=True, padding_value=pad_token)

    return x_padded


def collate_fn(batch, max_seq_length=None):
    code_input_ids = [item['code_input_ids'] for item in batch]
    code_attention_mask = [item['code_attention_mask'] for item in batch]
    past_input_ids = [item['past_input_ids'] for item in batch]
    past_attention_mask = [item['past_attention_mask'] for item in batch]

    y_list = [item['y'] for item in batch]
    records = [item['row'] for item in batch]

    # Truncate if necessary and pad each sequence to max_seq_length
    code_input_ids_padded = padding(code_input_ids, max_seq_length)
    code_attention_mask_padded = padding(code_attention_mask, max_seq_length)

    past_input_ids_padded = padding(past_input_ids, max_seq_length)
    past_attention_mask_padded = padding(past_attention_mask, max_seq_length)

    y_stacked = torch.stack(y_list)

    # return (
    #     code_input_ids_padded, code_attention_mask_padded,
    #     past_input_ids_padded, past_attention_mask_padded,
    #     y_stacked,
    #     records
    # )

    return {
        "code_input_ids": code_input_ids_padded,
        "code_attention_mask": code_attention_mask_padded,
        "past_input_ids": past_input_ids_padded,
        "past_attention_mask": past_attention_mask_padded,
        "labels": y_stacked,
        "records": records
    }


def trainer(
    model, train, test, y_scaler, device, max_seq_length=510,
    lr=1e-3, batch_size=8, epochs=1, checkpoint=1000, output_dir=None,
):
    train_dataset = TBCCDataset(data=train, scaler=y_scaler)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, max_seq_length=max_seq_length))

    test_dataset = TBCCDataset(data=test, scaler=y_scaler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, max_seq_length=max_seq_length))

    for batch in train_loader:
        # logger.info(f"[TRAINING] CODE: {batch[0].shape}, PAST: {batch[2].shape}, Y: {batch[-2].shape}")
        msg = {k: v.shape if torch.is_tensor(v) else len(v) for k, v in batch.items()}
        logger.info(f"[TRAINING] CODE: {msg}")
        break

    for batch in test_loader:
        # logger.info(f"[TESTING] CODE: {batch[0].shape}, PAST: {batch[2].shape}, Y: {batch[-2].shape}")
        msg = {k: v.shape if torch.is_tensor(v) else len(v) for k, v in batch.items()}
        logger.info(f"[TESTING] CODE: {msg}")
        break

    # Cross Entropy loss
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize step count
    total_samples = len(train_dataset)
    step_count = 0
    metrics = {}

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        # P-CORR
        item_list = []
        y_pred_list = []
        y_true_list = []

        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            batch_records = batch["records"]
            batch_labels = batch["labels"]

            # batch_code_input_ids, batch_code_attention_mask, batch_past_input_ids, batch_past_attention_mask, batch_labels, 
            # batch_code_input_ids, batch_code_attention_mask = batch_code_input_ids.to(device), batch_code_attention_mask.to(device)
            # batch_past_input_ids, batch_past_attention_mask = batch_past_input_ids.to(device), batch_past_attention_mask.to(device)
            # batch_labels = batch_labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            # logits = model(
            #     code_input_ids=batch_code_input_ids, 
            #     past_input_ids=batch_past_input_ids,
            #     code_attention_mask=batch_code_attention_mask,
            #     past_attention_mask=batch_past_attention_mask
            # )
            logits = model(**batch)

            y_pred = logits.cpu().detach().flatten().numpy().tolist()
            y_true = batch_labels.cpu().detach().flatten().numpy().tolist()

            item_list += [r["name"] for r in batch_records] if isinstance(batch_records, list) else [batch_records["name"]]
            y_pred_list += y_pred if isinstance(y_pred, list) else [y_pred]
            y_true_list += y_true if isinstance(y_true, list) else [y_true]

            # Compute loss
            loss = criterion(logits, batch_labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            # Update step count
            step_count += 1

            # Checkpoint based on steps
            if checkpoint > 0 and step_count % checkpoint == 0:
                logger.info(f'Epoch [{epoch + 1}/{epochs}], Step [{step_count}], Loss: {total_loss / (i + 1):.4f}')
                # torch.save(net.state_dict(), f'checkpoint_step_{step_count}.pth')

        # Log per-epoch statistics
        if y_scaler:
            # y_pred_list = torch.tensor(y_scaler.inverse_transform(np.array(y_pred_list).reshape(1, -1))).squeeze()
            # y_true_list = torch.tensor(y_scaler.inverse_transform(np.array(y_true_list).reshape(1, -1))).squeeze()
            y_pred_list = torch.tensor(y_pred_list).squeeze()
            y_true_list = torch.tensor(y_true_list).squeeze()
        else:
            y_pred_list = torch.tensor(y_pred_list).squeeze()
            y_true_list = torch.tensor(y_true_list).squeeze()

        # p_corr = pearson_corr(y_pred_list, y_true_list)
        # train_msg = f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / total_samples:.4f}, P-CORR: {p_corr}'
        # logger.info(train_msg)
        train_metric = report_metrics(y_pred_list.tolist(), y_true_list.tolist())
        metrics.update({"train": train_metric})
        train_msg = f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / total_samples:.4f} MSE: {train_metric["mse"]:.4f} MAE: {train_metric["mae"]:.4f} P-CORR: {train_metric["pcorr"]:.4f}'
        logger.info(train_msg)

        if output_dir:
            with open(os.path.join(output_dir, 'train_pred.csv'), "w", encoding="utf-8") as f:
                f.write(f"FILE\tTRUE\tPRED\n")
                for k, i, j in zip(item_list, y_true_list.tolist(), y_pred_list.tolist()):
                    f.write(f"{k}\t{i}\t{j}\n")

    total_samples = len(test_dataset)
    total_loss = 0.0

    # P-CORR
    item_list = []
    y_pred_list = []
    y_true_list = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            batch_records = batch["records"]
            batch_labels = batch["labels"]

            # batch_code_input_ids, batch_code_attention_mask, batch_past_input_ids, batch_past_attention_mask, batch_labels, 
            # batch_code_input_ids, batch_code_attention_mask = batch_code_input_ids.to(device), batch_code_attention_mask.to(device)
            # batch_past_input_ids, batch_past_attention_mask = batch_past_input_ids.to(device), batch_past_attention_mask.to(device)
            # batch_labels = batch_labels.to(device)

            # Forward pass
            # logits = model(
            #     code_input_ids=batch_code_input_ids, 
            #     past_input_ids=batch_past_input_ids,
            #     code_attention_mask=batch_code_attention_mask,
            #     past_attention_mask=batch_past_attention_mask
            # )
            logits = model(**batch)

            y_pred = logits.cpu().detach().flatten().numpy().tolist()
            y_true = batch_labels.cpu().detach().flatten().numpy().tolist()

            item_list += [r["name"] for r in batch_records] if isinstance(batch_records, list) else [batch_records["name"]]
            y_pred_list += y_pred if isinstance(y_pred, list) else [y_pred]
            y_true_list += y_true if isinstance(y_true, list) else [y_true]

            # Compute loss
            loss = criterion(logits, batch_labels)
            total_loss += loss.item()

        # Log per-epoch statistics
        if y_scaler:
            # y_pred_list = torch.tensor(y_scaler.inverse_transform(np.array(y_pred_list).reshape(1, -1))).squeeze()
            # y_true_list = torch.tensor(y_scaler.inverse_transform(np.array(y_true_list).reshape(1, -1))).squeeze()
            y_pred_list = torch.tensor(y_pred_list).squeeze()
            y_true_list = torch.tensor(y_true_list).squeeze()
        else:
            y_pred_list = torch.tensor(y_pred_list).squeeze()
            y_true_list = torch.tensor(y_true_list).squeeze()

        # p_corr = pearson_corr(y_pred_list, y_true_list)
        # eval_msg = f'Loss: {total_loss / total_samples:.4f}, P-CORR: {p_corr}'
        # logger.info(eval_msg)

        eval_metric = report_metrics(y_pred_list.tolist(), y_true_list.tolist())
        metrics.update({"eval": eval_metric})
        eval_msg = f'Loss: {total_loss / total_samples:.4f} MSE: {eval_metric["mse"]:.4f} MAE: {eval_metric["mae"]:.4f} P-CORR: {eval_metric["pcorr"]:.4f}'
        logger.info(eval_msg)

    if output_dir:
        torch.save(model.state_dict(), os.path.join(output_dir, f'model.pth'))
        with open(os.path.join(output_dir, 'output.json'), "w", encoding="utf-8") as fj:
            json.dump(metrics, fj, indent=2)

        with open(os.path.join(output_dir, 'eval_pred.csv'), "w", encoding="utf-8") as f:
            f.write(f"FILE\tTRUE\tPRED\n")
            for k, i, j in zip(item_list, y_true_list.tolist(), y_pred_list.tolist()):
                f.write(f"{k}\t{i}\t{j}\n")
