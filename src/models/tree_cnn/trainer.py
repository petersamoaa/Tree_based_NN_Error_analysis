"""
A script for training a neural network on tree-structured data.
"""

import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from metrics import pearson_corr_v2 as pearson_corr
from metrics import report_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def node_batch_samples(samples, batch_size, node_map):
    batch = ([], [])
    count = 0
    index_of = lambda x: node_map[x]
    for sample in samples:
        if sample['parent'] is not None:
            batch[0].append(index_of(sample['node']))
            batch[1].append(index_of(sample['parent']))
            count += 1
            if count >= batch_size:
                yield batch
                batch, count = ([], []), 0


def node_trainer(samples, model, node_map, device, lr=1e-3, batch_size=8, epochs=1, checkpoint=1000, output_dir=None):
    output_dir = output_dir if output_dir and len(output_dir) > 0 else None
    model = model.to(device)

    # Cross Entropy loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize step count
    step_count = 0

    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        sample_gen = node_batch_samples(samples, batch_size, node_map=node_map)
        total_samples = len(samples)

        for i, batch in enumerate(sample_gen):
            inputs, labels = batch
            inputs = torch.LongTensor(inputs).to(device)
            labels = torch.LongTensor(labels).to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(inputs)

            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update step count
            step_count += 1

            # Checkpoint based on steps
            if checkpoint > 0 and step_count % checkpoint == 0:
                logger.info(f'Epoch [{epoch + 1}/{epochs}], Step [{step_count}], Loss: {total_loss / (i + 1):.4f}')
                # torch.save(net.state_dict(), f'checkpoint_step_{step_count}.pth')

        # Log per-epoch statistics
        msg = f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / total_samples:.4f}'
        logger.info(msg)

        if output_dir:
            torch.save(model.state_dict(), os.path.join(output_dir, f'embedding_model.pth'))
            with open(os.path.join(output_dir, 'embedding_output.txt'), "w", encoding="utf-8") as f:
                f.write(f"msg: {msg}")

            torch.save(model.embeddings.weight.data.cpu().numpy(), os.path.join(output_dir, 'embeddings.bin'))
            torch.save(node_map, os.path.join(output_dir, 'node_map.bin'))

    return model


def gen_samples(records, vectors, vector_lookup):
    """Creates a generator that returns a tree in BFS order with each node
    replaced by its vector embedding, and a child lookup table."""

    for row in records:
        tree, label = row["_tree"], row["y"]

        nodes = []
        children = []
        label = [label]
        queue = [(tree, -1)]
        while queue:
            node, parent_ind = queue.pop(0)
            node_ind = len(nodes)
            # add children and the parent index to the queue
            queue.extend([(child, node_ind) for child in node['children']])
            # create a list to store this node's children indices
            children.append([])
            # add this child to its parent's child list
            if parent_ind > -1:
                children[parent_ind].append(node_ind)
            

            nodes.append(vectors[vector_lookup.get(node["node"], vector_lookup[None])])

        yield nodes, children, label, row


def batch_samples(gen, batch_size):
    """Batch samples from a generator"""
    nodes, children, labels, records = [], [], [], []
    samples = 0
    for n, c, l, r in gen:
        nodes.append(n)
        children.append(c)
        labels.append(l)
        records.append(r)

        samples += 1
        if samples >= batch_size:
            yield _pad_batch(nodes, children, labels, records)
            nodes, children, labels, records = [], [], [], []
            samples = 0

    if nodes:
        yield _pad_batch(nodes, children, labels, records)


def _pad_batch(nodes, children, labels, records):
    if not nodes:
        return [], [], [], []

    max_nodes = max([len(x) for x in nodes])
    max_children = max([len(x) for x in children])
    feature_len = len(nodes[0][0])
    child_len = max([len(c) for n in children for c in n])

    nodes = [n + [[0.0] * feature_len] * (max_nodes - len(n)) for n in nodes]
    # pad batches so that every batch has the same number of nodes
    children = [n + ([[]] * (max_children - len(n))) for n in children]
    # pad every child sample so every node has the same number of children
    children = [[c + [0.0] * (child_len - len(c)) for c in sample] for sample in children]

    return nodes, children, labels, records


def trainer(model, train_trees, test_trees, y_scaler, embeddings, embed_lookup, device, lr=1e-3, batch_size=8, epochs=1, checkpoint=1000, output_dir=None):
    output_dir = output_dir if output_dir and len(output_dir) > 0 else None
    # Cross Entropy loss
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize step count
    total_samples = len(train_trees)
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

        for i, batch in enumerate(batch_samples(
            gen_samples(train_trees, embeddings, embed_lookup),
            batch_size
        )):
            nodes, children, batch_labels, batch_records = batch

            nodes = torch.from_numpy(np.array(nodes)).to(device)
            children = torch.LongTensor(children).to(device)
            batch_labels = y_scaler.transform(batch_labels) if y_scaler else batch_labels
            batch_labels = torch.tensor(batch_labels).float().to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(nodes, children)

            y_pred = logits.cpu().detach().flatten().numpy().tolist()
            y_true = batch_labels.cpu().detach().flatten().numpy().tolist()

            item_list += [r["name"] for r in batch_records] if isinstance(batch_records, list) else [batch_records["name"]]
            y_pred_list += y_pred if isinstance(y_pred, list) else [y_pred]
            y_true_list += y_true if isinstance(y_true, list) else [y_true]

            # Compute loss
            loss = criterion(logits, batch_labels)
            total_loss += loss.item()

            # Backward pass and optimization
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

    total_samples = len(test_trees)
    total_loss = 0.0

    # P-CORR
    item_list = []
    y_pred_list = []
    y_true_list = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(batch_samples(
            gen_samples(test_trees, embeddings, embed_lookup),
            batch_size
        )):
            nodes, children, batch_labels, batch_records = batch
            nodes = torch.from_numpy(np.array(nodes)).to(device)
            children = torch.LongTensor(children).to(device)
            batch_labels = y_scaler.transform(batch_labels) if y_scaler else batch_labels
            batch_labels = torch.tensor(batch_labels).float().to(device)

            # Forward pass
            logits = model(nodes, children)

            # y_pred_list.append(logits.item())
            # y_true_list.append(batch_labels.item())
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

        eval_metrics = report_metrics(y_pred_list.tolist(), y_true_list.tolist())
        metrics.update({"eval": eval_metrics})
        eval_msg = f'Loss: {total_loss / total_samples:.4f} MSE: {eval_metrics["mse"]:.4f} MAE: {eval_metrics["mae"]:.4f} P-CORR: {eval_metrics["pcorr"]:.4f}'
        logger.info(eval_msg)

    if output_dir:
        torch.save(model.state_dict(), os.path.join(output_dir, f'model.pth'))
        with open(os.path.join(output_dir, 'output.json'), "w", encoding="utf-8") as fj:
            json.dump(metrics, fj, indent=2)

        with open(os.path.join(output_dir, 'eval_pred.csv'), "w", encoding="utf-8") as f:
            f.write(f"FILE\tTRUE\tPRED\n")
            for k, i, j in zip(item_list, y_true_list.tolist(), y_pred_list.tolist()):
                f.write(f"{k}\t{i}\t{j}\n")

    return model
