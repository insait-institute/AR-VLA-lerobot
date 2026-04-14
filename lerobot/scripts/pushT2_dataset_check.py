#!/usr/bin/env python

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Iterator
import numpy as np
import torch
import torch.utils.data
import tqdm
from collections import Counter

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from sklearn.cluster import KMeans


def weighted_lbg_vector_quantization(data, target_size=1024, split_epsilon=0.01, max_iters=20, center_boost=50.0):
    """
    LBG with Importance Sampling to force scanners into the center.
    
    Args:
        center_boost (float): How much heavier central points are. 
                              Higher value = more scanners in the center.
    """
    data = data.float()
    N, dim = data.shape
    
    # --- STEP 0: Calculate Weights ---
    # We define "Center" as the box from -5 to 5.
    # You can change this logic to be a radial distance if preferred.
    in_center_x = (data[:, 0] >= -5) & (data[:, 0] <= 5)
    in_center_y = (data[:, 1] >= -5) & (data[:, 1] <= 5)
    in_center_mask = in_center_x & in_center_y
    
    # Initialize weights
    weights = torch.ones(N, device=data.device)
    
    # Boost the weight of central points
    # Effectively, one point in the center counts as 'center_boost' points.
    weights[in_center_mask] = center_boost
    
    # Normalize weights so they sum to N (optional, but keeps math stable)
    weights = weights / weights.mean()
    
    # Reshape for broadcasting
    weights = weights.unsqueeze(1) 

    # --- Step 1: Initialization ---
    # Weighted mean of all data
    # sum(w * x) / sum(w)
    codebook = (data * weights).sum(dim=0, keepdim=True) / weights.sum()
    current_size = 1
    
    print(f"Starting Weighted LBG. Center boost: {center_boost}x")

    # --- Step 2: The Splitting Loop ---
    while current_size < target_size:
        # A. SPLIT
        epsilon_vec = torch.randn_like(codebook) * split_epsilon
        codebook_plus = codebook + epsilon_vec
        codebook_minus = codebook - epsilon_vec
        codebook = torch.cat([codebook_plus, codebook_minus], dim=0)
        current_size = codebook.shape[0]
        
        # B. RELAXATION (Weighted K-Means)
        for i in range(max_iters):
            # 1. Distance
            dists = torch.cdist(data, codebook)
            
            # 2. Assignment
            labels = torch.argmin(dists, dim=1)
            
            # 3. Update Centroids (Weighted)
            # Numerator: Sum of (weights * positions) for each cluster
            # Denominator: Sum of weights for each cluster
            
            # We use index_add_ to sum up the weighted data
            weighted_data = data * weights
            numerator = torch.zeros_like(codebook)
            numerator.index_add_(0, labels, weighted_data)
            
            # We sum up the weights for each cluster
            denominator = torch.zeros(current_size, 1, device=data.device)
            denominator.index_add_(0, labels, weights)
            
            # Avoid div by zero
            mask = denominator.squeeze() > 0
            
            new_codebook = torch.clone(codebook)
            new_codebook[mask] = numerator[mask] / denominator[mask]
            
            # Check shift
            shift = torch.norm(new_codebook - codebook).item()
            codebook = new_codebook
            if shift < 1e-4:
                break
                
    return codebook
class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def analyze_actions(
    dataset: LeRobotDataset,
    batch_size: int = 32,
    num_workers: int = 0,
) -> None:
    """Analyze all actions in the dataset to find unique values."""
    
    logging.info("Loading full dataset to analyze actions")
    
    # Create dataloader for entire dataset
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )
    
    # Collect all actions
    all_actions = []
    delta_actions = []
    total_samples = 0
    
    logging.info("Collecting all actions...")
    
    for batch in tqdm.tqdm(dataloader, desc="Processing batches"):
        if "action" in batch:
            # Convert to numpy and round to nearest integers
            actions = batch["action"].numpy()
            states = batch["observation.state"].numpy()
            # Round to handle values like 10.0 -> 10
            actions_rounded = np.round(actions).astype(int)
            delta_action = actions - states
            all_actions.append(actions_rounded)
            delta_actions.append(delta_action)
            total_samples += len(actions_rounded)
    
    # Concatenate all actions
    if all_actions:
        all_actions = np.concatenate(all_actions, axis=0)
        logging.info(f"Total action samples: {total_samples}")
        logging.info(f"Action shape: {all_actions.shape}")
        
        # Find unique actions
        unique_actions = np.unique(all_actions, axis=0)
        logging.info(f"Number of unique actions: {len(unique_actions)}")
        
        # Print some statistics
        print(f"\n=== ACTION ANALYSIS RESULTS ===")
        print(f"Total action samples: {total_samples}")
        print(f"Action dimensionality: {all_actions.shape[1]}D")
        print(f"Number of unique actions: {len(unique_actions)}")
        
        # Show action ranges
        min_vals = np.min(all_actions, axis=0)
        max_vals = np.max(all_actions, axis=0)
        print(f"Action ranges:")
        for dim in range(all_actions.shape[1]):
            print(f"  Dimension {dim}: [{min_vals[dim]}, {max_vals[dim]}]")
        
        # Show most common actions
        action_tuples = [tuple(action) for action in all_actions]
        action_counts = Counter(action_tuples)
        most_common = action_counts.most_common(10)
        
        print(f"\nTop 10 most frequent actions:")
        for i, (action, count) in enumerate(most_common, 1):
            percentage = (count / total_samples) * 100
            print(f"  {i:2d}. {action} - {count:6d} times ({percentage:.2f}%)")
        
        # Show unique actions (if not too many)
        if len(unique_actions) <= 50:
            print(f"\nAll unique actions:")
            for i, action in enumerate(unique_actions):
                count = action_counts[tuple(action)]
                percentage = (count / total_samples) * 100
                print(f"  {tuple(action)} - {count} times ({percentage:.2f}%)")
        else:
            print(f"\nToo many unique actions to display ({len(unique_actions)})")
            print("Showing first 20:")
            for i in range(20):
                action = unique_actions[i]
                count = action_counts[tuple(action)]
                percentage = (count / total_samples) * 100
                print(f"  {tuple(action)} - {count} times ({percentage:.2f}%)")
                
        vocab_size = 500
        
        # DO PER-DIM K-MEANS CLUSTERING
        per_dim_centroids = []
        for dim in range(all_actions.shape[1]):
            print(f"Fitting K-Means with {vocab_size} clusters on dimension {dim} with {all_actions.shape[0]} points...")
            kmeans = KMeans(n_clusters=vocab_size, n_init=10, max_iter=300)
            kmeans.fit(all_actions[:, dim].reshape(-1, 1))
            per_dim_centroids.append(kmeans.cluster_centers_.flatten())
            
        # Combine per-dim centroids into one tensor, into [dim, vocab_size]
        centroids = torch.tensor(np.stack(per_dim_centroids, axis=0), dtype=torch.float32)
        torch.save(centroids, "tokenizer/pushT2_per_dim_kmeans_centers.pt")
        # 3. Save the centroids
        # These are the geometric coordinates of your tokens
        # centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        # torch.save(centroids, "tokenizer/cube_kmeans_centers.pt")

        print("Saved 'kmeans_centers.pt'. You can now use the tokenizer.")
        

                
        # Additional analysis for delta actions
        if delta_actions and False:
            delta_actions = np.concatenate(delta_actions, axis=0)
            logging.info(f"Delta action shape: {delta_actions.shape}")
            
            print(f"doing the new LBG clustering for delta actions...")
            data_tensor = torch.tensor(delta_actions, dtype=torch.float32)
            centroids = weighted_lbg_vector_quantization(data_tensor, target_size=1024)
            torch.save(centroids, "lbg_delta_centers.pt")
            
            
            min_delta_vals = np.min(delta_actions, axis=0)
            max_delta_vals = np.max(delta_actions, axis=0)
            print(f"\nDelta action ranges:")
            for dim in range(delta_actions.shape[1]):
                print(f"  Dimension {dim}: [{min_delta_vals[dim]}, {max_delta_vals[dim]}]")
            # print a heatmap of delta actions frequency, because we are headless just save to a file
            # try:
            #     import matplotlib.pyplot as plt
            #     import seaborn as sns
                
            #     # print 2d heatmap for first two dimensions of delta actions!!!
                
            #     plt.figure(figsize=(8, 6))
            #     sns.jointplot(x=delta_actions[:, 0], y=delta_actions[:, 1], bins=50, kind="hex")
            #     plt.suptitle("Delta Action Heatmap (Dimensions 0 and 1)")
            #     plt.xlabel("Delta Action Dimension 0")
            #     plt.ylabel("Delta Action Dimension 1")
            #     plt.savefig("delta_action_heatmap.png")
            #     logging.info("Saved delta action heatmap to delta_action_heatmap.png")
                
            #     # also plot heatmap of action distribution for first two dimensions
            #     plt.figure(figsize=(8, 6))
            #     sns.jointplot(x=all_actions[:, 0], y=all_actions[:, 1], bins=50, kind="hex")
            #     plt.suptitle("Action Heatmap (Dimensions 0 and 1)")
            #     plt.xlabel("Action Dimension 0")
            #     plt.ylabel("Action Dimension 1")
            #     plt.savefig("action_heatmap.png")
            #     logging.info("Saved action heatmap to action_heatmap.png")
                
            # except ImportError:
            #     logging.warning("matplotlib or seaborn not installed, skipping delta action histogram.")
    
    else:
        logging.error("No actions found in the dataset!")


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    analyze_all_actions: bool = False,
) -> None:
    """Original visualization function with optional action analysis."""
    
    if analyze_all_actions:
        analyze_actions(dataset, batch_size, num_workers)
        return
    
    # Original episode-specific visualization code
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    logging.info("Logging episode data")

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        # iterate over the batch
        for i in range(len(batch["index"])):
            # display each dimension of action space (e.g. actuators command)
            if "action" in batch:
                for dim_idx, val in enumerate(batch["action"][i]):
                    print(f"action/{dim_idx}", val.item())

            # display each dimension of observed state space (e.g. agent position in joint space)
            if "observation.state" in batch:
                for dim_idx, val in enumerate(batch["observation.state"][i]):
                    print(f"state/{dim_idx}", val.item())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        default="you2who/pusht2-teleop",
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode to visualize (ignored if --analyze-all-actions is used).",
    )
    parser.add_argument(
        "--analyze-all-actions",
        action="store_true",
        help="Analyze all actions in the entire dataset instead of visualizing a single episode.",
    )
    parser.add_argument(
        "--local-files-only",
        type=int,
        default=0,
        help="Use local files only. By default, this script will try to fetch the dataset from the hub if it exists.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes of Dataloader for loading the data.",
    )

    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    repo_id = args.repo_id
    root = args.root
    local_files_only = bool(args.local_files_only)

    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, root=root, local_files_only=local_files_only)
    
    visualize_dataset(
        dataset, 
        args.episode_index, 
        args.batch_size, 
        args.num_workers,
        args.analyze_all_actions
    )


if __name__ == "__main__":
    main()