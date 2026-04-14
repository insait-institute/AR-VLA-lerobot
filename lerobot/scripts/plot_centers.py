# load a torch pt file with last dim 2, and plot all the points
import torch
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":

    pt_file = "tokenizer/kmeans_centers.pt"
    data = torch.load(pt_file)

    if data.ndim < 2 or data.shape[-1] != 2:
        print("Error: The loaded tensor must have at least 2 dimensions and the last dimension must be of size 2.")
        sys.exit(1)

    points = data.view(-1, 2).numpy()

    # this is a headless script, so we just plot and save to a file
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], s=1)
    # set the axis to [-2, 2   ] for both x and y
    # plt.xlim(-150, 150)
    # plt.ylim(-150, 150)
    plt.title("2D Points from Torch PT File")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.savefig(f"tokenizer/pusht_plot.png", dpi=300)
    print("Plot saved to points_plot.png")