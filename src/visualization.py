import matplotlib.pyplot as plt


def display_matrices(matrices: list, titles: list) -> None:
    count = len(matrices)

    fig, axes = plt.subplots(1, count, figsize=(5 * count, 5))

    for matrix, title, ax in zip(matrices, titles, axes):
        image = ax.imshow(matrix, cmap="gray", interpolation="nearest")
        ax.set_title(title)
        fig.colorbar(image, ax=ax)

    plt.tight_layout()
    plt.show()
