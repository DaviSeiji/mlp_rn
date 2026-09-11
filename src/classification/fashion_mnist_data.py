from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "results" / "figures"

#Transforma as imagens em tensores e normaliza os valores dos pixels
#Importante pois o MLP recebe entradas numéricas e não imagens
def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])


def check_class_distribution(targets_by_split, class_names, tolerance=0.03):
    if not 0 <= tolerance <= 1:
        raise ValueError("A tolerância deve estar entre 0 e 1.")

    proportions = {}
    for name, targets in targets_by_split.items():
        targets = torch.as_tensor(targets, dtype=torch.long)
        if targets.numel() == 0:
            raise ValueError(f"O conjunto {name} está vazio.")
        counts = torch.bincount(targets, minlength=len(class_names))
        proportions[name] = counts.to(torch.float64) / targets.numel()

    values = torch.stack(list(proportions.values()))
    differences = values.max(dim=0).values - values.min(dim=0).values
    print("\nDistribuição de classes (proporção em cada conjunto):")
    print(f"{'Classe':<24}" + "".join(f"{name:>12}" for name in proportions) + "  Dif. (p.p.)")
    violations = []
    for index, class_name in enumerate(class_names):
        print(
            f"{class_name:<24}"
            + "".join(f"{p[index].item():>11.2%} " for p in proportions.values())
            + f"{differences[index].item() * 100:>10.2f}"
        )
        if differences[index].item() > tolerance + 1e-12:
            violations.append(class_name)

    if violations:
        raise ValueError(
            f"Diferença acima de {tolerance * 100:g} pontos percentuais "
            f"nas classes: {', '.join(violations)}."
        )
    print(f"Distribuições dentro da tolerância de {tolerance * 100:g} p.p.\n")
    return proportions


def plot_class_distribution(proportions, class_names, output_dir=FIGURES_DIR):
    """Salva barras agrupadas em PNG (300 dpi) e PDF vetorial para o relatório."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ("#0072B2", "#E69F00", "#009E73")
    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(13, 6), layout="constrained")
        try:
            width = 0.8 / len(proportions)
            for i, (name, values) in enumerate(proportions.items()):
                positions = [x + (i - (len(proportions) - 1) / 2) * width
                             for x in range(len(class_names))]
                ax.bar(positions, values.detach().cpu().tolist(), width=width,
                       label=name, color=colors[i % len(colors)], zorder=3)

            ax.axhline(1 / len(class_names), color="#555555", linestyle="--",
                       linewidth=1, label="Distribuição uniforme", zorder=2)
            ax.set_title("Distribuição de classes — Fashion-MNIST", loc="left",
                         fontsize=17, fontweight="bold", pad=18)
            ax.set_ylabel("Proporção de amostras no conjunto")
            ax.set_xticks(range(len(class_names)), class_names,
                          rotation=30, ha="right")
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            ax.set_ylim(0, max(p.max().item() for p in proportions.values()) * 1.25)
            ax.grid(axis="y", alpha=0.2, zorder=0)
            ax.legend(frameon=False, ncol=4, loc="upper center")
            paths = [output_dir / f"fashion_mnist_class_distribution.{ext}"
                     for ext in ("png", "pdf")]
            for path in paths:
                fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
                print(f"Gráfico salvo em: {path}")
            return paths
        finally:
            plt.close(fig)


def load_fashion_mnist(train_batch_size=64, test_batch_size=64, seed=42,
                      class_distribution_tolerance=0.03, save_distribution_plot=True):
    """
    Carrega o Fashion-MNIST e divide o conjunto de treino em:
    - treino
    - validação
    - teste (conjunto do Fashion-MNIST já separado)

    Verifica as proporções de classes nos três conjuntos, com tolerância
    padrão de 3 %.
    """
    dataset_root = DATA_DIR / "fashion_mnist"
    dataset_root.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)

    train_full = datasets.FashionMNIST(
        root=str(dataset_root),
        train=True,
        download=True,
        transform=get_transforms(),
    )

    test_full = datasets.FashionMNIST(
        root=str(dataset_root),
        train=False,
        download=True,
        transform=get_transforms(),
    )

    # 80% treino e 20% validação dentro do conjunto de treino
    train_size = int(0.8 * len(train_full))
    val_size = len(train_full) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_full, [train_size, val_size], generator=generator)

    proportions = check_class_distribution(
        {
            "Treino": train_full.targets[train_ds.indices],
            "Validação": train_full.targets[val_ds.indices],
            "Teste": test_full.targets,
        },
        train_full.classes,
        tolerance=class_distribution_tolerance,
    )
    if save_distribution_plot:
        plot_class_distribution(proportions, train_full.classes)

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(test_full, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_fashion_mnist()
