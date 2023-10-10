import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_figs(titles, ytitle):
    data = pd.read_csv("logs/run.csv")
    compressed = data.columns.intersection(titles)
    sns.lineplot(x="epoch", y="conditional entropy", data=compressed)
    plt.show()


if __name__ == "__main__":
    titles = ["epoch", "120 neurons H(Y|X) train", "60 neurons H(Y|X) train"]
    plot_figs(titles, "Conditional entropy")
