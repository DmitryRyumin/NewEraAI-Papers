import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    years = ["2023"]
    conferences = ["INTERSPEECH", "ICCV", "ISMIR"]
    total_papers = [1142, 2156, 144]
    preprint_papers = [502, 1660, 55]
    open_code_papers = [247, 1419, 69]
    video_papers = [0, 275, 0]

    data = {
        "Conference": np.repeat(conferences, len(years)),
        "Year": np.tile(years, len(conferences)),
        "Total Papers": total_papers,
        "Preprint Papers": preprint_papers,
        "Open Code Papers": open_code_papers,
        "Video Papers": video_papers,
    }

    df = pd.DataFrame(data)
    return df


def preprocess_data(df):
    df_melted = pd.melt(
        df, id_vars=["Conference", "Year"], var_name="Category", value_name="Count"
    )
    return df_melted


def set_seaborn_style():
    sns.set_palette("muted")
    sns.set_style("whitegrid")


def plot_bar_chart(df_melted):
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        data=df_melted, x="Conference", y="Count", hue="Category", palette="viridis"
    )

    for p in ax.patches:
        height = int(p.get_height())
        if height != 0:
            ax.annotate(
                f"{height}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="center",
                xytext=(0, 12),
                textcoords="offset points",
                fontsize=14,
                color="black",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    edgecolor="none",
                    facecolor="lightgray",
                    alpha=0.7,
                ),
            )

    return ax


def customize_legend(ax):
    ax.legend(
        title="Category",
        title_fontsize="12",
        fontsize="10",
        loc="upper right",
        bbox_to_anchor=(1.0, 1),
    )


def set_plot_labels(df):
    plt.title(
        f'Conference Statistics Comparison ({" ".join(df["Year"].unique())})',
        fontsize=16,
    )
    plt.ylabel("Number of Papers", fontsize=12)
    plt.xlabel("Conference", fontsize=12)


def remove_spines():
    sns.despine()


def save_plot():
    plt.savefig(
        "conference_statistics_comparison.svg", format="svg", bbox_inches="tight"
    )


def main():
    df = load_data()
    df_melted = preprocess_data(df)

    set_seaborn_style()
    ax = plot_bar_chart(df_melted)
    customize_legend(ax)
    set_plot_labels(df)
    remove_spines()
    save_plot()

    plt.show()


if __name__ == "__main__":
    main()
