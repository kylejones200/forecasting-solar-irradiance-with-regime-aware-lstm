"""Generated from Jupyter notebook: Regime changing time series

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy import stats
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def animate(frame):
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.clear()
    current_df = df.iloc[:frame]
    for regime in [0, 1]:
        regime_data = current_df[current_df["True_Regime"] == regime]
        ax1.scatter(
            regime_data["Time"],
            regime_data["Data"],
            label=f"Regime {regime}",
            alpha=0.6,
            s=20,
        )
    ax1.set_title("Original Data with True Regimes")
    ax1.set_ylim(df["Data"].min() - 1, df["Data"].max() + 1)
    ax1.legend()
    if len(current_df) > 20:
        pred_regimes, result = get_regime_predictions(current_df["Data"])
        ax2.plot(
            current_df.index, current_df["True_Regime"], label="True Regime", alpha=0.6
        )
        ax2.plot(current_df.index, pred_regimes, label="Predicted Regime", alpha=0.6)
        ax2.set_title("True vs Predicted Regimes")
        ax2.set_ylim(-0.1, 1.1)
        ax2.legend()
        for regime in [0, 1]:
            regime_data = current_df[current_df["True_Regime"] == regime]["Data"]
            if len(regime_data) > 1:
                sns.kdeplot(data=regime_data, ax=ax3, label=f"Regime {regime}")
        ax3.set_title("Density Distribution by Regime")
        ax3.legend()
        confusion_matrix = (
            pd.crosstab(
                current_df["True_Regime"],
                pd.Series(pred_regimes, index=current_df.index),
            )
            .fillna(0)
            .values
        )
        if confusion_matrix.shape != (2, 2):
            temp_matrix = np.zeros((2, 2))
            for t in range(min(2, confusion_matrix.shape[0])):
                for p in range(min(2, confusion_matrix.shape[1])):
                    temp_matrix[t, p] = confusion_matrix[t, p]
            confusion_matrix = temp_matrix
        im = ax4.imshow(confusion_matrix, cmap="Blues", interpolation="nearest")
        for (j, i), label in np.ndenumerate(confusion_matrix):
            ax4.text(
                i,
                j,
                int(label),
                ha="center",
                va="center",
                color="white" if label > 50 else "black",
            )
        ax4.set_xticks([0, 1])
        ax4.set_yticks([0, 1])
        ax4.set_xticklabels(["Predicted 0", "Predicted 1"])
        ax4.set_yticklabels(["True 0", "True 1"])
        ax4.set_title("Confusion Matrix")
        if result is not None:
            transition_matrix = result.regime_transition.reshape(2, 2)
            im2 = ax5.imshow(
                transition_matrix, cmap="coolwarm", interpolation="nearest"
            )
            for (j, i), label in np.ndenumerate(transition_matrix):
                ax5.text(i, j, f"{label:.2f}", ha="center", va="center")
            ax5.set_title("Transition Probability Matrix")
            ax5.set_xticks([0, 1])
            ax5.set_yticks([0, 1])
            ax5.set_xlabel("To Regime")
            ax5.set_ylabel("From Regime")
    plt.suptitle(f"Frame {frame}/{len(df)}", y=1.02)
    plt.tight_layout()
    return []


def get_regime_predictions(data):
    """Helper function to get regime predictions safely"""
    try:
        model = MarkovRegression(data, k_regimes=2, trend="c", switching_variance=True)
        result = model.fit(disp=False)
        probs = result.smoothed_marginal_probabilities
        if probs is not None and probs.shape[1] >= 2:
            pred_regimes = (probs[:, 1] > 0.5).astype(int)
            return (pred_regimes, result)
        else:
            return (np.zeros(len(data)), None)
    except Exception:
        return (np.zeros(len(data)), None)


def main() -> None:
    np.random.seed(42)

    n = 500

    regimes = np.random.choice([0, 1], size=n, p=[0.7, 0.3])

    data = np.array(np.random.normal(0, np.where(regimes == 0, 1, 5)))

    plt.figure(figsize=(10, 6))

    plt.plot(data, label="Simulated Data")

    plt.legend()

    plt.title("Simulated Data")

    plt.show()

    model = MarkovRegression(data, k_regimes=2, trend="c", switching_variance=True)

    result = model.fit()

    print(result.summary())

    print("\nTransition Matrix:")

    print(result.regime_transition)

    np.random.seed(42)

    n = 500

    regimes = np.random.choice([0, 1], size=n, p=[0.7, 0.3])

    data = np.array(np.random.normal(0, np.where(regimes == 0, 1, 5)))

    plt.figure(figsize=(10, 6))

    plt.plot(data, label="Simulated Data")

    plt.legend()

    plt.title("Simulated Data")

    plt.show()

    model = MarkovRegression(data, k_regimes=2, trend="c", switching_variance=True)

    result = model.fit()

    print(result.summary())

    print("\nTransition Matrix:")

    print(result.regime_transition)

    smoothed_probs = result.smoothed_marginal_probabilities[1]

    plt.figure(figsize=(10, 6))

    plt.plot(smoothed_probs, label="Smoothed Probabilities of High Volatility")

    plt.legend()

    plt.title("Smoothed Probabilities of High Volatility Regime")

    plt.show()

    predicted_regimes = np.argmax(result.smoothed_marginal_probabilities, axis=1)

    plt.figure(figsize=(10, 6))

    plt.plot(predicted_regimes, label="Predicted Regimes")

    plt.legend()

    plt.title("Predicted Regimes")

    plt.ylim(-0.1, 1.1)

    plt.show()

    np.random.seed(42)

    n = 500

    regimes = np.random.choice([0, 1], size=n, p=[0.7, 0.3])

    data = np.array(np.random.normal(0, np.where(regimes == 0, 1, 5)))

    df = pd.DataFrame({"Data": data, "True_Regime": regimes, "Time": range(n)})

    model = MarkovRegression(data, k_regimes=2, trend="c", switching_variance=True)

    result = model.fit()

    df["Predicted_Prob_High"] = result.smoothed_marginal_probabilities[:, 1]

    df["Predicted_Regime"] = np.argmax(result.smoothed_marginal_probabilities, axis=1)

    plt.figure(figsize=(12, 6))

    for regime in [0, 1]:
        mask = df["True_Regime"] == regime
        plt.scatter(
            df[mask]["Time"], df[mask]["Data"], label=f"Regime {regime}", alpha=0.6
        )

    plt.title("Original Data with True Regimes")

    plt.legend()

    plt.savefig("original_data_regimes.png")

    plt.close()

    plt.figure(figsize=(12, 6))

    plt.plot(df["True_Regime"], label="True Regime", alpha=0.6)

    plt.plot(df["Predicted_Regime"], label="Predicted Regime", alpha=0.6)

    plt.title("True vs Predicted Regimes")

    plt.legend()

    plt.savefig("true_vs_predicted_regimes.png")

    plt.close()

    plt.figure(figsize=(12, 6))

    for regime in [0, 1]:
        sns.kdeplot(
            data=df[df["True_Regime"] == regime]["Data"], label=f"Regime {regime}"
        )

    plt.title("Density Distribution by Regime")

    plt.legend()

    plt.savefig("density_distribution.png")

    plt.close()

    plt.figure(figsize=(8, 6))

    transition_matrix = result.regime_transition.reshape(2, 2)

    sns.heatmap(transition_matrix, annot=True, cmap="coolwarm")

    plt.title("Transition Probability Matrix")

    plt.xlabel("To Regime")

    plt.ylabel("From Regime")

    plt.savefig("transition_matrix.png")

    plt.close()

    plt.figure(figsize=(8, 6))

    confusion_matrix = pd.crosstab(df["True_Regime"], df["Predicted_Regime"])

    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix: True vs Predicted Regimes")

    plt.xlabel("Predicted Regime")

    plt.ylabel("True Regime")

    plt.savefig("confusion_matrix.png")

    plt.close()

    print("\nModel Performance Metrics:")

    accuracy = (df["True_Regime"] == df["Predicted_Regime"]).mean()

    print(f"Prediction Accuracy: {accuracy:.2%}")

    print("\nRegime Statistics:")

    for regime in [0, 1]:
        regime_data = df[df["True_Regime"] == regime]["Data"]
        print(f"\nRegime {regime}:")
        print(f"Mean: {regime_data.mean():.2f}")
        print(f"Std: {regime_data.std():.2f}")
        print(f"Skewness: {stats.skew(regime_data):.2f}")
        print(f"Kurtosis: {stats.kurtosis(regime_data):.2f}")

    print("\nAverage Duration in Each Regime:")

    for regime in [0, 1]:
        regime_runs = (
            (df["Predicted_Regime"] == regime)
            .astype(int)
            .groupby(
                (df["Predicted_Regime"] != df["Predicted_Regime"].shift()).cumsum()
            )
            .sum()
        )
        print(f"Regime {regime}: {regime_runs.mean():.2f} periods")

    transitions = pd.DataFrame(
        {"From": df["Predicted_Regime"][:-1], "To": df["Predicted_Regime"][1:]}
    )

    print("\nTransition Counts:")

    print(pd.crosstab(transitions["From"], transitions["To"]))

    fig = plt.figure(figsize=(20, 10))

    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])

    ax2 = fig.add_subplot(gs[0, 1])

    ax3 = fig.add_subplot(gs[1, 0])

    ax4 = fig.add_subplot(gs[1, 1])

    plt.close()

    confusion_matrix = np.zeros((2, 2))

    step = 5

    frames = range(10, len(df), step)

    anim = FuncAnimation(fig, animate, frames=frames, interval=100, blit=False)

    writer = PillowWriter(fps=10)

    anim.save("regime_switching_animation.gif", writer=writer)

    fig = plt.figure(figsize=(15, 10))

    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])

    ax2 = fig.add_subplot(gs[0, 1])

    ax3 = fig.add_subplot(gs[0, 2])

    ax4 = fig.add_subplot(gs[1, 0])

    ax5 = fig.add_subplot(gs[1, 1])

    plt.close()

    step = 10

    frames = range(20, len(df), step)

    anim = FuncAnimation(fig, animate, frames=frames, interval=200, blit=False)

    writer = PillowWriter(fps=5)

    anim.save("regime_switching_animation.gif", writer=writer)


if __name__ == "__main__":
    main()
