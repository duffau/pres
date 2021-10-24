import locale
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


locale.setlocale(locale.LC_ALL, locale="da_DK")


def plot_timeline(timeline, filename, n_levels=6, figsize=(6, 4.5), color="red"):
    timeline = [
        (datetime.strptime(timestamp, "%Y-%m"), desc)
        for timestamp, desc in timeline
    ]
    timeline = [
        (timestamp, f"{timestamp.strftime('%Y')}: {desc}")
        for timestamp, desc in timeline
    ]

    dates, descs = zip(*timeline)

    levels = [lev for i in reversed(range(1, n_levels//2+1)) for lev in (i, -i)]
    levels = np.tile(levels, int(np.ceil(len(dates) / n_levels)))[
        : len(dates)
    ]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)

    ax.vlines(dates, 0, levels, color=f"tab:{color}")  # The vertical stems.
    ax.plot(
        dates, np.zeros_like(dates), "-o", color="w", markerfacecolor="w"
    ) 

    for d, l, r in zip(dates, levels, descs):
        ax.annotate(
            r,
            xy=(d, l),
            xytext=(-3, np.sign(l) * 3),
            textcoords="offset points",
            horizontalalignment="left",
            verticalalignment="bottom" if l > 0 else "top",
        )

    plt.setp(ax.get_xticklabels(), rotation=0, ha="left")
    ax.yaxis.set_visible(False)
    ax.spines[["left", "top", "right"]].set_visible(False)

    ax.margins(y=0.15)
    plt.tight_layout()
    plt.savefig(filename, transparent=True)
    plt.close()


timeline = [
    ("1984-11", "Valiant\nA theory of the learnable"),
    ("1988-06", "Kearns & Valiant\nIntroduce 'weak learners'"),
    ("1989-02", "Kearns & Valiant\nhypothesis boosting problem"),
    ("1990-06", "Shaphire\nRecursice algorithm solving the 'hypothesis boosting problem'"),
    ("1990-07", "Freund\nBoost By Majority: A practical boosting algorithm"),
    ("1995-06", "Schapire & Freund\nAdaboost"),
    ("2000-04", "Friedman, Hastie & Tibshirani\nAdaBoost as a logistic regression"),
    ("2001-10", "Friedman\nGradient boosting"),
    ("2014-06", "Chen\nXGBoost\nHiggs Kaggle competition"),
]

plt.style.use('dark_background')
plot_timeline(timeline, "boosting_timeline.svg", color="blue")
