import locale
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.scale as mscale


locale.setlocale(locale.LC_ALL, locale="da_DK")


def plot_timeline(timeline, title, filename, color="red"):
    timeline = [
        (datetime.strptime(timestamp, "%Y-%m"), desc)
        for timestamp, desc in timeline
    ]
    timeline = [
        (timestamp, f"{timestamp.strftime('%Y')}: {desc}")
        for timestamp, desc in timeline
    ]

    dates, descs = zip(*timeline)

    # Choose some nice levels
    levels = np.tile([-9, 9, -5, 5, -1, 1], int(np.ceil(len(dates) / 6)))[
        : len(dates)
    ]

    # Create figure and plot a stem plot with the date
    fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=False)
    # ax.set(title=title)

    ax.vlines(dates, 0, levels, color=f"tab:{color}")  # The vertical stems.
    ax.plot(
        dates, np.zeros_like(dates), "-o", color="w", markerfacecolor="w"
    )  # Baseline and markers on it.

    # annotate lines
    for d, l, r in zip(dates, levels, descs):
        ax.annotate(
            r,
            xy=(d, l),
            xytext=(-3, np.sign(l) * 3),
            textcoords="offset points",
            horizontalalignment="left",
            verticalalignment="bottom" if l > 0 else "top",
        )

    # ax.xaxis.set_major_locator(mdates.DayLocator())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%A %d/%m"))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="left")
    # remove y axis and spines
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
    ("2001-10", "Friedman\nGradient boosting"),
    ("2014-06", "Chen\nXGBoost\nHiggs Kaggle competition"),
]

plt.style.use('dark_background')
# with plt.xkcd():
plot_timeline(timeline, "Timeline of Boosting research", "boosting_timeline.svg", color="blue")
