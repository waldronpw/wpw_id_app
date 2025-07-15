import matplotlib.pyplot as plt

def plot_pie(data: dict, title: str, ax=None, colors=None):
    labels = list(data.keys())
    sizes = list(data.values())

    # Use passed-in axes or create a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = None  # don't return fig unless you create one

    # Draw pie chart
    wedges, _ = ax.pie(
        sizes,
        startangle=90,
        colors=colors,
        # wedgeprops=dict(width=0.6)
    )

    # Make it a circle and set title
    ax.axis("equal")
    ax.set_title(title, fontsize=9)

    # Return fig only if created here
    if fig:
        return fig