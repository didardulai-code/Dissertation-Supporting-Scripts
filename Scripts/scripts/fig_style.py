def apply(fig, title=12, label=11, tick=10, legend=9.5):
    def _size_legend(leg):
        if not leg:
            return
        for tx in leg.get_texts():
            tx.set_fontsize(legend)
        if leg.get_title() and leg.get_title().get_text().strip():
            leg.get_title().set_fontsize(legend)
    for ax in fig.axes:
        for t in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
            if t is not None and t.get_text().strip():
                t.set_fontsize(title); t.set_fontweight("bold")
        if ax.get_xlabel(): ax.xaxis.label.set_fontsize(label)
        if ax.get_ylabel(): ax.yaxis.label.set_fontsize(label)
        ax.tick_params(labelsize=tick)
        for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()) \
                 + list(ax.get_xticklabels(minor=True)) + list(ax.get_yticklabels(minor=True)):
            lbl.set_fontsize(tick)
        _size_legend(ax.get_legend())
    for leg in getattr(fig, "legends", []):
        _size_legend(leg)
