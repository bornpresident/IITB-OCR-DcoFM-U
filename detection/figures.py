def get_figure_hocr(fig):
    fig_hocr = f"<img bbox=\"{fig[0]} {fig[1]} {fig[2]} {fig[3]}\">\n"
    return fig_hocr