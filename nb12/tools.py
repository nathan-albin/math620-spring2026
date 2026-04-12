import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cycler import cycler
import io
import xml.etree.ElementTree as ET
import numpy as np
import plotly.graph_objects as go
from IPython.display import display, HTML


def init_plotting(config):
    """
    Configures Matplotlib global defaults based on the PlotConfig class.
    """

    # Disable interactive mode to prevent auto-display of figures in Jupyter
    plt.ioff()

    # Define the Okabe-Ito Palette (The Default)
    okabe_ito = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#000000",
    ]

    # Define High Contrast Palette (Higher luminance separation)
    high_contrast = [
        "#D55E00",
        "#0072B2",
        "#009E73",
        "#CC79A7",
        "#882255",
        "#44AA99",
        "#117733",
        "#000000",
    ]

    # Select base palette
    colors = high_contrast if config.HIGH_CONTRAST else okabe_ito

    # set colormap
    config.cmap = mcolors.ListedColormap(colors[:2], name="custom_cmap")

    # update DPI to scale images
    plt.rcParams["figure.dpi"] = 100 * config.SCALE

    # Define Shapes and Styles
    line_styles = ["-", "--", ":", "-."]

    # Scaling Logic
    base_line_width = 2.0 if not config.HIGH_CONTRAST else 3.5
    base_marker_size = 6.0 if not config.HIGH_CONTRAST else 10.0
    base_font_size = 10.0

    # Calculate final scaled values
    lw = base_line_width * config.SCALE
    ms = base_marker_size * config.SCALE
    fs = base_font_size * config.FONT_SCALE

    config.lw = lw
    config.ms = ms
    config.fs = fs

    # Build the Cycler (Combined properties)
    # We repeat line styles/markers to match the 8 colors in the palette
    plt_cycler = cycler(color=colors) + cycler(linestyle=line_styles * 2)

    # Apply to rcParams
    plt.rcParams.update(
        {
            # Colors and Styles
            "axes.prop_cycle": plt_cycler,
            # Line and Marker sizes
            "lines.linewidth": lw,
            "lines.markersize": ms,
            "patch.linewidth": lw,
            # Font Sizes
            "font.size": fs,
            "axes.titlesize": fs * 1.2,
            "axes.labelsize": fs * 1.1,
            "legend.fontsize": fs * 0.9,
            "xtick.labelsize": fs * 0.8,
            "ytick.labelsize": fs * 0.8,
            # Figure Layout
            "figure.autolayout": True,
            "figure.dpi": 100,
            "legend.frameon": True,
            "legend.edgecolor": "0.8",
        }
    )

    # Extra logic for High Contrast: Force thicker borders and darker ticks
    if config.HIGH_CONTRAST:
        plt.rcParams.update(
            {
                "axes.edgecolor": "black",
                "axes.linewidth": 2.0 * config.SCALE,
                "grid.alpha": 0.5,
                "grid.color": "black",
                "grid.linestyle": ":",
            }
        )


def figure(config, figsize=None):
    """
    Helper function to create a figure with the appropriate size and scaling
    based on the PlotConfig.
    """
    if figsize is None:
        figsize = (5, 5)
    figsize = (figsize[0] * config.SCALE, figsize[1] * config.SCALE)
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def to_svg(fig, title, description):
    """
    Converts a Matplotlib figure to a WCAG 2.1 compliant SVG string.

    Args:
        fig: The matplotlib figure object.
        title: A short, descriptive title (becomes the <title> tag).
        description: A longer explanation of trends/data (becomes the <desc> tag).
    """

    global PLOT_CONFIG

    # Save to buffer
    buf = io.BytesIO()
    fig.canvas.draw()
    fig.savefig(buf, format="svg", bbox_inches="tight")

    # Parse the SVG XML
    buf.seek(0)
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.parse(buf)
    root = tree.getroot()

    # Add Accessibility Attributes to the <svg> tag
    # role="img" tells screen readers this isn't just a document, it's a graphic.
    root.set("role", "img")
    root.set("aria-labelledby", "svg-title svg-desc")

    # Create and insert <title> and <desc>
    # These must be the first children for best screen reader support.
    title_elem = ET.Element("title", id="svg-title")
    title_elem.text = title

    desc_elem = ET.Element("desc", id="svg-desc")
    desc_elem.text = description

    root.insert(0, desc_elem)
    root.insert(0, title_elem)

    # Convert to HTML
    display(HTML(ET.tostring(root, encoding="unicode")))
    plt.close(fig)


def apply_layer(layer, X):
    """
    Applies a neural network layer to inputs.
    """

    # unpack the components of the layer
    W, b, f = layer

    # apply the layer
    return f(X @ W.T + b)


def apply_network(network, X):
    """
    Applies a list of neural network layers to inputs.
    """

    # loop over the layers
    for layer in network:
        # apply the layer. the output will become the input to the next layer
        X = apply_layer(layer, X)

    # when we're done, X will now contain the final output
    return X


def tabulate_values_2d(network, num_points):
    """
    Tabulates the output of a neural network with 2D input and 1D output.
    """

    # make a grid to evaluate on
    v = np.linspace(-2, 2, num_points)
    x1, x2 = np.meshgrid(v, v)
    X = np.c_[x1.ravel(), x2.ravel()]

    # apply the network to the grid points
    Y = apply_network(network, X).reshape(x1.shape)

    # create an HTML table to display the results
    html = """
    <table>
        <caption>Neural network output</caption>
        <thead>
            <tr>
                <th scope="col">x2 \\ x1</th>
    """
    for x in v:
        html += f"<th scope='col'>{x:.1f}</th>"
    html += "</tr></thead><tbody>"
    for i, y in reversed(list(enumerate(v))):
        html += f"<tr><th scope='row'>{y:.1f}</th>"
        for j, x in enumerate(v):
            html += f"<td>{Y[i, j]:.1f}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    display(HTML(html))


def visualize_network_2d(network, type="3d"):
    """
    Draws a 3D graph of a neural network with 2D input and 1D output.
    """

    # make a grid to evaluate on
    v = np.linspace(-2, 2, 100)
    x1, x2 = np.meshgrid(v, v)
    X = np.c_[x1.ravel(), x2.ravel()]

    # apply the network to the grid points
    Y = apply_network(network, X).reshape(x1.shape)

    # draw the function graph
    if type == "3d":
        fig = go.Figure(data=[go.Surface(x=x1, y=x2, z=Y)])
    else:
        fig = go.Figure(data=[go.Contour(x=v, y=v, z=Y)])
        fig.update_layout(
            width=600,
            height=600,
            plot_bgcolor="white",  # Removes background grid area
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
        )
    fig.show()


def describe_function(
    network, x_range=(-2, 2), num_points=1000, flat_threshold=0.1, heading_level=3
):
    """
    Generates a description of the function computed by a neural network.
    """

    # generate input values
    x = np.linspace(x_range[0], x_range[1], num_points)

    # compute the output of the network for these inputs
    y = apply_network(network, x[:, None])[:, 0]

    # compute the derivative using finite differences
    dy_dx = np.gradient(y, x)

    # identify flat regions and transition regions
    flat_regions = np.abs(dy_dx) < flat_threshold

    # identify flat intervals and transition intervals in order
    ordered_regions = []
    current_region = None
    current_start = None
    for i in range(len(x)):
        if flat_regions[i]:
            if current_region != "flat":
                if current_region == "transition":
                    ordered_regions.append(("transition", current_start, x[i]))
                current_region = "flat"
                current_start = x[i]
        else:
            if current_region != "transition":
                if current_region == "flat":
                    ordered_regions.append(("flat", current_start, x[i]))
                current_region = "transition"
                current_start = x[i]
    # handle the last interval
    if current_region == "flat":
        ordered_regions.append(("flat", current_start, x[-1]))
    else:
        ordered_regions.append(("transition", current_start, x[-1]))

    # generate an interleaved description of the regions
    description = f"{'#' * heading_level} Function Description\n"
    for region_type, start, end in ordered_regions:
        if region_type == "flat":
            value = y[(x >= start) & (x <= end)].mean()
            description += f"- constant region with value {value:.2f} from x={start:.2f} to x={end:.2f}\n"
        else:
            midpoint = (start + end) / 2
            width = end - start
            description += f"- transition region with midpoint x={midpoint:.2f} and width {width:.2f}\n"
    return description
