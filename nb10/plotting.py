import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from scipy.ndimage import label
from cycler import cycler
from IPython.display import Markdown, display, HTML, Javascript
import io
import xml.etree.ElementTree as ET
import numpy as np
import ipywidgets as widgets
import base64
import numpy as np
import ipywidgets as widgets
from scipy.io import wavfile
from scipy.interpolate import make_interp_spline
import json


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
    display(Markdown(description))
    plt.close(fig)


def summarize_values(names, values, small=0.1, large=2.0):

    # average value
    avg_values = [np.mean(np.abs(v)) for v in values]

    # median values
    median_values = [np.median(np.abs(v)) for v in values]

    # maximum values
    max_values = [np.max(np.abs(v)) for v in values]

    # number of small values (|v| < small)
    small_values = [np.sum(np.abs(v) < small) for v in values]

    # number of large values (|v| > large)
    large_values = [np.sum(np.abs(v) > large) for v in values]

    # create dataframe for reporting
    import pandas as pd

    df = pd.DataFrame(
        {
            "Method": names,
            "Avg Value": avg_values,
            "Median Value": median_values,
            "Max Value": max_values,
            f"# Small Values (<{small})": small_values,
            f"# Large Values (>{large})": large_values,
        }
    )

    display(df)


def create_sonification_panel(
    signals_dict, reference_signal=None, duration=2.5, fs=44100, level=2
):
    """
    Creates a sonification panel with global scaling.
    Uses Markdown headers and a reliable JS audio trigger.
    """
    # 1. Global Scaling
    all_arrays = list(signals_dict.values())
    if reference_signal is not None:
        all_arrays.append(reference_signal)

    all_values = np.concatenate(all_arrays)
    g_min, g_max = all_values.min(), all_values.max()

    # 2. JavaScript helper (Simplified version)
    js_play_code = """
    window.playAudio = function(base64Data) {
        var byteCharacters = atob(base64Data);
        var byteNumbers = new Array(byteCharacters.length);
        for (var i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        var byteArray = new Uint8Array(byteNumbers);
        var blob = new Blob([byteArray], {type: 'audio/wav'});
        var url = URL.createObjectURL(blob);
        var audio = new Audio(url);
        audio.play();
    };
    """
    display(Javascript(js_play_code))

    def get_waveform(data, t_audio):
        y_interp = np.interp(t_audio, np.linspace(0, duration, len(data)), data)
        f_min, f_max = 200, 1000

        if g_max == g_min:
            freqs = np.full_like(y_interp, 440)
        else:
            freqs = f_min + (f_max - f_min) * (y_interp - g_min) / (g_max - g_min)

        phase = 2 * np.pi * np.cumsum(freqs) / fs

        fade_len = int(0.05 * fs)
        fade = np.ones_like(t_audio)
        fade[:fade_len] = np.linspace(0, 1, fade_len)
        fade[-fade_len:] = np.linspace(1, 0, fade_len)
        return np.sin(phase) * fade

    def play_audio(b):
        t_audio = np.linspace(0, duration, int(fs * duration))
        sig_target = get_waveform(b.data, t_audio)

        if reference_signal is not None:
            # Stereo: Left = Reference, Right = Target
            sig_ref = get_waveform(reference_signal, t_audio)
            audio_data = np.vstack((sig_ref, sig_target)).T
        else:
            # Mono
            audio_data = sig_target

        # Export to WAV
        scaled = np.int16(audio_data * 32767)
        byte_io = io.BytesIO()
        wavfile.write(byte_io, fs, scaled)

        b64 = base64.b64encode(byte_io.getvalue()).decode("utf-8")
        display(Javascript(f"window.playAudio('{b64}')"))

    # 3. UI Construction with Markdown Header
    heading_level = "#" * level
    mode_title = (
        f"{heading_level} Stereo Comparison (Left: Reference, Right: Selected)"
        if reference_signal is not None
        else f"{heading_level} Audio Graph"
    )
    display(Markdown(f"{mode_title}\n*Tab to a signal and press Space to play.*"))

    buttons = []
    for label, data in signals_dict.items():
        btn = widgets.Button(description=label, button_style="primary")
        btn.data = data
        btn.on_click(play_audio)
        buttons.append(btn)

    return widgets.HBox(buttons)


def create_image_sonification_panel(
    signals_dict,
    reference_signal=None,
    rows=[0.25, 0.5, 0.75],
    duration=2.0,
    fs=44100,
    level=2,
):
    """
    Creates a grid of buttons for 2D image rows.
    Removes marker tones for a direct start to the signal.
    """
    # 1. Global Scaling
    all_imgs = list(signals_dict.values())
    if reference_signal is not None:
        all_imgs.append(reference_signal)

    g_min = min(img.min() for img in all_imgs)
    g_max = max(img.max() for img in all_imgs)

    sample_img = all_imgs[0]
    height, width = sample_img.shape

    # JavaScript helper (stable version)
    js_play_code = """
    window.playAudio = function(base64Data) {
        var byteCharacters = atob(base64Data);
        var byteNumbers = new Array(byteCharacters.length);
        for (var i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        var byteArray = new Uint8Array(byteNumbers);
        var blob = new Blob([byteArray], {type: 'audio/wav'});
        var url = URL.createObjectURL(blob);
        new Audio(url).play();
    };
    """
    display(Javascript(js_play_code))

    def get_waveform(data, t_audio):
        y_interp = np.interp(t_audio, np.linspace(0, duration, len(data)), data)
        f_min, f_max = 200, 1000
        if g_max == g_min:
            freqs = np.full_like(y_interp, 440)
        else:
            freqs = f_min + (f_max - f_min) * (y_interp - g_min) / (g_max - g_min)

        phase = 2 * np.pi * np.cumsum(freqs) / fs
        fade_len = int(0.05 * fs)
        fade = np.ones_like(t_audio)
        fade[:fade_len] = np.linspace(0, 1, fade_len)
        fade[-fade_len:] = np.linspace(1, 0, fade_len)
        return np.sin(phase) * fade

    def on_button_clicked(b):
        # Accessing the row and image data attached to the button
        row_data = b.data[b.row_idx, :]
        t_audio = np.linspace(0, duration, int(fs * duration))
        sig_target = get_waveform(row_data, t_audio)

        if reference_signal is not None:
            sig_ref = get_waveform(reference_signal[b.row_idx, :], t_audio)
            # Stereo: Ref on Left, Selection on Right
            audio_data = np.vstack((sig_ref, sig_target)).T
        else:
            # Mono: Selection in both ears
            audio_data = np.vstack((sig_target, sig_target)).T

        scaled = np.int16(audio_data * 32767)
        byte_io = io.BytesIO()
        wavfile.write(byte_io, fs, scaled)

        b64 = base64.b64encode(byte_io.getvalue()).decode("utf-8")
        display(Javascript(f"window.playAudio('{b64}')"))

    # 2. UI Construction
    heading_level = "#" * level
    display(Markdown(f"{heading_level} Image Row Audio Grid"))

    grid_rows = []
    for fraction in rows:
        row_idx = int(fraction * (height - 1))
        row_label = widgets.Label(value=f"Row {row_idx}:", layout={"width": "80px"})

        row_buttons = []
        for label, img_data in signals_dict.items():
            btn = widgets.Button(description=label, button_style="primary")
            btn.data = img_data
            btn.row_idx = row_idx
            btn.on_click(on_button_clicked)
            row_buttons.append(btn)

        grid_rows.append(widgets.HBox([row_label] + row_buttons))

    return widgets.VBox(grid_rows)
