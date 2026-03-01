import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from scipy.ndimage import label
from cycler import cycler
from IPython.display import Markdown, display, HTML
import io
import xml.etree.ElementTree as ET
import numpy as np


def init_plotting(config):
    """
    Configures Matplotlib global defaults based on the PlotConfig class.
    """
    
    # Disable interactive mode to prevent auto-display of figures in Jupyter
    plt.ioff()

    # Define the Okabe-Ito Palette (The Default)
    okabe_ito = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]

    # Define High Contrast Palette (Higher luminance separation)
    high_contrast = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#882255", "#44AA99", "#117733", "#000000"]

    # Select base palette
    colors = high_contrast if config.HIGH_CONTRAST else okabe_ito

    # set colormap
    config.cmap = mcolors.ListedColormap(colors[:2], name='custom_cmap')

    # update DPI to scale images
    plt.rcParams['figure.dpi'] = 100 * config.SCALE

    # Define Shapes and Styles
    line_styles = ['-', '--', ':', '-.']

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
    plt_cycler = (
        cycler(color=colors) + 
        cycler(linestyle=line_styles * 2)
    )

    # Apply to rcParams
    plt.rcParams.update({
        # Colors and Styles
        'axes.prop_cycle': plt_cycler,
        
        # Line and Marker sizes
        'lines.linewidth': lw,
        'lines.markersize': ms,
        'patch.linewidth': lw,
        
        # Font Sizes
        'font.size': fs,
        'axes.titlesize': fs * 1.2,
        'axes.labelsize': fs * 1.1,
        'legend.fontsize': fs * 0.9,
        'xtick.labelsize': fs * 0.8,
        'ytick.labelsize': fs * 0.8,
        
        # Figure Layout
        'figure.autolayout': True,
        'figure.dpi': 100,
        'legend.frameon': True,
        'legend.edgecolor': '0.8',
    })

    # Extra logic for High Contrast: Force thicker borders and darker ticks
    if config.HIGH_CONTRAST:
        plt.rcParams.update({
            'axes.edgecolor': 'black',
            'axes.linewidth': 2.0 * config.SCALE,
            'grid.alpha': 0.5,
            'grid.color': 'black',
            'grid.linestyle': ':'
        })

def figure(config, figsize=None):
    """
    Helper function to create a figure with the appropriate size and scaling
    based on the PlotConfig.
    """
    if figsize  is None:
        figsize = (5,5)
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
    fig.savefig(buf, format='svg', bbox_inches='tight')
    
    # Parse the SVG XML
    buf.seek(0)
    ET.register_namespace('', "http://www.w3.org/2000/svg")
    ET.register_namespace('xlink', "http://www.w3.org/1999/xlink")
    tree = ET.parse(buf)
    root = tree.getroot()
    
    # Add Accessibility Attributes to the <svg> tag
    # role="img" tells screen readers this isn't just a document, it's a graphic.
    root.set('role', 'img')
    root.set('aria-labelledby', 'svg-title svg-desc')
    
    # Create and insert <title> and <desc>
    # These must be the first children for best screen reader support.
    title_elem = ET.Element('title', id='svg-title')
    title_elem.text = title
    
    desc_elem = ET.Element('desc', id='svg-desc')
    desc_elem.text = description
    
    root.insert(0, desc_elem)
    root.insert(0, title_elem)
    
    # Convert to HTML
    display(HTML(ET.tostring(root, encoding='unicode')))
    plt.close(fig)


def classification_report(y_val, y_pred):
    """
    Simple wrapper for scikit-learn's classification report that places the
    report into a Pandas DataFrame instead of printing text.
    """
    from sklearn.metrics import classification_report as skcr
    report_dict = skcr(y_val, y_pred, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    return report_df.style.format("{:.2f}", subset=['precision', 'recall', 'f1-score']).format("{:.0f}", subset=['support']).set_caption('Classification report')

def format_confusion_matrix(cm, class_names):
    """
    Formats a scikit-learn confusion matrix as a Pandas dataframe.
    """
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.index.name = 'True Label'
    df_cm.columns.name = 'Predicted Label'

    styled_cm = df_cm.style.set_properties(**{
        'border': '1px solid black',
        'text-align': 'center',
        'min-width': '30px',
        'padding-top': '12px',
        'padding-bottom': '12px'
    }).background_gradient(cmap='viridis', axis=None)

    return styled_cm.set_caption("Confusion Matrix: True vs Predicted Labels")
