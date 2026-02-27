import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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


def svm_diagnostic(svc, X_train, y_train, X_val, y_val, config, ax):
    """
    Diagnostic suite for SVM hyperparameter exploration.
    """
    # Prediction grid
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Fill Logic: Use hatching if High Contrast is enabled
    if config.HIGH_CONTRAST:
        # Class 0: Diagonal Hatches, Class 1: Dots
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=['#FFFFFF', '#DDDDDD'], alpha=0.5)
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5], hatches=['///'], colors='none')
        ax.contourf(xx, yy, Z, levels=[0.5, 1.5], hatches=['...'], colors='none')
        # Add a thick boundary line
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=config.lw*1.5)
    else:
        ax.contourf(xx, yy, Z, cmap=config.cmap, alpha=0.3)

    ax.set_title(f"SVM Diagnostic (C={svc.C}, $\\gamma$={svc.gamma})", fontsize=12 * config.FONT_SCALE)

    # Calculate metrics
    train_err = (y_train != svc.predict(X_train)).sum()
    val_err = (y_val != svc.predict(X_val)).sum()
    sv_count = len(svc.support_)
    
    # Island Counting
    structure = np.ones((3, 3))
    _, isl0 = label(Z == 0, structure=structure)
    _, isl1 = label(Z == 1, structure=structure)
    total_islands = isl0 + isl1

    report = f"""
### 📋 Diagnostic Report

* **Training Points Mislabeled:** {train_err} / {len(y_train)} ({(train_err/len(y_train))*100 :.1f}% error)
* **Validation Points Mislabeled:** {val_err} / {len(y_val)} ({(val_err/len(y_val))*100 :.1f}% error)
* **Total Decision Regions:** {total_islands}
* **Support Vector Count:** {sv_count} points ({ (sv_count/len(y_train))*100 :.1f}% of data).
"""
    display(Markdown(report))
