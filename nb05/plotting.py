import pandas as pd
import seaborn as sns
import matplotlib as mpl
from IPython.display import Markdown, display, HTML
import io
import xml.etree.ElementTree as ET


PLOT_CONFIG = None

def init_plotting(PlotConfig):

    global PLOT_CONFIG
    PLOT_CONFIG = PlotConfig

    sns.set_context(
        "notebook", 
        font_scale=PlotConfig.FONT_SCALE,
        rc={"lines.linewidth": 2.0 * PlotConfig.SCALE, "axes.labelsize": 12 * PlotConfig.FONT_SCALE}
        )

    if PlotConfig.HIGH_CONTRAST:
        sns.set_style("ticks")
        # 'bright' or 'colorblind' offer much higher contrast than 'deep'
        sns.set_palette("bright")
        # Ensure spines (border lines) are bold
        mpl.rcParams['axes.linewidth'] = 2.0 * PlotConfig.SCALE
        mpl.rcParams['lines.markersize'] = 8 * PlotConfig.SCALE
    else:
        sns.set_style("darkgrid")
        sns.set_palette("deep")
        mpl.rcParams['axes.linewidth'] = 0.8 * PlotConfig.SCALE
        mpl.rcParams['lines.markersize'] = 6 * PlotConfig.SCALE


def accessible_svg(fig, title, description):
    """
    Converts a Matplotlib figure to a WCAG 2.1 compliant SVG string.
    
    Args:
        fig: The matplotlib figure object.
        title: A short, descriptive title (becomes the <title> tag).
        description: A longer explanation of trends/data (becomes the <desc> tag).
    """

    current_width, current_height = fig.get_size_inches()
    fig.set_size_inches(current_width * PLOT_CONFIG.SCALE, 
                        current_height * PLOT_CONFIG.SCALE)
    original_font_size = mpl.rcParams['font.size']
    mpl.rcParams['font.size'] = original_font_size * PLOT_CONFIG.FONT_SCALE
    
    # Save to buffer
    buf = io.BytesIO()
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

def plot_line(df, a, b, c, linestyle='--', label='separator'):

    # create a small dataframe for the line endpoints
    xl, xr = df['x'].min(), df['x'].max()
    line_df = pd.DataFrame({
        'x': [ xl, xr ],
        'y': [ (-a*xl + c)/b, (-a*xr + c)/b ],
    })

    # plot the line using seaborn
    sns.lineplot(data=line_df, x='x', y='y', linestyle=linestyle, label=label)



def table_generator(X, y=None, feature_names=None, title="Data Summary", level=1):
    """Generates a Markdown table as a textual alternative to plots."""

    if not isinstance(X, pd.DataFrame):
        names = feature_names or [f'x{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=names)
    else:
        df = X.copy()
    if y is not None: df['label'] = y
    
    summary = df.describe().to_markdown()
    sample = df.sample(5).to_markdown(index=False)
    
    hlevel = "#" * level

    display(Markdown(f"{hlevel} {title}\n{hlevel}# Statistics\n{summary}\n\n{hlevel}# Sample Rows\n{sample}"))

def check_transformation_feedback(y, z_values):
    """Provides textual feedback for nonlinear transformation discovery."""
    p1_z = z_values[y == 1]
    m1_z = z_values[y == -1]
    msg = f"### Transformation Feedback\n- **Class +1** height range: [{p1_z.min():.2f}, {p1_z.max():.2f}]\n"
    msg += f"- **Class -1** height range: [{m1_z.min():.2f}, {m1_z.max():.2f}]\n"
    
    if p1_z.min() > m1_z.max() or m1_z.min() > p1_z.max():
        msg += "\n**Result:** Success! The classes are vertically separated."
    else:
        msg += "\n**Result:** The classes still overlap in height. Try a different function."
    display(Markdown(msg))