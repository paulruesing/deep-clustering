# text width: 427.43153
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def set_size_old(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 427.43153
    elif width == "beamer":
        width_pt = 878.74
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    # golden_ratio = (5**.5 - 1) / 2
    golden_ratio = (5**0.5 + 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt  # a+b
    # b = a/golden_ratio
    # Figure height in inches
    #    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    fig_height_in = fig_width_in - fig_width_in / golden_ratio * (
        subplots[0] / subplots[1]
    )

    return (fig_width_in, fig_height_in)


def add_ttf_matplotlib(fontpath, mpl_font_family: str, font_name: str):
    '''
    Imports a new font from a .ttf file to matplotlib and sets it as default.

    Parameters
    ----------
    fontpath: string or path-like object
            Path referring to the font .ttf file to be used.
    mpl_font_family: string
            Font-family in matplotlib to add the font to. E.g. "sans-serif".
    font_name: string
            Name of the font, needs to match the .ttf file!
    '''
    mpl.font_manager.fontManager.addfont(fontpath)
    prop = mpl.font_manager.FontProperties(fname=fontpath)
    mpl.rc('font', family=mpl_font_family)
    mpl.rcParams.update({
        "font.size": 16,
        f"font.{mpl_font_family}": prop.get_name()
    })

    # Check if everything went right:
    assert plt.rcParams[f"font.{mpl_font_family}"][0] == font_name
    assert plt.rcParams["font.family"][0] == mpl_font_family


def file_title(title: str, dtype_suffix=".svg", short=False):
    '''
    Creates a file title containing the current time and a data-type suffix.

    Parameters
    ----------
    title: string
            File title to be used
    dtype_suffix: (default is ".svg") string
            Suffix determining the file type.
    Returns
    -------
    file_title: string
            String to be used as the file title.
    '''
    if short:
        return datetime.now().strftime('%Y%m%d') + " " + title + dtype_suffix
    else:
        return datetime.now().strftime('%Y%m%d %H_%M_%S') + " " + title + dtype_suffix


def rwth_palette(shade='blue'):
    '''
    Returns a matplotlib listed colormap of RWTH Colors.

    Paramters:
    ----------
    shade : (default is 'blue') string
        'blue' or 'colorful' defines the color set.

    Returns:
    ----------
    rwth_palette : matplotlib.colors.ListedColormap
        Palette containing the defined RWTH corporate design colors.
    '''
    if shade == 'blue':
        rgb_tones = [[0, 84/255, 159/255], [64/255, 127/255, 183/255],
                     [142/255, 186/255, 229/255], [199/255, 221/255, 242/255],
                     [232/255, 241/255, 250/255]]
    elif shade == 'colorful':
        rgb_tones = [[0, 84/255, 159/255], [0, 0, 0], [227/255, 0, 102/255],
                     [255/255, 237/255, 0], [87/255, 171/255, 39/255]]
    elif shade == 'colorful/yellow':
        rgb_tones = [[0, 84/255, 159/255], [0, 0, 0], [227/255, 0, 102/255],
                     [87/255, 171/255, 39/255]]
    return sns.color_palette(rgb_tones, as_cmap=True)