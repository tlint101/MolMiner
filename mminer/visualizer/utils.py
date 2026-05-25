"""
Utility functions for plotting
"""

import seaborn as sns
import matplotlib.colors as mcolors


def palette_hex(palette_name: str = None, n_colors: int = 5):
    """
    Get hex codes for Seaborn color schemes. Output will be given as a list of hex codes.

    Parameters:
    palette_name: str
        Name of the Seaborn color palette.
    n_colors: int
        Number of colors to get from the palette.
    """
    colors = []

    # Get the color palette
    palette = sns.color_palette(palette_name, n_colors=n_colors)

    # Convert the colors to hex codes
    hex_codes = [mcolors.to_hex(color) for color in palette]

    # Print the hex codes
    for i, hex_code in enumerate(hex_codes):
        colors.append(hex_code)

    return colors


if __name__ == "__main__":
    import doctest

    doctest.testmod()
