"""
Scripts to draw plots. Methods here are primarily for plotting chemical space
"""

from typing import Union, List
import matplotlib.pyplot as plt
from legendkit import legend
import seaborn as sns
import pandas as pd
import numpy as np

__all__ = ["Plots"]


class Plots:
    def __init__(self, data: pd.DataFrame = None, x: str = None, y: str = None, name_col: str = None,
                 smi_col: str = None, font: str = "Arial", font_size: int = 12):
        """
        (Optional) Initialize Plot with data from a pd.DataFrame.
        :param data: pd.DataFrame
            DataTable containing information needed for plotting.
        :param x: str
            The data from the table corresponding to the X-axis. Used for scatterplot.
        :param y: str
            The data from the table corresponding to the Y-axis. Used for scatterplot.
        :param name_col: str
            Column to correspond with molecule name.
        :param smi_col: str
            Column to correspond with molecule smiles string.
        """

        self.data = data
        self.x = x
        self.y = y
        self.name_col = name_col
        self.smi_col = smi_col

        # Remove gridlines
        sns.set_style('white')

        # set font
        plt.rcParams['font.family'] = font
        plt.rcParams['font.size'] = font_size

        # globally remove grid lines from plot
        plt.rcParams['axes.grid'] = False

        # set variable for clustermap csv file
        self.matrix_index = None
        self.diagonal_matrix_data = None

    def scatter(self, data: pd.DataFrame = None, x: str = None, y: str = None, hue: str = None, title: str = None,
                title_size: int = 20, legend_loc: str = None, legend_title: str = None, savefig: str = None,
                use_svg: bool = False, **kwargs):
        """
        Function to plot data from a DataFrame as a matplotlib/seaborn scatter plot. Function accepts data frame, x col
        and y column, and hue.

        :param data: pd.DataFrame
            Input data in pd.DataFrame format.
        :param x: str
            Column name for X axis.
        :param y: str
            Column name for Y axis
        :param hue: str
            Column denoting the group variable to generate color.
        :param title: str
            Set the title of the plot.
        :param title_size: int
            Set the size of the title.
        :param legend_loc: str
            Set the location of the figure legend.
        :param legend_title: str
            Set the title of the figure legend.
        :param savefig: str
            Set the file path to save the figure.
        :param use_svg: bool = False
            Determine to output image in SVG format. If set to False, assume image is png.
        :param kwargs:
            Keyword arguments for additional formatting. Arguments are taken from seaborn and matplotlib.

        :return: figure
        """
        # match instance variable
        if data is None:
            data = self.data
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        # set hue if needed
        if hue is not None:
            hue = data[hue]

        image = sns.scatterplot(data=data, x=data[x], y=data[y], hue=hue, **kwargs)

        # replace plt legend with legendkit
        if legend_loc:
            image.legend_.remove()
            legend(loc=legend_loc, title=legend_title)

        # add title
        if title:
            plt.title(title, fontsize=title_size)

        # optional - savefig
        if savefig:
            plt.tight_layout()
            plt.savefig(savefig, dpi=300)
        elif savefig and use_svg == True:
            plt.savefig(savefig, dpi=300, bbox_inches='tight', format='svg')
            plt.show()

        plt.tight_layout()
        plt.show()

        return image

    def scatter_grid(self, data: pd.DataFrame = None, x_cols: list = None, y_cols: list = None, grid: tuple = (1, 2),
                     figsize=(15, 5), title: str = None, sub_title: list = None, hue: str = None,
                     legend_loc: str = None, legend_title: str = None, savefig: str = None, use_svg: bool = False,
                     **kwargs):
        """
        Plot multiple scatter plots in a grid.
        :param data: pd.DataFrame
            Input data in pd.DataFrame format.
        :param x_cols: list
            List of column names for the x-axis.
        :param y_cols: list
            List of column names for the y-axis.
        :param grid: tuple
            Set teh grid layout.
        :param figsize: tuple
            Set the figsize of subplots.
        :param title: str
            Set overall figure title.
        :param sub_title: list
            Set the subplot title for each individual scatter plot.
        :param hue: str
            Column denoting the group variable to generate color.
        :param legend_loc: str
            Set the location of the figure legend.
        :param legend_title: str
            Set the title of the figure legend. CURRENTLY NOT WORKING!!!!
        :param savefig: str
            Set the file path to save the figure.
        :param use_svg: bool = False
            Determine to output image in SVG format. If set to False, assume image is png.
        :param kwargs:
            Keyword arguments for additional formatting. Arguments are taken from seaborn and matplotlib.
        :return:
        """
        # match instance variable
        if data is None:
            data = self.data

        # set hue if needed
        if hue is not None:
            hue = data[hue]

        # dynamically extract grid shape
        rows, cols = grid

        # create a figure and axes
        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        # flatten array for extraction
        axes = axes.flatten()

        # generate individual scatterp lot
        for ax, x, y, subtitle in zip(axes, x_cols, y_cols, sub_title):
            image = sns.scatterplot(data=data, x=x, y=y, hue=hue, ax=ax, **kwargs)
            image.set_title(subtitle)

            # replace plt legend with legendkit
            if legend_loc:
                ax.get_legend().remove()
                legend(loc=legend_loc)

        # add title
        if title:
            fig.suptitle(title)

        # optional - savefig
        if savefig:
            plt.tight_layout()
            plt.savefig(savefig, dpi=300)
        elif savefig and use_svg == True:
            plt.savefig(savefig, dpi=300, bbox_inches='tight', format='svg')
            plt.show()

        plt.tight_layout()
        plt.show()

        return fig

    def similarity_matrix(self, data: pd.DataFrame = None, cmap: str = None, dendrogram_ratio: float = 0.03,
                          vmin: float = 0.0, vmax: float = 1.0, cbar_orientation='vertical',
                          cbar_pos=(1, 0.4, 0.03, 0.2), figsize=(10, 10), savefig: str = None, use_svg: bool = False,
                          **kwargs):
        """
        Generate a clustermap. Data must be a pd.DataFrame of similarity matrixes.
        :param data: pd.DataFrame
            Input data in pd.DataFrame. Must be a table of similarity scores.
        :param cmap: str
            Set the color palette of the clustermap.
        :param dendrogram_ratio: float
            Set the dendrogram size ratio.
        :param vmin: float
            The minimum value for the colorbar.
        :param vmax: float
            The maximum value for the colorbar.
        :param cbar_orientation: str
            Set orientation of the colorbar.
        :param cbar_pos: tuple
            cbar position. The tuple sets for (x, y, width, height)
        :param figsize: tuple
            Set the figsize of image.
        :param savefig: str
            Set the file location to save the image in .png format.
        :param use_svg: bool = False
            Determine to output image in SVG format. If set to False, assume image is png.
        :param kwargs:
            Keyword arguments for additional formatting. Arguments are taken from seaborn and matplotlib.
        :return:
        """
        # match instance variable
        if data is None:
            data = self.data

        # cbar orientation and ticks
        if cbar_orientation == "vertical":
            orientation = "vertical"
        elif cbar_orientation == "horizontal":
            orientation = "horizontal"
        else:
            raise ValueError("cbar_orientation must be either 'vertical' or 'horizontal'")

        cbar_kws = {"orientation": orientation, 'ticks': [0, 0.5, 1]}

        image = sns.clustermap(data, dendrogram_ratio=dendrogram_ratio, cmap=cmap, cbar_pos=cbar_pos, cbar_kws=cbar_kws,
                               vmin=vmin, vmax=vmax, figsize=figsize, **kwargs)

        # optional - savefig
        if savefig:
            image.savefig(savefig, dpi=300, bbox_inches='tight')
            plt.show()
        elif savefig and use_svg == True:
            plt.savefig(savefig, dpi=300, bbox_inches='tight', format='svg')
            plt.show()

        # set instance variable
        self.matrix_index = 'original_recipe'

        plt.show()

        return image

    def diagonal_matrix(self, data: pd.DataFrame = None, cmap: str = None, dendrogram_ratio: float = 0.03,
                        vmin: float = 0.0, vmax: float = 1.0, cbar_orientation: str = 'vertical',
                        cbar_pos=(1, 0.4, 0.03, 0.2), savefig: str = None, use_svg: bool = False, **kwargs):
        """
        Generate a clustermap. Data must be a pd.DataFrame of similarity matrices.
        :param data: pd.DataFrame
            Input data in pd.DataFrame. Must be a table of similarity scores.
        :param cmap: str
            Set the color palette of the clustermap.
        :param dendrogram_ratio: float
            Set the dendrogram size ratio.
        :param vmin: float
            The minimum value for the colorbar.
        :param vmax: float
            The maximum value for the colorbar.
        :param cbar_orientation: str
            Set orientation of the colorbar.
        :param cbar_pos: tuple
            cbar position. The tuple sets for (x, y, width, height)
        :param savefig: str = None
            Set the file location to save the image in .png format.
        :param use_svg: bool = False
            Determine to output image in SVG format. If set to False, assume image is png.
        :param kwargs:
            Keyword arguments for additional formatting. Arguments are taken from seaborn and matplotlib.
        :return:
        """
        # match instance variable
        if data is None:
            data = self.data

        # Generate a mask for the upper triangle
        # Generate an initial matrix and extract the order of the data
        initial_cluster = sns.clustermap(data, cmap=cmap)
        reordered_data = initial_cluster.data2d
        plt.close()  # close plot to keep it from showing during run

        # Create a mask for the upper triangle based on the reordered data
        mask = np.triu(np.ones_like(reordered_data, dtype=bool))

        np.fill_diagonal(mask, False)

        # cbar orientation and ticks
        if cbar_orientation == "vertical":
            orientation = "vertical"
        elif cbar_orientation == "horizontal":
            orientation = "horizontal"
        else:
            raise ValueError("cbar_orientation must be either 'vertical' or 'horizontal'")

        cbar_kws = {"orientation": orientation, 'ticks': [0, 0.5, 1]}

        image = sns.clustermap(reordered_data, mask=mask, dendrogram_ratio=dendrogram_ratio, cmap=cmap, vmin=vmin,
                               vmax=vmax, yticklabels=False, cbar_kws=cbar_kws, cbar_pos=cbar_pos, **kwargs)

        # Remove Col Dendrogram
        image.ax_col_dendrogram.remove()

        # Remove labels
        ax = image.ax_heatmap
        ax.set_ylabel('')
        ax.set_xlabel('')

        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
            plt.show()
        elif savefig and use_svg == True:
            plt.savefig(savefig, dpi=300, bbox_inches='tight', format='svg')
            plt.show()

        # set instance variable
        self.matrix_index = 'diagonal'
        self.diagonal_matrix_data = reordered_data  # for cluster_csv

        plt.show()

        return image

    def matrix_to_csv(self, image, data: pd.DataFrame = None, savepath: str = None):
        """
        Save the cluster map data from the image as a separate .csv file.
        """

        if self.matrix_index == 'original_recipe':
            # Get the order of rows and columns
            row_order = image.dendrogram_row.reordered_ind
            col_order = image.dendrogram_col.reordered_ind

            # Reorder the DataFrame based on the cluster map order
            matrix_order = data.iloc[row_order, col_order]

            # Save the reordered data to a .csv file
            matrix_order.to_csv(savepath)
        elif self.matrix_index == 'diagonal':
            # Call table from instance variable
            matrix_order = self.diagonal_matrix_data

            # Save the reordered data to a .csv file
            matrix_order.to_csv(savepath)

    def plot_space(self, data: pd.DataFrame = None, grid: tuple = (1, 4), color: Union[str, list] = None,
                   figsize: tuple = (25, 4),
                   savefig: str = None, **kwargs):
        """
        Plot the chemical space of dataset. Function is primarily used with class chemical_space.LibrarySpace.
        :param data: pd.DataFrame
            Input pd.DataFrame with chemical descriptors. Must include 'mw', 'n_lipinski_hba', 'n_lipinski_hbd', and
            'clogp'.
        :param grid: tuple
            Set the grid size. Tuple corresponds to row x column in plot grid.
        :param color: Union[str, list]
            Set the color of the hist plot. Can be a string, which will give all plots a uniform color, or a list, which
            will give each plot a specific color.
        :param figsize: tuple
            Set the size of the plots.
        :param savepath: str
            Set the file location to save the image in .png format.
        :param kwargs:
            Additional params associated with the sns.histplot.
        :return:
        """
        global color1, color2, color3, color4
        # match instance variable
        if data is None:
            data = self.data

        # set colorscheme
        if isinstance(color, str):
            color1 = color
            color2 = color
            color3 = color
            color4 = color
        elif isinstance(color, list) and len(color) >= 2:
            color1 = color[0]
            color2 = color[1]
            color3 = color[2]
            color4 = color[3]
        elif isinstance(color, list) and len(color) > 4:
            raise ValueError('Only FOUR colors are allowed!')
        elif color is None:
            color1 = 'lightblue'
            color2 = 'lightblue'
            color3 = 'lightblue'
            color4 = 'lightblue'

        fig, axs = plt.subplots(nrows=grid[0], ncols=grid[1], figsize=figsize)
        axs = axs.flatten()

        sns.histplot(data=data['mw'], ax=axs[0], color=color1, **kwargs)
        axs[0].set_title('Molecular Weight')
        axs[0].set_xlabel('mw')
        axs[0].set_ylabel('Frequency')

        sns.histplot(data=data['n_lipinski_hba'], ax=axs[1], color=color2, **kwargs)
        axs[1].set_title('Number of Lipinski HBA')
        axs[1].set_xlabel('n_lipinski_hba')
        axs[1].set_ylabel('Frequency')

        sns.histplot(data=data['n_lipinski_hbd'], ax=axs[2], color=color3, **kwargs)
        axs[2].set_title('Number of Lipinski HBD')
        axs[2].set_xlabel('n_lipinski_hbd')
        axs[2].set_ylabel('Frequency')

        sns.histplot(data=data['clogp'], ax=axs[3], color=color4, **kwargs)
        axs[3].set_title('CLogP')
        axs[3].set_xlabel('clogp')
        axs[3].set_ylabel('Frequency')

        # optional - savefig
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

        plt.tight_layout()
        plt.show()

        return fig


if __name__ == "__main__":
    import doctest

    doctest.testmod()
