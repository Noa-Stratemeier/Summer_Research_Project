# Import external libraries
from dask import dataframe as dd


def formatted_galaxy_catalog():
    """
    Gets and filters the GLADE galaxy catalog.

    Returns:
        - glade_filtered_pd (pandas.DataFrame): Pandas dataframe containing only galaxies with a known redshift
    """
    glade = dd.read_csv('GLADE-.txt', delimiter=' ', usecols=[2, 7, 8, 9, 27, 31, 32, 33, 34, 38],
                        header=None,
                        names=['GWGC', 'type', 'ra', 'dec', 'z_h', 'z_err', 'd_L', 'd_L_err', 'flag', 'Merger_rate'],
                        low_memory=False,
                        blocksize='64MB')

    # Filter out quasars ('Q'), galaxies without a known redshift ('flag' == 0)
    glade_fil = glade[(glade['type'] != 'Q') & (glade['flag'] != 0)]

    # Compute to pandas dataframe
    glade_filtered_pd = glade_fil.compute()

    return glade_filtered_pd
