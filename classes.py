# Import local libraries
import graphing as graph


class Cell:

    def __init__(self, ra, dec, side_len, prob, am, galaxies, gal_prob):
        """
        Cell within a tiled sky region grid.

        Attributes:
            - ra (float): Right ascension of the cell's centre [degrees]
            - dec (float): Declination of the cell's centre [degrees]
            - side_len (float): Angular side length of the cell [degrees]
            - prob (float): Probability contained within the cell
            - am (float): Air mass at the cell's centre at the datetime the cell was created
            - galaxies (pandas.DataFrame): Galaxies contained within the intersection of the cell and
                                           sky region that the cell is tiling
            - gal_prob (float): Probability contained within the cell when probability is convolved with galaxy density
        """
        self.ra = ra
        self.dec = dec
        self.side_len = side_len
        self.prob = prob
        self.am = am
        self.galaxies = galaxies
        self.gal_prob = gal_prob

    def num_galaxies(self):
        """
        Finds the number of galaxies that lie within the cell.

        Returns:
            - num_galaxies (int): Number of galaxies contained within the cell
        """
        num_galaxies = self.galaxies.shape[0]

        return num_galaxies

    def highest_redshift(self):
        """
        Finds the highest heliocentric redshift of all galaxies within the cell.

        Returns:
            - max_redshift (float): Max heliocentric redshift out of the galaxies within cell
        """
        max_redshift = self.galaxies['z_h'].max()

        return max_redshift

    def show_redshift_distribution(self):
        """
        Graphs the redshift distribution of all galaxies within the cell.
        """
        graph.cell_redshift_distribution(self)

    # PLACEHOLDER FUNCTION, REPLACE WITH EXPOSURE TIME CALCULATOR
    def exposure_time(self, apparent_magnitude=19):
        time = ((apparent_magnitude ** 2) * self.am)
        return time


class GrbSkyErrorRegion:

    def __init__(self, ra, dec, error_radius, galaxies, tiled_region):
        """
        Sky error region of a GRB.

        Attributes:
            - ra (float): Right ascension of the region centre [degrees]
            - dec (float): Declination of the region centre [degrees]
            - error_radius (float): Angular error radius of the region (corresponding to 3 st_dev) [degrees]
            - galaxies (pandas.DataFrame): Galaxies contained within the region
            - tiled_region (list): List of cells that make up the tiled region
        """
        self.ra = ra
        self.dec = dec
        self.error_radius = error_radius
        self.galaxies = galaxies
        self.tiled_region = tiled_region

    def total_probability_in_tiles(self):
        """
        Finds the total probability covered by the tiled region.
        """
        total_prob = 0

        for tile in self.tiled_region:
            total_prob += tile.prob

        return total_prob

    def total_gal_probability_in_tiles(self):
        """
        Finds the total probability covered by the tiled region.
        """
        total_prob = 0

        for tile in self.tiled_region:
            total_prob += tile.gal_prob

        return total_prob
