# Import external libraries
import math
import astropy.coordinates as coord
import astropy.units as u
import astropy.cosmology.units as cu
from astropy.time import Time
from astroplan import Observer, FixedTarget
from pytz import timezone
from datetime import datetime
import numpy as np
import copy
import openturns as ot
from scipy.stats import gaussian_kde

# Import local libraries
import classes as cl
import galaxy_catalog_parser as gcp
import graphing as graph


"""Define the Greenhill observatory"""
latitude = -42.43  # degrees
longitude = 147.3  # degrees
height = 646  # m

bisdee_tier = coord.EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=height * u.m)
GHO = Observer(location=bisdee_tier, name='Greenhill Observatory', timezone='Australia/Tasmania')


"""Create galaxy catalog"""
GLADE_CATALOG = gcp.formatted_galaxy_catalog()


def is_between(start, end, mid):
    """
    Determines whether an angle falls between a given start and end angle,
    all given angles must be in the range 0-360 degrees.

    Params:
        - start (float): Start angle [degrees]
        - end (float): End angle [degrees]
        - mid (float): The angle that is being checked [degrees]

    Returns:
        - result (bool): True if 'mid' falls between 'start' and 'end', false if not.
    """
    end = np.where(((end - start) < 0.0), end - start + 360.0, end - start)
    mid = np.where(((mid - start) < 0.0), mid - start + 360.0, mid - start)

    result = mid < end

    return result


def angular_separation(ra1, dec1, ra2, dec2):
    """
    Finds the angular separation between two points on the celestial sphere.

    Params:
        - ra1, dec2 (float): Coordinates of the first point [degrees]
        - ra1, dec2 (float): Coordinates of the second point [degrees]

    Returns:
        - angular_sep (float): Angular separation between the two given points [degrees]
    """
    ra1_rad = np.radians(ra1)
    dec1_rad = np.radians(dec1)
    ra2_rad = np.radians(ra2)
    dec2_rad = np.radians(dec2)

    angular_sep = np.degrees(coord.angular_separation(dec1_rad, ra1_rad, dec2_rad, ra2_rad))

    return angular_sep


def pyramid_search(ra, dec, apothem, galaxy_catalog):
    """
    Finds all galaxies within a given position and angular apothem.
    Search is restricted to not intersect geographic poles.

    Params:
        - ra, dec (float): Coordinates of the centre of the pyramid search [degrees]
        - apothem (float): Half the angular side length of the pyramid base [degrees]
        - galaxy_catalog (pandas.DataFrame): Galaxy catalog that is being searched

    Returns:
        - selected_galaxies (pandas.DataFrame): All galaxies contained within the pyramid search
    """
    lwr_ra = (ra - apothem) % 360
    upr_ra = (ra + apothem) % 360
    lwr_dec = dec - apothem
    upr_dec = dec + apothem

    # If search intersects either geographic pole then exit with error
    if lwr_dec < -90 or upr_dec > 90:
        raise ValueError(f'Declination must fall within range (-90, 90), current range is ({lwr_dec}, {upr_dec})!')

    gc_copy = copy.deepcopy(galaxy_catalog)

    # Filter galaxies within the pyramid search
    selected_galaxies = gc_copy[(is_between(lwr_ra, upr_ra, gc_copy['ra'])) &
                                gc_copy['dec'].between(lwr_dec, upr_dec, inclusive='both')]

    return selected_galaxies


def cone_search(ra, dec, radius, galaxy_catalog):
    """
    Finds all galaxies within a given position and angular radius.

    Params:
        - ra, dec (float): Coordinates of the centre of the cone search [degrees]
        - radius (float): Angular radius of the cone search [degrees]
        - galaxy_catalog (pandas.DataFrame): Galaxy catalog that is being searched

    Returns:
        - selected_galaxies (pandas.DataFrame): All galaxies contained within the cone search
    """
    gc_copy = copy.deepcopy(galaxy_catalog)

    # Filter galaxies within the specified cone search radius
    selected_galaxies = gc_copy[angular_separation(ra, dec, galaxy_catalog['ra'], galaxy_catalog['dec']) <= radius]

    return selected_galaxies


def get_air_mass(ra, dec, dt):
    """
    Gets air mass at a certain time and location. Credit goes to Thomas Plunkett,
    function was adapted from https://github.com/tjplunkett/UTGO_ObsTools.

    Params:
        - ra, dec (float): The coordinates of the target [degrees]
        - dt (datetime): The datetime of observation

    Returns:
        - obj_air_mass (float): The air mass at the given time and location
    """
    # Define timezone and get UTC time of observations
    tz = timezone("Australia/Tasmania")
    aware1 = tz.localize(dt)
    utcoffset = aware1.utcoffset()
    time_utc = Time(dt) - utcoffset

    coordinate = coord.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
    target = FixedTarget(coord=coordinate)

    # Calculate air mass
    obj_air_mass = GHO.altaz(time_utc, target).secz

    return obj_air_mass


def sky_error_region_gaussian(ra, dec, error_radius):
    """
    Gaussian that simulates the GRB error distribution within the sky error region.

    Params:
        - ra, dec (float): Mean of the circular sky region [degrees]
        - error_radius (float): Angular error radius corresponding to 3 std dev [degrees]

    Returns:
        - multi_normal (openturns.Normal): Sky error gaussian
    """
    # Error radius corresponds to 3 standard deviations
    std_dev = error_radius / 3

    mean = np.array([ra, dec])
    cov = np.array([[std_dev**2, 0.0], [0.0, std_dev**2]])
    multi_normal = ot.Normal(mean, ot.CovarianceMatrix(cov))

    return multi_normal


def probability_in_sky_error_region_tile(sky_gaussian, cell_ra, cell_dec, cell_side_len):
    """
    Gets the probability contained within a sky error region tile.

    Params:
        - sky_gaussian (openturns.Normal): Gaussian representing the GRB error distribution
        - cell_ra, cell_dec (float): Coordinates of cell's centre [degrees]
        - cell_side_len (float): Angular side length of the cell [degrees]

    Returns:
        - probability (float): Probability within cell
    """
    min_ra = cell_ra - cell_side_len / 2
    max_ra = cell_ra + cell_side_len / 2
    min_dec = cell_dec - cell_side_len / 2
    max_dec = cell_dec + cell_side_len / 2

    probability = sky_gaussian.computeProbability(ot.Interval([min_ra, min_dec], [max_ra, max_dec]))

    return probability


def chord_point_coordinates(axis_offset, chord_length, separation):
    """
    Gets the coordinates of points along a horizontal or vertical chord given an axis offset.

    Params:
        - axis_offset (float): Offset of the chord from the origin
        - chord_length (float): Length of the chord
        - separation (float): Separation of points along the chord

    Returns:
        - coordinates (List): List of points over the given chord axis
    """
    # Number of points that lie on the chord to tile it
    n = int(chord_length / separation)

    # Optimise points for small sky error regions
    if n == 0 or n == 1:
        n += 1

    # Calculate the offset of the end points from the chord ends
    total_points_length = (n - 1) * separation
    remaining_space = chord_length - total_points_length
    space = remaining_space / 2

    # Calculate the coordinates of the points along the chord
    coordinates = np.linspace(space + axis_offset - chord_length/2, axis_offset + chord_length/2 - space, n)

    return coordinates


def tile_circular_sky_region(ra, dec, radius, fov):
    """
    Tile a circular sky region into squares matching a given square telescope field of view.

    Params:
        - ra, dec (float): Coordinates of the circular sky region origin [degrees]
        - radius (float): Angular error radius of the circular sky region [degrees]
        - fov (float): Angular side length of the telescope field of view [degrees]

    Returns:
        - sol_grid (list): List of cells that make up the tiled sky region
    """
    diameter = 2 * radius

    # Calculate the y-coordinates of the points
    y_coordinates = chord_point_coordinates(dec, diameter, fov)

    # Calculate the separation of each point from the circle's center
    point_separations = [round(np.sqrt((ra - ra) ** 2 + (y - dec) ** 2), 3) for y in y_coordinates]

    # Calculate chord lengths
    chord_lengths = [2 * np.sqrt((radius ** 2) - (s ** 2)) for s in point_separations]

    # Galaxies contained within circular sky region
    galaxies_in_region = cone_search(ra, dec, radius, GLADE_CATALOG)

    # Sky error region gaussian
    sky_gaussian = sky_error_region_gaussian(ra, dec, radius)

    sol_grid = []

    for y, chord_length in zip(y_coordinates, chord_lengths):
        x_dots = chord_point_coordinates(ra, chord_length, fov)

        for x in x_dots:
            cell_ra = x
            cell_dec = y

            cell_am = get_air_mass(cell_ra, cell_dec, datetime.now())
            cell_galaxies = pyramid_search(cell_ra, cell_dec, fov / 2, galaxies_in_region)
            cell_prob = probability_in_sky_error_region_tile(sky_gaussian, cell_ra, cell_dec, fov)
            cell_gal_prob = 0
            if len(cell_galaxies) != 0 and len(galaxies_in_region) != 0:
                cell_gal_prob = (cell_prob + len(cell_galaxies) / len(galaxies_in_region)) / 2
            cell_to_add = cl.Cell(cell_ra, cell_dec, fov, cell_prob, cell_am, cell_galaxies, cell_gal_prob)

            sol_grid.append(cell_to_add)

    return sol_grid


def angular_offset_radius(redshift):
    """
    A function to convert the radius of the galaxy's offset distribution into an angular radius.

    Return:
        angular_offset_radius (float): Angular radius of the galaxy's offset distribution in degrees
    """
    z = redshift * cu.redshift
    kpc_distance_from_earth = z.to(u.kpc, cu.redshift_distance())

    offset_radius = 80  # kpc
    ang_offset_radius = math.degrees(math.atan(offset_radius / kpc_distance_from_earth.value))

    return ang_offset_radius


def angular_fov_to_length_fov(redshift, fov):
    """
    A function to convert the angular fov of a telescope into a constant fov
    for a certain distance from Earth (determined by redshift).

    Return:
       length_fov (float): constant fov side length
    """
    z = redshift * cu.redshift
    kpc_distance_from_earth = z.to(u.kpc, cu.redshift_distance())

    length_fov = kpc_distance_from_earth.value * math.tan(math.radians(fov))
    return length_fov


def angular_co_to_length_co(redshift, coordinate):
    """
    A function to convert the angular coordinates to constant coordinates.

    Return:
       length_co (float): constant coordinates
    """
    z = redshift * cu.redshift
    kpc_distance_from_earth = z.to(u.kpc, cu.redshift_distance())

    length_co = kpc_distance_from_earth.value * math.tan(math.radians(coordinate))
    return length_co


def gaussian_kde_offset_distribution():
    """
    Gaussian KDE function for the offset distribution.
    """

    # Lists of x and y offsets [kpc]
    x_off, y_off = zip(*graph.OFFSET_DISTRIBUTION_COORDINATES)

    values = np.vstack([x_off, y_off])
    kernel = gaussian_kde(values)

    return kernel


def tile_circular_galaxy_region(ra, dec, sr_radius, gal_ra, gal_dec, redshift, fov):
    """
    Tile a circular galaxy region into squares matching a given square telescope field of view.

    Params:
        - ra, dec (float): Coordinates of the circular sky region origin [degrees]
        - gal_ra, gal_dec (float): Coordinates of the galaxy origin [degrees]
        - sr_radius (float): Angular error radius of the circular sky region [degrees]
        - fov (float): Angular side length of the telescope field of view [degrees]

    Returns:
        - sol_grid (list): List of cells that make up the tiled galaxy region
    """
    radius = angular_offset_radius(redshift)

    len_fov = angular_fov_to_length_fov(redshift, fov)

    diameter = 2 * radius

    # Calculate the y-coordinates of the points
    y_coordinates = chord_point_coordinates(gal_dec, diameter, fov)

    # Calculate the separation of each point from the circle's center
    point_separations = [round(np.sqrt((gal_ra - gal_ra) ** 2 + (y - gal_dec) ** 2), 3) for y in y_coordinates]

    # Calculate chord lengths
    chord_lengths = [2 * np.sqrt((radius ** 2) - (s ** 2)) for s in point_separations]

    # Sky error region gaussian
    sky_gaussian = gaussian_kde_offset_distribution()

    sol_grid = []

    for y, chord_length in zip(y_coordinates, chord_lengths):
        x_dots = chord_point_coordinates(gal_ra, chord_length, fov)

        for x in x_dots:
            cell_ra = x
            cell_dec = y

            cell_x = angular_co_to_length_co(redshift, gal_ra - cell_ra)
            cell_y = angular_co_to_length_co(redshift, gal_dec - cell_dec)

            min_ra = cell_x - len_fov / 2
            max_ra = cell_x + len_fov / 2
            min_dec = cell_y - len_fov / 2
            max_dec = cell_y + len_fov / 2

            cell_prob = sky_gaussian.integrate_box([min_ra, min_dec], [max_ra, max_dec])
            cell_am = get_air_mass(cell_ra, cell_dec, datetime.now())
            cell_to_add = cl.Cell(cell_ra, cell_dec, fov, cell_prob, cell_am, 0, 0)

            sol_grid.append(cell_to_add)

    # Filter out cells that lie outside the sky error region
    sol_grid = [cell for cell in sol_grid if ((cell.ra - ra) ** 2 + (cell.dec - dec) ** 2 <= sr_radius ** 2)]

    return sol_grid


# def tile_circular_sky_region_ref(ra, dec, radius, fov):
#     """
#     Tile a circular sky region into squares matching a given square telescope field of view (REFERENCE FUNCTION).
#
#     Params:
#         - ra, dec (float): Coordinates of the circular sky region origin [degrees]
#         - radius (float): Angular error radius of the circular sky region [degrees]
#         - fov (float): Angular side length of the telescope field of view [degrees]
#
#     Returns:
#         - sol_grid (list): List of cells that make up the tiled sky region
#     """
#     # Calculate the grid dimension rounded to the nearest odd integer
#     grid_dim = round(2 * radius / fov)
#     if grid_dim % 2 == 0:
#         grid_dim += 1
#
#     # Galaxies contained within circular sky region
#     galaxies_in_region = cone_search(ra, dec, radius, GLADE_CATALOG)
#
#     # Sky error region gaussian
#     sky_gaussian = sky_error_region_gaussian(ra, dec, radius)
#
#     sol_grid = []
#
#     # Loop through rows
#     for i in range(grid_dim):
#
#         # Loop through columns
#         for j in range(grid_dim):
#             # Calculate coordinates for the center of each cell
#             cell_ra = ra + (j - grid_dim // 2) * fov
#             cell_dec = dec + (i - grid_dim // 2) * fov
#
#             # Add cell to solution if cell centre lies within the circular sky region
#             if (cell_ra - ra) ** 2 + (cell_dec - dec) ** 2 <= radius ** 2:
#                 cell_am = get_air_mass(cell_ra, cell_dec, datetime.now())
#                 cell_galaxies = pyramid_search(cell_ra, cell_dec, fov / 2, galaxies_in_region)
#                 cell_prob = probability_in_sky_error_region_tile(sky_gaussian, cell_ra, cell_dec, fov)
#                 cell_to_add = cl.Cell(cell_ra, cell_dec, fov, cell_prob, cell_am, cell_galaxies)
#
#                 sol_grid.append(cell_to_add)
#
#     return sol_grid
