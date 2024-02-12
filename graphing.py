# Import external libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal, gaussian_kde


'''Offset distribution as a list of tuples [kpc]. Positive-y is aligned with galaxy's North.'''
offset_file = open("Offset Coordinates.txt", "r")
OFFSET_DISTRIBUTION_COORDINATES = eval(offset_file.read())


def gaussian_kde_offset_distribution():
    """
    Visualises the gaussian kernel density estimate of the GRB host galaxy offset distribution using several graphs.
    """
    # Create meshgrid
    x_min = -30
    x_max = 30
    y_min = -30
    y_max = 30
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

    # Lists of x and y offsets [kpc]
    x_off, y_off = zip(*OFFSET_DISTRIBUTION_COORDINATES)

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x_off, y_off])
    kernel = gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    colormap = 'coolwarm'

    ax.contourf(xx, yy, f, cmap=colormap)
    ax.imshow(np.rot90(f), cmap=colormap, extent=(x_min, x_max, y_min, y_max))
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('ΔRA (kpc)')
    ax.set_ylabel('ΔDEC (kpc)')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.title('2D Gaussian Kernel density estimation')

    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap=colormap, edgecolor='none')
    ax.set_xlabel('ΔRA (kpc)')
    ax.set_ylabel('ΔDEC (kpc)')
    ax.set_zlabel('PDF')
    ax.set_title('Surface plot of Gaussian 2D KDE')
    fig.colorbar(surf, shrink=0.5, aspect=5)  # add color bar indicating the PDF
    ax.view_init(60, 35)

    plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(xx, yy, f)
    ax.set_xlabel('ΔRA (kpc)')
    ax.set_ylabel('ΔDEC (kpc)')
    ax.set_zlabel('PDF')
    ax.set_title('Wireframe plot of Gaussian 2D KDE')

    plt.show()


def gaussian_offset_distribution():
    """
    Calculates then graphs the bi-variate gaussian distribution associated with the GRB host galaxy offset distribution.
    Goodness of fit using Mardia's tests showed poor correlation.
    """
    # Lists of x and y offsets [kpc]
    x_off, y_off = zip(*OFFSET_DISTRIBUTION_COORDINATES)

    # Convert the lists to NumPy arrays
    x_off_array = np.array(x_off)
    y_off_array = np.array(y_off)

    # Calculate the mean vector and covariance matrix of the gaussian distribution.
    mean_vector = np.array([np.mean(x_off_array), np.mean(y_off_array)])
    cov_matrix = np.cov(x_off_array, y_off_array)

    # Create a grid of x and y values
    x, y = np.meshgrid(np.linspace(-80, 80, 100), np.linspace(-80, 80, 100))
    pos = np.dstack((x, y))

    # Create a multivariate Gaussian distribution
    gaussian = multivariate_normal(mean=mean_vector, cov=cov_matrix)

    # Calculate the probability density function (PDF) values
    pdf_values = gaussian.pdf(pos)

    # Contour plot of the bi-variate normal distribution
    fig = plt.figure()
    ax = fig.gca()
    ax.contourf(x, y, pdf_values, cmap='coolwarm')
    ax.imshow(np.rot90(pdf_values), cmap='coolwarm', extent=(-80.0, 80.0, -80.0, 80.0))
    cset = ax.contour(x, y, pdf_values, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('ΔRA (kpc)')
    ax.set_ylabel('ΔDEC (kpc)')

    # 3d plot of the bi-variate gaussian distribution
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, pdf_values, cmap='plasma')

    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)

    ax.set_xlabel('ΔRA (kpc)')
    ax.set_ylabel('ΔDEC (kpc)')
    ax.set_zlabel('PDF')
    ax.set_title('Gaussian Offset Distribution')

    plt.show()


def grb_offset_distribution():
    """
    Graphs the distribution of GRB host galaxy offsets as seen in
    https://iopscience.iop.org/article/10.3847/1538-4357/ac91d0.
    """
    fig, ax = plt.subplots()

    median_radius = plt.Circle((0, 0), 7.7, ec='red', fc='none')
    kpc10_radius = plt.Circle((0, 0), 10, ec='black', fc='none', alpha=0.25, linestyle='--')
    kpc30_radius = plt.Circle((0, 0), 30, ec='black', fc='none', alpha=0.25, linestyle='--')
    kpc50_radius = plt.Circle((0, 0), 50, ec='black', fc='none', alpha=0.25, linestyle='--')
    plt.gca().add_patch(median_radius)
    plt.gca().add_patch(kpc10_radius)
    plt.gca().add_patch(kpc30_radius)
    plt.gca().add_patch(kpc50_radius)

    plt.text(-35, 15, '30 kpc', rotation=45)
    plt.text(-49, 29, '50 kpc', rotation=45)
    plt.text(-75, -75, 'Median = 7.7 kpc', color='red')

    # Add horizontal and vertical axes
    ax.axhline(0, color='black', linewidth=1, alpha=0.25, zorder=0)
    ax.axvline(0, color='black', linewidth=1, alpha=0.25, zorder=0)

    # Unpack the list of tuples into lists of x, y coordinates and graph them
    delta_ra, delta_dec = zip(*OFFSET_DISTRIBUTION_COORDINATES)
    plt.scatter(delta_ra, delta_dec, s=5)

    ax.set_aspect('equal')
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)

    plt.title('GRB Host Galaxy\nOffset Distribution', size=16,)
    plt.xlabel('ΔRA (kpc)')
    plt.ylabel('ΔDEC (kpc)')

    plt.show()


def circular_gaussian_kde(ra_values, dec_values, ra, dec, radius):
    """
    Given a circular sky region and any list (ra, dec) coordinates in it, graph a gaussian kde.
    Used for visualising galaxy density.
    """
    # Create mesh grid
    x_min = ra - 1.5 * radius
    x_max = ra + 1.5 * radius
    y_min = dec - 1.5 * radius
    y_max = dec + 1.5 * radius

    # Graph scatter of x, y values
    fig, ax = plt.subplots()
    sky_region_circle = plt.Circle((ra, dec), radius, ec='red', fc='none')
    ax.add_patch(sky_region_circle)
    plt.scatter(ra_values, dec_values, s=1, color='black')
    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([ra_values, dec_values])
    kernel = gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    ax.contourf(xx, yy, f, cmap='coolwarm')
    ax.imshow(np.rot90(f), cmap='coolwarm', extent=(x_min, x_max, y_min, y_max))
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('RA (degrees)')
    ax.set_ylabel('DEC (degrees)')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.title('Galaxy density for GRB sky error region')

    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
    ax.set_xlabel('RA (degrees)')
    ax.set_ylabel('DEC (degrees)')
    ax.set_zlabel('PDF')
    ax.set_title('Galaxy density for GRB sky error region')
    fig.colorbar(surf, shrink=0.5, aspect=5)  # add color bar indicating the PDF
    ax.view_init(60, 35)

    plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(xx, yy, f)
    ax.set_xlabel('RA (degrees)')
    ax.set_ylabel('DEC (degrees)')
    ax.set_zlabel('PDF')
    ax.set_title('Wireframe plot of Gaussian 2D KDE')

    plt.show()


def tiled_circular_sky_region(grb_ser):
    """
    Graphs the tiled circular sky region.

    Params:
        - grb_ser (classes.GrbSkyErrorRegion): Sky error region of a GRB
    """
    galaxy_circle = plt.Circle((grb_ser.ra, grb_ser.dec), grb_ser.error_radius, fc='none', ec='black', linewidth=2)
    plt.gca().add_patch(galaxy_circle)

    for cell in grb_ser.tiled_region:
        # Coordinates of the bottom left corner of the square being graphed
        bl_ra = cell.ra - cell.side_len / 2
        bl_dec = cell.dec - cell.side_len / 2

        rectangle = plt.Rectangle((bl_ra, bl_dec), cell.side_len, cell.side_len, fc='gray', ec='black', alpha=0.5)
        plt.gca().add_patch(rectangle)
        plt.scatter(cell.ra, cell.dec, c='black', s=40*cell.side_len)

    plt.axis('scaled')

    plt.xlabel("Right Ascension (degrees)")
    plt.ylabel("Declination (degrees)")

    plt.show()


def cell_redshift_distribution(cell):
    """
    Graphs the redshift distribution within a cell

    Params:
        - cell (classes.Cell): Cell whose galaxy redshift distribution is being graphed
    """
    cell_galaxy_redshifts = cell.galaxies['z_h']

    if cell_galaxy_redshifts.empty:
        print('Cell contains no galaxies with a recorded redshift.')
    else:
        plt.hist(cell_galaxy_redshifts, bins='auto')

        plt.title('Galaxy redshift distribution for cell')
        plt.xlabel('Redshift')
        plt.ylabel('Frequency')

        plt.show()
