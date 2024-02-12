# Import external libraries
import time

# Import local libraries
import graphing as graph
import functions as fn
import classes as cl


'''TELESCOPE FOV'''
# As only square FOVs are considered, FOV is given as a side length
FOV = 0.08  # degrees


if __name__ == '__main__':
    start_runtime = time.time()

    glade = fn.GLADE_CATALOG

    # GRB sky error region parameters
    ra = 0
    dec = 0
    error_radius = 0.3
    gal_in_region = fn.cone_search(ra, dec, error_radius, glade)
    tiled_grb_region = fn.tile_circular_sky_region(ra, dec, error_radius, FOV)

    # Decide whether to redistribute probability based on galaxy density (0 == false, 1 == true)
    use_gal_prob = 0

    # Tile Search
    if use_gal_prob == 1:
        # Filter out cells where gal_prob == 0
        filtered_cells = [cell for cell in tiled_grb_region if cell.gal_prob != 0]
        grb_sky_err_reg_filtered = cl.GrbSkyErrorRegion(ra, dec, error_radius, gal_in_region, filtered_cells)
        graph.tiled_circular_sky_region(grb_sky_err_reg_filtered)
        print('Cell Coordinates [ra, dec]:')
        for cell in filtered_cells:
            print(f'[{round(cell.ra, 3)}, {round(cell.dec, 3)}]')
    elif use_gal_prob == 0:
        grb_sky_err_reg = cl.GrbSkyErrorRegion(ra, dec, error_radius, gal_in_region, tiled_grb_region)
        graph.tiled_circular_sky_region(grb_sky_err_reg)
        print('Cell Coordinates [ra, dec]:')
        for cell in tiled_grb_region:
            print(f'[{round(cell.ra, 3)}, {round(cell.dec, 3)}]')

    # Galaxy Targeted Search (0 == false, 1 == true)
    use_gal_tar = 1

    # Galaxy parameters
    gal_ra = 0.1
    gal_dec = 0.1
    redshift = 0.008

    if use_gal_tar == 1:
        tiled_galaxy_region = fn.tile_circular_galaxy_region(ra, dec, error_radius, gal_ra, gal_dec, redshift, FOV)
        gal_sky_err_reg = cl.GrbSkyErrorRegion(ra, dec, error_radius, 0, tiled_galaxy_region)
        graph.tiled_circular_sky_region(gal_sky_err_reg)
        print('Cell Coordinates [ra, dec]:')
        for cell in tiled_grb_region:
            print(f'[{round(cell.ra, 3)}, {round(cell.dec, 3)}]')

    # graph.grb_offset_distribution()
    # graph.gaussian_kde_offset_distribution()
    # graph.circular_gaussian_kde(gal_in_region['ra'], gal_in_region['dec'], ra, dec, error_radius)
    # graph.cell_redshift_distribution(tiled_grb_region[0])

    end_runtime = time.time()

    print('Total runtime: ', end_runtime - start_runtime, 'sec')
