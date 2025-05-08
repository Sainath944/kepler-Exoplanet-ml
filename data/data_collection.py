import pandas as pd #type:ignore
import requests#type:ignore
print("helo")
# KOI Candidate Data - column info: https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html
# Set API url
api_url = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?'

# Columns to use, selected by inspection
col_to_pull = ['kepoi_name', 'koi_disposition', 'koi_comment', 'koi_count', 'koi_fpflag_nt',
               'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'ra', 'dec',
               'koi_gmag', 'koi_rmag', 'koi_imag', 'koi_zmag', 'koi_jmag', 'koi_hmag',
               'koi_kmag', 'koi_kepmag', 'koi_num_transits', 'koi_max_sngle_ev', 'koi_max_mult_ev',
               'koi_bin_oedp_sig', 'koi_ldm_coeff4', 'koi_ldm_coeff3', 'koi_ldm_coeff2', 'koi_ldm_coeff1',
               'koi_model_snr', 'koi_prad', 'koi_sma', 'koi_impact', 'koi_duration',
               'koi_depth', 'koi_period', 'koi_ror', 'koi_dor', 'koi_incl', 'koi_teq', 'koi_steff',
               'koi_slogg', 'koi_smet', 'koi_srad', 'koi_smass', 'koi_fwm_stat_sig', 'koi_fwm_sra',
               'koi_fwm_sdec', 'koi_fwm_srao', 'koi_fwm_sdeco', 'koi_fwm_prao', 'koi_fwm_pdeco',
               'koi_dicco_mra', 'koi_dicco_mdec', 'koi_dicco_msky', 'koi_insol', 'koi_srho']

# Aliases to rename columns
aliases = ['koi_name', 'disposition', 'false_positive_type', 
           'num_of_objects_around_star', 'fp_not_transit', 'fp_stellar_eclipse', 'fp_centroid_offset',
           'fp_contamination', 'right_ascension', 'declination', 'g_band_mag', 'r_band_mag', 'i_band_mag',
           'z_band_mag', 'j_band_mag', 'h_band_mag', 'k_band_mag', 'kepler_band_mag', 'num_of_transits',
           'max_single_event_stat', 'max_multi_event_stat', 'odd_even_depth_stat', 'limb_dark_co4',
           'limb_dark_co3', 'limb_dark_co2', 'limb_dark_co1', 'transit_signal_to_noise',
           'planet_radius_earth', 'orbit_semimajor_axis', 'impact', 'transit_duration', 'transit_depth',
           'orbital_period', 'planet_star_radius_ratio', 'planet_star_distance_radius', 'inclination',
           'planet_temp', 'star_temp', 'star_surface_gravity', 'star_metallicity', 'star_radius',
           'star_mass', 'flux_weight_offset_sig', 'centroid_right_ascension', 'centroid_declination',
           'centroid_right_ascension_offset', 'centroid_declination_offset', 'planet_star_right_ascension_offset',
           'planet_star_declination_offset', 'angular_offset_right_ascension', 'angular_offset_declination',
           'angular_offset_sky', 'insolation_flux', 'star_density']

# Format and send API request
select_string = ', '.join(col_to_pull)
params = {'select': '{}'.format(select_string), 'table': 'cumulative', 'format': 'csv'}
results = requests.get(api_url, params=params)

# Convert results into single text string, replacing header row with aliases
rows = results.text.split('\n')
rows[0] = ','.join(aliases)
data_string = '\n'.join(rows)

# Write to .csv file
with open('your/path/to/the/data.csv', 'w') as f:
    f.write(data_string)