import os
from typing import Dict
from scipy.signal import butter


def design_filter(xc_param: Dict, mem_info: Dict):
    """
    Design filter for fastxc, save the filter coefficients to file.

    Parameters
    ----------
    xc_param : dict, including frequency bands, e.g. {'bands': '0.1/0.5 0.5/1.0 1.0/2.0'}
    executing : dict, including output_dir, e.g. {'output_dir': 'output'}
    mem_info : dict, including delta, e.g. {'delta': 0.001}

    Returns
    -------
    None
    """

    # Parsing parameters
    bands = xc_param['bands']  # frequency bands
    output_dir = xc_param['output_dir']  # output directory
    fs = 1.0 / mem_info['delta']  # sampling frequency
    f_nyq = fs / 2.0  # Nyquist frequency
    order = 2  # filter order

    # check output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'butterworth_filter.txt')
    bands_str = bands.split()

    filters = []
    for band_str in bands_str:
        freq_low, freq_high = map(float, band_str.split('/'))

        if not 0 < freq_low < freq_high < f_nyq:
            print(f'Error: frequency bands {band_str} are not valid.')
            raise ValueError

        freq_low_norm = freq_low / f_nyq  # normalized low frequency
        freq_high_norm = freq_high / f_nyq  # normalized high frequency

        b, a = butter(order, [freq_low_norm, freq_high_norm], btype='bandpass')

        line_b = '\t'.join(f'{b_i:.18e}' for b_i in b)
        line_a = '\t'.join(f'{a_i:.18e}' for a_i in a)

        filters.append([line_b, line_a])

    try:
        with open(output_file, 'w') as f:
            for filter, band_str in zip(filters, bands_str):
                f.write(f'# {band_str}\n')
                f.write(filter[0] + '\n')
                f.write(filter[1] + '\n')
    except IOError as e:
        print(f"Filter file writing error: {e}")


if __name__ == "__main__":
    design_filter({'bands': '0.2/0.5 0.6/0.8'},
                  {'output_dir': './'}, {'delta': 0.01})
