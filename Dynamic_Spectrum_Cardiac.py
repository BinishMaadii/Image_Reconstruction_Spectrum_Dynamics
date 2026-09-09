### Initializing and invoking the libraries

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, shift
from scipy.optimize import curve_fit
from skimage.transform import radon, iradon


#### Setting the output 
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)


# Helper function to save plots
def save_plot(fig, filename):
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


##### Generating the Phantom in 2D
