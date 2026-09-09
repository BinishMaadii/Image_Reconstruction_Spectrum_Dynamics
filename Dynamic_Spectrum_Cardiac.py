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

print("Step [1] Generating phantom")
image_size = 96 
x, y = np.mgrid[0:image_size, 0:image_size] # to secure the spots where image should be created
center = image_size /2

###3 In order to calculate the distance from the center of each pixel

# $r(x, y) = \sqrt{(x - x_0)^2 + (y - y_0)^2}$
radius = np.sqrt( (x-center) ** 2  +  (y - center) ** 2)

# $\theta(x, y) = \left( \operatorname{arctan2}(y - y_0, x - x_0) \cdot \frac{180}{\pi} + 360 \right) \pmod{360}$
angles_deg = (np.degrees (np.arctan2(y - center, x - center)) ** 2) 




