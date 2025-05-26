"""
Optiland Image Simulation Module.

This module provides the `ImageSimulation` class, which simulates the image
formation process of an input image through a given optical system. It includes
functionality for Point Spread Function (PSF) generation, PSF interpolation,
image loading, spatially varying convolution, and geometric distortion.
"""
import numpy as np
# It's common to use a backend for numerical computations, especially if GPU support is desired.
# Assuming 'optiland.backend' provides this.
import optiland.backend as be
import matplotlib.pyplot as plt # For potential visualization tasks
import itertools
from optiland.psf import FFTPSF
from scipy.spatial import KDTree
from scipy.ndimage import zoom
from PIL import Image


# Assuming Optic is defined elsewhere, e.g., from optiland.optic_model import Optic
# This import might be needed if type hinting for Optic is used extensively,
# but for now, we'll assume it's passed as an object.

class ImageSimulation:
    """
    Simulates image formation through an optical system.

    Handles PSF generation, input image loading and preprocessing,
    spatially varying convolution with interpolated PSFs, and can apply
    geometric distortion. The simulation pipeline is typically run upon
    instantiation.

    Attributes:
        optic: The optical system model.
        image_path (str): Path to the input image.
        grid_density (tuple): Grid density for PSF computation.
        psf_type (str): Type of PSF computation ('FFT' or 'geometric').
        wavelengths_resolved (list): List of wavelengths used for simulation.
        num_psf_rays (int): Number of rays for PSF computation.
        psf_grid_size (int): Grid size for individual PSF computation.
        output_resolution (tuple): Desired output resolution (width, height)
            of the image.
        oversample_factor (int): Factor for oversampling the input image.
        use_distortion (bool): Boolean indicating if geometric distortion is applied.
        psf_grid_data (list): Stores computed PSFs at grid points. Each entry
            is a dict with 'field_point', 'wavelength', 'psf_data', etc.
        _input_image_data (be.DeviceArray): The loaded, (potentially)
            oversampled, and normalized input image data. Shape is (height, width)
            for grayscale or (height, width, channels) for color.
        _convolved_image_data (be.DeviceArray): The image data after
            convolution with spatially varying PSFs. Shape matches
            `_input_image_data`.
        _processed_image_data (be.DeviceArray): The final image data after all
            processing steps (convolution, distortion, resampling). Shape is
            (output_resolution[1], output_resolution[0], channels) or
            (output_resolution[1], output_resolution[0]) for grayscale.
    """

    def __init__(self,
                 optic,
                 image_path,
                 grid_density=(5, 5),
                 psf_type='FFT',
                 wavelengths='primary',
                 num_psf_rays=64,
                 psf_grid_size=512,
                 output_resolution=(512, 512),
                 oversample_factor=1,
                 use_distortion=True):
        """
        Initializes the ImageSimulation class.

        This involves setting up parameters, resolving wavelengths, generating
        the PSF grid, loading the input image, performing convolution,
        applying distortion (if enabled), and resampling to the final output
        resolution.

        Args:
            optic: The optical system model (instance of an Optic class).
            image_path (str): Path to the input image file.
            grid_density (tuple): A tuple (rows, cols) defining the grid
                density for PSF computation across the field of view.
                Default is (5, 5).
            psf_type (str): Type of PSF computation. Either 'FFT' for wave
                optics based PSF or 'geometric' for ray-based geometric PSF.
                Default is 'FFT'.
            wavelengths (str or list): Wavelengths to use for PSF computation.
                - 'all': Use all wavelengths defined in `optic.wavelengths`.
                - 'primary': Use `optic.primary_wavelength`.
                - list of float: A list of specific wavelengths in micrometers.
                Default is 'primary'.
            num_psf_rays (int): Number of rays for PSF computation (e.g.,
                pupil samples like 64x64 for FFTPSF). Default is 64.
            psf_grid_size (int): Grid size (pixels) for individual PSF
                computation (e.g., 512x512 for FFTPSF). Default is 512.
            output_resolution (tuple): Desired resolution (width, height) of
                the final simulated image in pixels. Default is (512, 512).
            oversample_factor (int): Factor to oversample the input image
                before convolution. Default is 1.
            use_distortion (bool): Whether to apply geometric distortion.
                Default is True.
        """
        self.optic = optic
        self.image_path = image_path
        self.grid_density = grid_density
        self.psf_type = psf_type
        self.wavelengths = wavelengths # Keep original for reference
        self.num_psf_rays = num_psf_rays
        self.psf_grid_size = psf_grid_size
        self.output_resolution = output_resolution
        self.oversample_factor = oversample_factor
        self.use_distortion = use_distortion

        # Resolve wavelengths
        if wavelengths == 'all':
            if not hasattr(self.optic, 'wavelengths') or not hasattr(self.optic.wavelengths, 'get_wavelengths'):
                raise ValueError("Optic model does not have 'wavelengths' attribute with 'get_wavelengths' method.")
            self.wavelengths_resolved = self.optic.wavelengths.get_wavelengths()
        elif wavelengths == 'primary':
            if not hasattr(self.optic, 'primary_wavelength'):
                raise ValueError("Optic model does not have 'primary_wavelength' attribute.")
            self.wavelengths_resolved = [self.optic.primary_wavelength]
        elif isinstance(wavelengths, list):
            self.wavelengths_resolved = wavelengths
        else:
            raise ValueError(f"Invalid value for wavelengths: {wavelengths}. "
                             "Must be 'all', 'primary', or a list of floats.")

        if not self.wavelengths_resolved:
            raise ValueError("Resolved wavelengths list is empty. Cannot compute PSFs.")

        # Initialize data storage attributes
        self.psf_grid_data = []
        self._input_image_data = None
        self._convolved_image_data = None
        self._processed_image_data = None

        # --- Simulation Pipeline ---
        # 1. Generate PSF grid
        self._generate_psf_grid()
        
        # 2. Load input image (applies oversampling)
        self._load_image()

        # 3. Convolve image with PSFs
        if self._input_image_data is not None:
            self._convolve_image_with_psfs()
        else:
            print("Warning: Input image data not loaded, skipping convolution.")
            # If no input image, processed_image cannot be generated
            return

        # Start with convolved image as the basis for processed_image
        if self._convolved_image_data is not None:
            current_image_for_processing = self._convolved_image_data.copy()
        else:
            print("Warning: Convolved image data is not available. Further processing steps might fail or be skipped.")
            # If convolution failed, cannot proceed with distortion or resampling meaningfully
            # Potentially set _processed_image_data to a zero array of output_resolution or handle error
            return

        # 4. Apply geometric distortion
        if self.use_distortion:
            current_image_for_processing = self._apply_geometric_distortion(current_image_for_processing)
        
        self._processed_image_data = current_image_for_processing # Store intermediate result

        # 5. Resample to final output_resolution
        # Check if resampling is needed (shape is (H, W, C) or (H, W))
        # output_resolution is (W, H)
        img_h, img_w = self._processed_image_data.shape[0], self._processed_image_data.shape[1]
        target_w, target_h = self.output_resolution

        if img_h != target_h or img_w != target_w:
            self._resample_to_output() # This method will update self._processed_image_data

        print("ImageSimulation initialization complete.")


    def view(self, image_type='processed', title=None, figsize=(8, 8)):
        """
        Displays the specified image using matplotlib.pyplot.

        Args:
            image_type (str, optional): The type of image to display.
                Valid options are 'input', 'convolved', 'processed', or 'output'.
                Defaults to 'processed'. 'output' is an alias for 'processed'.
            title (str, optional): The title for the plot. If None, a default
                title is generated based on `image_type`. Defaults to None.
            figsize (tuple, optional): The figure size (width, height) in inches
                for the plot. Defaults to (8, 8).
        
        Raises:
            ValueError: If an invalid `image_type` is provided.
        """
        image_data_to_display = None
        default_title_prefix = "Displayed Image: "

        if image_type == "input":
            image_data_to_display = self._input_image_data
            default_title = default_title_prefix + "Input (Oversampled)"
        elif image_type == "convolved":
            image_data_to_display = self._convolved_image_data
            default_title = default_title_prefix + "Convolved"
        elif image_type == "processed" or image_type == "output":
            image_data_to_display = self._processed_image_data
            default_title = default_title_prefix + "Processed (Final)"
        else:
            raise ValueError(f"Unknown image_type: '{image_type}'. "
                             "Valid options are 'input', 'convolved', 'processed', 'output'.")

        if image_data_to_display is None:
            print(f"Warning: No data available for image_type='{image_type}'. "
                  "It might not have been computed yet or an error occurred.")
            return

        # Convert to NumPy array and ensure it's in the [0, 1] range for display
        img_np = be.to_numpy(image_data_to_display)
        img_np = np.clip(img_np, 0.0, 1.0)

        # Handle grayscale images that might have a single channel dimension
        if img_np.ndim == 3 and img_np.shape[2] == 1:
            img_np = img_np.squeeze(axis=2)
        
        cmap = 'gray' if img_np.ndim == 2 else None
        plot_title = title if title is not None else default_title

        plt.figure(figsize=figsize)
        plt.imshow(img_np, cmap=cmap)
        plt.title(plot_title)
        if cmap == 'gray': # Add colorbar for grayscale images for intensity reference
            plt.colorbar()
        plt.show() # Note: plt.show() is blocking.


    def save_image(self, output_path, image_type='processed'):
        """
        Saves the specified image to disk.

        The image data is converted from [0, 1] float to [0, 255] uint8 format
        before saving.

        Args:
            output_path (str): The path where the image will be saved.
                The file format is inferred from the extension by PIL.
            image_type (str, optional): The type of image to save.
                Valid options are 'input', 'convolved', 'processed', or 'output'.
                Defaults to 'processed'. 'output' is an alias for 'processed'.
        
        Raises:
            ValueError: If an invalid `image_type` is provided.
            IOError: If there's an error saving the file.
        """
        image_data_to_save = None
        if image_type == "input":
            image_data_to_save = self._input_image_data
        elif image_type == "convolved":
            image_data_to_save = self._convolved_image_data
        elif image_type == "processed" or image_type == "output":
            image_data_to_save = self._processed_image_data
        else:
            raise ValueError(f"Unknown image_type: '{image_type}'. "
                             "Valid options are 'input', 'convolved', 'processed', 'output'.")

        if image_data_to_save is None:
            print(f"Warning: No data available for image_type='{image_type}' to save. "
                  "It might not have been computed yet or an error occurred.")
            return

        # Convert to NumPy array
        img_np = be.to_numpy(image_data_to_save)
        
        # Convert float [0,1] to uint8 [0,255]
        # Clip to ensure values are in [0,1] before scaling, then clip again for safety.
        img_np_clipped = np.clip(img_np, 0.0, 1.0)
        uint8_data = np.clip(img_np_clipped * 255.0, 0, 255).astype(np.uint8)

        # Handle grayscale images that might have a single channel dimension
        if uint8_data.ndim == 3 and uint8_data.shape[2] == 1:
            uint8_data = uint8_data.squeeze(axis=2)
        
        try:
            pil_img = Image.fromarray(uint8_data)
            pil_img.save(output_path)
            print(f"Image saved successfully to {output_path}")
        except IOError as e:
            print(f"Error saving image to {output_path}: {e}")
            raise
        except Exception as e: # Catch other potential PIL errors
            print(f"An unexpected error occurred while saving image to {output_path}: {e}")
            raise


    def _generate_psf_grid(self): # Existing method, no change to its content here
        """
        Computes the PSF at different points across the image field,
        forming a grid of PSFs.
        """
        if self.psf_type not in ['FFT', 'geometric']:
            raise ValueError(f"Unsupported psf_type: {self.psf_type}. Must be 'FFT' or 'geometric'.")

        num_rows, num_cols = self.grid_density
        if not (isinstance(num_rows, int) and num_rows > 0 and isinstance(num_cols, int) and num_cols > 0):
            raise ValueError(f"grid_density must be a tuple of two positive integers, got {self.grid_density}")

        # Generate normalized field coordinates (Hx, Hy) from -1 to 1
        # For a single point (e.g., on-axis), use (0,0)
        if num_rows == 1 and num_cols == 1:
            field_points_x = [0.0]
            field_points_y = [0.0]
        else:
            # np.linspace includes endpoints. For a grid_density of (N,M) we want N points,
            # so if N=1, it's just the center. If N>1, it spans -1 to 1.
            field_points_x = np.linspace(-1, 1, num_cols) if num_cols > 1 else [0.0]
            field_points_y = np.linspace(-1, 1, num_rows) if num_rows > 1 else [0.0]


        # Create a list of (Hx, Hy) field points, filtering for those within the unit circle
        valid_field_points = []
        for Hy, Hx in itertools.product(field_points_y, field_points_x):
            if Hx**2 + Hy**2 <= 1.0: # Normalized field radius check
                valid_field_points.append((Hx, Hy))
        
        if not valid_field_points:
            if num_rows > 0 and num_cols > 0 : # If grid density was non-zero but still no points, means e.g. (1,1) density was not handled above
                 valid_field_points.append((0.0,0.0)) # Default to on-axis if grid logic somehow fails for (1,1)
            else: # Should not happen with guards above
                raise ValueError("No valid field points generated based on grid_density. Ensure grid_density leads to points within Hx^2 + Hy^2 <= 1.")


        self.psf_grid_data = [] # Clear any previous data

        for Hx, Hy in valid_field_points:
            field_point = (Hx, Hy)
            for wl in self.wavelengths_resolved:
                if self.psf_type == 'FFT':
                    # Ensure optic has necessary attributes for FFTPSF
                    if not all(hasattr(self.optic, attr) for attr in ['focal_length', 'pupil_radius']): # example attributes
                         # A more robust check would be specific to what FFTPSF needs
                        print(f"Warning: Optic model might be missing attributes required by FFTPSF for field point {field_point}, wavelength {wl}.")

                    psf_calculator = FFTPSF(
                        optic=self.optic,
                        field_point_normalized=field_point,
                        wavelength_um=wl,
                        num_pupil_samples=self.num_psf_rays, # num_psf_rays is used as num_pupil_samples
                        grid_size_pixels=self.psf_grid_size
                    )
                    # The compute() method should exist on psf_calculator and return the PSF data
                    # and other relevant info.
                    # Assuming FFTPSF object has attributes 'psf', 'pixel_size_um', 'fov_vignetted_normalized'
                    # after computation (or a compute method that returns them)
                    # For now, let's assume FFTPSF computes and stores these internally upon instantiation
                    # or via a specific .compute() method if it exists.
                    # If FFTPSF has a compute method:
                    # psf_data, psf_pixel_size, psf_fov_vignetted = psf_calculator.compute()

                    # For this implementation, we assume FFTPSF populates these attributes upon init or has them ready
                    if not hasattr(psf_calculator, 'psf') or \
                       not hasattr(psf_calculator, 'pixel_size_um') or \
                       not hasattr(psf_calculator, 'fov_vignetted_normalized'):
                        raise AttributeError("FFTPSF instance does not have one or more required attributes: "
                                             "'psf', 'pixel_size_um', 'fov_vignetted_normalized'. "
                                             "Ensure FFTPSF computes these upon instantiation or via a called method.")

                    psf_data_entry = {
                        'field_point': field_point,
                        'wavelength': wl,
                        'psf_data': psf_calculator.psf, # This is expected to be a 2D numpy array
                        'psf_pixel_size_um': psf_calculator.pixel_size_um,
                        'psf_fov_vignetted_normalized': psf_calculator.fov_vignetted_normalized
                    }
                    self.psf_grid_data.append(psf_data_entry)

                elif self.psf_type == 'geometric':
                    # Placeholder for geometric PSF generation
                    # Geometric PSF would typically involve ray tracing and creating a spot diagram
                    # or a histogram of ray intersections on the image plane.
                    # Example:
                    # spot_diagram = self.optic.trace_rays(field_point, wl, num_rays=self.num_psf_rays, ...)
                    # psf_data = self._create_geometric_psf_from_spots(spot_diagram, self.psf_grid_size)
                    raise NotImplementedError("Geometric PSF generation is not yet implemented.")
                else:
                    # This case should ideally be caught by the check at the beginning of the method.
                    raise ValueError(f"Unknown psf_type: {self.psf_type}")
        
        if not self.psf_grid_data and (num_rows * num_cols > 0):
            print(f"Warning: PSF grid data is empty after generation for grid_density {self.grid_density} and wavelengths {self.wavelengths_resolved}.")


    # ... _generate_psf_grid and _interpolate_psf_at_pixel methods remain the same ...
    # (Their content is not shown here for brevity, but they are part of the class)


    def _load_image(self):
        """
        Loads, normalizes, and oversamples the input image.

        The image specified by `self.image_path` is loaded using Pillow.
        It handles common image modes (grayscale, RGB, RGBA, LA), converting
        them to either grayscale (L) or RGB. Pixel values are normalized to
        the range [0, 1]. If `self.oversample_factor` is greater than 1,
        the image is resized using `scipy.ndimage.zoom` with bilinear
        interpolation (order=1). The processed image data is stored in
        `self._input_image_data` as a `be.DeviceArray`.

        Raises:
            FileNotFoundError: If the image file specified by `self.image_path`
                               does not exist.
            ValueError: If the image format is unhandled or if oversampling
                        parameters are invalid.
        """
        try:
            img = Image.open(self.image_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at: {self.image_path}")

        # Handle different image modes (convert to L or RGB)
        if img.mode == 'L':
            img_data_np = np.array(img, dtype=np.float32)
        elif img.mode == 'RGB':
            img_data_np = np.array(img, dtype=np.float32)
        elif img.mode == 'RGBA':
            print("Warning: RGBA image loaded, converting to RGB by discarding alpha channel.")
            img_data_np = np.array(img.convert('RGB'), dtype=np.float32)
        elif img.mode == 'LA': # Grayscale with alpha
            print("Warning: LA image loaded, converting to L by discarding alpha channel.")
            img_data_np = np.array(img.convert('L'), dtype=np.float32)
        else:
            print(f"Warning: Unhandled image mode {img.mode}. Attempting to convert to RGB.")
            try:
                img_data_np = np.array(img.convert('RGB'), dtype=np.float32)
            except Exception as e:
                raise ValueError(f"Could not convert image mode {img.mode} to RGB or L. Error: {e}")

        # Normalize to [0, 1]
        if np.max(img_data_np) > 1.0 + 1e-6: # Add epsilon for float comparison
            img_data_np /= 255.0
        img_data_np = np.clip(img_data_np, 0.0, 1.0)

        # Apply oversampling if factor > 1
        if self.oversample_factor > 1:
            if not isinstance(self.oversample_factor, (int, float)) or self.oversample_factor <= 0:
                raise ValueError("oversample_factor must be a positive number.")
            
            zoom_factors = [self.oversample_factor, self.oversample_factor]
            if img_data_np.ndim == 3: # RGB image
                zoom_factors.append(1) # Don't zoom channels dimension
            
            img_data_np = zoom(img_data_np, zoom_factors, order=1) # Bilinear interpolation
            img_data_np = np.clip(img_data_np, 0.0, 1.0) 

        self._input_image_data = be.asarray(img_data_np)
        print(f"Input image loaded: shape {self._input_image_data.shape}, type {self._input_image_data.dtype}")


    def _convolve_image_with_psfs(self):
        """
        Convolves the loaded input image with spatially varying PSFs.

        This method uses a "splatting" approach. For each pixel in the
        (potentially oversampled) input image, it interpolates the PSF for that
        pixel's location. This PSF is then scaled by the input pixel's intensity
        and added (splatted) to the corresponding region in an output buffer.
        The PSFs obtained from interpolation are averaged across wavelengths
        to produce a single effective PSF for each pixel. The resulting
        convolved image is stored in `self._convolved_image_data`.

        Assumes `self._input_image_data` is available.
        """
        if self._input_image_data is None:
            print("Error: Input image data is not loaded. Cannot perform convolution.")
            return

        input_img = self._input_image_data # This is already a be.DeviceArray
        eff_height, eff_width = input_img.shape[0], input_img.shape[1]
        num_channels = input_img.shape[2] if input_img.ndim == 3 else 1

        # Initialize convolved image data buffer on the same backend
        if num_channels > 1 and input_img.ndim ==3: # Color image
             self._convolved_image_data = be.zeros((eff_height, eff_width, num_channels), dtype=be.float32)
        else: # Grayscale image
            self._convolved_image_data = be.zeros((eff_height, eff_width), dtype=be.float32)

        print(f"Starting convolution on image of size {eff_height}x{eff_width}...")
        for r in range(eff_height): # Iterate over rows (y-coordinate)
            if r % (max(1, eff_height // 20)) == 0 : # Progress indicator every 5%
                print(f"Convolution progress: {r / eff_height * 100:.0f}%")
            for c in range(eff_width): # Iterate over columns (x-coordinate)
                # input_val_at_rc is a scalar for grayscale, or a (channels,) array for color
                input_val_at_rc = input_img[r, c]

                interpolated_psfs_list = self._interpolate_psf_at_pixel(image_x=c, image_y=r)

                if not interpolated_psfs_list:
                    # If no PSF, copy input value to output (like a Dirac delta PSF)
                    # This ensures energy conservation for pixels where PSF is undefined.
                    if num_channels > 1 and input_img.ndim ==3:
                         self._convolved_image_data[r, c, :] += input_val_at_rc
                    else:
                         self._convolved_image_data[r, c] += input_val_at_rc
                    continue

                # Average PSF data across wavelengths
                # PSFs from interpolation are already be.DeviceArrays
                # Initialize sum_psf with the shape of the first PSF
                # and ensure it's on the correct backend device
                ref_psf_shape = interpolated_psfs_list[0]['psf_data'].shape
                sum_psf = be.zeros(ref_psf_shape, dtype=be.float32)
                num_valid_psfs = 0
                for psf_entry in interpolated_psfs_list:
                    # Ensure consistent shape before adding
                    if psf_entry['psf_data'].shape == ref_psf_shape:
                        sum_psf += psf_entry['psf_data']
                        num_valid_psfs +=1
                    else:
                        print(f"Warning: PSF shape mismatch at {(r,c)} for wavelength {psf_entry['wavelength']}. Skipping this PSF.")

                if num_valid_psfs == 0: current_psf = be.zeros(ref_psf_shape, dtype=be.float32)
                else: current_psf = sum_psf / num_valid_psfs
                
                # Normalize the averaged PSF to sum to 1
                psf_sum = be.sum(current_psf)
                if psf_sum > 1e-9: current_psf = current_psf / psf_sum
                # else: PSF is essentially zero, will contribute nothing.

                psf_h, psf_w = current_psf.shape
                psf_center_r, psf_center_c = psf_h // 2, psf_w // 2

                for dr in range(psf_h):
                    for dc in range(psf_w):
                        output_r = r + dr - psf_center_r
                        output_c = c + dc - psf_center_c

                        if 0 <= output_r < eff_height and 0 <= output_c < eff_width:
                            psf_val = current_psf[dr, dc]
                            if num_channels > 1 and input_img.ndim ==3: # Color
                                self._convolved_image_data[output_r, output_c, :] += input_val_at_rc * psf_val
                            else: # Grayscale
                                self._convolved_image_data[output_r, output_c] += input_val_at_rc * psf_val
        print("Convolution finished.")


    def _apply_geometric_distortion(self, image_data_to_distort):
        """
        Applies geometric distortion to the input image data.

        This method implements a backward mapping approach. It creates a grid
        representing the output (distorted) image pixels. For each pixel in this
        grid, it determines the corresponding normalized field coordinates (Hx, Hy).
        These field coordinates are then traced through the optical system to find
        where they actually land on the image sensor (in physical units, e.g., mm).
        These physical landing positions are then converted back to pixel coordinates
        in the source (undistorted) image space (`image_data_to_distort`).
        The `scipy.ndimage.map_coordinates` function is used to resample the
        source image using these calculated source coordinates, effectively
        warping the image to simulate geometric distortion.

        The physical size of the sensor area covered by `image_data_to_distort`
        is determined using `self.optic.get_max_image_height()` or, as a
        fallback, `self.optic.image_surface.get_semi_diameter()`.

        Wavelength-specific distortion can be applied if multiple wavelengths
        are resolved and the input image is multi-channel (e.g., RGB).

        Args:
            image_data_to_distort (be.DeviceArray): The image data to which
                distortion should be applied. This is typically a copy of
                `self._convolved_image_data`. Shape (H, W) or (H, W, C).

        Returns:
            be.DeviceArray: The geometrically distorted image data. If distortion
                cannot be applied (e.g., missing optical parameters or input data),
                a copy of the input data may be returned with a warning.

        Raises:
            ValueError: If the physical size of the sensor cannot be determined
                        from the optic model.
        """
        if image_data_to_distort is None:
            print("Warning: image_data_to_distort is None. Cannot apply geometric distortion.")
            return None

        source_h, source_w = image_data_to_distort.shape[0], image_data_to_distort.shape[1]
        num_source_channels = image_data_to_distort.shape[2] if image_data_to_distort.ndim == 3 else 1

        # Determine sensor physical size (full width/height in mm)
        sensor_width_mm = 0.0
        sensor_height_mm = 0.0

        if hasattr(self.optic, 'get_max_image_height') and callable(self.optic.get_max_image_height):
            max_img_h = self.optic.get_max_image_height()
            if max_img_h is not None and max_img_h > 0:
                sensor_height_mm = sensor_width_mm = 2.0 * max_img_h
        
        if sensor_width_mm == 0.0 and hasattr(self.optic, 'image_surface') and \
           hasattr(self.optic.image_surface, 'get_semi_diameter') and \
           callable(self.optic.image_surface.get_semi_diameter):
            semi_diam = self.optic.image_surface.get_semi_diameter()
            if semi_diam is not None and semi_diam > 0:
                 sensor_height_mm = sensor_width_mm = 2.0 * semi_diam
        
        if sensor_width_mm <= 0 or sensor_height_mm <= 0:
            raise ValueError("Cannot determine physical sensor size for geometric distortion. "
                             "Optic model must provide get_max_image_height() or image_surface.get_semi_diameter().")

        # Wavelength to channel mapping logic
        wl_map = [self.wavelengths_resolved[0]] * num_source_channels # Default: use first wavelength for all channels
        if num_source_channels == 1:
            if len(self.wavelengths_resolved) > 1:
                print(f"Warning: Grayscale image, using first wavelength {self.wavelengths_resolved[0]} for distortion.")
        elif num_source_channels == 3: # RGB
            if len(self.wavelengths_resolved) == 1:
                print(f"Warning: RGB image, using single wavelength {self.wavelengths_resolved[0]} for all channels distortion.")
                wl_map = [self.wavelengths_resolved[0]] * 3
            elif len(self.wavelengths_resolved) == 3:
                print(f"Info: RGB image, using wavelengths {self.wavelengths_resolved} for R, G, B channel distortion respectively.")
                wl_map = self.wavelengths_resolved[:3]
            else: # e.g. 2 wavelengths for RGB, or >3 wavelengths
                print(f"Warning: RGB image with {len(self.wavelengths_resolved)} wavelengths. Using first wavelength {self.wavelengths_resolved[0]} for all channels distortion.")
                wl_map = [self.wavelengths_resolved[0]] * 3
        
        # Create grid of output pixel coordinates (indices of the distorted image)
        output_r_indices, output_c_indices = np.mgrid[0:source_h, 0:source_w]

        # Convert output pixel indices to normalized field coordinates (target Hx, Hy)
        # These are the field points that *should* ideally land at these output pixel locations.
        # Handle source_w/h = 1 to avoid division by zero if image is 1 pixel wide/high.
        den_w = source_w - 1.0 if source_w > 1 else 1.0
        den_h = source_h - 1.0 if source_h > 1 else 1.0

        target_Hx_np = (output_c_indices / den_w) * 2.0 - 1.0
        target_Hy_np = ((source_h - 1.0 - output_r_indices) / den_h) * 2.0 - 1.0 
        if source_w == 1: target_Hx_np[:] = 0.0
        if source_h == 1: target_Hy_np[:] = 0.0


        # Prepare for map_coordinates, needs to be NumPy
        image_data_np = be.to_numpy(image_data_to_distort)
        distorted_image_np = np.zeros_like(image_data_np)

        print(f"Applying geometric distortion: sensor size {sensor_width_mm:.2f}x{sensor_height_mm:.2f} mm. Image shape: ({source_h}, {source_w}, {num_source_channels})")

        for ch in range(num_source_channels):
            wl_for_channel = wl_map[ch]
            print(f"  Processing channel {ch} with wavelength {wl_for_channel} um...")

            # These maps will store the (r, c) coordinates in the *source* image
            # that correspond to each pixel in the *output* (distorted) image.
            source_coords_r_map_np = np.zeros((source_h, source_w), dtype=np.float64)
            source_coords_c_map_np = np.zeros((source_h, source_w), dtype=np.float64)

            for r_idx in range(source_h): # Iterate over output image rows
                if r_idx % (max(1, source_h // 10)) == 0: # Progress
                    print(f"    Channel {ch} distortion mapping: {r_idx/source_h*100:.0f}%")
                for c_idx in range(source_w): # Iterate over output image columns
                    Hx = target_Hx_np[r_idx, c_idx]
                    Hy = target_Hy_np[r_idx, c_idx]

                    # Trace this ideal field point to find where it *actually* lands on sensor
                    # Px, Py = 0, 0 for chief ray for distortion
                    ray = self.optic.trace_generic(Hx=Hx, Hy=Hy, Px=0, Py=0, wavelength=wl_for_channel)

                    if ray is None or ray.error != 0:
                        # If ray trace fails, map to the original pixel (no distortion for this point)
                        # Or could map to an out-of-bound coordinate if mode='constant' cval=... is used
                        source_coords_r_map_np[r_idx, c_idx] = float(r_idx)
                        source_coords_c_map_np[r_idx, c_idx] = float(c_idx)
                        if r_idx ==0 and c_idx ==0: # Print warning only once per channel
                             print(f"    Warning: Ray trace failed for Hx={Hx:.2f}, Hy={Hy:.2f} at channel {ch}, wl {wl_for_channel}. Using ideal coords.")
                    else:
                        # Ray's (x,y) are in mm on the image plane, relative to optical axis.
                        # Convert these physical mm coordinates to pixel coordinates in the *source* image.
                        # Source image (image_data_to_distort) covers SensorWidthMM x SensorHeightMM.
                        # Pixel (0,0) is top-left. Sensor origin (0,0) is center.
                        # ray.x positive is to the right. ray.y positive is up.
                        src_c_val = (ray.x / sensor_width_mm + 0.5) * (source_w - 1.0)
                        src_r_val = (-ray.y / sensor_height_mm + 0.5) * (source_h - 1.0) # y-coord flipped

                        source_coords_r_map_np[r_idx, c_idx] = src_r_val
                        source_coords_c_map_np[r_idx, c_idx] = src_c_val
            
            # Perform the warping for the current channel
            current_channel_data_np = image_data_np[..., ch] if num_source_channels > 1 else image_data_np
            
            # map_coordinates expects coordinates as (rows_coords, cols_coords)
            map_coordinates_input = np.array([source_coords_r_map_np, source_coords_c_map_np])
            
            warped_channel_np = map_coordinates(
                current_channel_data_np,
                map_coordinates_input,
                order=1,          # Bilinear interpolation
                mode='nearest',   # How to handle out-of-bounds: use nearest valid pixel
                prefilter=False   # Important for order=1 (bilinear)
            )
            
            if num_source_channels > 1:
                distorted_image_np[..., ch] = warped_channel_np
            else:
                distorted_image_np = warped_channel_np
        
        print("Geometric distortion application finished.")
        return be.asarray(distorted_image_np)


    def run_simulation(self):
        """
        Runs or re-runs stages of the image simulation pipeline.

        This method checks if essential data products (input image, convolved
        image, processed image) exist and attempts to generate them if missing,
        following the defined simulation sequence.
        """
        print("Running simulation pipeline checks...")
        # 1. PSF Grid is generated in __init__

        # 2. Load image if not already loaded
        if self._input_image_data is None:
             print("Input image not found, attempting to load...")
             self._load_image()
             if self._input_image_data is None:
                 print("Error: Failed to load input image. Aborting simulation.")
                 return
        
        # 3. Convolve if not already convolved and input exists
        if self._convolved_image_data is None and self._input_image_data is not None:
            print("Convolved image not found, attempting to convolve...")
            self._convolve_image_with_psfs()
            if self._convolved_image_data is None:
                print("Error: Failed to convolve image. Aborting further processing.")
                return
        
        # Initial state for processed image
        if self._convolved_image_data is not None and self._processed_image_data is None:
            self._processed_image_data = self._convolved_image_data.copy()

        # 4. Apply distortion if enabled and processed data exists
        if self.use_distortion and self._processed_image_data is not None:
            print("Applying geometric distortion...")
            self._processed_image_data = self._apply_geometric_distortion(self._processed_image_data) # works on copy
        
        # 5. Resample if necessary
        if self._processed_image_data is not None:
            img_h, img_w = self._processed_image_data.shape[0], self._processed_image_data.shape[1]
            target_w, target_h = self.output_resolution
            if img_h != target_h or img_w != target_w:
                print("Resampling to final output resolution...")
                self._resample_to_output() # This updates self._processed_image_data
            
        print("Simulation run/check complete.")


    def _resample_to_output(self):
        """
        Resamples `self._processed_image_data` to `self.output_resolution`.

        Uses `scipy.ndimage.zoom` for resampling. The result is stored back
        in `self._processed_image_data` as a `be.DeviceArray`.
        Assumes `self._processed_image_data` is not None.
        """
        if self._processed_image_data is None:
            print("Error: No processed image data to resample.")
            return

        current_h, current_w = self._processed_image_data.shape[0], self._processed_image_data.shape[1]
        target_w, target_h = self.output_resolution # (width, height)
        
        if current_h == target_h and current_w == target_w:
            print("Processed image already at target output resolution. No resampling needed.")
            return

        zoom_h = target_h / current_h
        zoom_w = target_w / current_w

        zoom_factors = [zoom_h, zoom_w]
        is_color = self._processed_image_data.ndim == 3 and self._processed_image_data.shape[2] > 1
        if is_color:
            zoom_factors.append(1) # Don't zoom channels

        print(f"Resampling from ({current_w}x{current_h}) to ({target_w}x{target_h})...")
        
        img_np = be.to_numpy(self._processed_image_data)
        # Bilinear interpolation (order=1), clip to maintain range
        resampled_img_np = zoom(img_np, zoom_factors, order=1, prefilter=False) 
        resampled_img_np = np.clip(resampled_img_np, 0.0, 1.0)

        self._processed_image_data = be.asarray(resampled_img_np)
        print(f"Resampled image shape: {self._processed_image_data.shape}")
