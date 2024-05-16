import rasterio
import numpy as np
from pathlib import Path
from rasterio.enums import Resampling
from rasterio.windows import Window

class Resample:
    """
    This class upscales images and crops or pads them to a target size.
    """
    def __init__(self, image_path: str, upscale_factor: int, target_size = 512):
        self.image_path = image_path
        self.upscale_factor = upscale_factor
        self.base_image_size = target_size
        if self.upscale_factor not in [2, 4, 8, 16, 32]:
            raise ValueError("The resolution must be a power of 2, e.g. 2, 4, 8, 16, 32")
        self.upscaled_image_size = self.base_image_size * self.upscale_factor
        self.resampled_image_path = self.create_folders()
        
    def create_folders(self):
        """
        Creates the necessary folders for the data and sets the resampled image path.
        """
        self.base_path = Path(self.image_path).parent.parent
        self.upscale_path = self.base_path / "satellite_upscaled"
        self.upscale_path.mkdir(exist_ok=True)
        # Resampled image name = original image name 
        resampled_image_name = f"{Path(self.image_path).stem}.tif"
        resampled_image_path = self.upscale_path / resampled_image_name
        return resampled_image_path

    def resample_image(self):
        """
        Resamples the image to the new resolution using bicubic interpolation.
        """
        with rasterio.open(self.image_path) as src:
            # Define new dimensions based on scale factor
            scale = self.upscale_factor
            new_height, new_width = int(src.height * scale), int(src.width * scale)

            # Resample data to target shape:
            data = src.read(
                out_shape=(src.count, new_height, new_width),
                resampling=Resampling.cubic
            )

            # Scale image transform:
            new_transform = src.transform * src.transform.scale(
                src.width / new_width,
                src.height / new_height
            )

            # Prepare the new metadata
            new_meta = src.meta.copy()
            new_meta.update({
                'driver': 'GTiff',
                'height': new_height,
                'width': new_width,
                'transform': new_transform
            })

            # Write the upscaled image to disk:
            with rasterio.open(self.resampled_image_path, 'w', **new_meta) as dst:
                dst.write(data)

    def crop_or_pad_image(self):
        """
        Crops or pads the image to the target size.
        """
        with rasterio.open(self.resampled_image_path) as src:
            data = src.read()
            height, width = data.shape[1], data.shape[2]

            if height > self.upscaled_image_size or width > self.upscaled_image_size:
                # Cropping if image is too large:
                start_y = max(0, (height - self.upscaled_image_size) // 2)
                start_x = max(0, (width - self.upscaled_image_size) // 2)
                data_cropped = data[:, start_y:start_y + self.upscaled_image_size,
                                    start_x:start_x + self.upscaled_image_size]
                # Adjust the spatial reference of the image to the new window
                transform = rasterio.windows.transform(Window(start_x, start_y,
                                                              self.upscaled_image_size, self.upscaled_image_size),
                                                       src.transform)
            elif height < self.upscaled_image_size or width < self.upscaled_image_size:
                # Padding if the image is too small:
                pad_height = (self.upscaled_image_size - height) // 2
                pad_width = (self.upscaled_image_size - width) // 2
                data_cropped = np.pad(data, pad_width=((0, 0),
                                                       (pad_height, self.upscaled_image_size - height - pad_height),
                                                       (pad_width, self.upscaled_image_size - width - pad_width)),
                                      mode='constant', constant_values=0)
                transform = src.transform * src.transform.translation(-pad_width, -pad_height)
            else:
                return

            # Update metadata
            new_meta = src.meta.copy()
            new_meta.update({
                'driver': 'GTiff',
                'height': self.upscaled_image_size,
                'width': self.upscaled_image_size,
                'transform': transform
            })

            # Save the result to a new file
            with rasterio.open(self.resampled_image_path, 'w', **new_meta) as dst:
                dst.write(data_cropped)

# Example usage:
# resampler = Resample('path_to_image.tif', 4)
# resampler.resample_image()
# resampler.crop_or_pad_image()
