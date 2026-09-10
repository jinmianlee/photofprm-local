"""Explicit local image region for independent image-to-shape generation."""
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PhotoRegion(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)

    @model_validator(mode='after')
    def ordered(self):
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError('Region edges must be ordered.')
        return self

    def crop(self, image):
        bounds = (round(self.x0*image.width), round(self.y0*image.height),
                  round(self.x1*image.width), round(self.y1*image.height))
        if min(bounds[2]-bounds[0], bounds[3]-bounds[1]) < 256:
            raise ValueError('框选区域的短边至少需要 256 个原始像素，请扩大选区或上传更清晰的照片。')
        return image.crop(bounds), {
            'mode': 'independent_region', 'normalized_bounds': self.model_dump(),
            'pixel_bounds_after_exif': list(bounds), 'original_size_after_exif': list(image.size),
            'automatically_assembled': False,
        }
