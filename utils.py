import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback
from torchvision.transforms import functional as F


class FixedResize(dt.transforms.Transform, dt.transforms.v2.Resize):
    """Fixed Resize class that works with newer torchvision versions."""
    
    def __init__(self, size, interpolation=2, max_size=None, antialias=True, source="image", target="image"):
        dt.transforms.v2.Resize.__init__(self, size, interpolation, max_size, antialias)
        self.source = source
        self.target = target
    
    def __call__(self, x):
        img = self.nested_get(x, self.source)
        if isinstance(img, list):
            resized = [F.resize(i, self.size, self.interpolation, self.max_size, self.antialias) for i in img]
        else:
            resized = F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
        self.nested_set(x, resized, self.target)
        return x


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = FixedResize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")