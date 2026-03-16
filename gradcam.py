import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
import torchvision.models
import torch
import torch.nn
import torch.nn.functional as F
import numpy as np

from lerobot.utils.constants import OBS_IMAGES

class GradCAMVisualizer:
    def __init__(self, target_layer: torch.nn.Module):
        self._gradients = None
        self._activations = None
        
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def generate_heatmap(self):
        weights = torch.mean(self._gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self._activations, dim=1).squeeze()
        cam = torch.relu(cam)
        return cam.detach()
    
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)

dataset = LeRobotDataset("", "/home/nakamura/data/tomato_merged_20260225/user/id")
policy = DiffusionPolicy.from_pretrained("/home/nakamura/outputs/train/diffusion_20260225/checkpoints/200000/pretrained_model")
policy.cpu()
policy.train()
dataloader = torch.utils.data.DataLoader(dataset)
batch = next(iter(dataloader))
visualizers = []
for encoder in policy.diffusion.rgb_encoder:    
    target_layer = encoder.backbone[-1]
    visualizer = GradCAMVisualizer(target_layer)
    visualizers.append(visualizer)

batch[OBS_IMAGES] = torch.stack([batch[key] for key in policy.config.image_features], dim=-4)
policy._queues = populate_queues(policy._queues, batch)
actions = policy.predict_action_chunk(batch)
target = torch.sum(torch.abs(actions[0,0]))

policy.zero_grad()
target.backward()

images = [image[0] for key, image in batch.items() if key.startswith("observation.images.") and "eye" not in key]
heatmaps = [visualizer.generate_heatmap()[0] for visualizer in visualizers]

colored_images = []
resized_heatmaps = []

for image, heatmap in zip(images, heatmaps):
    image = image.cpu().numpy().transpose(1, 2, 0)
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    heatmap = F.interpolate(heatmap, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False).squeeze().numpy()
    colored_images.append(image)
    resized_heatmaps.append(heatmap)


image = np.concatenate(colored_images,axis=1)
image = np.mean(image, axis=2)
heatmap = np.concatenate(resized_heatmaps,axis=1)

plt.imshow(image, cmap='gray')
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.axis('off')
plt.show()