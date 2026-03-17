import matplotlib.pyplot as plt
import matplotlib.animation
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
import torchvision.models
import torch
import torch.nn
import torch.nn.functional as F
import numpy as np

from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
import tqdm

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

episode = 0
dataset = LeRobotDataset("", "/home/nakamura/data/eval_tomato_diffusion_A0_20260226", episodes=[episode])
policy = DiffusionPolicy.from_pretrained("/home/nakamura/outputs/train/diffusion_20260225/checkpoints/200000/pretrained_model")
policy.cuda()
policy.train()
dataloader = torch.utils.data.DataLoader(dataset)

visualizers = []
for encoder in policy.diffusion.rgb_encoder:    
    target_layer = encoder.backbone[-1]
    visualizer = GradCAMVisualizer(target_layer)
    visualizers.append(visualizer)

concat_image = None
concat_heatmap = None
obs_embed_cam = []

fig = plt.figure(figsize=(640*3/100, 480*2/100), dpi=100, frameon=False)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

for i, batch in enumerate(tqdm.tqdm(dataloader)):
    if i % 16 != 0:
        continue

    batch = {key: value.cuda() for key, value in batch.items() if isinstance(value, torch.Tensor)}
    batch[OBS_IMAGES] = torch.stack([batch[key] for key in policy.config.image_features], dim=-4)

    policy._queues = populate_queues(policy._queues, batch)
    actions = policy.predict_action_chunk(batch)
    target = torch.sum(torch.abs(actions[0,0]))

    policy.diffusion.global_cond.retain_grad()
    policy.zero_grad()
    target.backward()

    weights = policy.diffusion.global_cond.grad
    tmp = torch.relu(weights * policy.diffusion.global_cond).squeeze().cpu().detach().numpy()
    gradcam_camera_state = np.array([np.mean(tmp[0:19]), np.mean(tmp[19:83]), np.mean(tmp[83:147]), np.mean(tmp[147:211]), np.mean(tmp[211:275]), np.mean(tmp[275:339]), np.mean(tmp[339:403])])
    obs_embed_cam.append(gradcam_camera_state)

    heatmaps = [visualizer.generate_heatmap()[0].cpu() for visualizer in visualizers]
    images = [batch[key][0].cpu() for key in policy.config.image_features]

    colored_images = []
    resized_heatmaps = []

    for image, heatmap in zip(images, heatmaps):
        image = image.cpu().numpy().transpose(1, 2, 0)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        heatmap = F.interpolate(heatmap, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False).squeeze().numpy()
        colored_images.append(image)
        resized_heatmaps.append(heatmap)

    image = np.concatenate([
        colored_images[0],
        colored_images[3],
        colored_images[1],
        colored_images[4],
        colored_images[2],
        colored_images[5]], axis=1)
    image = np.mean(image, axis=2)
    heatmap = np.concatenate([
        resized_heatmaps[0],
        resized_heatmaps[3],
        resized_heatmaps[1],
        resized_heatmaps[4],
        resized_heatmaps[2],
        resized_heatmaps[5]], axis=1)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    if concat_image is None:
        concat_image = image.copy()
        concat_heatmap = heatmap.copy()
    else:
        concat_image = np.concatenate([concat_image, image], axis=0)
        concat_heatmap = np.concatenate([concat_heatmap, heatmap], axis=0)

plt.imshow(concat_image[0:len(concat_image)//480//2*480], cmap='gray')
plt.imshow(concat_heatmap[0:len(concat_heatmap)//480//2*480], cmap='jet', alpha=0.5)
plt.savefig(f"gradcam_{episode}_first.png")

plt.imshow(concat_image[len(concat_image)//480//2*480:], cmap='gray')
plt.imshow(concat_heatmap[len(concat_heatmap)//480//2*480:], cmap='jet', alpha=0.5)
plt.savefig(f"gradcam_{episode}_second.png")

plt.clf()
obs_embed_cam = np.stack(obs_embed_cam)
keys = list(policy.config.image_features.keys())

fig = plt.figure()
plt.plot(obs_embed_cam[:, 0])
plt.savefig(f"obs_embed_cam_{episode}.png")