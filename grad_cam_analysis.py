"""Show network train graphs and analyze training results."""
import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from common import FIGURES_DIR
from utils import load_dataset, load_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Arguments
def parse_args():
    """Parse script arguments.

    Returns:
        Namespace with model name, checkpoint path and dataset name.
    """
    parser = argparse.ArgumentParser(description='Analyze network performance.')
    parser.add_argument('--model', '-m',
                        default='XceptionBased', type=str,
                        help='Model name: SimpleNet or XceptionBased.')
    parser.add_argument('--checkpoint_path', '-cpp',
                        default='checkpoints/XceptionBased.pt', type=str,
                        help='Path to model checkpoint.')
    parser.add_argument('--dataset', '-d',
                        default='fakes_dataset', type=str,
                        help='Dataset: fakes_dataset or synthetic_dataset.')

    return parser.parse_args()


def get_grad_cam_visualization(test_dataset, model) -> tuple[np.ndarray, torch.Tensor]:
    """
    Returns a tuple of (Grad-CAM overlay, true label).

    Grad-CAM overlay is a 256x256x3 NumPy array (uint8),
    and the label is a torch.Tensor of shape (1,).
    """

    # 1. Sample a single image/label using DataLoader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    images, labels = next(iter(test_loader))  # images: (1,3,256,256), labels: (1,)

    # 2. Hooks to capture the forward activation and backward gradient of `model.conv3`
    activation = []
    gradient = []

    def forward_hook(module, inp, out):
        activation.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradient.append(grad_out[0])

    # Register hooks on the conv3 layer
    fwd_handle = model.conv3.register_forward_hook(forward_hook)
    bwd_handle = model.conv3.register_full_backward_hook(backward_hook)

    # Move inputs/model to same device
    device = next(model.parameters()).device
    images, labels = images.to(device), labels.to(device)

    model.eval()
    with torch.no_grad():
        _ = model(images)  # forward pass to populate activation (hook)

    # 3. Enable gradient on images; re-run forward & do backward to get conv3 gradients
    images.requires_grad_()
    activation.clear()
    gradient.clear()

    outputs = model(images)                # forward with grad
    target_score = outputs[0, labels[0]]   # pick the score for the ground-truth label
    target_score.backward()                # compute gradients w.r.t. conv3 features

    # 4. Construct the Grad-CAM
    conv_features = activation[-1]  # (1, C, H_feat, W_feat)
    conv_grads = gradient[-1]       # (1, C, H_feat, W_feat)

    # Channel-wise mean of gradients => weights
    weights = conv_grads.mean(dim=(2, 3), keepdim=True)  # shape: (1, C, 1, 1)

    # Weighted sum across channels
    grad_cam = torch.sum(weights * conv_features, dim=1, keepdim=True)
    grad_cam = torch.relu(grad_cam)  # keep only positive contributions

    # Upsample to 256x256
    grad_cam = torch.nn.functional.interpolate(
        grad_cam,
        size=(256, 256),
        mode='bilinear',
        align_corners=False
    )

    # Normalize heatmap to [0,1]
    grad_cam = grad_cam - grad_cam.min()
    grad_cam = grad_cam / (grad_cam.max() + 1e-8)
    heatmap = grad_cam.squeeze().detach().cpu().numpy()  # (256,256)

    # 5. Create an RGB heatmap using matplotlib
    cmap = plt.get_cmap('jet')
    heatmap_rgba = cmap(heatmap)          # shape: (256,256,4) in [0,1]
    heatmap_rgb = (heatmap_rgba[..., :3] * 255).astype(np.uint8)  # drop alpha, scale to [0,255]

    # Retrieve original image & normalize for visualization
    original_img = images[0].detach().cpu().numpy()  # (3,256,256)
    original_img = np.transpose(original_img, (1, 2, 0))  # -> (256,256,3)
    original_img = original_img - original_img.min()
    original_img = original_img / (original_img.max() + 1e-8)
    original_img = (original_img * 255).astype(np.uint8)

    # 6. Overlay the heatmap (simple alpha-blend)
    alpha = 0.5
    overlay = (alpha * original_img + (1 - alpha) * heatmap_rgb).astype(np.uint8)

    # 7. Cleanup: remove hooks
    fwd_handle.remove()
    bwd_handle.remove()

    return overlay, labels


def main():
    """Create two GradCAM images, one of a real image and one for a fake
    image for the model and dataset it receives as script arguments."""
    args = parse_args()
    test_dataset = load_dataset(dataset_name=args.dataset, dataset_part='test')

    model_name = args.model
    model = load_model(model_name)
    model.load_state_dict(torch.load(args.checkpoint_path)['model'])

    model.eval()
    seen_labels = []
    while len(set(seen_labels)) != 2:
        visualization, true_label = get_grad_cam_visualization(test_dataset,
                                                               model)
        grad_cam_figure = plt.figure()
        plt.imshow(visualization)
        title = 'Fake Image' if true_label == 1 else 'Real Image'
        plt.title(title)
        seen_labels.append(true_label.item())
        grad_cam_figure.savefig(
            os.path.join(FIGURES_DIR,
                         f'{args.dataset}_{args.model}_'
                         f'{title.replace(" ", "_")}_grad_cam.png'))


if __name__ == "__main__":
    main()
