"""Create Saliency Maps."""
import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader

from common import FIGURES_DIR
from utils import load_dataset, load_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Arguments
def parse_args():
    """Parse script arguments.

    Returns:
        Namespace with model name, checkpoints path and dataset name.
    """
    parser = argparse.ArgumentParser(description='Plot saliency maps.')
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


def compute_gradient_saliency_maps(samples: torch.Tensor,
                                   true_labels: torch.Tensor,
                                   model: nn.Module) -> torch.Tensor:
    """
    Compute vanilla gradient saliency maps for the given samples.
    
    Args:
        samples: The samples we want to compute saliency maps for. 
                 Shape: (B, 3, 256, 256).
        true_labels: The true labels (class indices) of the samples.
                     Shape: (B,).
        model: The model to use for forward and backward passes.
    
    Returns:
        saliency: Vanilla gradient saliency maps. Shape: (B, 256, 256).
    """
    # (1) Enable gradient tracking on samples.
    samples = samples.clone().detach().requires_grad_(True)

    # (2) Forward pass through the model.
    outputs = model(samples)  # outputs.shape: (B, num_classes, ...)

    # (3) Gather only the scores corresponding to the true labels.
    #     Assuming `outputs` is of shape (B, num_classes).
    #     For multi-class classification with class dimension=1, do:
    correct_scores = outputs[torch.arange(outputs.size(0)), true_labels]

    # (4) Backward pass on these scores.
    #     We create a gradient of ones for each correct score.
    grad_mask = torch.ones_like(correct_scores)
    correct_scores.backward(grad_mask)

    # (5) Collect the gradients from the samples.
    #     `samples.grad` holds the gradient of the scores wrt. each input pixel.
    gradients = samples.grad  # shape: (B, 3, 256, 256)

    # (6) Compute the absolute value (L1) of these gradients.
    gradients_abs = gradients.abs()  # shape: (B, 3, 256, 256)

    # (7) Pick the maximum value across the channel dimension for each pixel.
    saliency, _ = torch.max(gradients_abs, dim=1)  # shape: (B, 256, 256)

    return saliency


def main():  # pylint: disable=R0914, R0915
    """Parse script arguments, show saliency maps for 36 random samples,
    and the average saliency maps over all real and fake samples in the test
    set."""
    args = parse_args()

    # load dataset
    test_dataset = load_dataset(dataset_name=args.dataset, dataset_part='test')

    # load model
    model_name = args.model
    model = load_model(model_name)
    model.load_state_dict(torch.load(args.checkpoint_path)['model'])
    model.eval()

    # create sets of samples of images and their corresponding saliency maps
    all_samples = []
    all_saliency_maps = []
    sample_to_image = lambda x: np.transpose(x, (1, 2, 0))

    for _ in range(6):
        samples, true_labels = next(iter(DataLoader(test_dataset,
                                                    batch_size=6,
                                                    shuffle=True)))
        all_samples.append(torch.cat([sample_to_image(s).unsqueeze(0)
                                      for s in samples]))
        saliency_maps = compute_gradient_saliency_maps(samples.to(device),
                                     true_labels.to(device),
                                     model)
        all_saliency_maps.append(saliency_maps.cpu().detach())

    all_samples = torch.cat(all_samples)
    all_saliency_maps = torch.cat(all_saliency_maps)

    saliency_maps_and_images_pairs = plt.figure()
    plt.suptitle('Images and their saliency maps')
    for idx, (image, saliency_map) in enumerate(zip(all_samples,
                                                    all_saliency_maps)):
        plt.subplot(6, 6 * 2, 2 * idx + 1)
        # plot image
        image -= image.min()
        image /= image.max()
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        # plot saliency map
        plt.subplot(6, 6 * 2, 2 * idx + 2)
        saliency_map -= saliency_map.min()
        saliency_map /= saliency_map.max()
        plt.imshow(saliency_map)
        plt.xticks([])
        plt.yticks([])

    saliency_maps_and_images_pairs.set_size_inches((8, 8))
    saliency_maps_and_images_pairs.savefig(
        os.path.join(FIGURES_DIR,
                     f'{args.dataset}_{args.model}_'
                     f'saliency_maps_and_images_pairs.png'))

    # loop through the images in the test set and compute saliency map for
    # each image. Compute the average map of all real face image and
    # all fake face image images.
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    real_images_saliency_maps = []
    fake_images_saliency_maps = []

    for samples, true_labels in dataloader:
        fake_samples = samples[true_labels == 1].to(device)
        fake_labels = true_labels[true_labels == 1].to(device)
        real_samples = samples[true_labels == 0].to(device)
        real_labels = true_labels[true_labels == 0].to(device)
        saliency_maps = compute_gradient_saliency_maps(fake_samples,
                                                       fake_labels,
                                                       model)
        fake_images_saliency_maps.append(saliency_maps.cpu().detach())
        saliency_maps = compute_gradient_saliency_maps(real_samples,
                                                       real_labels,
                                                       model)
        real_images_saliency_maps.append(saliency_maps.cpu().detach())

    all_real_saliency_maps = torch.cat(real_images_saliency_maps)
    all_fake_saliency_maps = torch.cat(fake_images_saliency_maps)

    for idx in range(all_real_saliency_maps.shape[0]):
        all_real_saliency_maps[idx] -= all_real_saliency_maps[idx].min()
        all_real_saliency_maps[idx] /= all_real_saliency_maps[idx].max()

    for idx in range(all_fake_saliency_maps.shape[0]):
        all_fake_saliency_maps[idx] -= all_fake_saliency_maps[idx].min()
        all_fake_saliency_maps[idx] /= all_fake_saliency_maps[idx].max()

    mean_saliency_maps = plt.figure()
    plt.subplot(1, 2, 1)
    mean_map = all_fake_saliency_maps.mean(axis=0)
    mean_map -= mean_map.min()
    mean_map /= mean_map.max()
    plt.imshow(mean_map)
    plt.title('mean of fake images saliency maps')
    plt.subplot(1, 2, 2)
    mean_map = all_real_saliency_maps.mean(axis=0)
    mean_map -= mean_map.min()
    mean_map /= mean_map.max()
    plt.imshow(mean_map)
    plt.title('mean of real images saliency maps')
    mean_saliency_maps.set_size_inches((8, 6))
    mean_saliency_maps.savefig(
        os.path.join(FIGURES_DIR,
                     f'{args.dataset}_{args.model}_mean_saliency_maps.png'))


if __name__ == '__main__':
    main()
