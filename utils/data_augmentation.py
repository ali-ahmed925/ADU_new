import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import GaussianBlur

# def get_jigsaw_tensor(im_batch, resize, grid, device, is_gaussian_blur=True):
def get_jigsaw_tensor(im_batch, grid, device, is_gaussian_blur=True):
# Ensure the input tensor is on the CPU
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    im_batch = im_batch.to(device)

    # Container for all the jigsawed tensors
    jigsawed_tensors = []

    # Iterate over each tensor in the batch
    for b in range(im_batch.size(0)):
        im_tensor = im_batch[b]

        # Resize the image tensor
        # im_tensor = TF.resize(im_tensor, resize)

        # Calculate the size of each tile
        s = int(im_tensor.shape[2] / grid)

        tiles = []
        for n in range(grid**2):
            y1, x1 = s * int(n / grid), s * (n % grid)
            y2, x2 = y1 + s, x1 + s
            tile = im_tensor[:, y1:y2, x1:x2]
            tiles.append(tile)

        # Shuffle tiles
        tiles = torch.stack(tiles, dim=0)
        indices = torch.randperm(grid**2)
        shuffled_tiles = tiles[indices]

        # Construct the jigsaw image tensor
        rows = []
        for i in range(grid):
            row = torch.cat(tuple(shuffled_tiles[i*grid:(i+1)*grid]), dim=2)
            rows.append(row)
        jigsaw_tensor = torch.cat(rows, dim=1)
        if is_gaussian_blur:
            gaussian = GaussianBlur((5, 9), (10, 30))
            jigsaw_tensor = gaussian(jigsaw_tensor)

        jigsawed_tensors.append(jigsaw_tensor)

    # Stack all jigsawed tensors into a single tensor batch
    return torch.stack(jigsawed_tensors)