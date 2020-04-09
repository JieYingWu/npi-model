""" Combine images into gif. Adjust duration if necessary"""
import os
import imageio

def create_gif(path, output_path):
    """
    Args:
    - path: path to folder with only images
    - output_path: path to output gif 

    """
    f = [f.name for f in os.scandir(path)]
    name, ext = os.path.splitext(f[0])

    images = []
    for filename in f:
        images.append(imageio.imread(os.path.join(path,filename)))
    imageio.mimsave(output_path, images, duration=0.5)


if __name__ == '__main__':
    path = 'results/plots/usa'
    output = 'results/plots/test.gif'
    create_gif(path, output)