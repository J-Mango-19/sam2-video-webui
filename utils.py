import os
import cv2
import torch
import gradio as gr

def process_img_paths(sample_dir):
    """ 
    List and sort image file paths from the sample directory by numerical order.
    """
    images = os.listdir(sample_dir)
    images = sorted(images, key=lambda x: int(x.split('.')[0]))
    ### NOTE: images reduced in size to only the first 5 to speed up testing. Remove the [:5] later
    images = list(map(lambda x: os.path.join(sample_dir, x), images))[:5]
    return images

def set_sam_path(path_file='./paths.txt'):
    """
    Appends line 1 of the paths file (meant to be the path of the sam2 repo on your system) to PYTHONPATH
    """
    with open(path_file, 'r') as f:
        sam_path = f.read().splitlines()[0]

    current_pythonpath = os.getenv('PYTHONPATH', '')

    os.environ['PYTHONPATH'] = current_pythonpath + ':' + sam_path

def get_sample_dir(path_file='./paths.txt'):
    """
    sample dir is the directory containing the video you want to segment
    it should be a directory of ordered .jpg images
    """
    with open(path_file, 'r') as f:
        sample_dir_path = f.read().splitlines()[1]

    return sample_dir_path

def set_device():
    """
    Set CUDA (with optimizations) if available. 
    Otherwise, set to CPU with 4 threads (CRC frontend allows a maximum usage of 4 cores 
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    print(f'Using Device: {device}')

    if device == torch.device('cuda'):
        torch.autocast('cuda', dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    elif device == torch.device('cpu'):
        torch.set_num_threads(4) # CRC frontend allows usage of a maximum of 4 cpus

    return device

class Color():
    pos = (0, 255, 0) # 'positive' color is green
    neg = (255, 0, 0) # 'negative' color is red
    tmp = (0, 0, 255) # 'neutral'  color is blue

def draw_point_on_plot(image, points, color):
    """
    takes an image, list of all points, and an RGB color value
    draws a dot on the image, at the latest point, and in the given color
    """
    radius = image.shape[0] // 100
    for point in points:
        cv2.circle(image, point[:2], radius, color, 4)

