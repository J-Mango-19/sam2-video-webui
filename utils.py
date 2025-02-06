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
    #images = list(map(lambda x: os.path.join(sample_dir, x), images))[:5]
    images = list(map(lambda x: os.path.join(sample_dir, x), images))
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

    if device.type == 'cuda':
        torch.autocast('cuda', dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    elif device.type == 'cpu':
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

def draw_box_on_plot(img, boxes, color):
    """
    draw a box on the image with top left corner = box[0] and bottom right corner = box[1]
    """
    for box in boxes:
        cv2.rectangle(img, box[0], box[1], color, 4)

def point_str_to_list(points):
    points = list(map(eval, points.split('\n'))) if points else []
    return points

def point_list_to_str(points):
    if points is None:
        points = []
    points = '\n'.join(list(map(str, points)))
    return points

def parse_paths(file_path):
    """
    parse the paths.txt file
    """
    path_dict = {}
    with open(file_path, 'r') as f:
        kv_pairs = f.read().splitlines()

    for kv in kv_pairs:
        # Hashtags will act as comments in the paths file
        if kv.startswith("#") or kv == "":
            continue
        try:
            key = kv.split(': ')[0].strip()
            val = kv.split(': ')[1].strip()

        except IndexError:
            key = kv.split(':')[0].strip()
            val = kv.split(':')[1].strip()

        if key == "sam2_path":
            path_dict["sam2_path"] = val
        elif key == "sample_dir":
            path_dict["sample_dir"] = val
        elif key == "video_data_path":
            path_dict["video_data_path"] = val
        elif key == "mask_dir":
            path_dict["mask_dir"] = val
        elif key == "raw_video_path":
            path_dict["raw_video_path"] = val
        elif key == "out_video_path":
            path_dict["out_video_path"] = val
        elif key == "model_cfg":
            path_dict["model_cfg"] = val
        elif key == "sam2_checkpoint":
            path_dict["sam2_checkpoint"] = val

    return path_dict
