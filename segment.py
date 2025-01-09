import os
import sys
import cv2
import torch
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from functools import partial
from sam2.build_sam import build_sam2_video_predictor
from utils import *

set_sam_path() # appends sam2's path to PYTHONPATH variable
from sam2.build_sam import build_sam2_video_predictor

# Directory containing input images for processing.
sample_dir = get_sample_dir()

# Directory to save masks and paths for input and output videos.
mask_dir = './masks/'
raw_video_path = './test_video_raw.mp4'
out_video_path = './tmp_video_out.mp4'
fps = 30

# Paths for model checkpoint and configuration.
sam2_checkpoint = "/afs/crc.nd.edu/user/j/jmangion/Public/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "//afs/crc.nd.edu/user/j/jmangion/Public/segment-anything-2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"

# Initialize global variables 
predictor       = None                          # SAM2 model
inference_state = None                          # Dictionary closely linked to SAM2 model. Holds information pertaining to the current segmentation task (see SAM2VideoPredictor.init_state())
ann_obj_id      = 1                             # We are only annotating one object for now
images          = process_img_paths(sample_dir) # returns an odered list of img paths from sample_dir
video_segments  = [None] * len(images)          # A list of image-shaped numpy boolean arrays, where a True entry indicates that pixel is part of the segmented object; False means it's not
device          = set_device()                  # Moves SAM2 to CUDA if available. 
point_labels   = []                             # holds the labels (0:negative 1:positive) for all points prompts

def export_result():
    """
    Uses OpenCV video writer to save a video where for each frame, corresponding masks are applied as black regions
    """
    height, width, _ = cv2.imread(images[0]).shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height), True)

    for i, mask in enumerate(video_segments):
        mask = np.where(mask, 255, 0)
        cv2.imwrite(os.path.join(mask_dir, f"{i}.jpg"), mask)

    for image, mask in zip(images, video_segments):
        image = cv2.imread(image)
        mask = np.tile(mask[:,:,None], (1, 1, 3))
        image = np.where(mask, 0, image)
        writer.write(image)
    writer.release()
    return f'masked video exported to: {out_video_path}', out_video_path

def model_load():
    """ 
    loads SAM2 model into `predictor` with default inference state 
    """
    global predictor, inference_state
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(video_path=sample_dir)
    predictor.reset_state(inference_state)
    # NOTE: this reduces the cost of inference for testing. Remove this line (the next line) later.
    inference_state['images'] = inference_state['images'][:5]
    inference_state['num_frames']=5
    return f"SAM2 ({sam2_checkpoint.split('/')[-1][:-3]}) loaded successfully"

def video_slide(frame, all_points, image=None):
    """
    Ensures everything (image, points, masks) are displayed up to date for a given frame
    """
    if image is None:
        image = cv2.imread(images[frame])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f'Video_slide called. All points: {all_points}')
    all_points = list(map(eval, all_points.split('\n'))) if all_points else []
    if all_points != []:
        print(f'all_points is: {all_points}')
        positive_points = list(filter(lambda x: x[2]==frame and x[3]==1, all_points))
        positive_points = list(map(lambda x: x[:2], positive_points))
        negative_points = list(filter(lambda x: x[2]==frame and x[3]==0, all_points))
        negative_points = list(map(lambda x: x[:2], negative_points))
        draw_point_on_plot(image, positive_points, Color.pos)
        draw_point_on_plot(image, negative_points, Color.neg)

    if video_segments[frame] is not None:
        mask = np.zeros_like(image)
        mask[:, :, 0] = np.where(video_segments[frame], 255, 0)
        image = np.where(mask > 0.0, mask, image)

    return {'__type__':'update', 'value':image}

def get_select_coords(frame, all_points, evt: gr.SelectData):
    """
    Listens for image clicks by user and returns the HxW point they clicked
    Calls draw_point_on_plot to display the point in tmp color (blue)
    """
    image = cv2.imread(images[frame])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    point = evt.index
    draw_point_on_plot(image, [point], Color.tmp)
    image = video_slide(frame, all_points, image)['value']
    return {'__type__':'update', 'value':image}, str(point)[1:-1]

def point_add(all_points, new_point, frame, label):
    """
    Uses SAM2 to generate a new mask for the specified frame based on a new point and if it's positive/negative
    """
    global video_segments
    global point_labels
    # all points is the same form as new point -- that is, a list of points, where each point is [x_coord, y_coord, frame_num, pos_or_neg]
    new_point = list(eval(new_point)) + [frame, label]
    all_points = list(map(eval, all_points.split('\n'))) if all_points else []
    all_points.append(new_point)
    point_labels.append(int(label))
    all_points = '\n'.join(list(map(str, all_points)))
    points = np.array([new_point[:2]], dtype=np.float32)
    labels = np.array([label], np.int32)
    out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(inference_state, frame_idx=frame, obj_id=ann_obj_id, points=points, labels=labels)
    video_segments[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0] # converts model's confidence scores into binary masks
    image = video_slide(frame, all_points)['value']
    return {'__type__':'update', 'value':image}, all_points

def point_remove(all_points, frame):
    """
    removes the most recently added point
    """
    global video_segments
    global point_labels
    all_points = list(map(eval, all_points.split('\n'))) if all_points else [] # convert all_points string into a list of coordinates
    all_points = all_points[:-1]                                               # remove most recent point coordinate
    if all_points == []:
        all_points = None
    point_labels = point_labels[:-1]                                           # remove the most point label
    video_segments = [None]*len(images)                                        # remove current masks from images
    if all_points:
        # clear sam predictions and start over without the latest point
        predictor.reset_state(inference_state)
        print(f'Just removed a point. all_points: {all_points}')
        point_coords = np.array([point[:2] for point in all_points])
        print(f'Point_coords generated from all_points; {point_coords.shape=}, {point_coords=}')
        out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(inference_state, frame_idx=frame, obj_id=ann_obj_id, points=point_coords, labels=np.array(point_labels, dtype=np.int32))
        video_segments[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0] # converts model's confidence scores into binary masks
        all_points = '\n'.join(list(map(str, all_points)))                          # convert all_points list of coordinates into a string

    image = video_slide(frame, all_points)['value']                            # update the UI
    return {'__type__':'update', 'value':image}, all_points

def refresh_image(frame, all_points):
    """
    Updates the display using `video_slide` to include all annotations in masks in the UI view of the current frame
    """
    image = video_slide(frame, all_points)['value']
    return {'__type__':'update', 'value':image}

def video_propagate(all_points, frame):
    """
    Tells SAM2 to propagate the masks across the video
    """
    global video_segments
    video_segments = video_segments if video_segments else [None] * len(images)
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0] # converts SAM2 confidence scores into binary masks

    image = video_slide(frame, all_points)['value']
    return {'__type__':'update', 'value':image}

def clear_video_propagate(all_points, frame):
    """
    Deletes all binary prediction masks
    Informs SAM2 we are starting over from scratch
    Clears the UI
    """
    global video_segments
    video_segments = [None] * len(images)
    predictor.reset_state(inference_state)
    image = video_slide(frame, all_points)['value']
    return {'__type__':'update', 'value':image}

def point_show(point, frame, all_points):
    """
    Updates the UI, drawing a temporary point and ensuring all masks, permanent points, etc are shown
    Does not change inference_state yet bc user has not dediced if this is a positive or negative point
    """
    image = cv2.imread(images[frame])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    point = eval(point)
    draw_point_on_plot(image, [point], Color.tmp)
    image = video_slide(frame, all_points, image)['value']
    return {'__type__':'update', 'value':image}

with gr.Blocks(title="SAM2") as demo:
    gr.Markdown("""# Interact with SAM2 through your browser 
            ### Steps: 
            1. load model
            2. click on image
                a. make selection (positive or negative)
                b. repeat step 2 until desired mask reached
            3. Export masked video
              """)
    with gr.Tabs():
        with gr.TabItem("0 - Video Segmentation"):
            with gr.Row():
                with gr.Column(min_width=600):
                    with gr.Row():
                        image_input = gr.Image(images[0], label='Video Images -- Select Points here') # originally display first image
                    with gr.Row():
                        slider = gr.Slider(minimum=0, maximum=len(images)-1, step=1, value=0, interactive=True)
                    with gr.Row():
                        video_output = gr.Video(value=raw_video_path)
                with gr.Column(min_width=60): 
                    with gr.Row():
                        new_point = gr.Textbox(label="new point (not pos or neg yet)", value="", lines=1, min_width=80)
                    with gr.Row():
                        all_points = gr.Textbox(label="all points", value="", lines=20, interactive=False)
                with gr.Column(min_width=40):
                    with gr.Row():
                        load_model = gr.Button("load model", variant="secondary")
                        load_info = gr.Textbox(value="SAM2 not yet loaded.", lines=1, interactive=False, show_label=False, container=False)
                    with gr.Row(min_height=40): pass
                    with gr.Row():
                        refresh = gr.Button("refresh (update display)", variant="primary")
                    with gr.Row(min_height=40): pass
                    add_positive_point = gr.Button("set new point as positive", variant="huggingface")
                    add_negative_point = gr.Button("set new point as negative", variant="huggingface")
                    remove_point       = gr.Button("remove last point", variant="huggingface")
                    with gr.Row(min_height=40): pass
                    with gr.Row():
                        propagate_btn = gr.Button("propagate", variant="primary")
                        clear_propagate_btn = gr.Button("clear propagate", variant="stop")
                    with gr.Row(min_height=40): pass
                    with gr.Row():
                        export_button = gr.Button("export", variant="secondary")
                        export_info = gr.Textbox(value="export info", lines=1, interactive=False, show_label=False, container=False)

            image_input.select(get_select_coords, [slider, all_points], [image_input, new_point])
            slider.change(point_show, [new_point, slider, all_points], [image_input])
            
            load_model.click(model_load, [], [load_info])
            refresh.click(refresh_image, [slider, all_points], [image_input])

            add_positive_point.click(partial(point_add, label=1), [all_points, new_point, slider], [image_input, all_points])
            add_negative_point.click(partial(point_add, label=0), [all_points, new_point, slider], [image_input, all_points])
            remove_point.click(point_remove, [all_points, slider], [image_input, all_points])

            propagate_btn.click(video_propagate, [all_points, slider], [image_input])
            clear_propagate_btn.click(clear_video_propagate, [all_points, slider], [image_input])

            export_button.click(export_result, [], [export_info, video_output])

    _, local_url, public_url = demo.launch(share=True)
