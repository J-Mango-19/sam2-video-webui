import os
import sys
import cv2
import torch
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from functools import partial
from gradio_image_annotation import image_annotator # must install this separately from gradio library
from utils import *
set_sam_path() # appends sam2's path to PYTHONPATH variable
from sam2.build_sam import build_sam2_video_predictor
sys.stdout.reconfigure(line_buffering=True)

file_paths = parse_paths("paths.txt")
fps = 30

# Initialize global variables 
device          = set_device()                  # Moves SAM2 to CUDA if available. 
predictor       = None                          # SAM2 model
compile_sam     = False                         # Optionally compile SAM2 model. Compilation and first inference are slow, but compiled SAM2 is faster for large videos.
inference_state = None                          # Dictionary closely linked to SAM2 model. Holds information pertaining to the current segmentation task (see SAM2VideoPredictor.init_state())
images          = process_img_paths(file_paths['sample_dir']) # returns an odered list of img paths from sample_dir
video_segments  = [[None]*3 for _ in range(len(images))] # A list of lists, where each outer list corresponds to a video frame, and each sub-list corresponds to an object and is a numpy boolean array (true for a mask there, false otherwise)

def export_result():
    """
    Uses OpenCV video writer to save a video where for each frame, its corresponding masks are applied as black regions
    """
    os.makedirs(file_paths["mask_dir"], exist_ok=True)
    height, width, _ = cv2.imread(images[0]).shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(file_paths["out_video_path"], fourcc, fps, (width, height), True)

    for i, frame_segment in enumerate(video_segments):           # for each frame
        image = cv2.imread(images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.zeros_like(image)
        for k in range(len(frame_segment)):                       # for each masked object
            mask[:, :, k] = np.where(frame_segment[k], 255, 0)

            image[:, :, k] = np.where(frame_segment[k], 255, image[:, :, k])

        cv2.imwrite(os.path.join(file_paths["mask_dir"], f"{i}.jpg"), mask)
        writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    writer.release()
    return f"masked video exported to: {file_paths['out_video_path']}", file_paths['out_video_path']

def model_load():
    """ 
    loads SAM2 model into `predictor` with default inference state 
    """
    global predictor, inference_state
    predictor = build_sam2_video_predictor(file_paths["model_cfg"], file_paths["sam2_checkpoint"], device=device, vos_optimized=compile_sam)
    inference_state = predictor.init_state(video_path=file_paths["sample_dir"])
    predictor.reset_state(inference_state)
    # NOTE: this reduces the cost of inference for testing. Remove this line (the next line) later.
    # inference_state['images'] = inference_state['images'][:5]
    # inference_state['num_frames']=5
    return f"SAM2 ({file_paths['sam2_checkpoint'].split('/')[-1][:-3]}) loaded successfully"

def video_slide(frame, all_points, all_boxes, image=None):
    """
    Ensures everything (image, points, boxes, masks) are displayed up to date for a given frame
    """
    if image is None:
        image = cv2.imread(images[frame])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    all_points = point_str_to_list(all_points)
    if all_points != []:
        positive_points = list(filter(lambda x: x[2]==frame and x[3]==1, all_points))
        positive_points = list(map(lambda x: x[:2], positive_points))
        negative_points = list(filter(lambda x: x[2]==frame and x[3]==0, all_points))
        negative_points = list(map(lambda x: x[:2], negative_points))
        draw_point_on_plot(image, positive_points, Color.pos)
        draw_point_on_plot(image, negative_points, Color.neg)

    all_boxes = point_str_to_list(all_boxes)
    if all_boxes != [] and all_boxes != None and None not in all_boxes:
        all_boxes = list(map(lambda x: x[:2], all_boxes))
        print('video_slide: all_boxes', all_boxes)
        draw_box_on_plot(image, all_boxes, Color.pos)

    # apply colored masks from video_segments to specified image
    frame_segment = video_segments[frame]
    mask = np.zeros_like(image)
    for k in range(len(frame_segment)):
        if frame_segment[k] is not None:
            mask[:, :, k] = np.where(frame_segment[k], 255, 0) # creates a colored mask

    image = np.where(mask > 0.0, mask, image) # applies the colored mask to the image

    return {'__type__':'update', 'value':image}

def get_select_coords(frame, all_points, all_boxes, new_box, prompt_type, evt: gr.SelectData):
    """
    Decides whether it's working on a lone point or a box point and acts accordingly
    Listens for image clicks by user and returns the HxW point they clicked
    Calls draw_point_on_plot to display the point in tmp color (blue)
    """

    img = cv2.imread(images[frame])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_point = evt.index
    points = point_str_to_list(all_points)
    boxes = point_str_to_list(all_boxes)
    draw_point_on_plot(img, [new_point], (0, 0, 255))

    if prompt_type == 'point':
        points += [new_point]
        new_point = str(new_point)[1:-1]
        newest_box = ""

    if prompt_type == 'box':
        # if we have an empty list of boxes
        # or if all existing boxes already have both their corners filled out
        newest_box = point_str_to_list(new_box)
        if len(newest_box) == 1:
            newest_box = newest_box[0]

        # if we aren't currently constructing a new box then make a new one
        if newest_box == [] or (newest_box[0] and newest_box[1]):
            newest_box = [None]*2

        corner1 = newest_box[0]
        corner2 = newest_box[1]


        # first case: we are starting a new box from scratch
        if corner1 is None:
            corner1 = new_point
            newest_box[0] = corner1

        # second case: the new box already has one corner
        else:
            corner2 = new_point
            newest_box[1] = corner2

        if corner1 and corner2:
            draw_box_on_plot(img, [newest_box], (0, 0, 255))

        new_point = ""
        newest_box   = str(newest_box)#[1:-1] 

    img = video_slide(frame, all_points, all_boxes, img)['value']
    #return img, point_list_to_str(points), point_list_to_str(boxes)
    return {'__type__':'update', 'value':img}, new_point, newest_box

def prompt_add(prompt_type, all_points, new_point, all_boxes, new_box, frame, obj_id, label):
    """
    Uses SAM2 to generate a new mask for the specified frame based on a new point and if it's positive/negative
    """
    global video_segments

    if prompt_type == "point":
        # all points is the same form as new point -- that is, a list of points, where each point is [x_coord, y_coord, frame_num, pos_or_neg]
        new_point = list(eval(new_point)) + [frame, label, obj_id]
        all_points = point_str_to_list(all_points)
        all_points.append(new_point)
        all_points = point_list_to_str(all_points)
        points = np.array([new_point[:2]], dtype=np.float32)
        labels = np.array([label], np.int32)
        out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(inference_state, frame_idx=frame, obj_id=obj_id, points=points, labels=labels)
        # out_mask_logits is shape (obj_id, 1, H, W). 1 likely scales with batch size

    else:
        new_box = list(eval(new_box))
        all_boxes = point_str_to_list(all_boxes)
        new_box[2:] = [frame, label, obj_id]
        all_boxes.append(new_box)
        all_boxes = point_list_to_str(all_boxes)
        boxes = np.array(new_box[:2], dtype=np.float32).flatten()
        labels = np.array([label], np.int32)
        out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(inference_state, frame_idx=frame, obj_id=obj_id, box=boxes, labels=None)

    video_segments[frame][obj_id] = (out_mask_logits[obj_id] > 0.0).cpu().numpy()[0] # converts model's confidence scores into binary masks
    image = video_slide(frame, all_points, all_boxes)['value']
    return {'__type__':'update', 'value':image}, all_points, all_boxes

def prompt_remove(prompt_type, all_points, all_boxes, frame):
    """
    Removes the most recently added prompt of the given category (prompt_type). Clears predicted masks. re-predicts masks (this time without the point that was removed)
    """
    global video_segments
    video_segments = [[None]*3 for _ in range(len(images))]                     # delete current masks
    predictor.reset_state(inference_state)

    all_points = point_str_to_list(all_points)                                 # convert all_points string into a list of coordinates
    all_boxes = point_str_to_list(all_boxes)

    if prompt_type == "point":
        all_points = all_points[:-1]                                               # remove most recent point coordinate

    else:
        all_boxes = all_boxes[:-1]                                                 # remove the most recent box

    if all_points == []:
        all_points = None


    # if there are any valid box prompts remaining, re-predict masks (this time without latest box)
    if all_boxes != [] and all_boxes != None:
        print('prompt_remove: all_boxes', all_boxes)
        # extract coords, frames, and labels into separate lists for easy processing
        box_coords = [box[:2] for box in all_boxes]
        box_frames = [box[2]  for box in all_boxes]
        box_labels = [box[3]  for box in all_boxes]
        obj_ids    = [box[4]  for box in all_boxes]

        # if points are all on the same frame and object, we can submit them to sam2 all at once for efficiency
        for coord, frame, label, obj_id in zip(box_coords, box_frames, box_labels, obj_ids):
            out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(inference_state, frame_idx=frame, obj_id=obj_id, box=coord, labels=None) # labels=None since boxes are always positive
            video_segments[out_frame_idx][obj_id] = (out_mask_logits[obj_id] > 0.0).cpu().numpy()[0] # converts model's confidence scores into binary masks and applies them to the images

    # if there are any remaining point prompts, re-predict masks (this time without the latest point)
    if all_points:
        # extract coords, frames, and labels into separate lists for easy processing
        point_coords = [point[:2] for point in all_points]
        point_frames = [point[2]  for point in all_points]
        point_labels = [point[3]  for point in all_points]
        obj_ids      = [point[4]  for point in all_points]

        # if points are all on the same frame and object, we can submit them to sam2 all at once for efficiency
        if (sorted(point_frames)[0] == sorted(point_frames)[-1]) and (sorted(obj_ids)[0] == sorted(obj_ids)[-1]):
            frame  = point_frames[0]
            obj_id = obj_ids[0]
            out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(inference_state, frame_idx=frame, obj_id=obj_id, points=point_coords, labels=point_labels)
            video_segments[out_frame_idx][obj_id] = (out_mask_logits[obj_id] > 0.0).cpu().numpy()[0] # converts model's confidence scores into binary masks and applies them to the images

        # if points are on different frames and/or objects, then we must submit them to sam2 separately
        else: 
            for coord, frame, label, obj_id in zip(point_coords, point_frames, point_labels, obj_ids):
                coord = [coord]
                label = [label]

                out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(inference_state, frame_idx=frame, obj_id=obj_id, points=coord, labels=label)
                video_segments[out_frame_idx][obj_id] = (out_mask_logits[obj_id] > 0.0).cpu().numpy()[0] # converts model's confidence scores into binary masks and applies them to the images


    all_points = point_list_to_str(all_points)                                  # convert all_points into a string
    all_boxes = point_list_to_str(all_boxes)                                  # convert all_boxes into a string
    image = video_slide(frame, all_points, all_boxes)['value']                            # update the UI
    return {'__type__':'update', 'value':image}, all_points, all_boxes

def refresh_image(frame, all_points, all_boxes):
    """
    Updates the display using `video_slide` to include all annotations in masks in the UI view of the current frame
    """
    image = video_slide(frame, all_points, all_boxes)['value']
    return {'__type__':'update', 'value':image}

def video_propagate(all_points, all_boxes, frame, reverse):
    """
    Calls SAM2 to propagate masks forward and backward in the video from the given frame.
    Ensures that masks are propagated throughout the entire video while resolving conflicts.
    """
    global video_segments

    if not video_segments or len(video_segments) != len(images):
        video_segments = [[None] * 3 for _ in range(len(images))]

    # get all interactive frames and sort them by time
    interactive_frames = sorted(set(point[2] for point in map(eval, all_points.split("\n"))))


    # first propagate forward (from the earliest interactive frame)
    for interactive_frame in interactive_frames:
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=interactive_frame
        ):
            out_mask_numpy = out_mask_logits.cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids):
                new_mask = (out_mask_numpy[i] > 0.0)[0].astype(bool)  # make sure new_mask is boolean

                if video_segments[out_frame_idx][out_obj_id] is None:
                    video_segments[out_frame_idx][out_obj_id] = new_mask
                else:
                    prev_mask = video_segments[out_frame_idx][out_obj_id].astype(bool)  # make sure prev_mask is boolean
                    if prev_mask.shape != new_mask.shape:
                        print(f"Shape mismatch at frame {out_frame_idx}, object {out_obj_id}. Skipping IoU calculation.")
                        continue  # skip if the shapes do not match

                    iou = np.sum(prev_mask & new_mask) / (np.sum(prev_mask | new_mask) + 1e-6)

                    if iou < 0.5:
                        video_segments[out_frame_idx][out_obj_id] = new_mask

    # then propagate backward (from the latest interactive frame)
    for interactive_frame in reversed(interactive_frames):
        if interactive_frame > 0:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state, start_frame_idx=interactive_frame, reverse=True
            ):
                out_mask_numpy = out_mask_logits.cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids):
                    new_mask = (out_mask_numpy[i] > 0.0)[0].astype(bool)

                    if video_segments[out_frame_idx][out_obj_id] is None:
                        video_segments[out_frame_idx][out_obj_id] = new_mask
                    else:
                        prev_mask = video_segments[out_frame_idx][out_obj_id].astype(bool)
                        if prev_mask.shape != new_mask.shape:
                            print(f"Shape mismatch at frame {out_frame_idx}, object {out_obj_id}. Skipping IoU calculation.")
                            continue

                        iou = np.sum(prev_mask & new_mask) / (np.sum(prev_mask | new_mask) + 1e-6)
                        if iou < 0.5:
                            video_segments[out_frame_idx][out_obj_id] = new_mask

    # Update UI display
    image = video_slide(frame, all_points)['value']
    return {'__type__': 'update', 'value': image}

def clear_video_propagate(all_points, all_boxes, frame):
    """
    Deletes all binary prediction masks
    Informs SAM2 we are starting over from scratch
    Clears the UI
    """
    global video_segments
    video_segments = [[None] * 3 for _ in range(len(images))]
    predictor.reset_state(inference_state)
    image = video_slide(frame, all_points, all_boxes)['value']
    return {'__type__':'update', 'value':image}

def point_show(point, box, frame, all_points, all_boxes):
    """
    Updates the UI, drawing a temporary point and ensuring all masks, permanent points, etc are shown
    Does not change inference_state yet bc user has not dediced if this is a positive or negative point
    """
    image = cv2.imread(images[frame])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if point != "":
        point = eval(point)
        draw_point_on_plot(image, [point], Color.tmp)
    if box != "":
        box = eval(box)
        draw_box_on_plot(image, [box], Color.tmp)
        
    image = video_slide(frame, all_points, all_boxes, image)['value']
    return {'__type__':'update', 'value':image}

with gr.Blocks(title="SAM2") as demo:
    gr.Markdown("""# Interact with SAM2 through your browser 
            ## Instructions: 
            ### 0. modify paths to segmentation model, data, etc in `paths.txt`  
            ### 1. load model  
            ### 2. Prompt the model  
            * Select which object you are currently annotating (starting at 0)  
            * Select a type of prompt (box or point). 
            * **To make a box prompt**, select its radio button, then click twice on the image, one click for each corner.  
                * **For Box Prompts**: SAM2 is most effective when you click in the top left of your box as the first corner, and the bottom right as the second corner  
                - When using point prompts and box prompts on the same object, the box prompts must come *before* the point prompts  
            * If necessary, undo a prompt using the `remove last prompt` button
            ### 3. Propagate your masks  
            ### 4. Export masked video
              """)
    with gr.Tabs():
        with gr.TabItem("0 - Video Segmentation"):
            with gr.Row():
                with gr.Column():
                    obj_id = gr.Radio(choices=[('object 0', 0), ('object 1', 1), ('object 2', 2)], label='Choose which object to annotate (always start with 0, then 1, ...)', value=0)
                    prompt_type = gr.Radio(choices=[("Point Prompt", 'point'), ("Box Prompt", 'box')], label='Select a prompt type', value='point')
                with gr.Column():
                    new_point = gr.Textbox(label="new point (not a positive or negative prompt yet)", value="", lines=1, min_width=10)
                    new_box   = gr.Textbox(label="new bounding box (not positive or negative prompt yet)", value="", lines=1, min_width=10)
                with gr.Column(min_width=600):
                    with gr.Row():
                        image_input = gr.Image(images[0], label='Video Images -- Select Points here') # originally display first image
                    with gr.Row():
                        slider = gr.Slider(label="Slider - select which frame to annotate", minimum=0, maximum=len(images)-1, step=1, value=0, interactive=True)
                    with gr.Row():
                        video_output = gr.Video(value=file_paths["raw_video_path"])
                with gr.Column(min_width=60): 
                    with gr.Row():
                        all_points = gr.Textbox(label="all points (x_coord, y_coord, frame_num, pos_or_neg, obj_id)", value="", lines=10, interactive=False)
                        all_boxes  = gr.Textbox(label="all boxes (point1, point2, frame_num, pos_or_neg, obj_id)", value="", lines=10, interactive=False)
                with gr.Column(min_width=40):
                    with gr.Row():
                        load_model = gr.Button("load model", variant="secondary")
                        load_info = gr.Textbox(value="SAM2 not yet loaded.", lines=1, interactive=False, show_label=False, container=False)
                    #with gr.Row(min_height=40): pass
                    with gr.Row():
                        refresh = gr.Button("refresh (update display)", variant="primary")
                    #with gr.Row(min_height=40): pass
                    add_positive_prompt = gr.Button("set new prompt as positive", variant="primary")
                    add_negative_prompt = gr.Button("set new prompt as negative", variant="primary")
                    remove_point       = gr.Button("remove last prompt (of selected prompt type)", variant="secondary")
                    #with gr.Row(min_height=40): pass
                    with gr.Row():
                        forward_propagate_btn = gr.Button("propagate masks forward", variant="primary")
                        backward_propagate_btn = gr.Button("propagate masks backward", variant="primary")
                        clear_propagate_btn = gr.Button("clear all masks", variant="secondary")
                    with gr.Row(min_height=40): pass
                    with gr.Row():
                        export_button = gr.Button("export", variant="primary")
                        export_info = gr.Textbox(value="exported video path will appear here", lines=1, interactive=False, show_label=False, container=False)

            image_input.select(get_select_coords, [slider, all_points, all_boxes, new_box, prompt_type], [image_input, new_point, new_box])
            slider.change(point_show, [new_point, new_box, slider, all_points, all_boxes], [image_input])
            
            load_model.click(model_load, [], [load_info])
            refresh.click(refresh_image, [slider, all_points, all_boxes], [image_input])

            add_positive_prompt.click(partial(prompt_add, label=1), [prompt_type, all_points, new_point, all_boxes, new_box, slider, obj_id], [image_input, all_points, all_boxes])
            add_negative_prompt.click(partial(prompt_add, label=0), [prompt_type, all_points, new_point, all_boxes, new_box, slider, obj_id], [image_input, all_points, all_boxes])
            remove_point.click(prompt_remove, [prompt_type, all_points, all_boxes, slider], [image_input, all_points, all_boxes])

            forward_propagate_btn.click(partial(video_propagate, reverse=False), [all_points, all_boxes, slider], [image_input])
            backward_propagate_btn.click(partial(video_propagate, reverse=True), [all_points, all_boxes, slider], [image_input])
            clear_propagate_btn.click(clear_video_propagate, [all_points, all_boxes, slider], [image_input])

            export_button.click(export_result, [], [export_info, video_output])

    _, local_url, public_url = demo.launch(share=True)
