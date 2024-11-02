import os
import sys
import cv2
import torch
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from functools import partial
from sam2.build_sam import build_sam2_video_predictor

device = torch.device("cuda")    
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sample_dir = '/home/work/disk/vision/video_inpaint/sample'
# sample_dir = '/home/work/disk/sam2/notebooks/videos/bedroom'
images = os.listdir(sample_dir)
images = sorted(images, key=lambda x: int(x.split('.')[0]))
images = list(map(lambda x: os.path.join(sample_dir, x), images))

mask_dir = '/home/work/disk/vision/video_inpaint/image'
raw_video_path = '/home/work/disk/vision/video_inpaint/video/b_no_sub.mp4'
out_video_path = '/home/work/disk/vision/video_inpaint/video/tmp.mp4'
fps = 30

# sam2_checkpoint = "/home/work/disk/sam2/checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_checkpoint = "/home/work/disk/sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

predictor, inference_state, predictor = None, None, None
ann_obj_id = 1 
video_segments = [None] * len(images)

def export_result():
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
    return 'export success', out_video_path

def model_load():
    global predictor, inference_state, predictor
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(video_path=sample_dir)
    predictor.reset_state(inference_state)
    return 'load success'

class Color():
    pos = (0, 255, 0)
    neg = (0, 0, 255)
    tmp = (255, 0, 0)

def circle_plot(image, points, color):
    radius = image.shape[0] // 100
    for point in points:
        cv2.circle(image, point[:2], radius, color, 4)

def key_transfer(x):
    return x[2]*1e8+x[0]*1e4+x[1]

def video_slide(frame, all_points, image=None):
    if image is None:
        image = cv2.imread(images[frame])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    all_points = list(map(eval, all_points.split('\n'))) if all_points else []
    positive_points = list(filter(lambda x: x[2]==frame and x[3]==1, all_points))
    positive_points = list(map(lambda x: x[:2], positive_points))
    negative_points = list(filter(lambda x: x[2]==frame and x[3]==0, all_points))
    negative_points = list(map(lambda x: x[:2], negative_points))
    circle_plot(image, positive_points, Color.pos)
    circle_plot(image, negative_points, Color.neg)

    if video_segments[frame] is not None:
        mask = np.zeros_like(image)
        mask[:, :, 0] = np.where(video_segments[frame], 255, 0)
        image = np.where(mask > 0.0, mask, image)

    return {'__type__':'update', 'value':image}

def get_select_coords(frame, all_points, evt: gr.SelectData):
    image = cv2.imread(images[frame])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    point = evt.index
    circle_plot(image, [point], Color.tmp)
    image = video_slide(frame, all_points, image)['value']
    return {'__type__':'update', 'value':image}, str(point)[1:-1]

def point_show(point, frame, all_points):
    image = cv2.imread(images[frame])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    point = eval(point)
    circle_plot(image, [point], Color.tmp)
    image = video_slide(frame, all_points, image)['value']
    return {'__type__':'update', 'value':image}

def point_add(all_points, new_point, frame, label):
    global video_segments
    new_point = list(eval(new_point)) + [frame, label]
    all_points = list(map(eval, all_points.split('\n'))) if all_points else []
    all_points.append(new_point)
    all_points = '\n'.join(list(map(str, all_points)))

    points = np.array([new_point[:2]], dtype=np.float32)
    labels = np.array([label], np.int32)
    out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(inference_state, frame_idx=frame, obj_id=ann_obj_id, points=points, labels=labels)
    video_segments[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0]

    image = video_slide(frame, all_points)['value']
    return {'__type__':'update', 'value':image}, all_points

def refresh_image(frame, all_points):
    image = video_slide(frame, all_points)['value']
    return {'__type__':'update', 'value':image}

def video_propagate(all_points, frame):
    global video_segments
    video_segments = video_segments if video_segments else [None] * len(images)
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0]

    image = video_slide(frame, all_points)['value']
    return {'__type__':'update', 'value':image}
    
def clear_video_propagate(all_points, frame):
    global video_segments
    video_segments = [None] * len(images)
    predictor.reset_state(inference_state)
    image = video_slide(frame, all_points)['value']
    return {'__type__':'update', 'value':image}

with gr.Blocks(title="SAM2") as app:
    gr.Markdown(value="本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.")
    with gr.Tabs():
        with gr.TabItem("0-视频分割"):
            with gr.Row():
                with gr.Column(min_width=600):
                    with gr.Row():
                        image_input = gr.Image(images[0])
                    with gr.Row():
                        slider = gr.Slider(minimum=0, maximum=len(images)-1, step=1, value=0, interactive=True)
                    with gr.Row():
                        video_output = gr.Video(value=raw_video_path)
                with gr.Column(min_width=60): 
                    with gr.Row():
                        new_point = gr.Textbox(label="new point", value="", lines=1, min_width=80)
                    with gr.Row():
                        all_points = gr.Textbox(label="all points", value="", lines=20, interactive=False)
                with gr.Column(min_width=40):
                    with gr.Row():
                        load_model = gr.Button("load model", variant="secondary")
                        load_info = gr.Textbox(value="load info", lines=1, interactive=False, show_label=False, container=False)
                    with gr.Row(min_height=40): pass
                    with gr.Row():
                        show_point = gr.Button("show point", variant="primary")
                        refresh = gr.Button("refresh", variant="primary")
                    with gr.Row(min_height=40): pass
                    add_positive_point = gr.Button("add positive point ", variant="huggingface")
                    add_negative_point = gr.Button("add negative point", variant="huggingface")
                    with gr.Row(min_height=40): pass
                    with gr.Row():
                        propagate = gr.Button("propagate", variant="primary")
                        clear_propagate = gr.Button("clear propagate", variant="stop")
                    with gr.Row(min_height=40): pass
                    with gr.Row():
                        export_button = gr.Button("export", variant="secondary")
                        export_info = gr.Textbox(value="export info", lines=1, interactive=False, show_label=False, container=False)

            image_input.select(get_select_coords, [slider, all_points], [image_input, new_point])
            slider.change(video_slide, [slider, all_points], [image_input])
            
            load_model.click(model_load, [], [load_info])
            show_point.click(point_show, [new_point, slider, all_points], [image_input])
            refresh.click(refresh_image, [slider, all_points], [image_input])

            add_positive_point.click(partial(point_add, label=1), [all_points, new_point, slider], [image_input, all_points])
            add_negative_point.click(partial(point_add, label=0), [all_points, new_point, slider], [image_input, all_points])

            propagate.click(video_propagate, [all_points, slider], [image_input])
            clear_propagate.click(clear_video_propagate, [all_points, slider], [image_input])

            export_button.click(export_result, [], [export_info, video_output])

    app.queue().launch(server_name="127.0.0.1", inbrowser=True, share=False, server_port=9874, quiet=True)
