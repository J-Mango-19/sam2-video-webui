import os
import cv2

def extract_frames(raw_video, temp_img_dir):
    cap = cv2.VideoCapture(raw_video)
    os.system(f"rm {temp_img_dir}/*")
    count = 0
    success = True
    while success:
        success, frame = cap.read()
        if success:
            cv2.imwrite(os.path.join(temp_img_dir, f"{count}.jpg"), frame)
            count += 1
    cap.release()

if __name__ == "__main__":
    raw_video = '//home/work/disk/vision/video_inpaint/video/b_no_sub.mp4'
    temp_img_dir = '/home/work/disk/vision/video_inpaint/sample'
    extract_frames(raw_video, temp_img_dir)
