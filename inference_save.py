import asyncio
import time
import cv2
from model_utils import predictor
from ultralytics import YOLO
from coordinates import coordinates


async def save_rtsp_video(video_output, frame):
    video_output.write(frame)


async def inference_video():
    camera = 1
    multipier = 1.5
    model = predictor.YOLO("weights/bestv3.pt")
    cap = cv2.VideoCapture("media/2024-10-15/12-10.mp4")
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_output = cv2.VideoWriter("result/12-10.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 20, frame_size)

    cv2.namedWindow("frame", cv2.WINDOW_FULLSCREEN | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    count = 0
    start = time.time()
    while True:
        ret, frame = cap.read()
        if ret:
            results = model.track(frame, half=True, verbose=False, device="cuda")
            count += 1
            if count % 100 == 0:
                print(f"FPS: {100/(time.time() - start):.1f}")
                start = time.time()

            annotated_frame = results[0].plot()
            cv2.rectangle(annotated_frame, coordinates[camera][:-2], coordinates[camera][2:], (0, 0, 255), 2)
            await save_rtsp_video(video_output, annotated_frame)
        else:
            break

    cap.release()
    video_output.release()

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(inference_video())
    loop.close()
