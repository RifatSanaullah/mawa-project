import time
import cv2
from sympy import false

from model_utils import predictor
from ultralytics import YOLO
from coordinates import frame_coordinates, crop_coordinates

lanes_dict: dict = {
    "L3": 1, "L10": 2, "L1": 3, "L9": 4, "L8": 5, "L7": 6, "L2": 7, "L12": 8, "L6": 9, "L4": 10, "L5": 11, "L11": 12,
}

if __name__ == '__main__':
    lane = "L9"
    camera = lanes_dict.get(lane)
    model1 = predictor.YOLO("weights/bestv2.engine")
    model2 = predictor.YOLO("weights/bestv4.engine")
    # cap = cv2.VideoCapture(f"rtsp://admin:Mawa0304@119.148.25.122:554/Streaming/tracks/{camera}01?starttime=20241025T151000Z")
    cap = cv2.VideoCapture(f"rtsp://admin:Mawa0304@119.148.25.122:554/Streaming/Channels/{camera}01")
    # cap = cv2.VideoCapture("media/ch0004_20241004T011951Z_20241004T015546Z_X03000003329000000.mp4")

    count = 0
    start = time.time()
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
            frame = frame[
                    :, int(frame_coordinates[camera][0]):int(frame_coordinates[camera][2])
                    ]
            results1 = model1.track(frame, half=True, verbose=False, device="cuda")
            results2 = model2.track(frame, half=True, verbose=False, device="cuda", agnostic_nms=True)
            count += 1
            if count % 100 == 0:
                print(f"FPS: {100 / (time.time() - start):.1f}")
                start = time.time()

            annotated_frame1 = results1[0].plot()
            annotated_frame2 = results2[0].plot()

            x1, y1, x2, y2 = 0, crop_coordinates[camera][1], frame_coordinates[camera][2] - frame_coordinates[camera][0], crop_coordinates[camera][3]
            cv2.rectangle(annotated_frame1, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(annotated_frame2, (x1, y1), (x2, y2), (0, 0, 255), 2)
            im_h = cv2.hconcat([annotated_frame1, annotated_frame2])

            cv2.imshow("v2 ----- v4", im_h)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
