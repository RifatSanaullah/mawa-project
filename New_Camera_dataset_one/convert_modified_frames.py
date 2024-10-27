import cv2
import os

video_folder = '/home/mehedi/PycharmProjects/YOLO/New_Camera_dataset_one/'
frame_interval = 80

video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file {video_file}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing video: {video_file}")
    print(f"Original FPS: {fps}")

    frame_count = 0

    output_dir = os.path.join(video_folder, video_file[:-4])  # Remove the .mp4 extension
    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_filename, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()
    print(f"Finished processing {video_file}")
    print(f"Frames saved to: {output_dir}")

cv2.destroyAllWindows()
