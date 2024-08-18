import os
import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_video_predictor
from cog import BasePredictor, Input, Path
import logging
import shutil
import subprocess
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

class Predictor(BasePredictor):
    def setup(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.info("Starting setup")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

        self.checkpoint = "/sam2_hiera_large.pt"
        self.model_cfg = "sam2_hiera_l.yaml"

        try:
            self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint)
            logging.info("SAM2 predictor built successfully")
        except Exception as e:
            logging.exception(f"Error building SAM2 predictor: {e}")
            raise

        # Load a pre-trained Faster R-CNN model for body detection
        self.body_detector = fasterrcnn_resnet50_fpn(pretrained=True)
        self.body_detector.eval()
        self.body_detector.to(self.device)

        logging.info("Setup completed")

    def predict(
            self,
            input_video: Path = Input(description="Input video file"),
            bg_color: str = Input(description="Background color (hex code)", default="#00FF00")
    ) -> Path:
        bg_color = tuple(int(bg_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))[::-1]  # BGR for OpenCV

        frames_dir = "/frames"
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)

        logging.info(f"Input video path: {input_video}")
        logging.info(f"Input video exists: {os.path.exists(input_video)}")
        logging.info(f"Input video file size: {os.path.getsize(input_video)} bytes")

        try:
            ffmpeg_cmd = [
                "ffmpeg", "-i", str(input_video), "-q:v", "2", "-start_number", "0",
                f"{frames_dir}/%05d.jpg"
            ]
            logging.info(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            logging.info("FFmpeg command executed successfully")
            logging.debug(f"FFmpeg stdout: {result.stdout}")
            logging.debug(f"FFmpeg stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg command failed: {e.stderr}")
            raise RuntimeError(f"Failed to extract frames from video: {e.stderr}")

        frame_names = [p for p in os.listdir(frames_dir) if p.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG'))]
        logging.info(f"Number of frames extracted: {len(frame_names)}")

        if not frame_names:
            logging.error(f"No frames were extracted. Contents of {frames_dir}: {os.listdir(frames_dir)}")
            raise RuntimeError(
                f"No frames were extracted from the video. The video file may be corrupt or in an unsupported format.")

        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        try:
            inference_state = self.predictor.init_state(video_path=frames_dir)
            logging.info("Inference state initialized successfully")
        except Exception as e:
            logging.exception(f"Error initializing inference state: {e}")
            raise

        first_frame_path = os.path.join(frames_dir, frame_names[0])
        logging.info(f"Attempting to read first frame: {first_frame_path}")
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            logging.error(f"Failed to read the first frame. File exists: {os.path.exists(first_frame_path)}")
            raise RuntimeError(f"Failed to read the first frame: {frame_names[0]}")

        # Detect body keypoints in the first frame
        keypoints = self.detect_body_keypoints(first_frame)
        logging.info(f"Detected {len(keypoints)} keypoints")

        try:
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=keypoints,
                labels=np.ones(len(keypoints), dtype=np.int32),  # All points are positive
            )
            logging.info("New points added successfully")
        except Exception as e:
            logging.exception(f"Error adding new points: {e}")
            raise

        video_segments = {}
        try:
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: out_mask_logits[i].cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            logging.info("Video segments propagated successfully")
        except Exception as e:
            logging.exception(f"Error propagating video segments: {e}")
            raise

        output_frames_dir = '/output_frames'
        os.makedirs(output_frames_dir, exist_ok=True)

        frame_count = 0
        for out_frame_idx in range(len(frame_names)):
            frame_path = os.path.join(frames_dir, frame_names[out_frame_idx])
            frame = cv2.imread(frame_path)

            if frame is None:
                logging.error(f"Failed to read frame: {frame_path}")
                continue

            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                frame_with_bg_removed = self.remove_background(frame, out_mask, bg_color)

            output_frame_path = os.path.join(output_frames_dir, f"{out_frame_idx:05d}.jpg")
            cv2.imwrite(output_frame_path, frame_with_bg_removed)
            frame_count += 1

        output_video_path = '/output.mp4'

        try:
            final_video_cmd = [
                "ffmpeg", "-y",  # Add -y flag to force overwrite without prompting
                "-framerate", "30",
                "-i", f"{output_frames_dir}/%05d.jpg",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_video_path
            ]
            logging.info(f"Running final FFmpeg command: {' '.join(final_video_cmd)}")

            result = subprocess.run(final_video_cmd, capture_output=True, text=True, check=True)
            logging.info("Final video created successfully")
            logging.debug(f"Final FFmpeg stdout: {result.stdout}")
            logging.debug(f"Final FFmpeg stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error creating final video: {e.stderr}")
            raise RuntimeError(f"Failed to create final video: {e.stderr}")

        logging.info(f"Processed {frame_count} frames")
        logging.info(f"Background removed video saved as {output_video_path}")

        return Path(output_video_path)

    def detect_body_keypoints(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to tensor
        img_tensor = F.to_tensor(frame_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.body_detector(img_tensor)[0]

        # Get the bounding box with the highest score
        if len(prediction['boxes']) > 0:
            best_box = prediction['boxes'][prediction['scores'].argmax()].cpu().numpy()
            x1, y1, x2, y2 = best_box

            # Calculate center of the bounding box
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

            # Calculate the dimensions of the bounding box
            width, height = x2 - x1, y2 - y1

            # Define offset for surrounding points (20% of width/height)
            offset_x, offset_y = width * 0.2, height * 0.2

            # Define keypoints
            keypoints = np.array([
                [center_x, center_y],  # Center
                [center_x - offset_x, center_y],  # Left
                [center_x + offset_x, center_y],  # Right
                [center_x, center_y - offset_y],  # Top
                [center_x, center_y + offset_y],  # Bottom
            ], dtype=np.float32)

            # Ensure all points are within the bounding box
            keypoints[:, 0] = np.clip(keypoints[:, 0], x1, x2)
            keypoints[:, 1] = np.clip(keypoints[:, 1], y1, y2)

            return keypoints
        else:
            # If no person is detected, fall back to center point
            height, width = frame.shape[:2]
            center = np.array([[width // 2, height // 2]], dtype=np.float32)
            return np.tile(center, (5, 1))  # Return 5 identical center points as fallback

    def remove_background(self, frame, mask, bg_color):
        mask = mask.squeeze()
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        else:
            mask = (mask > 0).astype(np.uint8) * 255

        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create background
        bg = np.full(frame.shape, bg_color, dtype=np.uint8)

        # Apply the mask to keep the person and remove the background
        fg = cv2.bitwise_and(frame, frame, mask=mask)
        bg = cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(mask))

        # Combine foreground and background
        result = cv2.add(fg, bg)

        # Clean up the hair area
        result = self.clean_hair_area(frame, result, mask, bg_color)

        return result

    def clean_hair_area(self, original, processed, mask, bg_color):
        # Create a dilated mask to capture the hair edges
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)
        hair_edge_mask = cv2.subtract(dilated_mask, mask)

        # Calculate the average color of the removed background
        bg_sample = cv2.bitwise_and(original, original, mask=cv2.bitwise_not(dilated_mask))
        bg_average = cv2.mean(bg_sample)[:3]

        # Create a color distance map
        color_distances = np.sqrt(np.sum((original.astype(np.float32) - bg_average) ** 2, axis=2))

        # Normalize color distances
        color_distances = (color_distances - color_distances.min()) / (color_distances.max() - color_distances.min())

        # Create an alpha mask based on color distance
        alpha = (1 - color_distances) * (hair_edge_mask / 255.0)
        alpha = np.clip(alpha, 0, 1)

        # Blend the hair edge area
        for c in range(3):
            processed[:, :, c] = processed[:, :, c] * (1 - alpha) + bg_color[c] * alpha

        return processed