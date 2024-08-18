import os
import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_video_predictor
from cog import BasePredictor, Input, Path
import logging
import shutil

class Predictor(BasePredictor):
    def setup(self):
        logging.basicConfig(level=logging.INFO)
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
            logging.error(f"Error building SAM2 predictor: {e}")
            raise

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

        os.system(f"ffmpeg -i {input_video} -q:v 2 -start_number 0 {frames_dir}/%05d.jpg")

        frame_names = [p for p in os.listdir(frames_dir) if p.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG'))]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        try:
            inference_state = self.predictor.init_state(video_path=frames_dir)
            logging.info("Inference state initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing inference state: {e}")
            raise

        first_frame = cv2.imread(os.path.join(frames_dir, frame_names[0]))
        height, width, _ = first_frame.shape
        center_x, center_y = width // 2, height // 2

        points = np.array([[center_x, center_y]], dtype=np.float32)
        labels = np.array([1], np.int32)

        try:
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
            )
            logging.info("New points added successfully")
        except Exception as e:
            logging.error(f"Error adding new points: {e}")
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
            logging.error(f"Error propagating video segments: {e}")
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

        # Use FFmpeg to create the final video
        os.system(
            f"ffmpeg -framerate 30 -i {output_frames_dir}/%05d.jpg -c:v libx264 -pix_fmt yuv420p {output_video_path}")

        logging.info(f"Processed {frame_count} frames")
        logging.info(f"Background removed video saved as {output_video_path}")

        return Path(output_video_path)

    def remove_background(self, frame, mask, bg_color):
        mask = mask.squeeze()
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        else:
            mask = (mask > 0).astype(np.uint8) * 255

        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        bg = np.full(frame.shape, bg_color, dtype=np.uint8)
        result = np.where(mask[:, :, None] == 255, frame, bg)

        return result