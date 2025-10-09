import os
import cv2
import yaml
import argparse
from pathlib import Path
import random


class YoloVisualizer:
    MODE_TRAIN = 0
    MODE_VAL = 1
    MODE_TEST = 2

    def __init__(self, data_config, start_mode='val', shuffle=False):
        self.config = data_config
        self.class_names = list(self.config['names'])
        self.class_map = {i: c for i, c in enumerate(self.class_names)}
        # assign distinct colors
        random.seed(42)
        self.colors = {i: (int(random.randint(80,255)), int(random.randint(80,255)), int(random.randint(80,255))) for i in range(len(self.class_names))}
        self.shuffle = shuffle
        self.set_mode(start_mode)

    def _collect(self, root: Path):
        img_dir = root / 'images'
        lbl_dir = root / 'labels'
        images = [p for p in img_dir.glob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        images.sort()
        if self.shuffle:
            random.shuffle(images)
        pairs = []
        for img in images:
            label = lbl_dir / (img.stem + '.txt')
            pairs.append((img, label if label.exists() else None))
        return pairs

    def set_mode(self, mode_str='val'):
        self.mode_str = mode_str
        if mode_str == 'train':
            self.mode = self.MODE_TRAIN
            path_key = 'train'
        elif mode_str == 'val':
            self.mode = self.MODE_VAL
            path_key = 'val'
        else: # test
            self.mode = self.MODE_TEST
            path_key = 'test'

        path_val = self.config.get(path_key)
        if not path_val:
            raise FileNotFoundError(f"Dataset config missing path for split: '{path_key}'")

        # Handle single path or list of paths
        paths = [Path(p) for p in path_val] if isinstance(path_val, list) else [Path(path_val)]
        
        self.samples = []
        for p in paths:
            if not p.exists():
                print(f"[WARNING] Dataset split folder missing: {p}")
                continue
            self.samples.extend(self._collect(p))

        if not self.samples:
            raise RuntimeError(f"No images found in any specified '{mode_str}' paths.")
        
        if self.shuffle:
            random.shuffle(self.samples)

        self.frame_index = 0

    def next_frame(self):
        self.frame_index = (self.frame_index + 1) % len(self.samples)

    def previous_frame(self):
        self.frame_index = (self.frame_index - 1) % len(self.samples)

    def seek_frame(self, idx):
        img_path, lbl_path = self.samples[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            return None
        lines = []
        if lbl_path is not None:
            try:
                with open(lbl_path, 'r') as f:
                    lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            except Exception:
                pass
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            class_index, x, y, w, h = map(float, parts)
            class_index = int(class_index)
            h_img, w_img = image.shape[:2]
            cx = int(x * w_img)
            cy = int(y * h_img)
            w_px = int(w * w_img)
            h_px = int(h * h_img)
            x1 = max(cx - w_px // 2, 0)
            y1 = max(cy - h_px // 2, 0)
            x2 = min(x1 + w_px, w_img - 1)
            y2 = min(y1 + h_px, h_img - 1)
            color = self.colors.get(class_index, (0, 255, 0))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label_text = self.class_map.get(class_index, str(class_index))
            cv2.putText(image, label_text, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return image

    def overlay_info(self, frame):
        if frame is None:
            return frame
        h, w = frame.shape[:2]
        info = [
            f"Mode: {self.mode_str.upper()}",
            f"Image: {self.frame_index+1}/{len(self.samples)}",
            "Keys: [d] next  [a] prev  [t]rain [v]al [e]test [q]uit"
        ]
        y = 18
        for line in info:
            cv2.putText(frame, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            y += 16
        return frame

    def run(self, window='YOLO Dataset Visualizer', resize=(960, 720), autoplay=False, delay=750):
        try:
            while True:
                frame = self.seek_frame(self.frame_index)
                if frame is None:
                    self.next_frame()
                    continue
                if resize:
                    frame = cv2.resize(frame, resize)
                frame = self.overlay_info(frame)
                cv2.imshow(window, frame)
                key = cv2.waitKey(delay if autoplay else 0) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord('d'):
                    self.next_frame()
                elif key == ord('a'):
                    self.previous_frame()
                elif key == ord('t'):
                    self.set_mode('train')
                elif key == ord('v'):
                    self.set_mode('val')
                elif key == ord('e'):
                    self.set_mode('test')
                elif key == ord(' '):  # toggle autoplay
                    autoplay = not autoplay
        finally:
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='Path to dataset YAML file')
    ap.add_argument('--mode', type=str, default='val', choices=['train', 'val', 'test'], help='Dataset split to visualize')
    ap.add_argument('--shuffle', action='store_true', help='Shuffle dataset order')
    ap.add_argument('--autoplay', action='store_true', help='Automatically cycle through images')
    ap.add_argument('--delay', type=int, default=800, help='Delay in ms for autoplay')
    ap.add_argument('--size', type=int, default=1024, help='Display window width')
    args = ap.parse_args()

    try:
        with open(args.data, 'r') as f:
            data_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return

    # Resolve relative paths in YAML
    yaml_parent = Path(args.data).parent
    for key in ['train', 'val', 'test']:
        if key in data_config:
            path_val = data_config[key]
            if isinstance(path_val, list):
                 data_config[key] = [str((yaml_parent / p).resolve()) for p in path_val]
            else:
                 data_config[key] = str((yaml_parent / path_val).resolve())

    try:
        vis = YoloVisualizer(data_config, start_mode=args.mode, shuffle=args.shuffle)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        return

    cv2.namedWindow('YOLO Visualizer', cv2.WINDOW_NORMAL)

    while True:
        frame = vis.seek_frame(vis.frame_index)
        if frame is None:
            print(f"Warning: Could not load image at index {vis.frame_index}")
            vis.next_frame()
            continue

        frame = vis.overlay_info(frame)
        
        # Resize for display
        h, w = frame.shape[:2]
        new_w = args.size
        new_h = int(h * (new_w / w))
        cv2.resizeWindow('YOLO Visualizer', new_w, new_h)
        cv2.imshow('YOLO Visualizer', frame)

        key = cv2.waitKey(args.delay if args.autoplay else 0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('d') or (args.autoplay and key == -1):
            vis.next_frame()
        elif key == ord('a'):
            vis.previous_frame()
        elif key == ord('t'):
            vis.set_mode('train')
        elif key == ord('v'):
            vis.set_mode('val')
        elif key == ord('e'):
            vis.set_mode('test')

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
