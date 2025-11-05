from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml
import argparse
import time


# Function to predict and save images
def predict_and_save(model, image_path, output_path, output_path_txt):
    # Perform prediction
    results = model.predict(image_path,conf=0.5)

    result = results[0]
    # Draw boxes on the image
    img = result.plot()  # Plots the predictions directly on the image

    # Save the result
    cv2.imwrite(str(output_path), img)
    # Save the bounding box data
    with open(output_path_txt, 'w') as f:
        for box in result.boxes:
            # Extract the class id and bounding box coordinates
            cls_id = int(box.cls)
            x_center, y_center, width, height = box.xywhn[0].tolist()
            
            # Write bbox information in the format [class_id, x_center, y_center, width, height]
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run YOLO predictions on test set and report metrics.")
    parser.add_argument('--run', type=str, default=None, help='Explicit training run folder name under runs/detect (e.g. train2)')
    parser.add_argument('--latest', action='store_true', help='Automatically pick the most recently modified train* folder')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for predictions')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit on number of images to process')
    args = parser.parse_args()

    this_dir = Path(__file__).parent
    os.chdir(this_dir)
    with open(this_dir / 'yolo_params.yaml', 'r') as file:
        data_cfg = yaml.safe_load(file)
        if 'test' in data_cfg and data_cfg['test'] is not None:
            images_dir = Path(data_cfg['test']) / 'images'
        else:
            raise SystemExit("No test field found in yolo_params.yaml; add a test split path.")

    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"Images directory {images_dir} missing or not a directory")
    img_list = [p for p in images_dir.glob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    if not img_list:
        raise SystemExit(f"Images directory {images_dir} is empty")
    if args.limit:
        img_list = img_list[:args.limit]

    detect_path = this_dir / 'runs' / 'detect'
    train_folders = [f for f in os.listdir(detect_path) if (detect_path / f).is_dir() and f.startswith('train')]
    if not train_folders:
        raise SystemExit('No training folders found under runs/detect')

    selected = None
    if args.run:
        if args.run not in train_folders:
            raise SystemExit(f"Specified run '{args.run}' not found. Available: {train_folders}")
        selected = args.run
    elif args.latest:
        selected = max(train_folders, key=lambda d: (detect_path / d).stat().st_mtime)
    elif len(train_folders) == 1:
        selected = train_folders[0]
    else:
        # Fallback to previous interactive selection if neither --run nor --latest supplied.
        choices = list(range(len(train_folders)))
        choice = -1
        while choice not in choices:
            print("Select the training folder:")
            for i, folder in enumerate(train_folders):
                print(f"{i}: {folder}")
            raw = input('Choice: ')
            if raw.isdigit():
                choice = int(raw)
        selected = train_folders[choice]

    model_path = detect_path / selected / 'weights' / 'best.pt'
    if not model_path.exists():
        raise SystemExit(f"Weights file not found: {model_path}")
    print(f"Using run: {selected} -> {model_path}")
    model = YOLO(str(model_path))

    # Output directories
    output_dir = this_dir / 'predictions'
    images_output_dir = output_dir / 'images'
    labels_output_dir = output_dir / 'labels'
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    processed = 0
    for img_path in img_list:
        out_img = images_output_dir / img_path.name
        out_txt = labels_output_dir / img_path.with_suffix('.txt').name
        predict_and_save(model, img_path, out_img, out_txt)
        processed += 1
    dur = time.time() - start

    print(f"Processed {processed} images in {dur:.2f}s ({processed/dur if dur>0 else 0:.2f} img/s)")
    print(f"Predicted images -> {images_output_dir}")
    print(f"Prediction labels -> {labels_output_dir}")
    data_yaml = this_dir / 'yolo_params.yaml'
    print(f"Dataset config -> {data_yaml}")

    # Validation on test split for status metrics
    print('Running validation on test split...')
    metrics = model.val(data=str(data_yaml), split='test')
    # Ultralytics metrics access pattern
    try:
        mp = metrics.box.map  # mean AP 0.5:0.95
        mp50 = metrics.box.map50
        mp75 = metrics.box.map75
        maps = metrics.box.maps  # per-class
        print(f"Model status metrics: mAP50-95={mp:.4f} mAP50={mp50:.4f} mAP75={mp75:.4f}")
        if maps is not None:
            print(f"Per-class APs: {[round(m,4) for m in maps]}")
    except AttributeError:
        print('Warning: Could not extract detailed metrics from validation result.')

    print('Prediction + validation complete.')

