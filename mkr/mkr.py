from roboflow import Roboflow

rf = Roboflow(api_key="")
project = rf.workspace("myworkspace-cnsj5").project("my-first-project-xjias")
version = project.version(2)
dataset = version.download("yolov11")

import albumentations as A
import cv2
import os
import glob
from tqdm import tqdm

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.RGBShift(p=0.2),
    A.HueSaturationValue(p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels']))

train_images_path = os.path.join(dataset.location, "train/images")
train_labels_path = os.path.join(dataset.location, "train/labels")

images = glob.glob(os.path.join(train_images_path, "*.jpg")) + glob.glob(os.path.join(train_images_path, "*.png"))

print(f"Починаємо аугментацію {len(images)} зображень...")

count = 0
for img_path in tqdm(images):
    image = cv2.imread(img_path)
    if image is None: continue
    h, w, _ = image.shape

    label_path = img_path.replace("images", "labels").rsplit('.', 1)[0] + ".txt"
    if not os.path.exists(label_path): continue

    bboxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            bboxes.append(coords)
            class_labels.append(cls)

    if not bboxes: continue

    try:
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_img = transformed['image']
        aug_bboxes = transformed['bboxes']

        if not aug_bboxes: continue

        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        new_img_name = f"{name}_aug{ext}"
        new_txt_name = f"{name}_aug.txt"

        cv2.imwrite(os.path.join(train_images_path, new_img_name), aug_img)

        with open(os.path.join(train_labels_path, new_txt_name), 'w') as f:
            for bbox, cls in zip(aug_bboxes, class_labels):
                line = f"{cls} {' '.join(map(str, bbox))}\n"
                f.write(line)
        count += 1
    except Exception as e:
        pass

print(f"\nГотово! Створено {count} нових зображень.")



from ultralytics import YOLO

model = YOLO('yolo11n.pt')



model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    project='/kaggle/working/runs',
    name='my_yolo_model'
)



from ultralytics import YOLO


model = YOLO('/kaggle/working/runs/my_yolo_model5/weights/best.pt')


video_path = '/kaggle/input/test-video/test_video.mp4'


results = model.predict(
    source=video_path,
    save=True,
    conf=0.5,
    project='/kaggle/working/output_video',
    name='final_result'
)

print("Відео готове!")
