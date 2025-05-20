Sure! Here's a **`README.md`** for your fire and person detection project:

---

````markdown
# 🔥🧍 Fire and Person Detection using YOLOv3 and ViT

This project performs **real-time detection of people** using **YOLOv3**, and if any person is detected, it then performs **fire classification** on the entire frame using a **Vision Transformer (ViT)** model.

## 📷 Live Detection Features

- 🧍 Detects people using YOLOv3
- 🔥 Detects fire using the [EdBianchi/vit-fire-detection](https://huggingface.co/EdBianchi/vit-fire-detection) model
- 🎥 Real-time webcam processing using OpenCV
- 🚫 Shows alert if no person is detected
- ✅ Displays bounding boxes and labels

---

## 🧩 Requirements

Install all required packages using:

```bash
pip install -r requirements.txt
````

Or manually:

```bash
pip install opencv-python numpy torch torchvision pillow transformers
```

---

## 📁 Files

* `main.py` — The main Python script.
* `yolov3.cfg` — YOLOv3 model configuration file.
* `yolov3.weights` — YOLOv3 pre-trained weights **(do not upload to GitHub)**.
* `coco.names` — Class label file for YOLOv3.

---

## 📥 Download YOLOv3 Files

* **`yolov3.cfg`**:
  [https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
  (Right-click → Save As)

* **`yolov3.weights`**:
  [https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
  (248 MB)

* **`coco.names`**:
  [https://github.com/pjreddie/darknet/blob/master/data/coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

Place all files in the **same folder** as your `main.py`.

---

## 🚀 How to Run

```bash
python main.py
```

Press `q` to quit the video stream.

---

## 📌 Notes

* You **must have a webcam** connected.
* The fire detection is only triggered when a person is visible.
* Large model files like `yolov3.weights` should not be committed to GitHub. Add them to `.gitignore`.

---

---

## 📸 Example Output

> 🟩 Bounding boxes around people
> 🔴 "Fire detected!" warning if fire is present
> 🟡 "No person detected" message otherwise

---

## 🙌 Acknowledgements

* [YOLOv3 by Joseph Redmon](https://pjreddie.com/darknet/yolo/)
* [HuggingFace ViT Fire Detection Model](https://huggingface.co/EdBianchi/vit-fire-detection)
* [OpenCV](https://opencv.org/)

---

````

---

### ✅ Next steps

1. Save the content above as a file called `README.md` in your project folder.
2. Add and commit:

```bash
git add README.md
git commit -m "Add project README"
git push
````

