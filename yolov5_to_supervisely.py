import json
import os
import shutil
from pathlib import Path

import yolov5
from PIL import Image
from sahi.utils.file import list_files
from tqdm import tqdm
from yolov5.utils.dataloaders import IMG_FORMATS

WEIGHTS = "yolov5s.pt"
DEVICE = "cuda:0"
SOURCE_DIR = "images"
SAVE_DIR = "results/"
IMAGE_SIZE = 640
CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.45
META_JSON_PATH = "meta.json"


def get_class_title_to_id_from_meta_json(meta_json):
    class_title_to_id = {
        class_["title"]: class_["id"] for class_ in meta_json["classes"]
    }
    return class_title_to_id


def create_dir(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        print(f"folder created at {newpath}")
    else:
        print(f"{newpath} folder already exist")


def predict_and_export(
    supervisely_meta_json_path,
    weights="best.pt",
    device="cpu",
    source_dir="test_images/",
    save_dir="results/",
    img_size=640,
    conf_thres=0.1,
    iou_thres=0.45,
):
    with open(supervisely_meta_json_path) as json_file:
        meta_json = json.load(json_file)

    meta_export_path = Path(save_dir) / "meta.json"

    save_dir = Path(save_dir) / "dataset"

    # Initialize directories
    save_dir_img = Path(save_dir) / "img"
    create_dir(save_dir_img)
    save_dir_ann = Path(save_dir) / "ann"
    create_dir(save_dir_ann)

    class_title_to_id = get_class_title_to_id_from_meta_json(meta_json)
    model = yolov5.load(model_path=weights, device=device)
    model.conf = conf_thres
    model.iou = iou_thres
    image_files = list_files(source_dir, contains=IMG_FORMATS)

    for frame_path in tqdm(image_files):
        width, height = Image.open(frame_path).size

        json_frame = {}
        json_frame["description"] = ""
        json_frame["tags"] = []
        json_frame["size"] = {"height": height, "width": width}
        json_frame["objects"] = []

        pred = model(frame_path, augment=False, size=img_size)

        for ind in range(len(pred.xyxy[0])):
            xyxy = pred.xyxy[0][ind][:4]
            frame_id = Path(frame_path).stem

            class_title = pred.names[ind]
            class_id = class_title_to_id[class_title]

            json_frame["objects"].append(
                {
                    "id": f"{frame_id}{ind}",
                    "classId": class_id,
                    "description": "",
                    "geometryType": "rectangle",
                    "labelerLogin": "account@gmail.com",
                    "createdAt": "2021-09-14T13:35:58.348Z",
                    "updatedAt": "2021-09-14T13:36:20.612Z",
                    "tags": [],
                    "classTitle": class_title,
                    "points": {
                        "exterior": [
                            [int(xyxy[0]), int(xyxy[1])],
                            [int(xyxy[2]), int(xyxy[3])],
                        ],
                        "interior": [],
                    },
                }
            )

        # save image
        shutil.copyfile(frame_path, save_dir_img / Path(frame_path).name)

        # save json ann
        with open(f"{save_dir_ann}/{Path(frame_path).stem}.json", "w") as fp:
            json.dump(json_frame, fp, indent=4)

    with open(meta_export_path, "w") as fp:
        json.dump(meta_json, fp, indent=4)


if __name__ == "__main__":
    predict_and_export(
        META_JSON_PATH,
        WEIGHTS,
        DEVICE,
        SOURCE_DIR,
        SAVE_DIR,
        IMAGE_SIZE,
        CONF_THRESHOLD,
        IOU_THRESHOLD,
    )
