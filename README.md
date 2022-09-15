# yolov5-to-supervisely

Use your yolov5 predictions as supervisely annotations

# setup

- clone this repo

- install dependencies:

```bash
pip install -r requirements.txt
```

# usage

- set meta.json file

- set `WEIGHTS`, `DEVICE`, `SOURCE_DIR`, `SAVE_DIR`, `IMAGE_SIZE`, `CONF_THRESHOLD`, `IOU_THRESHOLD`, `META_JSON_PATH` params in  [yolov5_to_supervisely.py](yolov5_to_supervisely.py).

- run:

```bash
python yolov5_to_supervisely.py
```

- upload output folder into supervisely via `Import Via Apps/Import Images in Supervisely Format`