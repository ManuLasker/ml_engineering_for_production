import cv2
import io
import os
import numpy as np
import cvlib as cv
import uuid
from cvlib.object_detection import draw_bbox
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse, RedirectResponse


class Model(str, Enum):
    yolov3tiny = "yolov3-tiny"
    yolov3 = "yolov3"
    
    
def handle_file_name(image_file: UploadFile = File(...)) -> UploadFile:
    # 1. Validate input file to be an image
    filename = image_file.filename
    file_extension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not file_extension:
        raise HTTPException(status_code=415,
                            detail="Unsupported file provided: {}".format(filename.split(".")[-1]))
    return image_file

    
def download_yolo_files():
    dummy_image = cv2.imread("images/apple.jpg")
    bbox, label, conf = cv.detect_common_objects(dummy_image, model=Model.yolov3)
    bbox, label, conf = cv.detect_common_objects(dummy_image, model=Model.yolov3tiny)
    print(os.getcwd())
    os.makedirs("images_uploaded",
                exist_ok=True)

download_yolo_files()

app = FastAPI(title = "Deploying a ML Model with FastAPI")


    
@app.get("/", include_in_schema=False)
def docs_redirect():
    prefix = os.getenv("CLUSTER_ROUTE_PREFIX", "").strip("/")
    return RedirectResponse(prefix + "/docs")

@app.post("/predict")
def predict(model: Model, image_file: UploadFile = Depends(handle_file_name)):
    # 2. Transform raw image into cv2 image
    # Read image as a stream of bytes
    image_stream = io.BytesIO(image_file.file.read())
    # Start the stream from the beginnning (postition zero)
    image_stream.seek(0)
    # write the stram of bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    # Decode the numpy array as an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 3. Run Object detection model
    # run object detection model
    bbox, label, conf = cv.detect_common_objects(image, model=model)
    # create image that includes bounding box boxes and labels   
    output_image = draw_bbox(image, bbox, label, conf)
    # Save it in a folder within the server
    filename = uuid.uuid1().hex + "_{}.jpg"
    cv2.imwrite(f"images_uploaded/{filename.format('out')}", output_image)
    cv2.imwrite(f"images_uploaded/{filename.format('in')}", image)
    
    # 4. Stream response back to the client
    # open the saved image for reading in binary mode
    file_image = open(f"images_uploaded/{filename.format('out')}", mode="rb")
    
    # Return the image as a stream specifying meddia type
    return StreamingResponse(file_image, media_type="image/jpeg")