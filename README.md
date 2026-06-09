<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_boxmot_tracking</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_boxmot_tracking">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_boxmot_tracking">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_boxmot_tracking/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_boxmot_tracking.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Multiple object tracking algorithm using BoxMOT trackers for object detection, instance segmentation and keypoints.

BoxMOT provides several tracking algorithms such as OccluBoost, BoT-SORT, BoostTrack, StrongSORT, DeepOCSORT, ByteTrack, HybridSORT, OC-SORT and SFSORT.

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display
import cv2

# Init your workflow
wf = Workflow()

# Add object detection algorithm
detector = wf.add_task(name="infer_yolo_26", auto_connect=True)

# Add BoxMOT tracking algorithm
tracking = wf.add_task(name="infer_boxmot_tracking", auto_connect=True)

stream = cv2.VideoCapture(0)
while True:
    # Read image from stream
    ret, frame = stream.read()

    # Test if streaming is OK
    if not ret:
        continue

    # Run the workflow on current frame
    wf.run_on(array=frame)

    # Get results
    image_out = tracking.get_output(0)
    obj_track_out = tracking.get_output(1)

    # Display
    img_res = cv2.cvtColor(image_out.get_image_with_graphics(obj_track_out), cv2.COLOR_BGR2RGB)
    display(img_res, title="BoxMOT tracking", viewer="opencv")

    # Press 'q' to quit the streaming process
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the stream object
stream.release()
# Destroy all windows
cv2.destroyAllWindows()
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **tracker** (str) - Default 'occluboost': Tracking algorithm to use. Available values are 'occluboost', 'botsort', 'boosttrack', 'strongsort', 'deepocsort', 'bytetrack', 'hybridsort', 'ocsort' and 'sfsort'.
- **reid** (str) - Default 'osnet_x0_25_msmt17': Re-identification model used by trackers that support ReID.
- **config** (str) - Default 'None': Optional path to a custom tracker configuration file in YAML format.
- **categories** (str) - Default 'all': Categories of objects you want to track. Use a comma separated string to set multiple categories (ex: "dog,person,car").
- **cuda** (bool) - Default depends on CUDA availability: Enable CUDA acceleration when available.
- **half** (bool) - Default depends on CUDA availability: Enable half precision when CUDA is available.

Available ReID models in BoxMOT 21.0.0:

- `resnet50_market1501`
- `resnet50_dukemtmcreid`
- `resnet50_msmt17`
- `resnet50_fc512_market1501`
- `resnet50_fc512_dukemtmcreid`
- `resnet50_fc512_msmt17`
- `mlfn_market1501`
- `mlfn_dukemtmcreid`
- `mlfn_msmt17`
- `hacnn_market1501`
- `hacnn_dukemtmcreid`
- `hacnn_msmt17`
- `mobilenetv2_x1_0_market1501`
- `mobilenetv2_x1_0_dukemtmcreid`
- `mobilenetv2_x1_0_msmt17`
- `mobilenetv2_x1_4_market1501`
- `mobilenetv2_x1_4_dukemtmcreid`
- `mobilenetv2_x1_4_msmt17`
- `osnet_x1_0_market1501`
- `osnet_x1_0_dukemtmcreid`
- `osnet_x1_0_msmt17`
- `osnet_x0_75_market1501`
- `osnet_x0_75_dukemtmcreid`
- `osnet_x0_75_msmt17`
- `osnet_x0_5_market1501`
- `osnet_x0_5_dukemtmcreid`
- `osnet_x0_5_msmt17`
- `osnet_x0_25_market1501`
- `osnet_x0_25_dukemtmcreid`
- `osnet_x0_25_msmt17`
- `osnet_ibn_x1_0_msmt17`
- `osnet_ain_x1_0_msmt17`
- `lmbn_n_duke`
- `lmbn_n_market`
- `lmbn_n_cuhk03_d`
- `clip_market1501`
- `clip_duke`
- `clip_veri`
- `clip_vehicleid`

This list is read from BoxMOT at runtime and may change with future BoxMOT releases. Check the upstream [mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot) repository for the latest available models.

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add detection algorithm
detector = wf.add_task(name="infer_yolo_26", auto_connect=True)

# Add tracking algorithm
tracking = wf.add_task(name="infer_boxmot_tracking", auto_connect=True)

tracking.set_parameters({
    "tracker": "bytetrack",
    "categories": "person,car",
    "cuda": "True",
    "half": "True",
})
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add detection algorithm
detector = wf.add_task(name="infer_yolo_26", auto_connect=True)

# Add tracking algorithm
tracking = wf.add_task(name="infer_boxmot_tracking", auto_connect=True)

# Run on your image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_work.jpg")

# Iterate over outputs
for output in tracking.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

## :fast_forward: Advanced usage 
