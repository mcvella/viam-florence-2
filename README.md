# florence-2 modular service

This module implements the [rdk vision API](https://github.com/rdk/vision-api) in a mcvella:vision:florence-2 model.

This model leverages the [Florence-2 computer vision model](https://huggingface.co/microsoft/Florence-2-large) to allow for object detection, grounding detection, classification(captioning), and segmentation.

The Florence-2 model and inference will run locally, and therefore speed of inference is highly dependant on hardware.
Cuda GPU support exists, but can run on CPU.
Metal GPU is not currently supported.

## Build and Run

To use this module, follow these instructions to [add a module from the Viam Registry](https://docs.viam.com/registry/configure/#add-a-modular-resource-from-the-viam-registry) and select the `florence-2` model from the [`mcvella:vision:florence-2` module](https://app.viam.com/module/rdk/mcvella:vision:florence-2).

## Configure your vision service

> [!NOTE]  
> Before configuring your vision, you must [create a machine](https://docs.viam.com/manage/fleet/machines/#add-a-new-machine).

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/).
Click on the **Components** subtab and click **Create component**.
Select the `vision` type, then select the `mcvella:vision:florence-2` model.
Enter a name for your vision and click **Create**.

On the new component panel, copy and paste the following attribute template into your vision’s **Attributes** box:

```json
{
  "default_query": "house. car. zebra.",
  "caption_detail": "low"
}
```

> [!NOTE]  
> For more information, see [Configure a Robot](https://docs.viam.com/manage/configuration/).

### Attributes

The following attributes are available for `rdk:vision:mcvella:vision:florence-2`:

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `model_id` | string | Optional |  The HuggingFace model ID for the florence-2 model |
| `default_query` | string | Optional |  A list of grounding classes to look for in images. Each class must end in a period. Note that multi-word classes will often be detected as the base word, for example "man cooking" might detect "man". This is relevant only with detections, not classifications. If not set, all detected objects will be returned.|
|`caption_detail`| string | Optional | low, medium, or high (default low) For classification(captioning), this is the level of detail for the caption generated. Relevant for classifications only. |
|`detection_as_segmentation`| boolean | Optional | Defaults to false.  If set to true, will return a single segmentation array as single 1x1 pixel detections of the class specified. To use this, you'll likely need to transform it into another segmentation mask format. |

### Example Configuration

```json
{
  "default_query": "road. car. stop sign."
}
```

## API

The florence-2 resource provides the following methods from Viam's built-in [rdk:service:vision API](https://python.viam.dev/autoapi/viam/services/vision/client/index.html)

### get_detections(image=*binary*)

### get_detections_from_camera(camera_name=*string*)

### get_classifications(image=*binary*)

### get_classifications_from_camera(camera_name=*string*)

Note: if using this method, any cameras you are using must be set in the `depends_on` array for the service configuration, for example:

```json
      "depends_on": [
        "cam"
      ]
```

If you want to look for different grounding detection classes in an image, you can pass a different query as an extra parameter "query".
For example:

``` python
service.get_detections(image, extra={"query": "dog. cat. rat."})
```

If you want change the caption detail level for an image classification, you can pass a different query as an extra parameter "detail".
For example:

``` python
service.get_detections(image, extra={"detail": "high"})
```

If you want override the setting *detection_as_segmentation*, you can pass the extra parameter "segmentation" to get_detections.
You must also either pass a *query* or have *default_query* set.
For example:

``` python
service.get_detections(image, extra={"query": "person", "segmentation": true})
```
