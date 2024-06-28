import subprocess
from typing import ClassVar, Mapping, Sequence, Any, Dict, Optional, Tuple, Final, List, cast
from typing_extensions import Self

import sys
from typing import Any, Final, List, Mapping, Optional, Union

from viam.media.video import ViamImage
from viam.proto.common import PointCloudObject
from viam.proto.service.vision import Classification, Detection, GetPropertiesResponse
from viam.resource.types import RESOURCE_NAMESPACE_RDK, RESOURCE_TYPE_SERVICE, Subtype

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from viam.utils import ValueTypes
from viam.module.types import Reconfigurable
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, Vector3
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily
from viam.services.vision import Vision, CaptureAllResult
from viam.logging import getLogger
from viam.media.utils.pil import viam_to_pil_image

from PIL import Image
from viam.components.camera import Camera, ViamImage
from transformers import AutoProcessor, AutoModelForCausalLM 
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch
import os
import torch

LOGGER = getLogger(__name__)

class florence2(Vision, Reconfigurable):
    MODEL: ClassVar[Model] = Model(ModelFamily("mcvella", "vision"), "florence-2")

    model: AutoModelForCausalLM
    processor: AutoProcessor
    default_query = ""
    caption_task = "<CAPTION>"

    # Constructor
    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        my_class = cls(config.name)
        my_class.reconfigure(config, dependencies)
        return my_class

    # Validates JSON Configuration
    @classmethod
    def validate(cls, config: ComponentConfig):
        return

    def fixed_get_imports(self, filename: str | os.PathLike) -> list[str]:
        """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
        if not str(filename).endswith("/modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        imports.remove("flash_attn")
        return imports

    # Handles attribute reconfiguration
    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        self.DEPS = dependencies
        model_id = config.attributes.fields["model_id"].string_value or "microsoft/Florence-2-large"
       
        # note: mps is currently not set up to work
        device = "cpu"
        if torch.cuda.is_available():  
            device = "cuda"
            subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

       
        if device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        else:
            with patch("transformers.dynamic_module_utils.get_imports", self.fixed_get_imports):
                self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
                self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        self.default_query = config.attributes.fields["default_query"].string_value or ""

        if config.attributes.fields["caption_detail"].string_value != "":
            if config.attributes.fields["caption_detail"].string_value == "low":
                self.caption_task = "<DETAILED_CAPTION>"
            elif config.attributes.fields["caption_detail"].string_value == "medium":
                self.caption_task = "<DETAILED_CAPTION>"
            elif config.attributes.fields["caption_detail"].string_value == "high":
                self.caption_task = "<MORE_DETAILED_CAPTION>"
        return

    
    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        result = CaptureAllResult()
        if return_image:
            result.image = await self.get_cam_image(camera_name)
        if return_detections:
            result.detections = await self.get_detections(result.image)
        if return_classifications:
            result.classifications = await self.get_classifications(result.image, 1)
        return result

    async def get_cam_image(
        self,
        camera_name: str
    ) -> ViamImage:
        actual_cam = self.DEPS[Camera.get_resource_name(camera_name)]
        cam = cast(Camera, actual_cam)
        cam_image = await cam.get_image(mime_type="image/jpeg")
        return cam_image
    
    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        return await self.get_detections(await self.get_cam_image(camera_name), extra=extra)


    async def perform_task(self, image, task_prompt, text_input=""):
        if text_input == "":
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        generated_ids = self.model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
        LOGGER.debug(parsed_answer)
        response = []
        if task_prompt == "<OD>" or task_prompt == "<CAPTION_TO_PHRASE_GROUNDING>":
            # example response 
            # {'<OD>': {'bboxes': [[724.4169921875, 1793.70458984375, 3891.385009765625, 2903.323486328125], 
            # [4548.47705078125, 2396.992431640625, 5146.3232421875, 2942.824462890625], 
            # [4144.52734375, 1668.01953125, 5377.92138671875, 2946.41552734375], 
            # [1607.7210693359375, 2009.16455078125, 2043.987060546875, 2271.3076171875]], 'labels': ['cat', 'flowerpot', 'houseplant', 'houseplant']}}	{"log_ts": "2024-06-28T11:23:38.060Z"}
            index = 0
            for label in parsed_answer[task_prompt]['labels']:
                detection = { "confidence": 1, "class_name": label, 
                             "x_min": int(parsed_answer[task_prompt]['bboxes'][index][0]), "y_min": int(parsed_answer[task_prompt]['bboxes'][index][1]), 
                             "x_max": int(parsed_answer[task_prompt]['bboxes'][index][2]), "y_max": int(parsed_answer[task_prompt]['bboxes'][index][3]) }
                response.append(detection)
                index = index + 1
        else:
            # {'<CAPTION>': '\na cat sitting on top of a couch next to a litter box\n'}
            class_name = parsed_answer[task_prompt]
            class_name = class_name.replace('\n','')
            response.append({"class_name": class_name, "confidence": 1})
        return response
    
    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        text = self.default_query
        if extra != None and extra.get('query') != None:
            text = extra['query']
        detections = []
        image = viam_to_pil_image(image)
        task = "<OD>"
        if text != "":
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
        results = await self.perform_task(image, task, text)
        return results

    
    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        return await self.get_classifications(await self.get_cam_image(camera_name), count, extra=extra)


    
    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        task = self.caption_task
        if extra != None and extra.get('detail') != None:
            if extra.get('detail') == "low":
                task = "<DETAILED_CAPTION>"
            elif extra.get('detail') == "medium":
                task = "<DETAILED_CAPTION>"
            elif extra.get('detail') == "high":
                task = "<MORE_DETAILED_CAPTION>"
        classifications = []
        image = viam_to_pil_image(image)
        results = await self.perform_task(image, task)
        return results
    
    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        return
    
    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None) -> Mapping[str, ValueTypes]:
        return
    
    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> GetPropertiesResponse:
        return GetPropertiesResponse(
            classifications_supported=True,
            detections_supported=True,
            object_point_clouds_supported=False
            )
