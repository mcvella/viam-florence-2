"""
This file registers the model with the Python SDK.
"""

from viam.services.vision import Vision
from viam.resource.registry import Registry, ResourceCreatorRegistration

from .florence2 import florence2

Registry.register_resource_creator(Vision.API, florence2.MODEL, ResourceCreatorRegistration(florence2.new, florence2.validate))
