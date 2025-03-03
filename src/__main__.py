import asyncio
import sys

from viam.module.module import Module
from viam.services.vision import Vision
from .florence2 import florence2

async def main():
    """This function creates and starts a new module, after adding all desired resources.
    Resources must be pre-registered. For an example, see the `__init__.py` file.
    """
    module = Module.from_args()
    module.add_model_from_registry(Vision.API, florence2.MODEL)
    await module.start()

if __name__ == "__main__":
    asyncio.run(main())
