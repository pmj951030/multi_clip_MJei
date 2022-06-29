import transformers


class MCLIPConfig(transformers.PretrainedConfig):
    model_type = "M-CLIP"

    def __init__(self, modelBase='xlm-roberta-large', transformerDimSize=1024, imageDimSize=512, **kwargs): ##768 ->512
        self.transformerDimensions = transformerDimSize
        self.numDims = imageDimSize
        self.modelBase = modelBase
        super().__init__(**kwargs)
