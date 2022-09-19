from enum import Enum

class FeatureArg(str, Enum):
    predictions = "predictions"
    embeddings = "embeddings"


class MethodArg(str, Enum):
    k = "k"
    delaunay = "delaunay"
    intersection = "intersection"


class ModelsArg(str, Enum):
    graphsage = "graphsage"
    clustergcn = "clustergcn"
    graphsaint = "graphsaint"
    sign = "sign"
    shadow = "shadow"
    gat = "gat"
    gatv2 = "gatv2"
    mlp = "mlp"
