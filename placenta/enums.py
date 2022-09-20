from enum import Enum


class ModelsArg(str, Enum):
    graphsage = "graphsage"
    clustergcn = "clustergcn"
    graphsaint = "graphsaint"
    sign = "sign"
    shadow = "shadow"
    gat = "gat"
    gatv2 = "gatv2"
    mlp = "mlp"
