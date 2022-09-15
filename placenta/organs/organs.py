from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Cell:
    label: str
    name: str
    colour: str
    id: int

    def __str__(self):
        return f"{self.label}"


@dataclass(frozen=True)
class Tissue:
    label: str
    name: str
    colour: str
    id: int

    def __str__(self):
        return f"{self.label}"


class Organ:
    def __init__(self, cells: List[Cell], tissues: List[Tissue]):
        self.cells = cells
        self.tissues = tissues

    def cell_by_id(self, i: int):
        return self.cells[i]

    def cell_by_label(self, label):
        labels = {cell.label: cell.id for cell in self.cells}
        return self.cells[labels[label]]

    def tissue_by_label(self, label):
        labels = {tissue.label: tissue.id for tissue in self.tissues}
        return self.tissues[labels[label]]


Placenta = Organ(
    [
        Cell("CYT", "Cytotrophoblast", "#00E307", 0),
        Cell("FIB", "Fibroblast", "#C80B2A", 1),
        Cell("HOF", "Hofbauer Cell", "#FFDC3D", 2),
        Cell("SYN", "Syncytiotrophoblast", "#009FFA", 3),
        Cell("VEN", "Vascular Endothelial", "#FF6E3A", 4),
        Cell("MAT", "Maternal Decidua", "#008169", 5),
        Cell("VMY", "Vascular Myocyte", "#6A0213", 6),
        Cell("WBC", "Leukocyte", "#003C86", 7),
        Cell("MES", "Mesenchymal Cell", "#FF71FD", 8),
        Cell("EVT", "Extra Villus Trophoblast", "#FFCFE2", 9),
        Cell("KNT", "Syncytial Knot", "#7CFFFA", 10),
    ],
    [
        Tissue("Unlabelled", "Unlabelled", "#000000", 0),
        Tissue("Sprout", "Villus Sprout", "#ff3cfe", 1),
        Tissue("MVilli", "Mesenchymal Villi", "#f60239", 2),
        Tissue("TVilli", "Terminal Villi", "#ff6e3a", 3),
        Tissue("ImIVilli", "Immature Intermediate Villi", "#5a000f", 4),
        Tissue("MIVilli", "Mature Intermediate Villi", "#ffac3b", 5),
        Tissue("AVilli", "Anchoring Villi", "#ffcfe2", 6),
        Tissue("SVilli", "Stem Villi", "#ffdc3d", 7),
        Tissue("Chorion", "Chorionic Plate", "#005a01", 8),
        Tissue("Maternal", "Basal Plate/Septum", "#00cba7", 9),
        Tissue("Inflam", "Inflammatory Response", "#7cfffa", 10),
        Tissue("Fibrin", "Fibrin", "#0079fa", 11),
        Tissue("Avascular", "Avascular Villi", "#450270", 12),
    ],
)
