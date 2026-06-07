# Dubai Police encode the incident type as the Arabic acci_name with the
# severity word stripped off. These are English labels for the types we surface
# in the UI; the raw Arabic type is always kept alongside in the data.
TYPE_LABELS_EN = {
    "حادث دهس رجل": "Pedestrian run-over",
    "دهس": "Run-over (pedestrian)",
    "صدم دراجة نارية": "Hit motorcycle",
    "حادث صدم دراجة": "Hit motorcycle (rider)",
    "صدم دراجة هوائية": "Hit bicycle",
    "اصطدام بين عدة مركبات": "Multi-vehicle collision",
    "حادث اصطدام بين عدة سيارات": "Multi-car collision",
    "اصطدام بين مركبتين": "Two-vehicle collision",
    "حادث اصطدام بين سيارتين": "Two-car collision",
    "صدم عمود": "Hit pole",
    "حادث صدم عمود": "Hit pole (accident)",
    "صدم حاجز": "Hit barrier",
    "حادث صدم حاجز": "Hit barrier (accident)",
    "صدم رصيف": "Hit curb",
    "حادث صدم رصيف": "Hit curb (accident)",
    "صدم جدار": "Hit wall",
    "حادث صدم جدار": "Hit wall (accident)",
    "الصدم والهروب": "Hit-and-run",
    "حادث صدم و هروب": "Hit-and-run (accident)",
    "حادث ضد مجهول": "Collision with unknown party",
    "صدم باب": "Hit door",
    "انقلاب": "Rollover",
}


def label_en(arabic_type: str) -> str | None:
    return TYPE_LABELS_EN.get(arabic_type)
