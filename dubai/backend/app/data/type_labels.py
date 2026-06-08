# Dubai Police encode the incident type as the Arabic acci_name with the
# severity word stripped off. Two quirks make a naive 1:1 translation wrong:
#
#  1. Vocabulary migration (~mid-2021): the older phrasing carries the حادث
#     ("accident") prefix and uses سيارة ("car"); the newer drops the prefix and
#     uses مركبة ("vehicle"). Same real event, two strings — e.g. صدم عمود and
#     حادث صدم عمود are both "hit pole". The pairs are temporally disjoint, which
#     is how we know they're the same category renamed.
#  2. The old era was more granular: pedestrians were split by victim
#     (man/woman/child) before 2021, then merged into a single دهس.
#
# So MULTIPLE Arabic strings deliberately map to ONE canonical English label;
# grouping by the English label re-unites a category across the whole 2018–2026
# span. The raw Arabic type is always kept alongside in the data.
TYPE_LABELS_EN = {
    # pedestrians: new generic + old victim-specific + run-over-and-flee
    "دهس": "Pedestrian run-over",
    "حادث دهس رجل": "Pedestrian run-over",
    "حادث دهس امراة": "Pedestrian run-over",
    "حادث دهس طفل": "Pedestrian run-over",
    "دهس وهروب": "Pedestrian run-over",
    # motorcycle (old "صدم دراجة" is the generic two-wheeler of that era)
    "صدم دراجة نارية": "Hit motorcycle",
    "حادث صدم دراجة": "Hit motorcycle",
    # bicycle
    "صدم دراجة هوائية": "Hit bicycle",
    # two vehicles (old "سيارتين" = two cars)
    "اصطدام بين مركبتين": "Two-vehicle collision",
    "حادث اصطدام بين سيارتين": "Two-vehicle collision",
    # several vehicles (old "عدة سيارات" = several cars)
    "اصطدام بين عدة مركبات": "Multi-vehicle collision",
    "حادث اصطدام بين عدة سيارات": "Multi-vehicle collision",
    # heavy-vehicle collisions
    "اصطدام بين شاحنة ومركبة": "Truck collision",
    "اصطدام بين شاحنتين": "Truck collision",
    "اصطدام بين حافلة نقل عمال ومركبة": "Bus collision",
    "اصطدام بين حافلة مدرسية ومركبة": "Bus collision",
    # against an unknown party (spans both eras already)
    "حادث ضد مجهول": "Collision with unknown party",
    # fixed-object strikes (each: new bare phrase + old حادث-prefixed phrase)
    "صدم عمود": "Hit pole",
    "حادث صدم عمود": "Hit pole",
    "صدم جدار": "Hit wall",
    "حادث صدم جدار": "Hit wall",
    "صدم حاجز": "Hit barrier",
    "حادث صدم حاجز": "Hit barrier",
    "صدم حواجز": "Hit barrier",
    "صدم رصيف": "Hit curb",
    "حادث صدم رصيف": "Hit curb",
    "صدم شجرة": "Hit tree",
    "حادث صدم شجرة": "Hit tree",
    "صدم مبنى": "Hit building",
    "حادث صدم مبنى او بيوت او معرض": "Hit building",
    "صدم حيوان": "Hit animal",
    "حادث صدم حيوان": "Hit animal",
    "صدم باب": "Hit vehicle door",
    # roadside signs / signals / boards
    "صدم علامة مرورية": "Hit road sign",
    "حادث صدم اشارة مرورية": "Hit road sign",
    "حادث صدم لوحة إرشادية": "Hit road sign",
    "صدم إشارة ضوئية": "Hit road sign",
    # objects/obstacles in the carriageway
    "صدم جسم في الشارع": "Hit object in road",
    "صدم جسر من شاحنة عالية الارتفاع": "Hit object in road",
    # hit-and-run (vehicle)
    "الصدم والهروب": "Hit-and-run",
    "حادث صدم و هروب": "Hit-and-run",
    # single-vehicle rollover / overturn (تدهور), old era + the bare انقلاب
    "انقلاب": "Vehicle rollover",
    "حادث تدهور مركبة خفيفة": "Vehicle rollover",
    "حادث تدهور مركبة ثقيلة": "Vehicle rollover",
    "حادث تدهور دراجة": "Vehicle rollover",
    "حادث تدهور باص عمال او موظفين": "Vehicle rollover",
    "حادث تدهور باص مدرسة": "Vehicle rollover",
    "حادث تدهور صهريج لمواد قابلة للاشتعال": "Vehicle rollover",
    # vehicle fire
    "حادث حريق في مركبة": "Vehicle fire",
    # rail
    "حادث صدم ترام": "Hit train/tram",
    "صدم ترام": "Hit train/tram",
    "صدم قطار": "Hit train/tram",
}


def label_en(arabic_type: str) -> str | None:
    return TYPE_LABELS_EN.get(arabic_type)
