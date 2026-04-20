import os

# Folder containing your images
image_folder = "../../datasets/Car_detection_snow/images"

# Prefix groups
train_prefixes = [
    "2022-12-04 Bjenberg 02",
    "2022-12-23 Asjo 01_HD 5x stab",
    "2022-12-02 Asjo 01_stabilized"
]

val_prefixes = [
    "2022-12-03 Nyland 01_stabilized"
]

test_prefixes = [
    "2022-12-23 Bjenberg 02"
]

# Supported image extensions
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Lists to collect matches
train_list = []
val_list = []
test_list = []

# Loop through files
for filename in os.listdir(image_folder):
    if not filename.lower().endswith(image_extensions):
        continue

    filepath = os.path.join("./images/", filename)

    if any(filename.startswith(prefix) for prefix in train_prefixes):
        train_list.append(filepath)

    elif any(filename.startswith(prefix) for prefix in val_prefixes):
        val_list.append(filepath)

    elif any(filename.startswith(prefix) for prefix in test_prefixes):
        test_list.append(filepath)

# Sort alphabetically
#train_list.sort()
val_list.sort()
test_list.sort()

# Write to files
with open("../../datasets/Car_detection_snow/trainProj.txt", "w") as f:
    f.write("\n".join(train_list))

with open("../../datasets/Car_detection_snow/valProj.txt", "w") as f:
    f.write("\n".join(val_list))

with open("../../datasets/Car_detection_snow/testProj.txt", "w") as f:
    f.write("\n".join(test_list))

print("Done! Sorted files written.")