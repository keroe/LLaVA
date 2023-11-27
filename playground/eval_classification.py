import yaml

f = "/tmp/tl_crops_label.yaml"

with open(f, "r") as fi:
	d = yaml.load(fi, Loader=yaml.BaseLoader)

colors = ["green", "red", "yellow"]
labels = ["Green", "Red", "Yellow"]
for color, label in zip(colors, labels):
	num_label = len(d[color]["labels"])
	correct = 0
	for l in d[color]["labels"]:
		if l == label:
			correct = correct + 1
	print(f"For {label } Out of {num_label} images, {correct} are correctly classified.")
