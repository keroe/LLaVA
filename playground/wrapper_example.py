from llava.model.llava_wrapper import LlavaWrapper
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-folder", type=str)
    llava_wrapper = LlavaWrapper("liuhaotian/llava-v1.5-7b")
    query = "Can you tell me which state the traffic light has? Possible states can be off, red, yellow, green. Please only answer with one word, the corresponding state."
    args = parser.parse_args()
    base_folder = Path(args.base_folder)
    tl_dir = {"red": {"files": [], "labels": []}, "green": {"files": [], "labels": []}, "yellow": {"files": [], "labels": []}}
    for color in ["red", "yellow", "green"]:
            tl_dir[color]["files"] = [str(x) for x in (base_folder / Path(color)).glob("*.png")]

    for color in ["red", "yellow", "green"]:
            for file in tqdm(tl_dir[color]["files"]):
                    tl_dir[color]["labels"].append(llava_wrapper.run(query, str(file)))

    with open("/tmp/tl_crops_label.yaml", "w") as fi:
            yaml.dump(tl_dir, fi)
