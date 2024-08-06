###### This is a anonymized version of the preprocessing which runs on FIGR-8
###### to allow reviewers to better understand what is happening when we prepare our data for training. 
###### we tried to add useful comments


import os.path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
from svglib.svg import SVG
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent import futures
from skimage import measure

ROOT_DIR = "/scratch/datasets/svg/figr8"


def invert(thread_index, block):
    img_id, img_path, img_class, split = [], [], [], []
    for idx, sample in tqdm(block.iterrows(), total=len(block), disable=(thread_index != 0)):

        png_path = sample["Image"]
        folder, filename = png_path.split("/")

        # simplified SVG path
        simp_svg_filepath = os.path.join(ROOT_DIR, "svg_simplified", folder, filename.replace(".png", ".svg"))
        if not os.path.exists(simp_svg_filepath):

            img = cv2.imread(os.path.join(ROOT_DIR, "Data", png_path))
            # img = 255 - img
            w, h, _ = img.shape
            # add a border to ensure outline contours are not lost for shapes that touch the borders
            border_size = 5
            img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            contours = measure.find_contours(img)
            if len(contours) == 0 or len(contours) > 250:  # filters images where we don't get any border or we get too many (tiny) ones
                continue

            # ############
            # # simplified svg -> parse the contours as SVG path and run deep svg preprocessing
            Path(os.path.join(ROOT_DIR, "svg_simplified", folder)).mkdir(parents=True, exist_ok=True)
            svg_file = f'<svg width="100%" height="100%" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
            for c in contours:
                svg_file += '<path d="M'
                for i in range(len(c)):
                    y, x = c[i]
                    svg_file += f"{x} {y} "
                svg_file += '" fill="none" stroke="#000000" stroke-width="2"/>'
            svg_file += "</svg>"

            # from here on follows deep svg preproc similarly to IconShop
            svg = SVG.from_str(svg_file)
            svg.fill_(False)
            svg.normalize()
            svg.zoom(0.9)
            svg.canonicalize()
            svg = svg.simplify_heuristic(epsilon=0.001)
            svg.save_svg(simp_svg_filepath)

        # logging this new file
        img_id.append(sample["ID"])
        img_path.append(os.path.join(ROOT_DIR, "svg_simplified", folder, filename.replace(".png", ".svg")))
        img_class.append(sample['Class'])
        split.append(sample['Split'])


    new_csv = pd.DataFrame({
                "ID": img_id,
                "file_path": img_path,
                "class": img_class,
                "split": split,
            })
    new_csv.to_csv(os.path.join(ROOT_DIR, "svg_simplified", f"{thread_index}_thread.csv"), index=False)


if __name__ == '__main__':

    if os.path.exists(os.path.join(ROOT_DIR, "split.csv")):
        print("Loading existing splits")
        df = pd.read_csv(os.path.join(ROOT_DIR, "split.csv"))
    else:
        print("Generating train/test split...")
        df = pd.read_csv(os.path.join(ROOT_DIR, "data.csv"), header=None, names=['ID', 'Class', 'Image', 'Author', 'License'])
        unique_classes = df['Class'].unique()
        for class_name in tqdm(unique_classes, total=len(unique_classes)):
            class_df = df[df['Class'] == class_name]
            if len(class_df) > 1:
                train_df, test_df = train_test_split(class_df, test_size=0.1, random_state=42)
                df.loc[train_df.index, 'Split'] = 'train'
                df.loc[test_df.index, 'Split'] = 'test'
            else:
                df.loc[class_df.index, 'Split'] = 'train'

        df.to_csv(os.path.join(ROOT_DIR, "split.csv"), index=False)

    print("Vectorizing and simplifying...")
    shards = 500
    print(f"Using {shards} shards, a thread for each of them.")
    csv_blocks = np.array_split(df, shards)

    with futures.ProcessPoolExecutor(max_workers=len(csv_blocks)) as executor:
        thread_indices = range(len(csv_blocks))
        preprocess_requests = [executor.submit(invert, thread_index, csv_block)
                               for thread_index, csv_block in zip(thread_indices, csv_blocks)]

    print("Process completed. merging.")
    csv_files = [f for f in os.listdir(os.path.join(ROOT_DIR, "svg_simplified")) if f.endswith("_thread.csv")]
    merged_df = pd.concat([pd.read_csv(os.path.join(ROOT_DIR, "svg_simplified", csv_file)) for csv_file in csv_files],
                          ignore_index=True)
    merged_df.to_csv(os.path.join(ROOT_DIR, "svg_simplified", "split.csv"), index=False)
    print("merged all svgs in ", os.path.join(ROOT_DIR, "svg_simplified", "split.csv"))
