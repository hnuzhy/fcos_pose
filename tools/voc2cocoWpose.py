'''
https://github.com/yukkyo/voc2coco

[usage]
python voc2coco.py \
    --ann_dir sample/Annotations \
    --ann_ids sample/dataset_ids/test.txt \
    --labels sample/labels.txt \
    --output sample/bccd_test_cocoformat.json \
    --ext xml
    
[example]
[warning][Please allocate different global_count for train set and val set]
python voc2cocoWpose.py \
    --ann_dir Annotations \
    --kpts_dir Keypoints_JSON \
    --ann_ids ImageSets/Main/train.txt \
    --labels labels.txt \
    --output coco_jsons/coco_dangerdet_Wpose_train.json \
    --ext xml

python voc2cocoWpose.py \
    --ann_dir Annotations \
    --kpts_dir Keypoints_JSON \
    --ann_ids ImageSets/Main/val.txt \
    --labels labels.txt \
    --output coco_jsons/coco_dangerdet_Wpose_val.json \
    --ext xml
'''

import os
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re

global global_count
global total_ids_dict

def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, 'r') as f:
        labels_str = f.read().split()
    labels_ids = list(range(1, len(labels_str)+1))
    return dict(zip(labels_str, labels_ids))


def get_annpaths(ann_dir_path: str = None,
                 kpts_dir_path: str = None,
                 ann_ids_path: str = None,
                 ext: str = '',
                 annpaths_list_path: str = None) -> List[str]:
    # If use annotation paths list
    if annpaths_list_path is not None:
        with open(annpaths_list_path, 'r') as f:
            ann_paths = f.read().split()
        return ann_paths

    # If use annotaion ids list
    ext_with_dot = '.' + ext if ext != '' else ''
    with open(ann_ids_path, 'r') as f:
        ann_ids = f.read().split()
    ann_paths = [os.path.join(ann_dir_path, aid+ext_with_dot) for aid in ann_ids]
    
    kpts_paths = [os.path.join(kpts_dir_path, aid+".json") for aid in ann_ids]
    
    # return ann_paths
    return ann_paths, kpts_paths


def get_image_info(annotation_root, anno_name, extract_num_from_imgid=True):
    # path = annotation_root.findtext('path')
    # if path is None:
        # filename = annotation_root.findtext('filename')
    # else:
        # filename = os.path.basename(path)
    # img_name = os.path.basename(filename)
    # img_id = os.path.splitext(img_name)[0]
    # if extract_num_from_imgid and isinstance(img_id, str):
        # img_id = int(re.findall(r'\d+', img_id)[0])

    filename = anno_name + ".jpg"
    img_id = anno_name
    
    global global_count
    global total_ids_dict
    if img_id not in total_ids_dict:
        total_ids_dict[img_id] = global_count
        global_count += 1
        
    
    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        # 'id': img_id
        'id': total_ids_dict[img_id]
    }
    return image_info


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(float(bndbox.findtext('xmin'))) - 1
    ymin = int(float(bndbox.findtext('ymin'))) - 1
    xmax = int(float(bndbox.findtext('xmax')))
    ymax = int(float(bndbox.findtext('ymax')))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    # return ann
    return ann, category_id


def convert_xmls_to_cocojson(annotation_paths: List[str],
                             keypoints_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print('Start converting !')
    ind = 0
    
    nokeypoints_list = []
    for a_path in tqdm(annotation_paths):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()
        
        anno_name = os.path.splitext(os.path.split(a_path)[-1])[0]
        
        img_info = get_image_info(annotation_root=ann_root, anno_name=anno_name,
                                  extract_num_from_imgid=extract_num_from_imgid)
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            # ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann, category_id = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
        
        # Read keypoints json
        keypoints_list = json.load(open(keypoints_paths[ind], "r"))
        # assert len(keypoints_list) != 0, "This json file has no keypoints: %s"%(keypoints_paths[ind])   
        if len(keypoints_list) == 0:
            print("This json file has no keypoints: %s"%(keypoints_paths[ind]))
            nokeypoints_list.append(keypoints_paths[ind].split("/")[-1][:-5])
            
        for keypoints_dict in keypoints_list:
            keypoints = keypoints_dict["keypoints"]
            face_box = keypoints_dict["face_box"]
            bbox = keypoints_dict["bbox"]
            score = keypoints_dict["score"]
            
            new_keypoints = []
            for i in range(len(keypoints)//3):
                new_keypoints.append(keypoints[3*i])
                new_keypoints.append(keypoints[3*i+1])
                if keypoints[3*i+2] == 0:
                    new_keypoints.append(0)  # 0 is invisible
                else:
                    new_keypoints.append(2)  # 2 is visible, 1 is occlusion
                    
            ann = {"iscrowd": 0, "score": score, "keypoints": new_keypoints, 
                "category_id": category_id,  # one category_id is enough for being a legal annotaion for coco API
                "image_id": img_id, "id": bnd_id}
                
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
        
        ind += 1
        
    print(nokeypoints_list)
    
    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


def main():
    parser = argparse.ArgumentParser(
        description='This script support converting voc format xmls to coco format json')
    parser.add_argument('--ann_dir', type=str, default=None,
                        help='path to annotation files directory. It is not need when use --ann_paths_list')    
    parser.add_argument('--kpts_dir', type=str, default=None,
                        help='path to kpts detection files directory. It is not need when use --ann_paths_list')
    parser.add_argument('--ann_ids', type=str, default=None,
                        help='path to annotation files ids list. It is not need when use --ann_paths_list')
    parser.add_argument('--ann_paths_list', type=str, default=None,
                        help='path of annotation paths list. It is not need when use --ann_dir and --ann_ids')
    parser.add_argument('--labels', type=str, default=None,
                        help='path to label list.')
    parser.add_argument('--output', type=str, default='output.json', help='path to output json file')
    parser.add_argument('--ext', type=str, default='', help='additional extension of annotation file')
    parser.add_argument('--extract_num_from_imgid', action="store_true",
                        help='Extract image number from the image filename')
    args = parser.parse_args()
    label2id = get_label2id(labels_path=args.labels)
    # ann_paths = get_annpaths(
    ann_paths, kpts_paths = get_annpaths(
        ann_dir_path=args.ann_dir,
        kpts_dir_path=args.kpts_dir,
        ann_ids_path=args.ann_ids,
        ext=args.ext,
        annpaths_list_path=args.ann_paths_list
    )
    convert_xmls_to_cocojson(
        annotation_paths=ann_paths,
        keypoints_paths=kpts_paths,
        label2id=label2id,
        output_jsonpath=args.output,
        extract_num_from_imgid=args.extract_num_from_imgid
    )


if __name__ == '__main__':
    global_count = 1500
    total_ids_dict = {}
    main()
    
    print("global_count: ", global_count)