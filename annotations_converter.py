import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse
from pycocotools.coco import COCO


def convert_to_pandas(bbox_list, columns=('filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax')):
    return pd.DataFrame(bbox_list, columns=columns)

def xml_to_csv(path):
    xml_list = []
    tree = ET.parse(path)
    root = tree.getroot()
    for member in root.findall('object'):
        value = (root.find('filename').text,
                 int(root.find('size')[0].text),
                 int(root.find('size')[1].text),
                 member[0].text,
                 int(member[4][0].text),
                 int(member[4][1].text),
                 int(member[4][2].text),
                 int(member[4][3].text)
                 )
        xml_list.append(value)
    return convert_to_pandas(xml_list)

def json_to_csv(path):
    coco = COCO(path)
    imgIds = coco.getImgIds()
    imgs_data = coco.loadImgs(imgIds)
    results = []
    for img_data in imgs_data:
        gt_annIds = coco.getAnnIds(imgIds=[img_data['id']])
        gt_anns = coco.loadAnns(gt_annIds)
        image_width, image_height = img_data['width'], img_data['height']
        for gt_ann in gt_anns:
            x1, y1, x2, y2 = round(gt_ann['bbox'][0]), round(gt_ann['bbox'][1]), \
                             round(gt_ann['bbox'][0] + gt_ann['bbox'][2]), \
                             round(gt_ann['bbox'][1] + gt_ann['bbox'][3])
            bbox_class = gt_ann['category_txt']
            filename= img_data['file_name']
            results.append((filename, image_width, image_height, bbox_class, x1,y1,x2,y2))
    return convert_to_pandas(results)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate SimpleDet GroundTruth Database')
    parser.add_argument('input_annotation', metavar='annotation_file.[xml|json]', type=str)
    parser.add_argument('--output_path', default='bboxes.csv', type=str)
    args = parser.parse_args()
    return args

def get_parser_func(filename):
    file_ext = os.path.splitext(filename)[1]
    if file_ext == '.xml':
        return xml_to_csv
    elif file_ext == '.json':
        return json_to_csv
    else:
        raise Exception("Unknown input file. Should be json or xml")

def main():
    args = parse_args()
    convertion_func = get_parser_func(filename=args.input_annotation)
    csv_df = convertion_func(path=args.input_annotation)
    csv_df.to_csv((args.output_path), index=None)
    print('Successfully converted to csv.')


main()

if __name__ == "__main__":
    main()
