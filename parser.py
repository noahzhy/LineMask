# parser of json
import os
import sys
import json
import glob
import random
# svg 2 json
import xmltodict
import json
# xml
import xml.etree.ElementTree as ET

import numpy as np
import cv2


# parse svg
def get_rect_node(xml_path):
    # find all rect node
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # print(root)

    # find all rect node
    rect_nodes = root.findall('''.//rect''')
    rects = []
    for rect_node in rect_nodes:
        node = rect_node.attrib
        # find x, y, w, h, if not exist, skip
        if 'x' not in node or 'y' not in node or 'width' not in node or 'height' not in node or 'fill' not in node:
            continue
        else:
            x = float(node['x'])
            y = float(node['y'])
            w = float(node['width'])
            h = float(node['height'])
            color = node['fill']
            rects.append([x, y, w, h, color])

    return rects


def svg2json(path):
    with open(path, 'r') as f:
        svg = f.read()
    o = json.loads(json.dumps(xmltodict.parse(svg)))


    rects = get_rect_node(path)
    if len(rects) == 0:
        return []

    txt = []
    # nosel
    data = o['svg']['g']['g'][-1]
    # print(data['g'])

    # <g transform="translate(73,0)" class="nosel">
    offset = o['svg']['g']['@transform']
    offset = offset.replace('translate(', '')
    offset = offset.replace(')', '')
    # split via ,
    offset = offset.split(',')
    offset_x = float(offset[0])
    offset_y = float(offset[1])

    for rect in rects:
        x, y, w, h, color = rect
        # print(x, y, w, h, color)
        x = x + offset_x
        y = y + offset_y
        txt.append([x, y, w, h, color])

    return txt


# main
if __name__ == "__main__":
    # svg file path
    paths = "E:/dataset/charts/receive/images/*.svg"
    # # random pick one
    # path = random.choice(glob.glob(path))

    for path in glob.glob(paths):
        print(path)
        # rp png
        png = path.replace('.svg', '.png')
        
        try:
            data = svg2json(path)
            if len(data) == 0:
                continue
            # save as txt
            txt = path.replace('.svg', '.txt')
            with open(txt, 'w') as f:
                for d in data:
                    x, y, w, h, color = d
                    f.write(f'{x},{y},{w},{h},{color}\n')

            # # draw
            # img = cv2.imread(png)

            # for d in data:
            #     x, y, w, h, color = d
            #     x, y, w, h = int(x), int(y), int(w), int(h)
            #     # print(x, y, w, h, color)
            #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)

            # # show
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        except Exception as e:
            print(e)
            pass

    
