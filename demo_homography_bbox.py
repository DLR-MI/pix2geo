#!/usr/bin/env python3
"""
demo_homography_bbox.py
Author: Borja Carrillo-Perez (carr_br)
Description: Homography‐based georeferencing demo using bounding‐box references only.
Reads the bbox model path, image and classes from config under:
  model.bbox, image, and classes (e.g. [8] for boats, [2] for cars).
Computes homography from pixel↔GPS CSV, then applies it to
the bottom‐center pixel of each selected detection.
Outputs an HTML with the annotated image and interactive map.

Usage:
    python demo_homography_bbox.py configs/config_homography.yaml
"""
import argparse
import io
import base64
import webbrowser

import cv2
import folium
import numpy as np
import pandas as pd
import utm
from jinja2 import Template
from PIL import Image
from ultralytics import YOLO

from pix2geo.homography.compute_homography import compute_homography
from pix2geo.reference_selector import select_bottom_center_bbox
from pix2geo.config import load_config


def main(cfg_path):
    # 1) Load config & correspondences
    cfg     = load_config(cfg_path)
    df_corr = pd.read_csv(
        cfg['correspondences_csv'],
        delimiter=cfg.get('csv',{}).get('delimiter',',')
    )
    pixel_pts  = df_corr[['px','py']].values.tolist()
    latlon_pts = df_corr[['lat','lon']].values.tolist()

    # 2) Compute homography (pixel → UTM)
    H = compute_homography(pixel_pts, latlon_pts)

    # 3) Determine UTM zone from first correspondence
    first_lat, first_lon = latlon_pts[0]
    _, _, zone_number, zone_letter = utm.from_latlon(first_lat, first_lon)

    # 4) Load image (for later annotation)
    img = cv2.imread(cfg['image'])
    H_img, W_img = img.shape[:2]

    # 5) Run YOLO bbox inference, filter by classes
    classes = cfg.get('classes', [8])
    model   = YOLO(cfg['model']['bbox'])
    results = model(cfg['image'], classes=classes)[0]

    # 6) Rescale detections from network size → full resolution
    in_h, in_w = results.orig_shape
    sx, sy     = W_img / in_w, H_img / in_h

    georefs = []
    for det in results.boxes.xywh.cpu().numpy():
        cx, cy, bw, bh = det
        cx *= sx; cy *= sy; bw *= sx; bh *= sy
        # top-left:
        x1 = cx - bw/2
        y1 = cy - bh/2
        # bottom-center ref
        px, py = select_bottom_center_bbox((x1, y1, bw, bh))

        # 7) Homography: pixel → UTM → lat/lon
        pt      = np.array([[[px, py]]], dtype=np.float32)
        utm_pt  = cv2.perspectiveTransform(pt, H)[0][0]
        easting, northing = float(utm_pt[0]), float(utm_pt[1])
        lat, lon = utm.to_latlon(easting, northing, zone_number, zone_letter)

        georefs.append((px, py, lat, lon))
        print(f"Det(px={px:.1f},py={py:.1f}) → lat={lat:.6f}, lon={lon:.6f}")

    return georefs, results, (sx, sy), img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Homography bbox georef demo'
    )
    parser.add_argument('config', help='path to config_homography.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    dets, results, (sx, sy), orig = main(args.config)

    # --- Annotate image with green bboxes + red ref‐dots ---
    dot_r  = max(3, int(orig.shape[1]*0.005))
    box_th = max(2, int(orig.shape[1]*0.003))

    # redraw bboxes
    for det in results.boxes.xywh.cpu().numpy():
        cx, cy, bw, bh = det
        cx *= sx; cy *= sy; bw *= sx; bh *= sy
        x1 = int(cx - bw/2); y1 = int(cy - bh/2)
        x2, y2 = x1+int(bw), y1+int(bh)
        cv2.rectangle(orig, (x1,y1), (x2,y2), (0,255,0), box_th)

    # draw red dots
    for px, py, _, _ in dets:
        cv2.circle(orig, (int(px),int(py)), dot_r, (0,0,255), -1)

    # Resize for display
    max_w, max_h = 1280, 720
    h, w = orig.shape[:2]
    scale = min(1.0, max_w/w, max_h/h)
    if scale < 1.0:
        orig = cv2.resize(orig, (int(w*scale), int(h*scale)))

    # Encode image as base64
    rgb  = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    pil  = Image.fromarray(rgb)
    buf  = io.BytesIO(); pil.save(buf,'PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    html_img = (
        '<img src="data:image/png;base64,{}" '
        'style="width:100%; height:auto;" />'
    ).format(img_b64)

    # Build Folium map centered on detections (or fallback to camera)
    if dets:
        ctr_lat = np.mean([lat for *_,lat,_ in dets])
        ctr_lon = np.mean([lon for *_,_,lon in dets])
    else:
        cam = cfg['camera']
        ctr_lat, ctr_lon = cam['lat'], cam['lon']

    m = folium.Map(location=[ctr_lat, ctr_lon], zoom_start=15)
    for idx, (*_, lat, lon) in enumerate(dets):
        folium.Marker([lat, lon], tooltip=f"Det {idx}").add_to(m)
    map_html = m.get_root().render()

    # Combine into a single HTML page
    page = Template("""
    <!DOCTYPE html><html><head><meta charset="utf-8"><title>Homography BBox</title></head>
    <body><div style="display:flex;height:100vh">
      <div style="width:50%;overflow:auto;padding:5px;background:#f9f9f9">{{ image|safe }}</div>
      <div style="width:50%">{{ map|safe }}</div>
    </div></body></html>
    """).render(image=html_img, map=map_html)

    out_path = 'map_homography_bbox.html'
    with open(out_path,'w') as f:
        f.write(page)
    webbrowser.open(out_path)