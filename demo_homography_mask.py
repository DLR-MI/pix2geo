#!/usr/bin/env python3
"""
demo_homography_mask.py
Author: Borja Carrillo-Perez (carr_br)
Description: Homography‐based georeferencing demo using segmentation‐mask references only.
Reads the segment model path, image and classes from config under:
  model.segment, image, and classes.
Computes homography from pixel↔GPS CSV, then applies it to
the bottom‐mode pixel of each detected mask polygon.
Outputs an HTML with the annotated image and interactive map.

Usage:
    python demo_homography_mask.py configs/config_homography.yaml
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
from pix2geo.reference_selector import select_bottom_mode_pixel
from pix2geo.config import load_config


def polygons_to_mask(polygons, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for pts in polygons:
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def main(cfg_path):
    # 1) Load config & correspondences
    cfg     = load_config(cfg_path)
    df_corr = pd.read_csv(
        cfg['correspondences_csv'],
        delimiter=cfg.get('csv',{}).get('delimiter',',')
    )
    pixel_pts  = df_corr[['px','py']].values.tolist()
    latlon_pts = df_corr[['lat','lon']].values.tolist()

    # 2) Compute homography
    H = compute_homography(pixel_pts, latlon_pts)

    # 3) Determine UTM zone
    first_lat, first_lon = latlon_pts[0]
    _, _, zone_number, zone_letter = utm.from_latlon(first_lat, first_lon)

    # 4) Load image for annotation
    img    = cv2.imread(cfg['image'])
    H_img, W_img = img.shape[:2]

    # 5) Run YOLO segmentation inference
    classes = cfg.get('classes', [8])
    model   = YOLO(cfg['model']['segment'])
    res     = model(cfg['image'], classes=classes)[0]

    georefs = []
    for idx, poly in enumerate(res.masks.xy):
        # poly: Nx2 float coords in image‐space
        mask_full = polygons_to_mask([poly], (H_img, W_img))
        px, py    = select_bottom_mode_pixel(mask_full.astype(bool))

        # homography → UTM → lat/lon
        pt      = np.array([[[px, py]]], dtype=np.float32)
        utm_pt  = cv2.perspectiveTransform(pt, H)[0][0]
        easting, northing = float(utm_pt[0]), float(utm_pt[1])
        lat, lon = utm.to_latlon(easting, northing, zone_number, zone_letter)

        georefs.append((px, py, lat, lon))
        print(f"Mask Det {idx}: px={px},py={py} → lat={lat:.6f},lon={lon:.6f}")

    return georefs, res, img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Homography mask georef demo'
    )
    parser.add_argument('config', help='path to config_homography.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    dets, res, orig = main(args.config)

    # --- Annotate masks and reference points ---
    dot_r  = max(3, int(orig.shape[1]*0.005))
    line_w = max(2, int(orig.shape[1]*0.003))

    # redraw mask outlines
    for poly in res.masks.xy:
        pts = poly.astype(np.int32)
        cv2.polylines(orig, [pts], isClosed=True, color=(0,255,0), thickness=line_w)

    # draw red bottom‐mode pixels
    for px, py, _, _ in dets:
        cv2.circle(orig, (int(px),int(py)), dot_r, (0,0,255), -1)

    # Resize for display
    max_w, max_h = 1280, 720
    h, w = orig.shape[:2]
    scale = min(1.0, max_w / w, max_h / h)
    if scale < 1.0:
        orig = cv2.resize(orig, (int(w*scale), int(h*scale)))

    # Encode annotated image
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO(); pil.save(buf,'PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    html_img = (
        '<img src="data:image/png;base64,{}" '
        'style="width:100%; height:auto;" />'
    ).format(img_b64)

    # Build Folium map
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

    # Combine into one HTML
    page = Template("""
    <!DOCTYPE html><html><head><meta charset="utf-8"><title>Homography Mask</title></head>
    <body><div style="display:flex;height:100vh">
      <div style="width:50%;overflow:auto;padding:5px;background:#f9f9f9">{{ image|safe }}</div>
      <div style="width:50%">{{ map|safe }}</div>
    </div></body></html>
    """).render(image=html_img, map=map_html)

    out_path = 'map_homography_mask.html'
    with open(out_path,'w') as f:
        f.write(page)
    webbrowser.open(out_path)