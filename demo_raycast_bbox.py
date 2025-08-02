#!/usr/bin/env python3
"""
demo_raycast_bbox.py
Author: Borja Carrillo-Perez (carr_br)
Description: Raycast georeferencing demo using bounding‑box references only.
Reads the bbox model path from config under `model.bbox`.
"""
import argparse
import numpy as np
import utm
import cv2
from ultralytics import YOLO

from pix2geo.config import load_config
from pix2geo.raycasting.raycasting_utils import calcRotationMatrix, calcCameraRay, intersectPlane
from pix2geo.reference_selector import select_bottom_center_bbox


def main(cfg_path):
    cfg = load_config(cfg_path)
    cam = cfg['camera']
    lat, lon = cam['lat'], cam['lon']
    height = cam['height']
    hFOV_deg = cam['fov']
    pitch, yaw, roll = cam['pitch'], cam['yaw'], cam['roll']
    plane = cfg['plane']
    plane_n = np.array(plane['normal'], dtype=float)
    plane_p = np.array(plane['point'], dtype=float)
    utm_x, utm_y, zn, zl = utm.from_latlon(lat, lon)
    R = calcRotationMatrix(np.radians(pitch), np.radians(yaw), np.radians(roll))
    Rt = np.vstack([np.c_[R, [utm_x, utm_y, height]], [0, 0, 0, 1]])
    img = cv2.imread(cfg['image'])
    H_img, W_img = img.shape[:2]
    ar = W_img / float(H_img)
    hFOV = np.radians(hFOV_deg)
    vFOV = 2 * np.arctan(np.tan(hFOV / 2) / ar)
    model_path = cfg['model']['bbox']
    classes = cfg.get('classes', [8])
    model = YOLO(model_path)
    res = model(cfg['image'], classes=classes)[0]
    boxes = res.boxes.xywh.cpu().numpy()
    in_h, in_w = res.orig_shape
    sx, sy = W_img / in_w, H_img / in_h
    geolocs = []
    ref_points = []

    for idx, b in enumerate(boxes):
        cx, cy, bw, bh = b
        cx *= sx; cy *= sy; bw *= sx; bh *= sy
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        bbox_xywh = (x1, y1, bw, bh)
        px, py = select_bottom_center_bbox(bbox_xywh)
        ref_points.append((px, py))
        ray_d, ray_o = calcCameraRay(px, py, vFOV, W_img, H_img, Rt)
        hit, dist = intersectPlane(plane_n, plane_p, ray_o, ray_d)
        if not hit:
            continue
        wp = ray_o + ray_d * dist
        lat2, lon2 = utm.to_latlon(wp[0], wp[1], zn, zl)
        print(f"BBox Det {idx}: lat={lat2:.6f}, lon={lon2:.6f}")
        geolocs.append((lat2, lon2))
    return geolocs, (lat, lon), res, (sx, sy), img, ref_points


if __name__ == '__main__':
    import io
    import base64
    import webbrowser
    from PIL import Image
    from jinja2 import Template
    import folium

    p = argparse.ArgumentParser()
    p.add_argument('config', help='Path to YAML/JSON config')
    args = p.parse_args()
    pts, cam_ll, res, (sx, sy), orig_img, ref_points = main(args.config)

    # Annotate image
    box_thickness = max(2, int(orig_img.shape[1] * 0.003))
    dot_radius = max(3, int(orig_img.shape[1] * 0.005))

    for idx, b in enumerate(res.boxes.xywh.cpu().numpy()):
        cx, cy, bw, bh = b
        cx *= sx; cy *= sy; bw *= sx; bh *= sy
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(x1 + bw)
        y2 = int(y1 + bh)

        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
        if idx < len(ref_points):
            px, py = ref_points[idx]
            cv2.circle(orig_img, (int(px), int(py)), dot_radius, (0, 0, 255), -1)

    max_w, max_h = 1280, 720
    h, w = orig_img.shape[:2]
    scale = min(1.0, max_w / w, max_h / h)
    if scale < 1.0:
        orig_img = cv2.resize(orig_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    encoded_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    html_img = f'<img src="data:image/png;base64,{encoded_img}" style="width:100%; height:auto;">'

    center = (sum(a for a, _ in pts) / len(pts), sum(b for _, b in pts) / len(pts)) if len(pts) > 0 else cam_ll
    m = folium.Map(location=center, zoom_start=15)
    for i, (a, b) in enumerate(pts):
        folium.Marker((a, b), tooltip=f"Det {i}").add_to(m)
    map_html = m.get_root().render()

    final_html = Template(f"""
    <!DOCTYPE html>
    <html>
    <head><title>BBox Detections with Map</title></head>
    <body><div style='display:flex;'>
    <div style='width:50%'>{html_img}</div>
    <div style='width:50%'>{map_html}</div>
    </div></body>
    </html>
    """).render()

    out_path = 'map_bbox_with_img.html'
    with open(out_path, 'w') as f:
        f.write(final_html)
    webbrowser.open(out_path)