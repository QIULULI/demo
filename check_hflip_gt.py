# save as check_hflip_gt.py
from PIL import Image, ImageDraw
import json, os

img_path = "/userhome/liqiulu/data/drone_rgb_clear_day/00002/clear_day/matrix-300-RTK=m210-rtk/images_rgb/10.3753.png"
ann_path = "data/drone_sim_coco/val.json"  # 改成你的 val.json

with open(ann_path,'r') as f:
    coco = json.load(f)

# 找到这张图的 image_id
img_info = next(i for i in coco["images"] if i["file_name"]=="00002/clear_day/matrix-300-RTK=m210-rtk/images_rgb/10.3753.png")
W,H = img_info["width"], img_info["height"]
img_id = img_info["id"]

# 取出它的所有 bbox
bboxes = [a["bbox"] for a in coco["annotations"] if a["image_id"]==img_id]

img = Image.open(img_path).convert("RGB")
draw = ImageDraw.Draw(img)
for (x,y,w,h) in bboxes:
    draw.rectangle([x,y,x+w,y+h], outline=(255,0,0), width=2)
img.save("gt_as_is.png")

# 再画“水平镜像后的 GT”：x' = W - x - w
img2 = Image.open(img_path).convert("RGB")
draw2 = ImageDraw.Draw(img2)
for (x,y,w,h) in bboxes:
    xx = W - x - w
    draw2.rectangle([xx,y,xx+w,y+h], outline=(0,255,0), width=2)
img2.save("gt_hflipped.png")

print("W,H=",W,H, "saved gt_as_is.png (红框=原GT), gt_hflipped.png (绿框=镜像GT)")
