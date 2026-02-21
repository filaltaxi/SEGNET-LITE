
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pynq_dpu import DpuOverlay
import time

# =====================================================
# USER CONFIG
# =====================================================
CASE_ID = "10019"

IMG_ROOT  = "/home/xilinx/jupyter_notebooks/MRI/dataset/images"
MASK_ROOT = "/home/xilinx/jupyter_notebooks/MRI/dataset/mask"

BIT_FILE        = "dpu.bit"
PG_XMODEL       = "segnet_lite.xmodel"
LESION_XMODEL   = "segnet_lite_lesion.xmodel"

PG_THRESH     = 0.5
LESION_THRESH = 0.5

PG_RESIZE = 192
CROP_SIZE = 64
EPS = 1e-6
# =====================================================


# =====================================================
# Helper functions
# =====================================================
def zscore(img):
    img = img.astype(np.float32)
    return (img - img.mean()) / (img.std() + 1e-5)

def overlay(img, mask, color):
    rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    rgb[mask == 1] = color
    return rgb

def metrics(gt, pd):
    TP = np.sum(gt * pd)
    FP = np.sum((1-gt) * pd)
    FN = np.sum(gt * (1-pd))
    TN = np.sum((1-gt) * (1-pd))

    dice = (2*TP + EPS) / (2*TP + FP + FN + EPS)
    iou  = (TP + EPS) / (TP + FP + FN + EPS)
    prec = (TP + EPS) / (TP + FP + EPS)
    rec  = (TP + EPS) / (TP + FN + EPS)
    acc  = (TP + TN + EPS) / (TP + TN + FP + FN + EPS)
    return dice, iou, prec, rec, acc


def crop_patch(img, mask, patch=64):
    _, H, W = img.shape
    ys, xs = np.where(mask[0] > 0)

    if len(xs) == 0:
        cy, cx = H//2, W//2
    else:
        cy, cx = int(ys.mean()), int(xs.mean())

    s = patch // 2
    y1, y2 = cy - s, cy + s
    x1, x2 = cx - s, cx + s

    pad_t = max(0, -y1); pad_b = max(0, y2 - H)
    pad_l = max(0, -x1); pad_r = max(0, x2 - W)

    img  = np.pad(img,  ((0,0),(pad_t,pad_b),(pad_l,pad_r)))
    mask = np.pad(mask, ((0,0),(pad_t,pad_b),(pad_l,pad_r)))

    y1 += pad_t; y2 += pad_t
    x1 += pad_l; x2 += pad_l

    return img[:, y1:y2, x1:x2], mask[:, y1:y2, x1:x2], y1, y2, x1, x2


# =====================================================
# Load MRI + GT
# =====================================================
print("\n📥 Loading images and GT")
# >>> TIMING <<<
t_total_start = time.time()
# >>> TIMING <<<

t2w = cv2.imread(f"{IMG_ROOT}/{CASE_ID}/{CASE_ID}_t2w.png", 0)
adc = cv2.imread(f"{IMG_ROOT}/{CASE_ID}/{CASE_ID}_adc.png", 0)
hbv = cv2.imread(f"{IMG_ROOT}/{CASE_ID}/{CASE_ID}_hbv.png", 0)

pg_gt = cv2.imread(f"{MASK_ROOT}/{CASE_ID}.png", 0)
ls_gt = cv2.imread(f"{CASE_ID}.png", 0)

pg_gt = (pg_gt > 0).astype(np.uint8)
ls_gt = (ls_gt > 0).astype(np.uint8)

print("✔ Images & GT loaded")


# =====================================================
# ============ STAGE 1: PROSTATE (SegNet) =============
# =====================================================
print("\n🟦 Prostate Gland Segmentation")

img_pg = np.stack([adc, hbv, t2w], axis=-1).astype(np.float32) / 255.0

overlay_pg = DpuOverlay(BIT_FILE)
overlay_pg.load_model(PG_XMODEL)
print(f"✔ Prostate XMODEL loaded → {PG_XMODEL}")

dpu_pg = overlay_pg.runner
inp_pg = dpu_pg.get_input_tensors()[0]
out_pg = dpu_pg.get_output_tensors()[0]

img_pg_q = np.round(img_pg * (2 ** inp_pg.get_attr("fix_point"))).astype(np.int8)
img_pg_q = np.expand_dims(img_pg_q, 0)

out_pg_buf = [np.empty(out_pg.dims, dtype=np.int8)]

# >>> TIMING <<<
t_pg_start = time.time()
jid = dpu_pg.execute_async([img_pg_q], out_pg_buf)
dpu_pg.wait(jid)
t_pg_end = time.time()

pg_inference_time = (t_pg_end - t_pg_start) * 1000.0   # ms
# >>> TIMING <<<


pg_logits = out_pg_buf[0].astype(np.float32) / (2 ** out_pg.get_attr("fix_point"))
pg_prob   = 1 / (1 + np.exp(-np.clip(pg_logits, -20, 20)))
pg_pred   = (pg_prob[0,:,:,0] > PG_THRESH).astype(np.uint8)

pg_dice, pg_iou, pg_prec, pg_rec, pg_acc = metrics(pg_gt, pg_pred)

print("\n📊 Prostate Segmentation Metrics")
print(f"Dice      : {pg_dice:.4f}")
print(f"IoU       : {pg_iou:.4f}")
print(f"Precision : {pg_prec:.4f}")
print(f"Recall    : {pg_rec:.4f}")
print(f"Accuracy  : {pg_acc:.4f}")
print("✔ Prostate inference done")

# =====================================================
# ============ STAGE 2: LESION (TRAINING MATCH) =======
# =====================================================
print("\n🟥 Lesion Segmentation")

t2w_192 = cv2.resize(t2w, (192,192))
hbv_192 = cv2.resize(hbv, (192,192))
adc_192 = cv2.resize(adc, (192,192))
ls_gt_192 = cv2.resize(ls_gt, (192,192), interpolation=cv2.INTER_NEAREST)

img_ls = np.stack([
    zscore(t2w_192),
    zscore(hbv_192),
    zscore(adc_192)
], axis=0)

mask_ls = ls_gt_192[None].astype(np.float32)

img_c, gt_c, y1, y2, x1, x2 = crop_patch(img_ls, mask_ls, CROP_SIZE)

# visualization crops
t2w_c_vis = t2w_192[y1:y2, x1:x2]
adc_c_vis = adc_192[y1:y2, x1:x2]
hbv_c_vis = hbv_192[y1:y2, x1:x2]

img_c = np.transpose(img_c, (1,2,0))
img_c = np.expand_dims(img_c, 0)

overlay_ls = DpuOverlay(BIT_FILE)
overlay_ls.load_model(LESION_XMODEL)
print(f"✔ Lesion XMODEL loaded → {LESION_XMODEL}")

dpu_ls = overlay_ls.runner
inp_ls = dpu_ls.get_input_tensors()[0]
out_ls = dpu_ls.get_output_tensors()[0]

img_ls_q = np.round(img_c * (2 ** inp_ls.get_attr("fix_point"))).astype(np.int8)

out_ls_buf = [np.empty(out_ls.dims, dtype=np.int8)]

# >>> TIMING <<<
t_ls_start = time.time()
jid = dpu_ls.execute_async([img_ls_q], out_ls_buf)
dpu_ls.wait(jid)
t_ls_end = time.time()

ls_inference_time = (t_ls_end - t_ls_start) * 1000.0   # ms
# >>> TIMING <<<


ls_logits = out_ls_buf[0].astype(np.float32) / (2 ** out_ls.get_attr("fix_point"))
ls_prob   = 1 / (1 + np.exp(-np.clip(ls_logits, -20, 20)))
ls_pred   = (ls_prob[0,:,:,0] > LESION_THRESH).astype(np.uint8)

ls_dice, ls_iou, ls_prec, ls_rec, ls_acc = metrics(gt_c[0], ls_pred)

print("\n📊 Lesion Segmentation Metrics")
print(f"Dice      : {ls_dice:.4f}")
print(f"IoU       : {ls_iou:.4f}")
print(f"Precision : {ls_prec:.4f}")
print(f"Recall    : {ls_rec:.4f}")
print(f"Accuracy  : {ls_acc:.4f}")
print("✔ Lesion inference done")

# =====================================================
# ================= VISUALIZATION =====================
# =====================================================
plt.figure(figsize=(18,22))

plt.subplot(6,3,1); plt.title("T2W"); plt.imshow(t2w,cmap="gray"); plt.axis("off")
plt.subplot(6,3,2); plt.title("ADC"); plt.imshow(adc,cmap="gray"); plt.axis("off")
plt.subplot(6,3,3); plt.title("HBV"); plt.imshow(hbv,cmap="gray"); plt.axis("off")

plt.subplot(6,3,4); plt.title("T2W + GT Prostate"); plt.imshow(overlay(t2w, pg_gt,[255,0,0])); plt.axis("off")
plt.subplot(6,3,5); plt.title("ADC + GT Prostate"); plt.imshow(overlay(adc, pg_gt,[255,0,0])); plt.axis("off")
plt.subplot(6,3,6); plt.title("HBV + GT Prostate"); plt.imshow(overlay(hbv, pg_gt,[255,0,0])); plt.axis("off")

plt.subplot(6,3,7); plt.title("T2W + Pred Prostate"); plt.imshow(overlay(t2w, pg_pred,[0,0,255])); plt.axis("off")
plt.subplot(6,3,8); plt.title("ADC + Pred Prostate"); plt.imshow(overlay(adc, pg_pred,[0,0,255])); plt.axis("off")
plt.subplot(6,3,9); plt.title("HBV + Pred Prostate"); plt.imshow(overlay(hbv, pg_pred,[0,0,255])); plt.axis("off")

plt.subplot(6,3,10); plt.title("Cropped T2W"); plt.imshow(t2w_c_vis,cmap="gray"); plt.axis("off")
plt.subplot(6,3,11); plt.title("Cropped ADC"); plt.imshow(adc_c_vis,cmap="gray"); plt.axis("off")
plt.subplot(6,3,12); plt.title("Cropped HBV"); plt.imshow(hbv_c_vis,cmap="gray"); plt.axis("off")

plt.subplot(6,3,13); plt.title("Cropped T2W + GT"); plt.imshow(overlay(t2w_c_vis, gt_c[0],[255,0,0])); plt.axis("off")
plt.subplot(6,3,14); plt.title("Cropped ADC + GT"); plt.imshow(overlay(adc_c_vis, gt_c[0],[255,0,0])); plt.axis("off")
plt.subplot(6,3,15); plt.title("Cropped HBV + GT"); plt.imshow(overlay(hbv_c_vis, gt_c[0],[255,0,0])); plt.axis("off")

plt.subplot(6,3,16); plt.title("Cropped T2W + Pred"); plt.imshow(overlay(t2w_c_vis, ls_pred,[0,0,255])); plt.axis("off")
plt.subplot(6,3,17); plt.title("Cropped ADC + Pred"); plt.imshow(overlay(adc_c_vis, ls_pred,[0,0,255])); plt.axis("off")
plt.subplot(6,3,18); plt.title("Cropped HBV + Pred"); plt.imshow(overlay(hbv_c_vis, ls_pred,[0,0,255])); plt.axis("off")

plt.tight_layout()
plt.show()
# >>> TIMING <<<
t_total_end = time.time()

total_latency = (t_total_end - t_total_start) * 1000.0  # ms
total_dpu_time = pg_inference_time + ls_inference_time  # ms

fps = 1000.0 / total_dpu_time
throughput = fps
# >>> TIMING <<<


print("\n==============================")
print("⏱️  PERFORMANCE SUMMARY")
print("==============================")
print(f"Prostate DPU inference time : {pg_inference_time:.2f} ms")
print(f"Lesion   DPU inference time : {ls_inference_time:.2f} ms")
print(f"Total DPU inference time    : {total_dpu_time:.2f} ms")
print(f"End-to-end latency          : {total_latency:.2f} ms")
print(f"Throughput                  : {throughput:.2f} samples/sec")
print(f"FPS                         : {fps:.2f}")
print("==============================")

print("✅ SEGNET TWO-STAGE INFERENCE FINISHED")




for th in [0.4, 0.5, 0.6, 0.65, 0.7, 0.1]:
    pred = (ls_prob[0,:,:,0] > th).astype(np.uint8)
    d,i,p,r,a = metrics(gt_c[0], pred)
    print(f"th={th:.2f} | Dice={d:.3f} | Prec={p:.3f} | Rec={r:.3f}")





 


