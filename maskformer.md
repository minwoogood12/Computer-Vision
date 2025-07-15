***MaskFormer - Swin-B

python demo.py \
  --config-file ../configs/coco-panoptic/swin/maskformer_panoptic_swin_base_IN21k_384_bs64_554k.yaml \
  --input input1.png \
  --output ./output \
  --opts MODEL.WEIGHTS ./checkpoints/model_final_4b7f49.pkl MODEL.DEVICE cpu

***MaskFormer - Swin_L

python demo.py \
  --config-file ../configs/coco-panoptic/swin/maskformer_panoptic_swin_large_IN21k_384_bs64_554k.yaml \
  --input input1.png \
  --output ./output \
  --opts MODEL.WEIGHTS ./checkpoints/model_final_7505c4.pkl MODEL.DEVICE cpu
