# simple training on cbad19

segfblla --precision 16 -d cuda:0 segtrain -F 20 -mr text:'Handwritten\\u0020text' -mr text:'Machine\\u0020Printed\\u0020text' -mr text:'Image/Drawing/Figure' -mr text:Seal -mr text:Signature -mr text:image -mb default:'Machine\\u0020Printed\\u0020text' -mb default:'Handwritten\\u0020text' --workers 32 -B 16 --partition 0.95 --augment --loss bce --epochs 600 --lrate 6e-4 --weight-decay 5e-4 --warmup 1500 --optimizer 'AdamW' --schedule cosine --cos-max 600 --cos-min-lr 1e-8 -o /mnt/nfs_data/experiments/segfblla/cbad19 /dev/shm/cbad19/*.xml
