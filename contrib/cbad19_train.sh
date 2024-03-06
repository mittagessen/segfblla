# simple training on cbad19

segfblla --precision 16 -d cuda:0 segtrain -mr text:'Handwritten\\u0020text' -mr text:'Machine\\u0020Printed\\u0020text' -mr text:'Image/Drawing/Figure' -mr text:Seal -mr text:Signature -mr text:image -mb default:'Machine\\u0020Printed\\u0020text' -mb default:'Handwritten\\u0020text' --workers 32 -B 16 --partition 0.95 --augment --loss gdl --epochs 300 --lrate 0.003 --warmup 250 --schedule reduceonplateau /dev/shm/cbad19/*.xml
