import os, random

img_dir = 'data/Kvasir-SEG/images'
names = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.jpg')]
random.shuffle(names)

# 按 880/120 切分
train, val = names[:880], names[880:1000]

with open('data/Kvasir-SEG/train.txt', 'w') as f:
    f.write('\n'.join(train))
with open('data/Kvasir-SEG/val.txt', 'w') as f:
    f.write('\n'.join(val))