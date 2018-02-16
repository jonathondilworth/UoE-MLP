# #!/usr/bin/bash
# ###################################################
# # 2 conv layers on clothes
# ###################################################

# python baseline.py 0 1 -n 10 -d 1 -s 100
python baseline.py 0 1 -n 10 -d 1 -s 75
# python baseline.py 0 1 -n 10 -d 1 -s 50
# python baseline.py 0 1 -n 10 -d 1 -s 25
# python baseline.py 0 1 -n 10 -d 1 -s 10
# python baseline.py 0 1 -n 10 -d 1 -s 1

# ###################################################
# # 2 conv layers on clothes DATA AUGMENTATION
# ###################################################

# python baseline.py 1 1 -n 10 -d 1 -a y -s 100
# python baseline.py 1 1 -n 10 -d 1 -a y -s 75
# python baseline.py 1 1 -n 10 -d 1 -a y -s 50
# python baseline.py 1 1 -n 10 -d 1 -a y -s 25
# python baseline.py 1 1 -n 10 -d 1 -a y -s 10
# python baseline.py 1 1 -n 10 -d 1 -a y -s 1

# ###################################################
# # 2 conv layers on faces
# ###################################################

python baseline.py 2 1 -n 10 -d 2 -s 100
# python baseline.py 2 1 -n 10 -d 2 -s 75
# python baseline.py 2 1 -n 10 -d 2 -s 50
# python baseline.py 2 1 -n 10 -d 2 -s 25
# python baseline.py 2 1 -n 10 -d 2 -s 10
# python baseline.py 2 1 -n 10 -d 2 -s 1

# ###################################################
# # 2 conv layers on faces DATA AUGMENTATION
# ###################################################

python baseline.py 3 1 -n 10 -d 2 -a y -s 100
# python baseline.py 3 1 -n 10 -d 2 -a y -s 75
# python baseline.py 3 1 -n 10 -d 2 -a y -s 50
# python baseline.py 3 1 -n 10 -d 2 -a y -s 25
# python baseline.py 3 1 -n 10 -d 2 -a y -s 10
# python baseline.py 3 1 -n 10 -d 2 -a y -s 1