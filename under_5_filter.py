import glob
import shutil

dirpath = './learning'
all_files = glob.glob(dirpath + '/*')
for file in all_files:
  images = glob.glob(file + "/*")
  if len(images) < 5:
    dest = file.replace('./learning', './learning_under_5')
    shutil.move(file, dest)
