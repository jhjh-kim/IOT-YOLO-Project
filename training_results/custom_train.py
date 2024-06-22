from model import YOLO

model = YOLO('your_model.pt') # adjust as you want 
data = 'dataset.yaml'
img_size = 640
epoch_num = 100 # adjust as you want 
batch_size = 32 # adjust as you want 
task = 'detect'
mode = 'train'
save_dir = '/path/to/save_dir' # adjust as you want 
results = model.train(data=data,
                      imgsz=img_size,
                      epochs=epoch_num,
                      batch=batch_size,
                      task=task,
                      mode=mode,
                      project=save_dir)
