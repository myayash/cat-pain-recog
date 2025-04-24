import torch

pt_file = input('Enter pt/pth file: ')

try:
  pt_load = torch.load(pt_file, map_location = torch.device('cpu'))
except Exception as e:
  print(f'\nException {e}')
  print(f'Trying ultralytics for YOLO...')
  try:
    import ultralytics
    from ultralytics import YOLO
    pt_load = YOLO(pt_file)
    print(f'\nLoading using ULTRALYTICS SUCCESSFUL')
  except:
    print('\nUsing ULTRALYTICS FAILED')

print('\nPICKLE file TYPE:')
print(type(pt_load))

if isinstance(pt_load, dict):
  print('\nFile is a DICT TYPE, showing keys...')
  print(pt_load.keys())
elif isinstance(pt_load, ultralytics.models.yolo.model.YOLO):
  print('\nFile is a YOLO MODEL TYPE, showing keys...')
  print(pt_load.model.state_dict().keys())


