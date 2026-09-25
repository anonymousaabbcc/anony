import sys
import torch
import transformers
print('Python:', sys.executable)
print('Python version:', sys.version.split()[0])
print('torch:', torch.__version__)
print('CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
print('transformers:', transformers.__version__)
try:
    import peft
    print('peft:', peft.__version__)
except Exception as e:
    print('peft: ERROR:', repr(e))
try:
    from PIL import Image
    import PIL
    print('Pillow:', PIL.__version__)
except Exception as e:
    print('Pillow: ERROR:', repr(e))
