import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = 'LR/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    
    # Read LR image and display its resolution
    img_lr = cv2.imread(path, cv2.IMREAD_COLOR)
    lr_height, lr_width, _ = img_lr.shape
    print(f"LR Image Resolution - Height: {lr_height}, Width: {lr_width}")
    
    # Read LR image and process through the model
    img_lr = img_lr * 1.0 / 255
    img_lr = torch.from_numpy(np.transpose(img_lr[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_lr_tensor = img_lr.unsqueeze(0)
    img_lr_tensor = img_lr_tensor.to(device)

    with torch.no_grad():
        output = model(img_lr_tensor).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    
    # Display resolution of the resulting image and save it
    result_height, result_width, _ = output.shape
    print(f"Resulting Image Resolution - Height: {result_height}, Width: {result_width}")
    
    # Calculate percentage difference in Psnr/Resolution
    height_diff_percent = ((result_height - lr_height) / lr_height) * 100
    width_diff_percent = ((result_width - lr_width) / lr_width) * 100
    print(f"Percentage Difference in Psnr of the final Image and the original Image: {width_diff_percent:.2f}%")
    
    cv2.imwrite('results/{:s}_rlt.png'.format(base), output)
