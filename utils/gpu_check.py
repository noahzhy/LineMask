import torch
import torch.nn as nn


# check gpu
def gpu_check():
    gpu_num = torch.cuda.device_count()
    if gpu_num == 0:
        print("There is no GPU available on this device")
        return False
    else:
        print("GPU number:{}".format(gpu_num))
        for i in range(gpu_num):
            print("GPU {} name:{}".format(i, torch.cuda.get_device_name(i)))
        return True


# main
if __name__ == "__main__":
    gpu_check()
