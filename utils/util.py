import torch

def gpu_device_check(gpu_index):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_index)
        device = torch.device(f'cuda:{gpu_index}')
        device_number = torch.cuda.current_device()  # 獲取當前 GPU 編號
        device_name = torch.cuda.get_device_name(device_number)  # 獲取 GPU 名稱
        print(f"Using device: {device} (GPU {device_number}: {device_name})")
        
    else:
        print(f"Using device: {device}")
    
    return device

#test function
if __name__ == "__main__":
    gpu_device_check()
