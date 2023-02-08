import pickle
import torch
from torch.utils.data import Dataset

def load_config(data_path, fold="train"):
    with open(data_path.format(fold, "config", "pkl"), "rb") as f:
        config = pickle.load(file=f)
    n_units = config["n_units"]
    n_treated = config["n_treated"]
    n_units_total = config["n_units_total"]
    step = config["step"]
    train_step = config["train_step"]
    control_sample = config["control_sample"]
    noise = config["noise"]
    n_basis = config["n_basis"]
    n_cluster = config["n_cluster"]
    return n_units, n_treated, n_units_total, step, train_step, control_sample, noise, n_basis, n_cluster


def load_tensor(data_path, fold="train"):
    print(data_path.format(fold, "x_full", "pth"))
    x_full = torch.load(data_path.format(fold, "x_full", "pth"))
    t_full = torch.load(data_path.format(fold, "t_full", "pth"))
    mask_full = torch.load(data_path.format(fold, "mask_full", "pth"))
    batch_ind_full = torch.load(data_path.format(fold, "batch_ind_full", "pth"))
    y_full = torch.load(data_path.format(fold, "y_full", "pth"))
    y_control = torch.load(data_path.format(fold, "y_control", "pth"))
    y_mask_full = torch.load(data_path.format(fold, "y_mask_full", "pth"))
    m = torch.load(data_path.format(fold, "m", "pth"))
    sd = torch.load(data_path.format(fold, "sd", "pth"))
    treatment_effect = torch.load(data_path.format(fold, "treatment_effect", "pth"))
    return x_full, t_full, mask_full, batch_ind_full, y_full, y_control, y_mask_full, m, sd, treatment_effect


class LDL_Stim_Dataset(Dataset):
    def __init__(self, data_path, fold,device):
        # Get the data
        (self.x_full,self.t_full,self.mask_full,self.batch_ind_full,
         self.y_full,self.y_control,self.y_mask_full,
         self.m,self.sd,self.treatment_effect,) = load_tensor(data_path, fold)
        print(self.m,self.sd)
        # print(self.x_full.max(),self.x_full.min(),self.x_full.mean())
        self.x_full = torch.moveaxis(self.x_full,1,0)
        # self.x_full = torch.moveaxis(self.x_full,1,-1)
        self.t_full = torch.moveaxis(self.t_full,1,0)
        # self.t_full = torch.moveaxis(self.t_full,1,-1)
        self.mask_full = torch.moveaxis(self.mask_full,1,0)
        # self.mask_full = torch.moveaxis(self.mask_full,1,-1)
        self.y_full = torch.moveaxis(self.y_full,1,0).squeeze()
        self.y_control = torch.moveaxis(self.y_control,1,0)
        self.treatment_effect = torch.moveaxis(self.treatment_effect,1,0)
        # print(self.batch_ind_full)
        self.device = device
        # for i in range (self.x_full.shape[-1]):
        #     self.x_full[:,:,i] = (self.x_full[:,:,i] - self.m[i])/self.sd[i]
        # print(self.x_full.max(),self.x_full.min(),self.x_full.mean())
        print(f'x_full: {self.x_full.shape}') ### Temporal Covariates
        print(f't_full: {self.t_full.shape}') ###  Time -25 to 4
        print(f'mask_full: {self.mask_full.shape}') ### Masking vector
        print(f'batch_ind_full: {self.batch_ind_full.shape}') ### Batch indexes
        print(f'y_full: {self.y_full.shape}')   ### y_i ### need to predict this
        print(f'y_control: {self.y_control.shape}') #### y_i(0)
        print(f'treatment_effect: {self.treatment_effect.shape}')  #### y_i(1)
        print(f'y_mask_full: {self.y_mask_full.shape}') ### if outcome not available during 
        print(f'm: {self.m.shape}') 
        print(f'sd: {self.sd.shape}')

        
    def __len__(self):
        return len(self.x_full)

    def __getitem__(self, idx):
        x = self.x_full[idx].to(self.device)    
        t = self.t_full[idx].to(self.device)    
        m = self.mask_full[idx].to(self.device)    
        y = self.y_full[idx].to(self.device)    
        y_mask = self.y_mask_full[idx].unsqueeze(-1).to(self.device)    
        return x,t,m,y,y_mask

    
    
def read_data_inference(data_path, fold,device,group = 'Treated'):
    (x_full,t_full,mask_full,batch_ind_full,
    y_full,y_control,y_mask_full,
    m,sd,treatment_effect,) = load_tensor(data_path, fold)
    
    x_full = torch.moveaxis(x_full,1,0).to(device)
    t_full = torch.moveaxis(t_full,1,0).to(device)
    mask_full = torch.moveaxis(mask_full,1,0).to(device)
    y_full = torch.moveaxis(y_full,1,0).squeeze().to(device)
    y_control = torch.moveaxis(y_control,1,0).to(device)
    treatment_effect = torch.moveaxis(treatment_effect,1,0).to(device)
    
    print(f"Loading {group} Group")
    if group == 'Treated':
        x_full = x_full[y_mask_full==0]
        t_full = t_full[y_mask_full==0]
        mask_full = mask_full[y_mask_full==0]
        y_full = y_full[y_mask_full==0]
        batch_ind_full = batch_ind_full[y_mask_full==0]
        return x_full,t_full,mask_full,y_full,y_mask_full,batch_ind_full,treatment_effect
        
    elif group == 'Control':
        x_full = x_full[y_mask_full==1]
        t_full = t_full[y_mask_full==1]
        mask_full = mask_full[y_mask_full==1]
        y_full = y_full[y_mask_full==1]
        batch_ind_full = batch_ind_full[y_mask_full==1]
        return x_full,t_full,mask_full,y_full,y_mask_full,batch_ind_full





    # print(f'x_full: {x_full.shape}') ### Temporal Covariates
    # print(f't_full: {t_full.shape}') ###  Time -25 to 4
    # print(f'mask_full: {mask_full.shape}') ### Masking vector
    # print(f'batch_ind_full: {batch_ind_full.shape}') ### Batch indexes
    # print(f'y_full: {y_full.shape}')   ### y_i ### need to predict this
    # print(f'y_control: {y_control.shape}') #### y_i(0)
    # print(f'treatment_effect: {treatment_effect.shape}')  #### y_i(1)
    # print(f'y_mask_full: {y_mask_full.shape}') ### if outcome not available during 
    # print(f'm: {m.shape}') 
    # print(f'sd: {sd.shape}')
    
    
    # x,t,m,y,y_mask,batch_ind