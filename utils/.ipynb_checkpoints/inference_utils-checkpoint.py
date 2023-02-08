import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from config.config import DEVICE
from utils.loss import matching_loss, AverageMeter

def get_representation(encoder,x,m):
    c_hat = encoder(x*m)#[:,0,:]
    c_hat = torch.flatten(c_hat,start_dim=1)
    return c_hat


def learn_B_for_one_sample(c_target,c_non_target,num_iterations=1000,is_plot = False):
    B = torch.zeros(len(c_non_target),1).to(DEVICE).requires_grad_(True)
    print(f"Tensor Shapes, B: {B.shape}, c_target: {c_target.shape}, c_non_target: {c_non_target.shape}")
    
    if is_plot:
        plt.figure(figsize=(30,10))
        plt.plot(B.squeeze().detach().cpu().numpy())
    
    opt =  torch.optim.Adam([B], lr=0.0001, betas=(0.9, 0.999))
    matching_losses = []
    for i in range(num_iterations):
        opt.zero_grad()
        c_est = B*c_non_target
        c_est = torch.sum(c_est,axis=0).unsqueeze(0)
        
        loss = matching_loss(c_est,c_target.unsqueeze(0))#/len(target_ind) 
        loss.backward()
        opt.step()
        matching_losses.append(loss.data.item())
        
    
    if is_plot:
        plt.plot(B.squeeze().detach().cpu().numpy())
    
    if is_plot:
        plt.figure()
        plt.plot(matching_losses)
        plt.show()
                
    
        fig, axs = plt.subplots(1, 2, figsize = (20,10))
        im = axs[0].plot(c_target.squeeze().detach().cpu().numpy())
        axs[0].set_title(f"C Target")
        im = axs[1].plot(c_est.squeeze().detach().cpu().numpy())#,cmap = 'jet')
        axs[1].set_title(f"C Est")

    return B


def estimate_y_one_sample(B,y_non_target):
    with torch.no_grad():
        print(B.shape,y_non_target.shape)
        y_hat = torch.matmul(B.T,y_non_target)
    return y_hat
    
def calculate_mae(y_hat,y_treat, treatment_effect):
    ite_est = y_treat-y_hat
    mae = torch.abs(treatment_effect - ite_est).mean()
    mae_sd = torch.std(torch.abs(treatment_effect - ite_est)).item() / np.sqrt(treatment_effect.shape[0])
    
    return mae.item(), mae_sd


def learn_B(c_target,c_non_target,num_iterations=1000,is_plot = False):
    B = torch.zeros(len(c_target),len(c_non_target)).to(DEVICE).requires_grad_(True)
    print(f"Tensor Shapes, B: {B.shape}, c_target: {c_target.shape}, c_non_target: {c_non_target.shape}")
    
    if is_plot:
        plt.figure(figsize=(30,10))
        im = plt.imshow(B.squeeze().detach().cpu().numpy(),cmap = 'jet')
        plt.colorbar(shrink = 0.5)
        plt.title('B before optimization')
        plt.show()
    
    opt =  torch.optim.Adam([B], lr=0.0001, betas=(0.9, 0.999))
    matching_losses = []
    matching_loss_per_iter = AverageMeter()
    target_ind_tensor = torch.arange(c_target.shape[0])
    target_ind_tensor = torch.split(target_ind_tensor,10)
    for i in range(num_iterations):
        opt.zero_grad()
        for target_ind  in target_ind_tensor: 
            c_target_sample = c_target[target_ind]
            with torch.no_grad():
                weight_norm = B/B.sum(dim=1, keepdim=True)
                B.copy_(B)
            c_estimate = torch.matmul(B[target_ind],c_non_target)
            loss = matching_loss(c_estimate,c_target_sample)#/len(target_ind) 
            loss.backward()
            opt.step()
            matching_loss_per_iter.update(loss.data.item())
        matching_losses.append(matching_loss_per_iter.avg)
        
    if is_plot:
        plt.figure(figsize=(30,10))
        plt.imshow(B.squeeze().detach().cpu().numpy(),cmap = 'jet')
        plt.colorbar(shrink = 0.5)
        plt.title('B after optimization')
        plt.show()

        plt.figure()
        plt.plot(matching_losses)
        plt.title("Matching Loss")
        plt.show()
        
        with torch.no_grad():
            c_estimate = torch.matmul(B,c_non_target)
        
        fig, axs = plt.subplots(1, 2, figsize = (40,20))
        im = axs[0].imshow(c_target.squeeze().detach().cpu().numpy(),cmap = 'jet')
        axs[0].set_title(f"C Target")
        plt.colorbar(im, ax=axs[0],shrink = 0.3)

        im = axs[1].imshow(c_estimate.squeeze().detach().cpu().numpy(),cmap = 'jet')
        axs[1].set_title(f"C Est")
        plt.colorbar(im, ax=axs[1],shrink = 0.3)
    
    return B


def estimate_y(B,y_non_target):
    with torch.no_grad():
        y_hat = torch.matmul(B.squeeze(),y_non_target)
    return y_hat


def get_tsne(c_treat,c_control,n_components = 2, random_state = 0,is_plot = True):
    features = torch.cat((c_control,c_treat),axis = 0)
    flattenend_images = torch.flatten(features,start_dim = 1).cpu()#np.array([i.cpu().flatten() for i in images])
    print(f'Features Shape :{flattenend_images.shape}')
    print(f'Flattened Features Shape :{flattenend_images.shape}')
    
    print("Fitting T-SNE on features ======>")
    tsne = TSNE(
        random_state=random_state,
        n_components=n_components,perplexity = 80).fit_transform(flattenend_images)
    
    
    if is_plot:
        fig = plt.figure(figsize = (10,10))
        plt.scatter(tsne[:c_control.shape[0],0], tsne[:c_control.shape[0],1], label = "Control" )
        plt.scatter(tsne[c_control.shape[0]:,0], tsne[c_control.shape[0]:,1], label = "Treated" )
        plt.legend(loc='best')
        plt.title("T-SNE on learned representations")
        plt.show()
    return tsne



