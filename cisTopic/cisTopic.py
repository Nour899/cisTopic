import torch
import torch_scatter
import numpy as np
import scanpy as sc
from tqdm import tqdm
import time


class cisTopic:

    def __init__(self, sc_data_atac, n_topics, alpha, beta):

        self.sc_data_atac = sc_data_atac
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
       

    def sc_to_tensor(self):
    
        n = self.sc_data_atac.X.nonzero()[0].shape[0]
        atac = torch.zeros([2,n])
        atac[0,:] = torch.from_numpy(self.sc_data_atac.X.nonzero()[0])
        atac[1,:] = torch.from_numpy(self.sc_data_atac.X.nonzero()[1])
        atac = atac.type(torch.LongTensor)
       
        self.atac = atac

    def create_cell_batches (self,n_batches,batch_size,n_cells):
    ## this function is used to create batch of cells for the atac data 
    # that should be transfered one at a time to the GPU to avoid GPU running 
     # out of memory. Because of the atac data format used to feed share-topic this function
       # is necessary to find the coordinates of the non-zero reads per batch.
        c = torch.arange(0,n_batches,batch_size)
        c = torch.hstack((c,torch.tensor(n_cells)))
        
            
        a,b,rep_c = torch.unique(self.atac[0,:],return_inverse =True,return_counts=True)
        rep_c=rep_c.type(torch.LongTensor) 
        cell_array = torch.arange(n_cells)
        rep_c_ = torch.repeat_interleave(cell_array, rep_c, dim = 0)
        

        q = 1
        t = 0
        t_ = 0
            
        atac_cell_batches = torch.zeros(c.shape[0],dtype = torch.int64)   
        for i in  torch.arange(batch_size,n_batches,batch_size):
            atac_cell_batches[q] = t+torch.sum(rep_c[t_:i])
            t = int(atac_cell_batches[q].item())
            t_ = i
            q+=1
            
        atac_cell_batches[q] = t+torch.sum(rep_c[t_:])
        
        return atac_cell_batches, c, rep_c,rep_c_, cell_array
    

    def create_region_batches(self, atac_cell_batches, cell_array):
        ###This function take the regions presented per batch of cells in the atac data with share-topic format. The presented regions
        # are ord
        indices = torch.zeros([self.atac.shape[1]])
        indices2 = torch.zeros([self.atac.shape[1]])
        regions = torch.tensor([])
        region_rep = torch.tensor([])
        region_rep_ = torch.tensor([])
        region_batching = torch.zeros([atac_cell_batches.shape[0]])
        for i in torch.arange(1,atac_cell_batches.shape[0]):
            
            sorted, indices[int(atac_cell_batches[i-1].item()):
                            int(atac_cell_batches[i].item())] = torch.sort(
                self.atac[1,int(atac_cell_batches[i-1].item()):
                            int(atac_cell_batches[i].item())])
            sorted2,indices2[int(atac_cell_batches[i-1].item()):
                                int(atac_cell_batches[i].item())] = torch.sort(
                indices[int(atac_cell_batches[i-1].item()):
                        int(atac_cell_batches[i].item())])
                
            a,b,rep_r = torch.unique(
                sorted,return_inverse =True,return_counts=True)
            regions = torch.cat((regions,a),0)
            region_batching[i] = a.shape[0]+region_batching[i-1] ##list of regiosn per batch
            region_rep = torch.cat((region_rep, rep_r),0) ##how many times presented region in a batch are repeated
            rep_r = rep_r.type(torch.LongTensor) 
            cell_array = torch.arange(a.shape[0]) 
            rep_r_ = torch.repeat_interleave(cell_array, rep_r,dim=0)
            region_rep_ = torch.cat((region_rep_, rep_r_),0) 

        region_rep = region_rep.type(torch.LongTensor) 
        regions = regions.type(torch.LongTensor) 
        indices = indices.int()
        indices2 = indices2.int()
        region_rep_ = region_rep_.type(torch.int64)
        return regions, region_batching, region_rep, region_rep_, indices, indices2
    
    def initialization(self, n_cells,device):
        m = torch.distributions.dirichlet.Dirichlet(self.alpha[0,:]) 
        theta_tmp = m.sample([n_cells])
        theta_tmp = theta_tmp.to(device)
        
        m =  torch.distributions.dirichlet.Dirichlet(self.beta[0,:])
        phi_tmp = m.sample([self.n_topics])
        phi_tmp = phi_tmp.to(device)
        
        return theta_tmp, phi_tmp
    
    def move_to_GPU(self, c,region_rep, regions, indices, indices2, rep_c, rep_c_, region_rep_, R, device):
    
        c=c.to(device)
        region_rep = region_rep.to(device)
        regions = regions.to(device)
        indices = indices.to(device)
        indices2 = indices2.to(device)
        rep_c = rep_c.to(device)
        rep_c_ = rep_c_.to(device)
        region_rep_ = region_rep_.to(device)
        t = torch.arange(0,self.n_topics,dtype=torch.uint8,device=device).reshape(self.n_topics,1)
        self.alpha = torch.ones([1,self.n_topics,],device=device)*(50/self.n_topics)
        self.beta = torch.ones([1,R],device=device)*0.1
        
        return  c, region_rep, regions, indices, indices2, rep_c, rep_c_, region_rep_, t
    
    def Gibbs_Sampler(self, n_samples,n_burnin,
                       n_b_samples, batch_size, n_cells,R, c, region_rep,
                         regions,region_batching, indices, indices2, rep_c, rep_c_, region_rep_, 
                         t,atac_cell_batches , theta_tmp, phi_tmp, device):
    
        
         
        theta = torch.zeros([n_b_samples,n_cells, self.n_topics])
        phi = torch.zeros([n_b_samples,self.n_topics, R])
        L_r = torch.zeros(n_b_samples, device=device)
        with tqdm(total=n_samples, desc="Processing") as pbar:
            for sample in range(0,n_samples):
                
                n_t_c = torch.zeros([n_cells,self.n_topics],device=device)
                n_t_r = torch.zeros([R,self.n_topics],device=device)
                
                L_r[int(sample/n_burnin)] = 0
                
                for i in torch.arange(1,c.shape[0]):

                    theta_ = theta_tmp[c[i-1]:c[i],:]
                    phi_ = phi_tmp[:,regions[int(region_batching[i-1].item()):int(region_batching[i].item())]]
                    n_regions = phi_.shape[1]
                    z_atac_ = torch.repeat_interleave(
                        theta_, rep_c[c[i-1]:c[i]],dim=0)##cells in this batch
                    phi_ = torch.repeat_interleave(
                        phi_,region_rep[int(region_batching[i-1].item()):int(region_batching[i].item())],dim=1)##regions in this batch
                    phi_ = torch.index_select(phi_,1, indices2[int(atac_cell_batches[i-1].item()):
                                                                int(atac_cell_batches[i].item())])##reorder the regions acording to data
                    
                    z_atac_ = torch.multiply(z_atac_.T,phi_)
                    z_L_r = torch.sum(z_atac_,axis=0)
                    z_L_r = torch.sum(torch.log(z_L_r))
                    L_r[int(sample/n_burnin)]+= z_L_r
                    z_atac_ = z_atac_/torch.sum(z_atac_,axis=0)
                    
                    z_atac_ = torch.cumsum(z_atac_,axis=0)
                    u_atac = torch.rand([1,z_atac_.shape[1]],device=device)
                    
                    z_atac_ = torch.searchsorted( (z_atac_.T).contiguous(),(u_atac.T).contiguous()).contiguous()
                    
                    z_atac_ = torch.gt((z_atac_==t.T), 0).int()
                    
                    h = rep_c_[int(atac_cell_batches[i-1].item()):
                                int(atac_cell_batches[i].item())]%batch_size###update for the cells in this batch
                    
                    n_t_c[c[i-1]:c[i],:]+= (
                        torch_scatter.scatter(z_atac_,h,dim=0,reduce="sum").reshape([int((c[i]-c[i-1]).item()),self.n_topics]))
                    
                    z_atac_ = torch.index_select(z_atac_,0,indices[int(atac_cell_batches[i-1].item()):
                                                                    int(atac_cell_batches[i].item())])
                    
                    h = region_rep_[int(atac_cell_batches[i-1].item()):
                                    int(atac_cell_batches[i].item())]
                    
                    n_t_r[regions[int(region_batching[i-1].item()):int(region_batching[i].item())],:]+= (
                        torch_scatter.scatter(z_atac_,h,dim=0,reduce="sum").reshape([n_regions,self.n_topics]))
                    
                #######################################Sampling theta and phi##########################
                
                m = torch.distributions.dirichlet.Dirichlet(self.beta+n_t_r.T)
                phi_tmp = m.sample()
                phi[int(sample/n_burnin),:,:] = phi_tmp 
                m = torch.distributions.dirichlet.Dirichlet(self.alpha+n_t_c)
                
                theta_tmp = m.sample()
                theta[int(sample/n_burnin),:,:] = theta_tmp
                time.sleep(0.1)
                pbar.update(1)
        return theta, phi 
    

    def fit(self,batch_size,n_samples,n_burnin,dev= "cpu",
                save_data=True,path=""):
    
        self.sc_to_tensor()
       
        
        if save_data:
            torch.save(self.atac, path+str("atac_share_topic.txt"))
           
        ### add path to save the files
        
        device = torch.device(dev)
        
        
        n_cells = self.atac.shape[0]
        l = torch.unique(self.atac[1,:])
        R = l.shape[0]
        self.alpha = torch.ones([1, self.n_topics])*self.alpha
        self.beta = torch.ones([1,R])*self.beta
        n_b_samples = int(n_samples/n_burnin)
        n_batches = n_cells+batch_size-(n_cells%batch_size)
        
        atac_cell_batches, c, rep_c,rep_c_, cell_array = self.create_cell_batches(n_batches,batch_size,n_cells)
        
        regions, region_batching, region_rep, region_rep_, indices, indices2 = self.create_region_batches(atac_cell_batches, cell_array)
        
        theta_tmp, phi_tmp = self.initialization(n_cells,device)
        
        if dev!="cpu":
            c, region_rep, regions, indices, indices2, rep_c, rep_c_, region_rep_, t = self.move_to_GPU(c,region_rep, regions, 
                                                                                                        indices, indices2, rep_c, rep_c_, region_rep_, R, device)
        
        
        else:
            t = torch.arange(0,self.n_topics,dtype=torch.uint8,device=device).reshape(self.n_topics,1)
            self.alpha = torch.ones([1,self.n_topics,],device=device)*(50/self.n_topics)
            self.beta = torch.ones([1,R],device=device)*0.1

        theta,  phi = self.Gibbs_Sampler(n_samples,n_burnin,
                                        n_b_samples, batch_size, n_cells,R, c, region_rep,
                                            regions,region_batching, indices, indices2, rep_c, rep_c_, region_rep_, 
                                            t,atac_cell_batches , theta_tmp, phi_tmp, device)
        
        torch.cuda.empty_cache()
        
        return theta, phi
        

        