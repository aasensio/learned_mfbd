import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import time
import model_encdec as model



class deep_mfbd(object):
    def __init__(self):
        self.cuda = torch.cuda.is_available()
        self.n_frames = 7
        self.depth = self.n_frames-1
        
        self.device = torch.device("cuda" if self.cuda else "cpu")

        self.model = model.deconv_block(n_frames=self.n_frames).to(self.device)
                        
        self.checkpoint = 'encdec_network/2018-01-22-10:13.pth.tar'

        print("=> loading checkpoint '{}'".format(self.checkpoint))
        if (self.cuda):
            checkpoint = torch.load(self.checkpoint)
        else:
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])        
        print("=> loaded checkpoint '{}'".format(self.checkpoint))

        
    def test(self):

        arcsec_per_px = 0.059
        self.model.eval()

        ims = np.expand_dims(np.load('data/ims.npy'), axis=0)

        data = torch.from_numpy((ims / 1e3).astype('float32')).to(self.device)

        with torch.no_grad():            
            start = time.time()
            out = self.model(data)                
            print('Elapsed time : {0} s'.format(time.time()-start))

            out = 1e3 * np.squeeze(out.to("cpu").data.numpy())
                        
            pl.close('all')
            fig, ax = pl.subplots(ncols=2, nrows=2, figsize=(10,10))
            ax[0,0].imshow(ims[0,0,:,:], extent=(0,960*arcsec_per_px,0,960*arcsec_per_px))                
            ax[0,1].imshow(out, extent=(0,960*arcsec_per_px,0,960*arcsec_per_px))
            ax[1,0].imshow(ims[0,0,100:500,100:500], extent=(0,400*arcsec_per_px,0,400*arcsec_per_px))                
            ax[1,1].imshow(out[100:500,100:500], extent=(0,400*arcsec_per_px,0,400*arcsec_per_px))
            ax[0,0].set_title('Frame')
            ax[0,1].set_title('NN')

            pl.show()
            
                                
if (__name__ == '__main__'):    
    
    deep_mfbd_network = deep_mfbd()
    deep_mfbd_network.test()
