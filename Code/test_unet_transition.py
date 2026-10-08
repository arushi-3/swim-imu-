"""Checks for U-Net output alignment, padding masks, and chunk coverage."""
import unittest
import numpy as np
import torch
import unet_transition as u


class UNetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.model=u.UNet()

    def test_sample_alignment_and_gradient(self):
        output=self.model(torch.randn(2,6,240))
        self.assertEqual(tuple(output.shape),(2,240))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_padding_is_excluded_and_sections_stay_separate(self):
        sections=[dict(x=np.ones((37,6),np.float32),y=np.ones(37,np.float32)),
                  dict(x=np.full((45,6),2,np.float32),y=np.zeros(45,np.float32))]
        dataset=u.chunks(sections,np.zeros(6,np.float32),np.ones(6,np.float32))
        x,y,mask=dataset.tensors
        self.assertEqual(tuple(x.shape),(2,6,240))
        self.assertEqual(mask.sum().item(),82)
        self.assertTrue((x[0,:,:37]==1).all() and (x[1,:,:45]==2).all())
        criterion=torch.nn.BCEWithLogitsLoss(reduction='none')
        first=(criterion(torch.zeros_like(y),y)*mask).sum()
        logits=torch.zeros_like(y); logits[mask==0]=100
        second=(criterion(logits,y)*mask).sum()
        self.assertEqual(first.item(),second.item())

    def test_inference_covers_short_and_tail_sections(self):
        for size in [37,240,241,301,3901]:
            section=dict(x=np.zeros((size,6),np.float32),y=np.zeros(size))
            p=u.infer(self.model,section,np.zeros(6,np.float32),np.ones(6,np.float32))
            self.assertEqual(len(p),size)
            self.assertTrue(np.isfinite(p).all() and ((p>=0)&(p<=1)).all())


if __name__=='__main__': unittest.main()
