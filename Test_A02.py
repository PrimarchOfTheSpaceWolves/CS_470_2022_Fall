import unittest
from unittest.mock import patch
import shutil
from pathlib import Path

import sys

import os
import subprocess as sub
import cv2
import numpy as np
import A02

RTOL=1e-07 
ATOL=1e-07 

base_dir = "assign02"
image_dir = base_dir + "/" + "images"
ground_dir = base_dir + "/" + "ground"
out_dir = base_dir + "/" + "output"

kernelArgs = [
    "3 3 0.125 127 1 2 1 0 0 0 -1 -2 -1",
    "3 3 0.125 127 1 0 -1 2 0 -2 1 0 -1",
    "3 3 0.0625 0 1 2 1 2 4 2 1 2 1",
    "3 3 0.125 127 0 1 0 1 -4 1 0 1 0",
    "3 1 0.125 127 1 0 -1",
    "1 3 0.125 127 1 0 -1",
    "7 5 0.0015873015 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35"
    ]

class Test_A02(unittest.TestCase):
    def do_test_one_filter_image(self, filename, ground_index):
        # Get kernel
        kernel = self.get_kernel_from_ground(ground_index)

        # Get alpha, beta
        alphaValue, betaValue = self.get_alpha_beta(ground_index)

        # Load input image
        image = cv2.imread(image_dir + "/" + filename, cv2.IMREAD_GRAYSCALE)
        input_stem = Path(filename).stem

        # Get filtered image
        out_image = A02.applyFilter(image, kernel)

        # Scale image accordingly
        out_image = cv2.convertScaleAbs(out_image, alpha=alphaValue, beta=betaValue)
                
        # Load ground image
        ground_image = self.load_ground_image(input_stem, ground_index) 
        
        # Is it correct?
        np.testing.assert_allclose(out_image, ground_image, rtol=RTOL, atol=ATOL)

    def get_kernel_from_ground(self, ground_index):
        # Get tokens
        tokens = kernelArgs[ground_index].split(" ")

        # Get shape
        kshape = tuple(map(int, tokens[0:2]))

        # Get kernel itself
        kernel = np.array(list(map(float, tokens[4:])))
        
        # Reshape
        kernel = kernel.reshape(kshape)

        return kernel

    def get_alpha_beta(self, ground_index):
        # Get tokens
        tokens = kernelArgs[ground_index].split(" ")

        # Get alpha and beta
        alpha = float(tokens[2])
        beta = float(tokens[3])

        return alpha, beta

    def test_filter_test_0(self):        
        self.do_test_one_filter_image("test.jpg", 0)

    def test_filter_test_1(self):        
        self.do_test_one_filter_image("test.jpg", 1)
    
    def test_filter_test_2(self):        
        self.do_test_one_filter_image("test.jpg", 2)

    def test_filter_test_3(self):        
        self.do_test_one_filter_image("test.jpg", 3)

    def test_filter_test_4(self):        
        self.do_test_one_filter_image("test.jpg", 4)
    
    def test_filter_test_5(self):        
        self.do_test_one_filter_image("test.jpg", 5)

    def test_filter_test_6(self):        
        self.do_test_one_filter_image("test.jpg", 6)

    def test_filter_ds_0(self):        
        self.do_test_one_filter_image("ds.png", 0)

    def test_filter_ds_1(self):        
        self.do_test_one_filter_image("ds.png", 1)
    
    def test_filter_ds_2(self):        
        self.do_test_one_filter_image("ds.png", 2)

    def test_filter_ds_3(self):        
        self.do_test_one_filter_image("ds.png", 3)

    def test_filter_ds_4(self):        
        self.do_test_one_filter_image("ds.png", 4)
    
    def test_filter_ds_5(self):        
        self.do_test_one_filter_image("ds.png", 5)

    def test_filter_ds_6(self):        
        self.do_test_one_filter_image("ds.png", 6)
        
    def load_ground_image(self, input_stem, ground_index):        
        ground_filepath = ground_dir + "/" + "OUT_" + input_stem + "_" + str(ground_index) +".png"
        ground_image = cv2.imread(ground_filepath, cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(ground_image, "Ground images are not loading: " + ground_filepath)
        return ground_image

    def run_main_normally(self, input_filename, ground_index):
        # Remove and recreate the output directory to be safe
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        
        # Get stem
        input_stem = Path(input_filename).stem
        
        # Set output path
        output_path = out_dir + "/" + "OUT_" + input_stem + "_" + str(ground_index) + ".png"

        # Remove file just to be safe        
        if os.path.exists(output_path):
            os.remove(output_path)
        
        # Run as usual
        other_params = kernelArgs[ground_index].split(" ")
        with patch('sys.argv', ['', image_dir + '/' + input_filename, output_path] + other_params):
            A02.main()

        # Is the output image there?        
        self.assertTrue(os.path.exists(output_path), "Output image not saved to: " + output_path)

        out_image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(out_image, "Output image exists but does not load:" + output_path)      

        # Load ground truth image    
        ground_image = self.load_ground_image(input_stem, ground_index)       

        # Is it correct?
        np.testing.assert_allclose(out_image, ground_image, rtol=RTOL, atol=ATOL)

    def test_main_insufficient_args(self):
        # Does it exit when insufficient arguments are provided?
        with self.assertRaises(SystemExit):
            with patch('sys.argv', ['']):
                A02.main()

        with self.assertRaises(SystemExit):
            with patch('sys.argv', ['', image_dir + '/' + 'test.jpg']):
                A02.main()

        with self.assertRaises(SystemExit):
            with patch('sys.argv', ['', image_dir + '/' + 'test.jpg', out_dir + '/' + 'output.png']):
                A02.main()

        other_params = kernelArgs[0].split(" ")
        for i in range(len(other_params)):
            sub_list = other_params[:(i+1)]
            with self.assertRaises(SystemExit):
                with patch('sys.argv', ['', image_dir + '/' + 'tartar_sauce.jpg', out_dir + '/' + 'output.png'] + sub_list):
                    A02.main()
        
    def test_main_bad_filename(self):
        # Does it exit when the image cannot be found? 
        other_params = kernelArgs[0].split(" ")       
        with self.assertRaises(SystemExit):
            with patch('sys.argv', ['', image_dir + '/' + 'tartar_sauce.jpg', out_dir + '/' + 'output.png'] + other_params):
                A02.main()       

    def test_main_normal(self):
        # Run normally...
        for i in range(len(kernelArgs)):
            self.run_main_normally("test.jpg", i)
        
        for i in range(len(kernelArgs)):
            self.run_main_normally("ds.png", i)       

def main():
    runner = unittest.TextTestRunner()
    runner.run(unittest.makeSuite(Test_A02))

if __name__ == '__main__':    
    main()

