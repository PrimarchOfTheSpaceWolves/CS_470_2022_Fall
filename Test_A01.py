import unittest
from unittest.mock import patch
import shutil
from pathlib import Path

import sys

import os
import subprocess as sub
import cv2
import numpy as np
import A01

RTOL=1e-07 
ATOL=1e-07 

class Test_A01(unittest.TestCase):
    def do_test_one_slice_image(self, filename, lower_bound, upper_bound):
        # Load input image
        image = cv2.imread("images/" + filename, cv2.IMREAD_GRAYSCALE)
        input_stem = Path(filename).stem

        # Get sliced image
        out_image = A01.slice_image(image, lower_bound, upper_bound)
        
        # Load ground image
        ground_image = self.load_ground_image(input_stem, lower_bound, upper_bound) 
        
        # Is it correct?
        np.testing.assert_allclose(out_image, ground_image, rtol=RTOL, atol=ATOL)

    def test_slice_image_0_0(self):        
        self.do_test_one_slice_image("test.jpg", 0, 0)

    def test_slice_image_0_100(self):      
        self.do_test_one_slice_image("test.jpg", 0, 100)

    def test_slice_image_0_200(self):      
        self.do_test_one_slice_image("test.jpg", 0, 200)

    def test_slice_image_0_255(self):      
        self.do_test_one_slice_image("test.jpg", 0, 255)

    def test_slice_image_50_100(self):      
        self.do_test_one_slice_image("test.jpg", 50, 100)

    def test_slice_image_50_200(self):      
        self.do_test_one_slice_image("test.jpg", 50, 200)

    def test_slice_image_50_255(self):      
        self.do_test_one_slice_image("test.jpg", 50, 255)

    def test_slice_image_100_200(self):      
        self.do_test_one_slice_image("test.jpg", 100, 200)

    def test_slice_image_200_255(self):      
        self.do_test_one_slice_image("test.jpg", 200, 255)

    def test_slice_image_255_255(self):      
        self.do_test_one_slice_image("test.jpg", 255, 255)

    def test_slice_image_256_255(self):      
        self.do_test_one_slice_image("test.jpg", 256, 255)

    def test_slice_image_other_0_200(self):      
        self.do_test_one_slice_image("other.jpg", 0, 200)

    def test_slice_image_other_200_255(self):      
        self.do_test_one_slice_image("other.jpg", 200, 255)

    def test_slice_image_other_240_255(self):      
        self.do_test_one_slice_image("other.jpg", 240, 255)   

    def get_out_filename(self, input_stem, lower_bound, upper_bound):
        lower_bound = str(lower_bound)
        upper_bound = str(upper_bound)
        return "OUT_" + input_stem + "_" + lower_bound + "_" + upper_bound +".png"  
        
    def load_ground_image(self, input_stem, lower_bound, upper_bound):
        ground_dir = "ground"
        ground_filepath = ground_dir + "/" + self.get_out_filename(input_stem, lower_bound, upper_bound)
        ground_image = cv2.imread(ground_filepath, cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(ground_image, "Ground images are not loading: " + ground_filepath)
        return ground_image

    def run_main_normally(self, input_stem, lower_bound, upper_bound):
        
        # Set output directory
        out_dir = 'output'

        # Remove and recreate the output directory to be safe
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        # Run as usual
        with patch('sys.argv', ['', 'images/' + input_stem + '.jpg', str(lower_bound), str(upper_bound), out_dir]):
            A01.main()

        # Is the output image there?
        out_filename = self.get_out_filename(input_stem, lower_bound, upper_bound)
        out_path = out_dir + "/" + out_filename
        self.assertTrue(os.path.exists(out_path))

        out_image = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(out_image, "Output image exists but does not load:" + out_path)      

        # Load ground truth image    
        ground_image = self.load_ground_image(input_stem, lower_bound, upper_bound)       

        # Is it correct?
        np.testing.assert_allclose(out_image, ground_image, rtol=RTOL, atol=ATOL)

    def test_main_insufficient_args(self):
        # Does it exit when insufficient arguments are provided?
        with self.assertRaises(SystemExit):
            with patch('sys.argv', ['']):
                A01.main()

        with self.assertRaises(SystemExit):
            with patch('sys.argv', ['', 'images/test.jpg']):
                A01.main()

        with self.assertRaises(SystemExit):
            with patch('sys.argv', ['', 'images/test.jpg', '100']):
                A01.main()

        with self.assertRaises(SystemExit):
            with patch('sys.argv', ['', 'images/tartar_sauce.jpg', '100', '200']):
                A01.main()
        
    def test_main_bad_filename(self):
        # Does it exit when the image cannot be found?        
        with self.assertRaises(SystemExit):
            with patch('sys.argv', ['', 'images/tartar_sauce.jpg', '100', '200', 'output']):
                A01.main()       

    def test_main_normal(self):
        # Run normally...
        self.run_main_normally("test", 100, 200)
        self.run_main_normally("test", 50, 100)  
        self.run_main_normally("other", 0, 200)
        self.run_main_normally("other", 200, 255)            

def main():
    runner = unittest.TextTestRunner()
    runner.run(unittest.makeSuite(Test_A01))

if __name__ == '__main__':    
    main()
