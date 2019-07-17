import torch
import imm
import unittest

batch_size = 8
image_channels = 1
num_features = 32

class TransporterTest(unittest.TestCase):
    def testGaussianMap(self):
        num_keypoints = 5
        features = torch.zeros(batch_size, num_keypoints * 2, 32, 32)
        features = imm.renormalize(imm.spatial_softmax(features))
        heatmaps = imm.gaussian_map(features)

        self.assertEqual(heatmaps.shape, (batch_size, num_keypoints, 32, 32))


if __name__ == "__main__":
    unittest.main()