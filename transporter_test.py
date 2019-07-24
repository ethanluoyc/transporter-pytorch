import torch
import transporter
import unittest

batch_size = 8
image_channels = 1
num_features = 32
k = 1

class TransporterTest(unittest.TestCase):
    def testEncoder(self):
        out_channels = 128
        encoder = transporter.FeatureEncoder(
            in_channels=image_channels
        )
        sample_image = torch.randn(batch_size, image_channels, 32, 32)
        feature_maps = encoder(sample_image)
        self.assertEqual(feature_maps.shape, (batch_size, out_channels, 8, 8))

    def testPointNet(self):
        pose_regressor = transporter.PoseRegressor(image_channels, k)
        sample_image = torch.randn(batch_size, image_channels, 32, 32)
        heatmaps = pose_regressor(sample_image)
        self.assertEqual(heatmaps.shape, (batch_size, k, 8, 8))

    def testRefineNet(self):
        refine_net = transporter.RefineNet(image_channels)
        sample_image = torch.randn(batch_size, 128, 8, 8)
        heatmaps = refine_net(sample_image)
        self.assertEqual(heatmaps.shape, (batch_size, image_channels, 32, 32))

    def testGaussianMap(self):
        num_keypoints = 5
        features = torch.zeros(batch_size, num_keypoints, 32, 32)
        heatmaps = transporter.gaussian_map(features)
        self.assertEqual(heatmaps.shape, (batch_size, num_keypoints, 32, 32))

    def testTransport(self):
        N, K, H, W, D = 1, 2, 32, 32, 4
        source_keypoints = torch.zeros(N, K, H, W)
        target_keypoints = torch.zeros(N, K, H, W)
        source_features  = torch.zeros(N, D, H, W)
        target_features = torch.zeros(N, D, H, W)
        transported = transporter.transport(
            source_keypoints,
            target_keypoints,
            source_features,
            target_features
        )
        self.assertEqual(transported.shape, target_features.shape)

if __name__ == "__main__":
    unittest.main()