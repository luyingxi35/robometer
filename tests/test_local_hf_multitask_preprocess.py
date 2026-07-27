import unittest
from types import SimpleNamespace

from robometer.data.scripts.preprocess_local_hf_datasets import LocalHFDatasetPreprocessor


class TestLocalHFMultitaskPreprocess(unittest.TestCase):
    def _make_preprocessor(self, max_frames=32):
        preprocessor = object.__new__(LocalHFDatasetPreprocessor)
        preprocessor.config = SimpleNamespace(max_frames_for_preprocessing=max_frames)
        return preprocessor

    def test_equal_video_and_progress_lengths_use_same_indices(self):
        preprocessor = self._make_preprocessor(max_frames=4)

        video_indices, progress_indices, offset = preprocessor._sample_aligned_video_progress_indices(10, 10)

        self.assertEqual(video_indices, progress_indices)
        self.assertEqual(offset, 0)
        self.assertEqual(len(video_indices), 4)

    def test_one_extra_initial_video_frame_is_dropped(self):
        preprocessor = self._make_preprocessor(max_frames=32)

        video_indices, progress_indices, offset = preprocessor._sample_aligned_video_progress_indices(21, 20)

        self.assertEqual(offset, 1)
        self.assertEqual(video_indices[0], 1)
        self.assertEqual(video_indices[-1], 20)
        self.assertEqual(progress_indices[0], 0)
        self.assertEqual(progress_indices[-1], 19)

    def test_other_video_progress_mismatches_are_rejected(self):
        preprocessor = self._make_preprocessor()

        with self.assertRaisesRegex(ValueError, "video/progress length mismatch"):
            preprocessor._sample_aligned_video_progress_indices(22, 20)


if __name__ == "__main__":
    unittest.main()
