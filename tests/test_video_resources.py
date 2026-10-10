"""Ensure failed inference or video output releases both stream handles."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from nn.hrnet.tools.infer import _run_video_cli


class VideoResourceTests(unittest.TestCase):
    def test_video_failures_release_capture_and_writer(self) -> None:
        for writer_opens in (False, True):
            with self.subTest(writer_opens=writer_opens), tempfile.TemporaryDirectory() as folder:
                args = argparse.Namespace(
                    source="input.mp4", output_video=str(Path(folder) / "out.mp4"),
                    frame_stride=1, output_fps=30.0, start_frame=0, end_frame=-1,
                    max_frames=-1, no_draw_scores=False,
                )
                capture = Mock()
                capture.isOpened.return_value = True
                capture.get.return_value = 30.0
                capture.read.return_value = (True, np.zeros((30, 30, 3), dtype=np.uint8))
                writer = Mock()
                writer.isOpened.return_value = writer_opens
                pipeline = Mock()
                pipeline.predict_numpy.side_effect = RuntimeError("inference failed")
                expected = "inference failed" if writer_opens else "Cannot create output video"
                with (
                    patch("nn.hrnet.tools.infer._build_pipeline", return_value=pipeline),
                    patch("cv2.VideoCapture", return_value=capture),
                    patch("cv2.VideoWriter", return_value=writer),
                    self.assertRaisesRegex(RuntimeError, expected),
                ):
                    _run_video_cli(args)
                capture.release.assert_called_once_with()
                writer.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
