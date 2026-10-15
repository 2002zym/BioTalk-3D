import unittest


from EATalker_data_template.generic_data_processor import GenericDataProcessor


class GenericDataProcessorTimecodeTests(unittest.TestCase):
    def setUp(self):
        self.proc = GenericDataProcessor("liu1")

    def test_parse_ff_dot_subframe_not_milliseconds(self):
        # HH:MM:SS:FF.mmm where .mmm is sub-frame fraction, not seconds-millisecond.
        got = self.proc.parse_timecode_to_seconds("10:38:35:16.995")
        expected = 10 * 3600 + 38 * 60 + 35 + (16 + 0.995) / 60.0
        self.assertAlmostEqual(got, expected, places=6)

    def test_parse_ffmmm_without_dot(self):
        # Compatibility format: HH:MM:SS:FFmmm
        got = self.proc.parse_timecode_to_seconds("10:38:35:16995")
        expected = 10 * 3600 + 38 * 60 + 35 + (16 + 0.995) / 60.0
        self.assertAlmostEqual(got, expected, places=6)

    def test_parse_ff_only(self):
        got = self.proc.parse_timecode_to_seconds("10:38:35:16")
        expected = 10 * 3600 + 38 * 60 + 35 + 16 / 60.0
        self.assertAlmostEqual(got, expected, places=6)


if __name__ == "__main__":
    unittest.main()
