import unittest


from tools.build_biotalk_splits import infer_language, normalize_sheet_name


class BioTalkSplitBuilderTests(unittest.TestCase):
    def test_infer_language_zh_en_mix(self):
        self.assertEqual(infer_language("你好，今天怎么样？"), "zh")
        self.assertEqual(infer_language("How are you today?"), "en")
        self.assertEqual(infer_language("你好 world"), "mix")

    def test_normalize_sheet_name(self):
        self.assertEqual(normalize_sheet_name("session1"), "session1")
        self.assertEqual(normalize_sheet_name("s1"), "session1")
        self.assertEqual(normalize_sheet_name("S4"), "session4")
        self.assertEqual(normalize_sheet_name("unknown"), "unknown")


if __name__ == "__main__":
    unittest.main()
