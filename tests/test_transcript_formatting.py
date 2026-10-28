import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from transcribe import available_output_base, group_by_speaker


class TranscriptFormattingTests(unittest.TestCase):
    def test_unpunctuated_segments_become_readable_sentences(self):
        segments = [
            (0.0, 8.0, "i found that position wasn't for me at all"),
            (8.0, 16.0, "i found it restrictive and limiting in our approaches"),
            (16.0, 24.0, "after about ten months i went out on my own"),
        ]
        turns = [(0.0, 24.0, "SPEAKER_00")]

        blocks = group_by_speaker(segments, turns)

        self.assertEqual(
            blocks[0][2],
            [
                "I found that position wasn't for me at all.",
                "I found it restrictive and limiting in our approaches.",
                "After about ten months I went out on my own.",
            ],
        )

    def test_existing_punctuation_is_preserved(self):
        segments = [(0.0, 3.0, "why did I leave?"), (3.0, 6.0, "It was limiting!")]
        turns = [(0.0, 6.0, "SPEAKER_00")]

        blocks = group_by_speaker(segments, turns)

        self.assertEqual(blocks[0][2], ["Why did I leave?", "It was limiting!"])

    def test_existing_word_file_selects_a_fresh_output_name(self):
        with TemporaryDirectory() as folder:
            base = Path(folder) / "interview.m4a"
            base.with_suffix(".docx").touch()

            selected = available_output_base(base, "both")

            self.assertEqual(selected.name, "interview_2.m4a")
            self.assertFalse(selected.with_suffix(".docx").exists())
            self.assertFalse(selected.with_suffix(".md").exists())

    def test_suffix_increments_until_both_outputs_are_free(self):
        with TemporaryDirectory() as folder:
            base = Path(folder) / "interview.m4a"
            base.with_suffix(".md").touch()
            (Path(folder) / "interview_2.docx").touch()

            selected = available_output_base(base, "both")

            self.assertEqual(selected.name, "interview_3.m4a")


if __name__ == "__main__":
    unittest.main()
