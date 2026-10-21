import unittest

from transcribe import group_by_speaker


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


if __name__ == "__main__":
    unittest.main()
