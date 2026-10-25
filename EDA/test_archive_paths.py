import tempfile
import unittest
from pathlib import Path
from archive_paths import resolve_recorded_path


class ArchivePathChecks(unittest.TestCase):
    def test_missing_historical_path_resolves_to_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            eda=Path(temporary)/'EDA'
            archived=eda/'legacy'/'old'/'validation.json'
            archived.parent.mkdir(parents=True)
            archived.write_text('{}')
            self.assertEqual(resolve_recorded_path(eda/'old'/'validation.json',eda),archived)
            self.assertEqual(resolve_recorded_path('EDA/old/validation.json',eda),archived)

    def test_active_path_takes_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            eda=Path(temporary)/'EDA'
            active=eda/'data.csv'
            archived=eda/'legacy'/'data.csv'
            archived.parent.mkdir(parents=True)
            active.write_text('current')
            archived.write_text('old')
            self.assertEqual(resolve_recorded_path(active,eda),active)

    def test_missing_or_unrelated_paths_fail_loudly(self):
        with tempfile.TemporaryDirectory() as temporary:
            eda=Path(temporary)/'EDA'
            for missing in [eda/'missing.csv',eda/'legacy'/'missing.csv',Path(temporary)/'unrelated.csv']:
                with self.assertRaises(FileNotFoundError):
                    resolve_recorded_path(missing,eda)


if __name__=='__main__':unittest.main()
