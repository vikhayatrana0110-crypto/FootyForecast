import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.db_manager import DatabaseManager
from src.database.models import ModelVersion, create_all_tables


class TestSaveModelVersion(unittest.TestCase):
    """
    Exercise save_model_version against a throwaway in-memory database.

    DatabaseManager builds a PostgreSQL URL from the environment, so the
    constructor is bypassed and the engine swapped for SQLite. Only the row
    bookkeeping is under test here, which is backend independent.
    """

    def setUp(self):
        self.db = DatabaseManager.__new__(DatabaseManager)
        self.db.engine = create_engine('sqlite:///:memory:')
        self.db.SessionLocal = sessionmaker(bind=self.db.engine)
        self.db.session = None
        create_all_tables(self.db.engine)

    def _rows(self):
        session = self.db.SessionLocal()
        try:
            return session.query(ModelVersion).order_by(ModelVersion.model_id).all()
        finally:
            session.close()

    def test_repeated_saves_reuse_one_row(self):
        """Retraining must update the existing row, not append a new one.

        Every pipeline run previously inserted a fresh row per model, so the
        table reached 51 rows describing 3 models and the training history
        became unreadable.
        """
        ids = [
            self.db.save_model_version(
                'outcome_classifier', '1.0',
                {'accuracy': 0.59, 'log_loss': 0.89, 'f1_macro': 0.44},
                'src/app/models/outcome_classifier_v1.0.joblib')
            for _ in range(4)
        ]

        self.assertEqual(len(set(ids)), 1, "each save should return the same row id")
        self.assertEqual(len(self._rows()), 1, "four retrains must leave one row")

    def test_latest_metrics_win(self):
        """The retained row must carry the most recent run's metrics."""
        self.db.save_model_version('outcome_classifier', '1.0',
                                   {'accuracy': 0.5417, 'log_loss': 0.95, 'f1_macro': 0.40},
                                   'old/path.joblib')
        self.db.save_model_version('outcome_classifier', '1.0',
                                   {'accuracy': 0.5944, 'log_loss': 0.8940, 'f1_macro': 0.4442},
                                   'src/app/models/outcome_classifier_v1.0.joblib')

        row = self._rows()[0]
        self.assertAlmostEqual(row.accuracy, 0.5944)
        self.assertEqual(row.model_path, 'src/app/models/outcome_classifier_v1.0.joblib')
        self.assertTrue(row.is_active)

    def test_distinct_versions_get_distinct_rows(self):
        """Bumping the version must create a new row rather than overwrite."""
        first = self.db.save_model_version('outcome_classifier', '1.0',
                                           {'accuracy': 0.59}, 'a.joblib')
        second = self.db.save_model_version('outcome_classifier', '2.0',
                                            {'accuracy': 0.61}, 'b.joblib')

        self.assertNotEqual(first, second)
        rows = self._rows()
        self.assertEqual(len(rows), 2)
        active = [r for r in rows if r.is_active]
        self.assertEqual(len(active), 1, "only the newest version stays active")
        self.assertEqual(active[0].version, '2.0')

    def test_separate_models_are_independent(self):
        """Saving one model must not deactivate a different model."""
        self.db.save_model_version('outcome_classifier', '1.0',
                                   {'accuracy': 0.59}, 'a.joblib')
        self.db.save_model_version('home_goals_regressor', '1.0',
                                   {'mae': 1.05, 'rmse': 1.37}, 'b.joblib')

        rows = {r.model_name: r for r in self._rows()}
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows['outcome_classifier'].is_active)
        self.assertTrue(rows['home_goals_regressor'].is_active)
        # Regressor metrics must not leak into the classifier columns.
        self.assertIsNone(rows['home_goals_regressor'].accuracy)
        self.assertAlmostEqual(rows['home_goals_regressor'].mae, 1.05)


if __name__ == '__main__':
    unittest.main()
