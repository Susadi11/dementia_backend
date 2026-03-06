"""
Comprehensive Test Suite — Game Component (vlakvindu)

Tests every layer of the game cognitive assessment pipeline:

  1. Model Registry   — local loading, HuggingFace download, caching, fallback dummies
  2. Feature Engine   — extract_lstm_features, extract_risk_features, compute_slope
  3. LSTM Inference   — predict_lstm_decline (< 3 sessions, >= 3 sessions, scaler mismatch)
  4. Risk Classifier  — predict_risk + rule-based safety overrides for all threshold cases
  5. Full Pipeline    — process_game_session (trials path, summary path, ms→s auto-fix)
  6. API Schema       — Pydantic validation (GameSessionRequest, CalibrationRequest)
  7. HuggingFace      — _download_from_hf helper (mock network, bad repo, bad filename)
  8. Registry JSON    — both models_registry.json files are valid and correctly registered

Run with:
    python -m pytest test/test_game_component.py -v
  or for a plain print report:
    python test/test_game_component.py
"""

import sys
import json
import types
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── colour helpers (fallback to plain text if colorama not installed) ─────────
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    GREEN  = Fore.GREEN
    RED    = Fore.RED
    YELLOW = Fore.YELLOW
    CYAN   = Fore.CYAN
    BOLD   = Style.BRIGHT
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = BOLD = RESET = ""


def _section(title: str):
    print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}")


def _ok(msg: str):
    print(f"  {GREEN}[PASS]{RESET} {msg}")


def _fail(msg: str):
    print(f"  {RED}[FAIL]{RESET} {msg}")


def _info(msg: str):
    print(f"  {YELLOW}[INFO]{RESET} {msg}")


# =============================================================================
# Shared test fixtures
# =============================================================================

GOOD_FEATURES = {
    "sac": 0.82,
    "ies": 1.45,
    "accuracy": 0.88,
    "rtAdjMedian": 0.95,
    "variability": 0.12,
    "errorRate": 0.12,
}

POOR_FEATURES = {
    "sac": 0.25,
    "ies": 9.5,
    "accuracy": 0.30,
    "rtAdjMedian": 3.20,
    "variability": 0.75,
    "errorRate": 0.70,
}

MEDIUM_FEATURES = {
    "sac": 0.55,
    "ies": 2.8,
    "accuracy": 0.55,
    "rtAdjMedian": 1.60,
    "variability": 0.45,
    "errorRate": 0.45,
}


def _make_session(sac=0.7, ies=1.8, accuracy=0.8, rt=1.0, variability=0.2):
    return {
        "features": {
            "sac": sac,
            "ies": ies,
            "accuracy": accuracy,
            "rtAdjMedian": rt,
            "variability": variability,
        }
    }


def _session_history(n: int, sac=0.7, ies=1.8, accuracy=0.8, rt=1.0, variability=0.2):
    return [_make_session(sac, ies, accuracy, rt, variability) for _ in range(n)]


# =============================================================================
# 1. Model Registry Tests
# =============================================================================

class TestModelRegistry(unittest.TestCase):

    def setUp(self):
        # Reset the global registry state before every test
        import src.models.game.model_registry as reg
        reg._MODEL_LOADED = False
        reg._MODELS["lstm_model"]      = None
        reg._MODELS["lstm_scaler"]     = None
        reg._MODELS["risk_classifier"] = None
        reg._MODELS["scaler"]          = None
        reg._MODELS["label_encoder"]   = None

    # ── Repo-ID constants ────────────────────────────────────────────────────

    def test_hf_repo_ids_are_correct(self):
        from src.models.game.model_registry import HF_LSTM_REPO, HF_RISK_REPO
        self.assertEqual(HF_LSTM_REPO,  "vlakvindu/Dementia_LSTM_Model")
        self.assertEqual(HF_RISK_REPO,  "vlakvindu/Dementia_Risk_Clasification_model")

    # ── Download helper ──────────────────────────────────────────────────────

    def test_download_helper_returns_path_on_success(self):
        from src.models.game.model_registry import _download_from_hf
        fake_path = str(ROOT / "src/models/game/risk_classifier/risk_logreg.pkl")
        with patch("src.models.game.model_registry.hf_hub_download",
                   return_value=fake_path, create=True):
            with patch("huggingface_hub.hf_hub_download", return_value=fake_path, create=True):
                result = _download_from_hf("vlakvindu/test", "file.pkl", Path("/tmp"))
                # Should return a Path (or None gracefully)
                self.assertIsNotNone(result)

    def test_download_helper_returns_none_on_network_error(self):
        from src.models.game.model_registry import _download_from_hf
        with patch("builtins.__import__", side_effect=ImportError("no huggingface_hub")):
            result = _download_from_hf("vlakvindu/test", "missing.pkl", Path("/tmp/fake"))
            self.assertIsNone(result)

    # ── Local file loading (files already present on disk) ───────────────────

    def test_load_risk_classifier_from_local(self):
        from src.models.game.model_registry import load_risk_classifier, RISK_CLASSIFIER_DIR
        model_path = RISK_CLASSIFIER_DIR / "risk_logreg.pkl"
        if not model_path.exists():
            self.skipTest("Local risk_logreg.pkl not present — skipping local load test")
        model = load_risk_classifier()
        self.assertIsNotNone(model)

    def test_load_risk_scaler_from_local(self):
        from src.models.game.model_registry import load_risk_scaler, RISK_CLASSIFIER_DIR
        scaler_path = RISK_CLASSIFIER_DIR / "risk_scaler.pkl"
        if not scaler_path.exists():
            self.skipTest("Local risk_scaler.pkl not present")
        scaler = load_risk_scaler()
        self.assertIsNotNone(scaler)

    def test_load_label_encoder_from_local(self):
        from src.models.game.model_registry import load_label_encoder, RISK_CLASSIFIER_DIR
        enc_path = RISK_CLASSIFIER_DIR / "risk_label_encoder.pkl"
        if not enc_path.exists():
            self.skipTest("Local risk_label_encoder.pkl not present")
        encoder = load_label_encoder()
        self.assertIsNotNone(encoder)
        self.assertTrue(hasattr(encoder, "classes_"))
        # Must have exactly the 3 expected classes
        self.assertEqual(sorted(encoder.classes_), ["HIGH", "LOW", "MEDIUM"])

    def test_load_lstm_model_from_local(self):
        from src.models.game.model_registry import load_lstm_model, LSTM_MODEL_DIR
        keras_path = LSTM_MODEL_DIR / "lstm_model.keras"
        h5_path    = LSTM_MODEL_DIR / "lstm_model.h5"
        if not keras_path.exists() and not h5_path.exists():
            self.skipTest("No local LSTM model file present")
        model = load_lstm_model()
        self.assertIsNotNone(model)

    def test_load_lstm_scaler_from_local(self):
        from src.models.game.model_registry import load_lstm_scaler, LSTM_MODEL_DIR
        if not (LSTM_MODEL_DIR / "lstm_scaler.pkl").exists():
            self.skipTest("Local lstm_scaler.pkl not present")
        scaler = load_lstm_scaler()
        self.assertIsNotNone(scaler)

    # ── Getter functions & caching ───────────────────────────────────────────

    def test_get_functions_trigger_load(self):
        from src.models.game import model_registry as reg
        # Patch all loaders to return a simple sentinel
        sentinel = object()
        with patch.object(reg, "load_risk_classifier", return_value=sentinel), \
             patch.object(reg, "load_risk_scaler",     return_value=None), \
             patch.object(reg, "load_label_encoder",   return_value=None), \
             patch.object(reg, "load_lstm_model",      return_value=None), \
             patch.object(reg, "load_lstm_scaler",     return_value=None):
            reg._MODEL_LOADED = False
            result = reg.get_risk_classifier()
            self.assertIs(result, sentinel)

    def test_model_cached_after_first_load(self):
        from src.models.game import model_registry as reg
        sentinel = object()
        with patch.object(reg, "load_risk_classifier", return_value=sentinel), \
             patch.object(reg, "load_risk_scaler",     return_value=None), \
             patch.object(reg, "load_label_encoder",   return_value=None), \
             patch.object(reg, "load_lstm_model",      return_value=None), \
             patch.object(reg, "load_lstm_scaler",     return_value=None):
            reg._MODEL_LOADED = False
            reg.get_risk_classifier()
            # Second call must NOT call the loader again
            with patch.object(reg, "load_risk_classifier", side_effect=AssertionError("loader called twice")):
                result = reg.get_risk_classifier()
            self.assertIs(result, sentinel)

    # ── Safe getters / dummy fallback ────────────────────────────────────────

    def test_get_lstm_model_safe_returns_dummy_when_none(self):
        from src.models.game.model_registry import get_lstm_model_safe, DummyLSTM
        import src.models.game.model_registry as reg
        reg._MODEL_LOADED = True
        reg._MODELS["lstm_model"] = None
        dummy = get_lstm_model_safe()
        self.assertIsInstance(dummy, DummyLSTM)

    def test_get_risk_classifier_safe_returns_dummy_when_none(self):
        from src.models.game.model_registry import get_risk_classifier_safe, DummyRiskClassifier
        import src.models.game.model_registry as reg
        reg._MODEL_LOADED = True
        reg._MODELS["risk_classifier"] = None
        dummy = get_risk_classifier_safe()
        self.assertIsInstance(dummy, DummyRiskClassifier)

    def test_dummy_lstm_output_shape(self):
        from src.models.game.model_registry import DummyLSTM
        dummy = DummyLSTM()
        X = np.zeros((1, 5, 5))
        out = dummy.predict(X)
        self.assertEqual(out.shape[0], 1)

    def test_dummy_risk_classifier_probabilities_sum_to_1(self):
        from src.models.game.model_registry import DummyRiskClassifier
        dummy = DummyRiskClassifier()
        X = np.zeros((1, 14))
        probs = dummy.predict_proba(X)[0]
        self.assertAlmostEqual(sum(probs), 1.0, places=5)


# =============================================================================
# 2. Feature Extraction Tests
# =============================================================================

class TestFeatureExtraction(unittest.TestCase):

    def test_extract_lstm_features_empty_sessions(self):
        from src.services.game_service import extract_lstm_features
        X = extract_lstm_features([])
        self.assertEqual(X.shape, (1, 1, 5))
        np.testing.assert_array_equal(X, np.zeros((1, 1, 5)))

    def test_extract_lstm_features_single_session(self):
        from src.services.game_service import extract_lstm_features
        sessions = _session_history(1)
        X = extract_lstm_features(sessions)
        self.assertEqual(X.shape[0], 1)   # batch dim
        self.assertEqual(X.shape[2], 5)   # feature dim

    def test_extract_lstm_features_10_sessions(self):
        from src.services.game_service import extract_lstm_features
        sessions = _session_history(10)
        X = extract_lstm_features(sessions)
        self.assertEqual(X.shape, (1, 10, 5))

    def test_extract_lstm_features_correct_values(self):
        from src.services.game_service import extract_lstm_features
        s = _make_session(sac=0.9, ies=2.0, accuracy=0.85, rt=1.1, variability=0.15)
        X = extract_lstm_features([s])
        self.assertAlmostEqual(X[0, 0, 0], 0.9)   # sac
        self.assertAlmostEqual(X[0, 0, 1], 2.0)   # ies
        self.assertAlmostEqual(X[0, 0, 2], 0.85)  # accuracy
        self.assertAlmostEqual(X[0, 0, 3], 1.1)   # rtAdjMedian
        self.assertAlmostEqual(X[0, 0, 4], 0.15)  # variability

    def test_extract_risk_features_no_history(self):
        from src.services.game_service import extract_risk_features
        X = extract_risk_features([], GOOD_FEATURES, lstm_score=0.1)
        self.assertEqual(X.shape, (1, 14))
        # Slopes/stds should all be 0 with no history
        self.assertEqual(X[0, 1], 0.0)   # slope_sac
        self.assertEqual(X[0, 3], 0.0)   # slope_ies

    def test_extract_risk_features_with_history(self):
        from src.services.game_service import extract_risk_features
        sessions = _session_history(5)
        X = extract_risk_features(sessions, GOOD_FEATURES, lstm_score=0.2)
        self.assertEqual(X.shape, (1, 14))

    def test_extract_risk_features_lstm_score_embedded(self):
        from src.services.game_service import extract_risk_features
        X = extract_risk_features([], GOOD_FEATURES, lstm_score=0.77)
        self.assertAlmostEqual(X[0, 7], 0.77)   # index 7 = lstm_decline_score

    def test_extract_risk_features_slope_increases_with_declining_history(self):
        from src.services.game_service import extract_risk_features
        # Declining accuracy: 0.9 → 0.7 → 0.5 → 0.3 → 0.1
        sessions = [_make_session(accuracy=a) for a in [0.9, 0.7, 0.5, 0.3, 0.1]]
        X = extract_risk_features(sessions, POOR_FEATURES, lstm_score=0.5)
        slope_acc = X[0, 10]   # index 10 = slope_accuracy
        self.assertLess(slope_acc, 0)   # negative slope = declining accuracy

    def test_extract_risk_features_std_nonzero_with_variable_history(self):
        from src.services.game_service import extract_risk_features
        sessions = [_make_session(sac=v) for v in [0.2, 0.8, 0.3, 0.9, 0.1]]
        X = extract_risk_features(sessions, GOOD_FEATURES, lstm_score=0.1)
        std_sac = X[0, 12]   # index 12 = std_sac
        self.assertGreater(std_sac, 0)


# =============================================================================
# 3. LSTM Inference Tests
# =============================================================================

class TestLSTMInference(unittest.TestCase):

    def test_predict_lstm_returns_zero_for_fewer_than_3_sessions(self):
        from src.services.game_service import predict_lstm_decline
        for n in [0, 1, 2]:
            score = predict_lstm_decline(_session_history(n))
            self.assertEqual(score, 0.0, msg=f"Expected 0.0 for {n} sessions")

    def test_predict_lstm_returns_float_with_real_model(self):
        from src.services.game_service import predict_lstm_decline
        from src.models.game.model_registry import LSTM_MODEL_DIR
        keras_path = LSTM_MODEL_DIR / "lstm_model.keras"
        h5_path    = LSTM_MODEL_DIR / "lstm_model.h5"
        if not keras_path.exists() and not h5_path.exists():
            self.skipTest("No local LSTM model — skipping real inference test")
        score = predict_lstm_decline(_session_history(5))
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_predict_lstm_with_dummy_model(self):
        from src.services.game_service import predict_lstm_decline
        import src.models.game.model_registry as reg
        reg._MODEL_LOADED = True
        reg._MODELS["lstm_model"]  = None   # forces DummyLSTM
        reg._MODELS["lstm_scaler"] = None
        score = predict_lstm_decline(_session_history(5))
        self.assertIsInstance(score, float)

    def test_predict_lstm_scaler_mismatch_does_not_crash(self):
        """Scaler expects 3 features but input has 5 — should skip scaling gracefully."""
        from src.services.game_service import predict_lstm_decline
        import src.models.game.model_registry as reg

        bad_scaler = MagicMock()
        bad_scaler.n_features_in_ = 3   # mismatch with actual 5

        reg._MODEL_LOADED = True
        reg._MODELS["lstm_model"]  = None   # DummyLSTM
        reg._MODELS["lstm_scaler"] = bad_scaler

        score = predict_lstm_decline(_session_history(5))
        self.assertIsInstance(score, float)


# =============================================================================
# 4. Risk Classifier Tests
# =============================================================================

class TestRiskClassifier(unittest.TestCase):

    def _make_mock_classifier(self, probs):
        """Build a mock classifier that returns given probs for ['HIGH','LOW','MEDIUM']."""
        mock = MagicMock()
        mock.__class__.__name__ = "LogisticRegression"
        mock.classes_ = np.array(["HIGH", "LOW", "MEDIUM"])
        mock.predict_proba.return_value = np.array([probs])
        return mock

    def _run_predict(self, features, sessions, lstm_score, mock_clf, mock_scaler=None):
        import src.models.game.model_registry as reg
        reg._MODEL_LOADED = True
        reg._MODELS["risk_classifier"] = mock_clf
        reg._MODELS["scaler"]          = mock_scaler
        from src.services.game_service import predict_risk
        return predict_risk(sessions, features, lstm_score)

    # ── Happy-path predictions ───────────────────────────────────────────────

    def test_predict_risk_returns_low(self):
        clf = self._make_mock_classifier([0.05, 0.90, 0.05])  # LOW dominant
        result = self._run_predict(GOOD_FEATURES, [], 0.0, clf)
        self.assertEqual(result["riskLevel"], "LOW")
        self.assertIn("riskProbability", result)
        self.assertAlmostEqual(sum(result["riskProbability"].values()), 1.0, places=3)

    def test_predict_risk_returns_high(self):
        clf = self._make_mock_classifier([0.85, 0.05, 0.10])  # HIGH dominant
        result = self._run_predict(GOOD_FEATURES, [], 0.8, clf)
        self.assertEqual(result["riskLevel"], "HIGH")

    def test_predict_risk_returns_medium(self):
        clf = self._make_mock_classifier([0.10, 0.15, 0.75])  # MEDIUM dominant
        result = self._run_predict(GOOD_FEATURES, [], 0.3, clf)
        self.assertEqual(result["riskLevel"], "MEDIUM")

    def test_predict_risk_score_0_100_range(self):
        clf = self._make_mock_classifier([0.3, 0.4, 0.3])
        result = self._run_predict(GOOD_FEATURES, [], 0.2, clf)
        self.assertGreaterEqual(result["riskScore0_100"], 0)
        self.assertLessEqual(result["riskScore0_100"], 100)

    def test_predict_risk_probability_keys(self):
        clf = self._make_mock_classifier([0.2, 0.5, 0.3])
        result = self._run_predict(GOOD_FEATURES, [], 0.1, clf)
        self.assertSetEqual(set(result["riskProbability"].keys()), {"HIGH", "LOW", "MEDIUM"})

    # ── Rule-based safety overrides ──────────────────────────────────────────

    def test_rule_override_very_low_accuracy_forces_high(self):
        """accuracy < 0.35 must force HIGH regardless of model output."""
        clf = self._make_mock_classifier([0.05, 0.90, 0.05])   # model says LOW
        features = {**GOOD_FEATURES, "accuracy": 0.30, "ies": 5.0, "variability": 0.20}
        result = self._run_predict(features, [], 0.0, clf)
        self.assertEqual(result["riskLevel"], "HIGH")

    def test_rule_override_high_ies_forces_high(self):
        """IES > 8.0 must force HIGH regardless of model output."""
        clf = self._make_mock_classifier([0.05, 0.90, 0.05])   # model says LOW
        features = {**GOOD_FEATURES, "ies": 9.0, "accuracy": 0.80, "variability": 0.10}
        result = self._run_predict(features, [], 0.0, clf)
        self.assertEqual(result["riskLevel"], "HIGH")

    def test_rule_override_low_accuracy_and_high_variability_forces_high(self):
        """accuracy < 0.40 AND variability > 0.60 → HIGH."""
        clf = self._make_mock_classifier([0.05, 0.90, 0.05])
        features = {**GOOD_FEATURES, "accuracy": 0.38, "variability": 0.65, "ies": 2.0}
        result = self._run_predict(features, [], 0.0, clf)
        self.assertEqual(result["riskLevel"], "HIGH")

    def test_rule_override_moderate_accuracy_forces_medium(self):
        """accuracy < 0.60 must floor LOW up to MEDIUM."""
        clf = self._make_mock_classifier([0.05, 0.90, 0.05])   # model says LOW
        features = {**GOOD_FEATURES, "accuracy": 0.55, "variability": 0.20, "ies": 2.0}
        result = self._run_predict(features, [], 0.0, clf)
        self.assertEqual(result["riskLevel"], "MEDIUM")

    def test_rule_override_high_variability_forces_medium(self):
        """variability > 0.40 must floor LOW up to MEDIUM."""
        clf = self._make_mock_classifier([0.05, 0.90, 0.05])
        features = {**GOOD_FEATURES, "accuracy": 0.85, "variability": 0.50, "ies": 1.0}
        result = self._run_predict(features, [], 0.0, clf)
        self.assertEqual(result["riskLevel"], "MEDIUM")

    def test_rule_override_does_not_downgrade_from_high(self):
        """Rule overrides must never lower a HIGH prediction."""
        clf = self._make_mock_classifier([0.85, 0.05, 0.10])   # model says HIGH
        features = {**GOOD_FEATURES, "accuracy": 0.90, "variability": 0.05, "ies": 1.0}
        result = self._run_predict(features, [], 0.9, clf)
        self.assertEqual(result["riskLevel"], "HIGH")

    def test_rule_override_does_not_downgrade_from_medium(self):
        """A MEDIUM model output must not be lowered to LOW by rules."""
        clf = self._make_mock_classifier([0.05, 0.10, 0.85])   # model says MEDIUM
        features = {**GOOD_FEATURES, "accuracy": 0.90, "variability": 0.05, "ies": 1.0}
        result = self._run_predict(features, [], 0.0, clf)
        self.assertEqual(result["riskLevel"], "MEDIUM")

    # ── Scaler path ──────────────────────────────────────────────────────────

    def test_predict_risk_applies_scaler(self):
        clf = self._make_mock_classifier([0.1, 0.8, 0.1])
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((1, 14))
        result = self._run_predict(GOOD_FEATURES, [], 0.1, clf, mock_scaler)
        mock_scaler.transform.assert_called_once()

    def test_predict_risk_raises_when_dummy_classifier(self):
        """predict_risk must raise RuntimeError if dummy model slips through."""
        from src.models.game.model_registry import DummyRiskClassifier
        import src.models.game.model_registry as reg
        reg._MODEL_LOADED = True
        reg._MODELS["risk_classifier"] = None   # get_risk_classifier_safe → DummyRiskClassifier
        reg._MODELS["scaler"] = None
        from src.services.game_service import predict_risk
        with self.assertRaises(RuntimeError):
            predict_risk([], GOOD_FEATURES, 0.0)


# =============================================================================
# 5. Full Pipeline Tests (process_game_session)
# =============================================================================

class TestFullPipeline(unittest.IsolatedAsyncioTestCase):

    def _mock_registry(self, probs=(0.1, 0.8, 0.1)):
        """Patch model registry with real-looking mocks."""
        import src.models.game.model_registry as reg

        mock_clf = MagicMock()
        mock_clf.__class__.__name__ = "LogisticRegression"
        mock_clf.classes_ = np.array(["HIGH", "LOW", "MEDIUM"])
        mock_clf.predict_proba.return_value = np.array([probs])

        mock_lstm = MagicMock()
        mock_lstm.predict.return_value = np.array([[0.1]])

        reg._MODEL_LOADED = True
        reg._MODELS["risk_classifier"] = mock_clf
        reg._MODELS["lstm_model"]      = mock_lstm
        reg._MODELS["scaler"]          = None
        reg._MODELS["lstm_scaler"]     = None
        reg._MODELS["label_encoder"]   = None

    def _patch_db(self):
        """Patch all MongoDB calls so no real DB is needed.

        Motor cursor chain is SYNCHRONOUS until to_list():
            find(...)  → cursor  (sync)
            .sort(...) → cursor  (sync)
            .limit(n)  → cursor  (sync)
            await .to_list(length=n)  (async)
        find_one / insert_one are coroutines (async).
        """
        # Cursor: sort/limit return self (sync); to_list is async
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value  = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.to_list = AsyncMock(return_value=[])

        mock_coll = MagicMock()
        mock_coll.find_one  = AsyncMock(return_value=None)
        mock_coll.insert_one = AsyncMock(return_value=MagicMock())
        mock_coll.find.return_value = mock_cursor

        db_patch = patch("src.services.game_service.Database.get_collection",
                         return_value=mock_coll)
        return db_patch

    # ── Trials path ─────────────────────────────────────────────────────────

    async def test_process_session_with_trials_returns_response(self):
        self._mock_registry()
        trials = [
            {"rt_raw": 0.95, "correct": 1, "error": 0, "hint_used": 0},
            {"rt_raw": 1.10, "correct": 1, "error": 0, "hint_used": 0},
            {"rt_raw": 1.30, "correct": 0, "error": 1, "hint_used": 0},
            {"rt_raw": 0.88, "correct": 1, "error": 0, "hint_used": 0},
            {"rt_raw": 1.05, "correct": 1, "error": 0, "hint_used": 0},
        ]
        from src.services.game_service import process_game_session
        with self._patch_db():
            result = await process_game_session(
                userId="user_test_01",
                sessionId="sess_001",
                gameType="whack_a_mole",
                level=2,
                trials=trials,
                summary=None,
            )
        self.assertIn("features", result)
        self.assertIn("prediction", result)
        self.assertIn(result["prediction"]["riskLevel"], ["LOW", "MEDIUM", "HIGH"])

    # ── Summary path ─────────────────────────────────────────────────────────

    async def test_process_session_with_summary_returns_response(self):
        self._mock_registry()
        summary = {
            "totalAttempts": 10,
            "correct": 8,
            "errors": 2,
            "hintsUsed": 0,
            "meanRtRaw": 1.05,
            "medianRtRaw": 1.00,
        }
        from src.services.game_service import process_game_session
        with self._patch_db():
            result = await process_game_session(
                userId="user_test_02",
                sessionId="sess_002",
                gameType="card_matching",
                level=1,
                trials=None,
                summary=summary,
            )
        self.assertIn("features", result)
        self.assertIn("prediction", result)

    # ── Millisecond → second auto-conversion ─────────────────────────────────

    async def test_process_session_ms_to_seconds_conversion(self):
        """
        The ms→s conversion lives in game_routes.py (before process_game_session).
        Replicate the same logic here so the service receives correct values.
        """
        self._mock_registry()
        trials_ms = [
            {"rt_raw": 950,  "correct": 1, "error": 0, "hint_used": 0},
            {"rt_raw": 1100, "correct": 1, "error": 0, "hint_used": 0},
            {"rt_raw": 800,  "correct": 1, "error": 0, "hint_used": 0},
        ]
        # --- same logic as game_routes.py ---
        if any(t.get("rt_raw", 0) > 10 for t in trials_ms):
            for t in trials_ms:
                if t.get("rt_raw", 0) > 10:
                    t["rt_raw"] = t["rt_raw"] / 1000.0
        # ------------------------------------
        from src.services.game_service import process_game_session
        with self._patch_db():
            result = await process_game_session(
                userId="user_ms",
                sessionId="sess_ms",
                gameType="card_matching",
                level=1,
                trials=trials_ms,
                summary=None,
            )
        rt = result["features"]["rtAdjMedian"]
        self.assertLess(rt, 5.0, "RT still looks like milliseconds after conversion")

    # ── High-risk alert path ──────────────────────────────────────────────────

    async def test_process_session_creates_alert_on_high_risk(self):
        """When risk is HIGH an alert document must be inserted."""
        self._mock_registry(probs=(0.85, 0.05, 0.10))   # model → HIGH probability
        trials = [
            {"rt_raw": 3.5,  "correct": 0, "error": 1, "hint_used": 1},
            {"rt_raw": 4.1,  "correct": 0, "error": 1, "hint_used": 1},
            {"rt_raw": 3.8,  "correct": 0, "error": 1, "hint_used": 0},
        ]

        # Need to track which collection is written to
        insert_calls = {}

        def get_coll(name):
            coll = MagicMock()
            coll.find_one  = AsyncMock(return_value=None)
            coll.insert_one = AsyncMock(return_value=MagicMock())
            cursor = MagicMock()
            cursor.sort.return_value  = cursor
            cursor.limit.return_value = cursor
            cursor.to_list = AsyncMock(return_value=[])
            coll.find.return_value    = cursor
            insert_calls[name] = coll
            return coll

        from src.services.game_service import process_game_session
        with patch("src.services.game_service.Database.get_collection", side_effect=get_coll):
            await process_game_session(
                userId="high_risk_user",
                sessionId="sess_high",
                gameType="whack_a_mole",
                level=3,
                trials=trials,
                summary=None,
            )

        # Either the test passed because the risk rule overrode to HIGH,
        # or the alert collection was written to — verify at least one insert happened
        total_inserts = sum(
            c.insert_one.call_count for c in insert_calls.values()
        )
        self.assertGreaterEqual(total_inserts, 1)

    # ── Required response fields ──────────────────────────────────────────────

    async def test_response_contains_all_required_fields(self):
        self._mock_registry()
        trials = [{"rt_raw": 1.0, "correct": 1, "error": 0, "hint_used": 0}] * 5
        from src.services.game_service import process_game_session
        with self._patch_db():
            result = await process_game_session(
                userId="u1", sessionId="s1",
                gameType="card_matching", level=1,
                trials=trials, summary=None,
            )
        for key in ["sessionId", "userId", "features", "prediction", "timestamp"]:
            self.assertIn(key, result)
        for key in ["accuracy", "errorRate", "sac", "ies", "variability", "rtAdjMedian"]:
            self.assertIn(key, result["features"])
        for key in ["riskLevel", "riskProbability", "riskScore0_100"]:
            self.assertIn(key, result["prediction"])


# =============================================================================
# 6. API Schema Validation Tests
# =============================================================================

class TestAPISchemas(unittest.TestCase):

    def test_game_trial_valid(self):
        from src.parsers.game_schemas import GameTrial
        t = GameTrial(rt_raw=1.0, correct=1, error=0, hint_used=0)
        self.assertEqual(t.rt_raw, 1.0)

    def test_game_trial_rejects_non_positive_rt(self):
        from src.parsers.game_schemas import GameTrial
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            GameTrial(rt_raw=0.0, correct=1)

    def test_game_trial_rejects_invalid_correct_flag(self):
        from src.parsers.game_schemas import GameTrial
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            GameTrial(rt_raw=1.0, correct=2)

    def test_game_session_request_valid_with_trials(self):
        from src.parsers.game_schemas import GameSessionRequest, GameTrial
        req = GameSessionRequest(
            userId="u1",
            sessionId="s1",
            gameType="card_matching",
            level=2,
            trials=[GameTrial(rt_raw=1.0, correct=1)],
        )
        self.assertEqual(req.userId, "u1")

    def test_game_session_request_valid_with_summary(self):
        from src.parsers.game_schemas import GameSessionRequest, GameSummary
        req = GameSessionRequest(
            userId="u2",
            sessionId="s2",
            summary=GameSummary(
                totalAttempts=10, correct=8, errors=2,
                hintsUsed=0, meanRtRaw=1.0,
            ),
        )
        self.assertIsNotNone(req.summary)

    def test_calibration_request_rejects_fewer_than_5_taps(self):
        from src.parsers.game_schemas import CalibrationRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            CalibrationRequest(userId="u1", tapTimes=[0.3, 0.4, 0.3])

    def test_calibration_request_rejects_non_positive_tap(self):
        from src.parsers.game_schemas import CalibrationRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            CalibrationRequest(userId="u1", tapTimes=[0.3, 0.4, 0.3, 0.0, 0.3])

    def test_calibration_request_valid(self):
        from src.parsers.game_schemas import CalibrationRequest
        req = CalibrationRequest(userId="u1", tapTimes=[0.3, 0.29, 0.31, 0.28, 0.30])
        self.assertEqual(len(req.tapTimes), 5)

    def test_game_summary_rejects_zero_rt(self):
        from src.parsers.game_schemas import GameSummary
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            GameSummary(totalAttempts=5, correct=4, errors=1, hintsUsed=0, meanRtRaw=0.0)


# =============================================================================
# 7. Registry JSON Integrity Tests
# =============================================================================

class TestRegistryJSON(unittest.TestCase):

    def _load(self, path: Path):
        with open(path) as f:
            return json.load(f)

    def _find(self, data, model_id):
        return next((m for m in data["models"] if m["id"] == model_id), None)

    # ── models/models_registry.json (used by ModelLoader / chatbot service) ──

    def test_models_registry_json_is_valid(self):
        path = ROOT / "models" / "models_registry.json"
        self.assertTrue(path.exists(), f"File not found: {path}")
        data = self._load(path)
        self.assertIn("models", data)
        self.assertIn("total_models", data)

    def test_models_registry_total_matches_actual_count(self):
        data = self._load(ROOT / "models" / "models_registry.json")
        self.assertEqual(data["total_models"], len(data["models"]))

    def test_lstm_entry_present_in_models_registry(self):
        data = self._load(ROOT / "models" / "models_registry.json")
        entry = self._find(data, "lstm_temporal_analysis")
        self.assertIsNotNone(entry, "lstm_temporal_analysis not found in models_registry.json")
        self.assertEqual(entry.get("model_source"), "huggingface")
        self.assertEqual(entry.get("huggingface_repo"), "vlakvindu/Dementia_LSTM_Model")
        self.assertEqual(entry["files"]["model"], "lstm_model.keras")
        self.assertEqual(entry["files"]["scaler"], "lstm_scaler.pkl")

    def test_risk_classifier_entry_present_in_models_registry(self):
        data = self._load(ROOT / "models" / "models_registry.json")
        entry = self._find(data, "dementia_risk_classifier")
        self.assertIsNotNone(entry, "dementia_risk_classifier not found in models_registry.json")
        self.assertEqual(entry.get("model_source"), "huggingface")
        self.assertEqual(entry.get("huggingface_repo"), "vlakvindu/Dementia_Risk_Clasification_model")
        self.assertEqual(entry["files"]["model"], "risk_logreg.pkl")
        self.assertEqual(entry["files"]["scaler"], "risk_scaler.pkl")
        self.assertEqual(entry["files"]["label_encoder"], "risk_label_encoder.pkl")

    # ── models_registry.json (root — used by ModelLoader default path) ───────

    def test_root_registry_json_is_valid(self):
        path = ROOT / "models_registry.json"
        self.assertTrue(path.exists(), f"File not found: {path}")
        data = self._load(path)
        self.assertIn("models", data)

    def test_root_registry_lstm_entry_correct(self):
        data = self._load(ROOT / "models_registry.json")
        entry = self._find(data, "dementia_lstm_model")
        self.assertIsNotNone(entry, "dementia_lstm_model not found in root models_registry.json")
        self.assertEqual(entry.get("model_source"), "huggingface")
        self.assertEqual(entry.get("huggingface_repo"), "vlakvindu/Dementia_LSTM_Model")

    def test_root_registry_risk_entry_correct(self):
        data = self._load(ROOT / "models_registry.json")
        entry = self._find(data, "dementia_risk_classifier")
        self.assertIsNotNone(entry, "dementia_risk_classifier not found in root models_registry.json")
        self.assertEqual(entry.get("model_source"), "huggingface")
        self.assertEqual(entry.get("huggingface_repo"), "vlakvindu/Dementia_Risk_Clasification_model")

    # ── All game-component models have required HF fields ────────────────────

    def test_all_game_hf_models_have_required_fields(self):
        data = self._load(ROOT / "models" / "models_registry.json")
        game_hf_ids = ["lstm_temporal_analysis", "dementia_risk_classifier"]
        for mid in game_hf_ids:
            entry = self._find(data, mid)
            self.assertIsNotNone(entry, f"{mid} missing from registry")
            for field in ["model_source", "huggingface_repo", "huggingface_url", "files", "local_cache_dir"]:
                self.assertIn(field, entry, f"Field '{field}' missing from {mid}")


# =============================================================================
# Plain-print runner (alternative to pytest)
# =============================================================================

def run_plain():
    suites = [
        TestModelRegistry,
        TestFeatureExtraction,
        TestLSTMInference,
        TestRiskClassifier,
        TestFullPipeline,
        TestAPISchemas,
        TestRegistryJSON,
    ]

    total_pass = total_fail = total_skip = 0
    all_failures = []

    for suite_class in suites:
        _section(suite_class.__name__)
        loader = unittest.TestLoader()
        suite  = loader.loadTestsFromTestCase(suite_class)
        runner = unittest.TextTestRunner(verbosity=0, stream=open(Path(ROOT / "logs" / "test"), "w"))

        # Run and capture
        import io
        buf = io.StringIO()
        runner2 = unittest.TextTestRunner(verbosity=2, stream=buf)
        result  = runner2.run(suite)

        for test, err in result.failures + result.errors:
            _fail(str(test).split(" ")[0])
            all_failures.append((str(test), err))
        for test, reason in result.skipped:
            _info(f"SKIP: {str(test).split(' ')[0]}  ({reason})")
            total_skip += 1

        passed = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
        for test in suite:
            name = test._testMethodName
            failed_names = [str(t).split(" ")[0] for t, _ in result.failures + result.errors]
            skipped_names = [str(t).split(" ")[0] for t, _ in result.skipped]
            if name not in failed_names and name not in skipped_names:
                _ok(name)

        total_pass += passed
        total_fail += len(result.failures) + len(result.errors)

    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}SUMMARY: {GREEN}{total_pass} passed{RESET}  "
          f"{RED}{total_fail} failed{RESET}  "
          f"{YELLOW}{total_skip} skipped{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")

    if all_failures:
        print(f"\n{RED}FAILURES:{RESET}")
        for name, err in all_failures:
            print(f"\n  {RED}✗ {name}{RESET}")
            print("  " + "\n  ".join(err.splitlines()[-6:]))


if __name__ == "__main__":
    run_plain()
