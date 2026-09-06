"""Unit tests verifying UI Benchmarking Dashboard and Client Integration (Level 5.5 Phase 4)."""

from pathlib import Path
import unittest

from fastapi.testclient import TestClient
from src.api.app import create_app
from src.api.dependencies import create_default_services


class TestBenchmarkingUi(unittest.TestCase):
    """Test suite verifying UI structure, view mounting, API client, and scientific boundaries."""

    def setUp(self) -> None:
        self.services = create_default_services()
        self.app = create_app(services=self.services)
        self.client = TestClient(self.app)
        self.ui_dir = Path(__file__).resolve().parent.parent.parent.parent / "src" / "ui"

    # 1. HTML Container and Navigation Button
    def test_index_html_contains_benchmarking_tab_and_viewport(self) -> None:
        index_path = self.ui_dir / "index.html"
        self.assertTrue(index_path.is_file(), "index.html must exist")
        html_content = index_path.read_text(encoding="utf-8")

        self.assertIn('data-view="benchmarking"', html_content)
        self.assertIn('id="view-benchmarking"', html_content)
        self.assertIn('Benchmarking</span>', html_content)

    # 2. View JS Component Architecture
    def test_benchmarking_view_js_structure_and_phase3_semantics(self) -> None:
        view_path = self.ui_dir / "js" / "views" / "benchmarking.js"
        self.assertTrue(view_path.is_file(), "benchmarking.js view must exist")
        js_content = view_path.read_text(encoding="utf-8")

        # Class declaration and mounting
        self.assertIn("class BenchmarkingView", js_content)
        self.assertIn("renderResult", js_content)
        self.assertIn("handleRunBenchmark", js_content)
        self.assertIn("handleResetBenchmarks", js_content)

        # Preserves Phase 3 event matching semantics
        self.assertIn("DETECTED_MATCH", js_content)
        self.assertIn("DETECTED_EARLY", js_content)
        self.assertIn("MISSED_EVENT", js_content)

        # Handles NOT_EVALUABLE status
        self.assertIn("NOT_EVALUABLE", js_content)

        # Zero browser-side math calculations
        self.assertNotIn("Math.sqrt", js_content)
        self.assertNotIn("Math.pow", js_content)

        # Reset does not call twin reset
        self.assertIn("api.resetBenchmark", js_content)
        self.assertNotIn("api.resetTwin", js_content)

    # 3. API Client Integration
    def test_api_client_defines_benchmarking_methods(self) -> None:
        api_path = self.ui_dir / "js" / "api.js"
        self.assertTrue(api_path.is_file(), "api.js must exist")
        api_content = api_path.read_text(encoding="utf-8")

        self.assertIn("getBenchmarkDatasets()", api_content)
        self.assertIn("getBenchmarkDataset(", api_content)
        self.assertIn("runBenchmark(", api_content)
        self.assertIn("getLatestBenchmark(", api_content)
        self.assertIn("getBenchmarkHistory(", api_content)
        self.assertIn("getBenchmarkResult(", api_content)
        self.assertIn("resetBenchmark(", api_content)

    # 4. App Bootstrapper Mounts BenchmarkingView
    def test_app_js_imports_and_mounts_benchmarking_view(self) -> None:
        app_path = self.ui_dir / "js" / "app.js"
        self.assertTrue(app_path.is_file(), "app.js must exist")
        app_content = app_path.read_text(encoding="utf-8")

        self.assertIn("import { BenchmarkingView }", app_content)
        self.assertIn("benchmarking: document.getElementById('view-benchmarking')", app_content)
        self.assertIn("new BenchmarkingView(", app_content)

    # 5. Static Route Serving
    def test_static_assets_serve_benchmarking_view(self) -> None:
        res_view = self.client.get("/ui/js/views/benchmarking.js")
        self.assertEqual(res_view.status_code, 200)
        self.assertIn("BenchmarkingView", res_view.text)


if __name__ == "__main__":
    unittest.main()
