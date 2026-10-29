"""
SENTINEL-AI — Environment Verification Script
===============================================

Verifies that all Python dependencies are correctly installed and importable.
Run this after setting up the virtual environment to confirm everything works.

Run:
    python infra/scripts/verify_environment.py

Expected output: All checks PASSED
"""

import sys
import os

# Fix OpenBLAS memory issue on low-RAM machines
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def check_import(module_name: str, display_name: str = None, version_attr: str = "__version__"):
    """Try to import a module and report success/failure."""
    display_name = display_name or module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, version_attr, "OK")
        if callable(version):
            version = version()
        print(f"  [OK]   {display_name:25s} {version}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {display_name:25s} FAILED -- {e}")
        return False
    except Exception as e:
        print(f"  [WARN] {display_name:25s} IMPORTED but error -- {e}")
        return True  # Import worked, just version check failed


def main():
    print("\n" + "=" * 60)
    print("  SENTINEL-AI — Environment Verification")
    print("=" * 60)
    print(f"\n  Python: {sys.version}")
    print(f"  Executable: {sys.executable}\n")

    results = []

    # ---- Core ML ----
    print("Core ML Libraries:")
    results.append(check_import("numpy", "NumPy"))
    results.append(check_import("pandas", "Pandas"))
    results.append(check_import("sklearn", "Scikit-learn", version_attr="__version__"))
    results.append(check_import("tensorflow", "TensorFlow"))
    results.append(check_import("keras", "Keras"))

    # ---- Data Processing ----
    print("\nData Processing:")
    results.append(check_import("pyarrow", "PyArrow"))
    results.append(check_import("imblearn", "imbalanced-learn"))

    # ---- Explainability ----
    print("\nExplainability (XAI):")
    results.append(check_import("shap", "SHAP"))

    # ---- Network ----
    print("\nNetwork / Capture:")
    results.append(check_import("scapy", "Scapy", version_attr="VERSION"))

    # ---- Visualization ----
    print("\nVisualization:")
    results.append(check_import("matplotlib", "Matplotlib"))
    results.append(check_import("seaborn", "Seaborn"))

    # ---- Communication ----
    print("\nCommunication Bridge:")
    results.append(check_import("socketio", "python-socketio"))
    results.append(check_import("flask", "Flask"))

    # ---- Reporting ----
    print("\nReporting:")
    results.append(check_import("reportlab", "ReportLab"))

    # ---- Database ----
    print("\nDatabase:")
    results.append(check_import("influxdb_client", "influxdb-client"))

    # ---- Utilities ----
    print("\nUtilities:")
    results.append(check_import("dotenv", "python-dotenv"))
    results.append(check_import("tqdm", "tqdm"))
    results.append(check_import("joblib", "Joblib"))
    results.append(check_import("yaml", "PyYAML"))

    # ---- Testing ----
    print("\nTesting:")
    results.append(check_import("pytest", "pytest"))

    # ---- Project Modules ----
    print("\nProject Modules (SENTINEL-AI):")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    results.append(check_import("ml.data.data_loader", "ml.data.data_loader", version_attr=None))
    results.append(check_import("ml.data.preprocessing", "ml.data.preprocessing", version_attr=None))
    results.append(check_import("ml.models.cnn_lstm", "ml.models.cnn_lstm", version_attr=None))
    results.append(check_import("ml.models.trainer", "ml.models.trainer", version_attr=None))
    results.append(check_import("ml.explainability.shap_explainer", "ml.explainability.shap_explainer", version_attr=None))
    results.append(check_import("ml.ingestion.packet_sniffer", "ml.ingestion.packet_sniffer", version_attr=None))
    results.append(check_import("ml.ingestion.flow_aggregator", "ml.ingestion.flow_aggregator", version_attr=None))
    results.append(check_import("ml.inference.pipeline", "ml.inference.pipeline", version_attr=None))

    # ---- Summary ----
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 60}")
    if passed == total:
        print(f"  [OK] ALL CHECKS PASSED ({passed}/{total})")
    else:
        print(f"  [!!] {passed}/{total} PASSED, {total - passed} FAILED")
        print(f"  Run: pip install -r ml/requirements.txt")
    print(f"{'=' * 60}\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
