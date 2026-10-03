"""
SENTINEL-AI — Smoke Tests

Basic import and instantiation tests to verify the ML pipeline scaffolding.
Run: pytest ml/tests/ -v
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestImports:
    """Verify all ML pipeline modules can be imported."""

    def test_import_data_loader(self):
        from ml.data.data_loader import DataLoader
        assert DataLoader is not None

    def test_import_preprocessor(self):
        from ml.data.preprocessing import Preprocessor, CICIDS_LABEL_MAP, CLASS_ORDER
        assert Preprocessor is not None
        assert len(CLASS_ORDER) == 6
        assert len(CICIDS_LABEL_MAP) > 0

    def test_import_cnn_lstm(self):
        from ml.models.cnn_lstm import build_cnn_lstm_model
        assert build_cnn_lstm_model is not None

    def test_import_trainer(self):
        from ml.models.trainer import Trainer
        assert Trainer is not None

    def test_import_shap_explainer(self):
        from ml.explainability.shap_explainer import SHAPExplainer
        assert SHAPExplainer is not None

    def test_import_packet_sniffer(self):
        from ml.ingestion.packet_sniffer import PacketSniffer
        assert PacketSniffer is not None

    def test_import_flow_aggregator(self):
        from ml.ingestion.flow_aggregator import FlowAggregator
        assert FlowAggregator is not None

    def test_import_inference_pipeline(self):
        from ml.inference.pipeline import InferencePipeline
        assert InferencePipeline is not None


class TestConfigurations:
    """Verify configuration files are valid."""

    def test_feature_schema_exists(self):
        import json
        schema_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'feature_schema.json')
        assert os.path.exists(schema_path), f"Feature schema not found at {schema_path}"
        with open(schema_path) as f:
            schema = json.load(f)
        assert "features" in schema
        assert "label_schema" in schema
        assert schema["label_schema"]["num_classes"] == 6

    def test_model_config_exists(self):
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'model_config.yaml')
        assert os.path.exists(config_path), f"Model config not found at {config_path}"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert "model" in config
        assert "training" in config
        assert config["model"]["architecture"]["num_classes"] == 6

    def test_alert_schema_exists(self):
        import json
        schema_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'alert_schema.json')
        assert os.path.exists(schema_path), f"Alert schema not found at {schema_path}"
        with open(schema_path) as f:
            schema = json.load(f)
        assert "properties" in schema
        assert "label" in schema["properties"]


class TestDataLoader:
    """Test DataLoader instantiation and validation."""

    def test_create_data_loader(self):
        from ml.data.data_loader import DataLoader
        loader = DataLoader(data_dir="data/raw")
        assert loader.data_dir.name == "raw"

    def test_validate_dataset(self):
        import pandas as pd
        import numpy as np
        from ml.data.data_loader import DataLoader

        loader = DataLoader()
        # Create a small test DataFrame
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, np.inf, 4.0],
            "feature2": [10.0, np.nan, 30.0, 40.0],
            "Label": ["BENIGN", "DDoS", "BENIGN", "DoS Hulk"],
        })
        report = loader.validate_dataset(df, "test_dataset")
        assert report["shape"] == (4, 3)
        assert report["missing_values"] == 1
        assert report["inf_values"] == 1
        assert report["label_column"] == "Label"


class TestPreprocessor:
    """Test Preprocessor label mapping."""

    def test_label_mapping_keys(self):
        from ml.data.preprocessing import CICIDS_LABEL_MAP, UNSW_LABEL_MAP, CLASS_ORDER
        # All mapped values should be in CLASS_ORDER
        for label in CICIDS_LABEL_MAP.values():
            assert label in CLASS_ORDER, f"CIC-IDS label '{label}' not in CLASS_ORDER"
        for label in UNSW_LABEL_MAP.values():
            assert label in CLASS_ORDER, f"UNSW label '{label}' not in CLASS_ORDER"


class TestFlowAggregator:
    """Test FlowAggregator basic functionality."""

    def test_create_aggregator(self):
        from ml.ingestion.flow_aggregator import FlowAggregator
        agg = FlowAggregator(timeout=60)
        assert agg.timeout == 60.0
        assert len(agg.flow_table) == 0

    def test_add_packet(self):
        from ml.ingestion.flow_aggregator import FlowAggregator
        
        results = []
        agg = FlowAggregator(timeout=60, callback=lambda f: results.append(f))

        # Add a packet
        packet = {
            "timestamp": 1000.0,
            "src_ip": "192.168.1.1",
            "dst_ip": "192.168.1.2",
            "src_port": 12345,
            "dst_port": 80,
            "protocol": 6,
            "protocol_name": "TCP",
            "ip_len": 1500,
            "header_len": 20,
            "tcp_flags": 0x02,  # SYN
        }
        agg.add_packet(packet)
        assert len(agg.flow_table) == 1

        # Stop to emit remaining flows
        agg.stop()
        assert len(results) == 1
        assert results[0]["syn_flag_count"] == 1
        assert results[0]["src_ip"] == "192.168.1.1"
