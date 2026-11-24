# Alert contract

Machine-readable schema: [ml/config/alert_schema.json](../../ml/config/alert_schema.json).

This is the object Phase 5, the database, and the report generator all consume. Field names and enums stay as they are in that file.

## Events

| Event | Direction | Payload |
| --- | --- | --- |
| `new_alert` | Python inference service to the Node process | One alert object |
| `alert` | Node rebroadcast to browser clients | The same object |

The Node listener is `socket.on("new_alert")` in [backend/src/server.js](../../backend/src/server.js). The Python client connects only when the service is started with `--emit`.

## Actions

| Action | When | Firewall |
| --- | --- | --- |
| `ALLOW` | Predicted label is Benign | None. `mitigation_status` is `n/a`. `explanation` is null. |
| `ALERT` | Any attack class with confidence below 0.98 | None. `mitigation_status` is `n/a`. |
| `MITIGATE` | Any attack class with confidence at or above 0.98 | None in Phase 4. `MITIGATE` means the flow would be blocked. `mitigation_status` is `pending` until Phase 6 inserts a rule. |

The threshold is `inference.confidence_threshold` in `ml/config/model_config.yaml`, overridable with `CONFIDENCE_THRESHOLD`.

## Explanation

`explanation` is either null or this object:

```json
{
  "top_features": [
    {
      "feature": "SYN Flag Count",
      "shap_value": 0.2,
      "direction": "positive",
      "pct_vs_baseline": "+400%"
    }
  ],
  "human_readable": "SYN Flag Count: +400% vs baseline"
}
```

`ALLOW` always sends null. `ALERT` and `MITIGATE` include that object when SHAP finished in time. If the explain queue (length 2) is full, the alert is still emitted and `explanation` is null. A missing explanation is null, not an empty string and not a placeholder object.

## Labels

Class names come from `label_encoder.classes_` inside `ml/models/artifacts/preprocessor.pkl`:

`Benign`, `BruteForce`, `DoS/DDoS`, `Infiltration`, `PortScan`, `WebAttack`

That order is alphabetical because sklearn's `LabelEncoder` sorted the training labels. It is not the order previously hard-coded in `InferencePipeline`. Scoring code indexes `model.predict` with this encoder list.

## Features the model sees

Live flows are the 78 CIC-IDS column names in `data/processed/feature_names.json`, scaled with the saved `StandardScaler` at classify time. `X_test.npy` is already scaled and is not passed through that scaler. The 42 snake_case aliases from `feature_schema.json` are display names. They are not a second model input.
