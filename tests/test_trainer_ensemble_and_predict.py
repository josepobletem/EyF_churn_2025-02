# tests/test_trainer_ensemble_and_predict.py
import os
import io
import yaml
import pickle
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

# Import del módulo bajo prueba
# Asegúrate de que tests/ esté fuera de src/ o añade el path si es necesario.
import src.trainer_ensemble_and_predict as mod


# ===== Dummy model picklable =====
class DummyModel:
    """Modelo LightGBM simulado, picklable y con método predict."""
    def __init__(self, constant=0.123):
        self.constant = constant

    def predict(self, X):
        # Devuelve una probabilidad constante del tamaño de X
        n = len(X)
        return np.full(n, self.constant, dtype=float)


class TestConfigAndHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmpdir = self.tmp.name

        # Config mínimo funcional
        self.cfg_dict = {
            "paths": {
                "raw_dataset": os.path.join(self.tmpdir, "raw.csv"),
                "processed_dataset": os.path.join(self.tmpdir, "proc.csv"),
                "feature_dataset": os.path.join(self.tmpdir, "feat.csv"),
            },
            "columns": {
                "id_column": "numero_de_cliente",
                "period_column": "foto_mes",
                "target_column_full": "clase_ternaria",
                "binary_target_col": "clase_binaria2",
                "peso_col": "clase_peso",
            },
            "train": {
                "models_dir": os.path.join(self.tmpdir, "models"),
                "train_months": [202101, 202102, 202103, 202104],
                "test_month": 202104,
                "drop_cols": ["lag_3_ctrx_quarter"],
                "weight_baja2": 1.00002,
                "weight_baja1": 1.00001,
                "weight_continua": 1.0,
                "seed": 12345,
                "n_models": 3,
                "decision_threshold": 0.025,
                "ganancia_acierto": 780000.0,
                "costo_estimulo": 20000.0,
            }
        }

        # Escribir config.yaml en disco
        self.cfg_path = os.path.join(self.tmpdir, "config.yaml")
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg_dict, f, sort_keys=False, allow_unicode=True)

        # DF pequeño de features
        self.df_feat = pd.DataFrame({
            "numero_de_cliente": [1, 2, 3, 4, 5, 6],
            "foto_mes": [202101, 202101, 202102, 202103, 202104, 202104],
            "clase_ternaria": ["BAJA+1", "BAJA+2", "CONTINUA", "BAJA+1", "CONTINUA", "BAJA+2"],
            "lag_3_ctrx_quarter": [1, 2, 3, 4, 5, 6],   # col a dropear
            "x1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "x2": [1, 0, 1, 0, 1, 0],
        })
        self.df_feat_path = self.cfg_dict["paths"]["feature_dataset"]
        self.df_feat.to_csv(self.df_feat_path, index=False)

        # Carpeta models
        os.makedirs(self.cfg_dict["train"]["models_dir"], exist_ok=True)

        # best_params_v2.yaml
        self.best_params = {"best_params": {"objective": "binary", "learning_rate": 0.1}, "best_iteration": 50}
        self.best_params_path = os.path.join(self.cfg_dict["train"]["models_dir"], "best_params_v2.yaml")
        with open(self.best_params_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.best_params, f, sort_keys=False, allow_unicode=True)

    def test_load_config_ok(self):
        cfg = mod.load_config(self.cfg_path)
        self.assertEqual(cfg.columns.id_column, "numero_de_cliente")
        self.assertEqual(cfg.paths.feature_dataset, self.df_feat_path)

    # tests/test_trainer_ensemble_and_predict.py  (dentro de TestConfigAndHelpers)
    def test_ensure_binarias_y_peso(self):
        cfg = mod.load_config(self.cfg_path)
        tr = cfg.train or mod.TrainConfig()
        df2 = mod.ensure_binarias_y_peso(self.df_feat, cfg.columns, tr)

        self.assertIn("clase_binaria1", df2.columns)
        self.assertIn("clase_binaria2", df2.columns)
        self.assertIn("clase_peso", df2.columns)

        # BAJA+2 -> binaria1 = 1 (índice 1 en el DF de la fixture)
        self.assertEqual(int(df2.loc[1, "clase_binaria1"]), 1)

        # BAJA+1 -> binaria2 = 1 (índice 0)
        self.assertEqual(int(df2.loc[0, "clase_binaria2"]), 1)

        # CONTINUA -> binaria2 = 0 (índice 2)
        self.assertEqual(int(df2.loc[2, "clase_binaria2"]), 0)


    def test_build_train_matrix(self):
        cfg = mod.load_config(self.cfg_path)
        tr = cfg.train or mod.TrainConfig()
        df2 = mod.ensure_binarias_y_peso(self.df_feat, cfg.columns, tr)
        X, y, w, feats = mod.build_train_matrix(df2, cfg, tr)
        # drop col fue aplicada
        self.assertNotIn("lag_3_ctrx_quarter", X.columns)
        # solo tipos numéricos/bool
        self.assertTrue(set(X.dtypes.unique()).issubset(
            {np.dtype(np.int8), np.dtype(np.int16), np.dtype(np.int32), np.dtype(np.int64),
             np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.uint32), np.dtype(np.uint64),
             np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64), np.dtype(bool)}
        ))
        self.assertEqual(len(y), X.shape[0])
        self.assertEqual(len(w), X.shape[0])

    def test_generate_seeds(self):
        cfg = mod.load_config(self.cfg_path)
        tr = cfg.train
        tr.n_models = 4
        tr.seed = 42
        seeds = mod._generate_seeds(tr)
        self.assertEqual(seeds, [42, 43, 44, 45])

        tr.seeds = [7, 8, 9, 10, 11]
        tr.n_models = 3
        seeds2 = mod._generate_seeds(tr)
        self.assertEqual(seeds2, [7, 8, 9])


class TestTrainingAndScoring(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmpdir = self.tmp.name

        self.cfg_dict = {
            "paths": {
                "raw_dataset": os.path.join(self.tmpdir, "raw.csv"),
                "processed_dataset": os.path.join(self.tmpdir, "proc.csv"),
                "feature_dataset": os.path.join(self.tmpdir, "feat.csv"),
            },
            "columns": {
                "id_column": "numero_de_cliente",
                "period_column": "foto_mes",
                "target_column_full": "clase_ternaria",
                "binary_target_col": "clase_binaria2",
                "peso_col": "clase_peso",
            },
            "train": {
                "models_dir": os.path.join(self.tmpdir, "models"),
                "train_months": [202101, 202102, 202103],
                "test_month": 202104,
                "drop_cols": [],
                "seed": 101,
                "n_models": 2,
                "decision_threshold": 0.3,
                "ganancia_acierto": 100.0,
                "costo_estimulo": 10.0,
            }
        }
        self.cfg_path = os.path.join(self.tmpdir, "config.yaml")
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg_dict, f, sort_keys=False, allow_unicode=True)

        self.df_feat = pd.DataFrame({
            "numero_de_cliente": [10, 11, 12, 13],
            "foto_mes": [202101, 202101, 202104, 202104],
            "clase_ternaria": ["BAJA+1", "CONTINUA", "BAJA+2", "CONTINUA"],
            "x1": [0.0, 0.2, 0.8, 0.6],
            "x2": [1, 0, 1, 0],
        })
        self.df_feat_path = self.cfg_dict["paths"]["feature_dataset"]
        self.df_feat.to_csv(self.df_feat_path, index=False)

        os.makedirs(self.cfg_dict["train"]["models_dir"], exist_ok=True)

        self.best_params_info = {"best_params": {"objective": "binary", "learning_rate": 0.05}, "best_iteration": 20}
        self.best_params_path = os.path.join(self.cfg_dict["train"]["models_dir"], "best_params_v2.yaml")
        with open(self.best_params_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.best_params_info, f, sort_keys=False, allow_unicode=True)

    @patch("lightgbm.train")
    def test_train_single_model(self, mock_lgb_train):
        # lgb.train devolverá un DummyModel
        mock_lgb_train.return_value = DummyModel(constant=0.7)

        cfg = mod.load_config(self.cfg_path)
        tr = cfg.train or mod.TrainConfig()
        df2 = mod.ensure_binarias_y_peso(self.df_feat, cfg.columns, tr)
        X, y, w, feats = mod.build_train_matrix(df2, cfg, tr)

        model, y_hat, params = mod._train_single_model(
            X, y, w,
            best_params={"objective": "binary"},
            best_iteration=15,
            seed_value=999
        )
        self.assertIsInstance(model, DummyModel)
        self.assertEqual(len(y_hat), X.shape[0])
        self.assertIn("seed", params)
        self.assertEqual(params["seed"], 999)

    @patch("lightgbm.train")
    def test_train_final_ensemble_writes_metadata_and_models(self, mock_lgb_train):
        # siempre devuelve un DummyModel, picklable
        mock_lgb_train.return_value = DummyModel(constant=0.6)

        res = mod.train_final_ensemble(config_path=self.cfg_path)
        self.assertIn("ensemble_metadata_path", res)
        self.assertTrue(os.path.exists(res["ensemble_metadata_path"]))

        # metadata contiene paths de modelos
        with open(res["ensemble_metadata_path"], "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)
        model_paths = meta.get("model_paths", [])
        self.assertEqual(len(model_paths), self.cfg_dict["train"]["n_models"])

        # Los modelos fueron pickeados
        for p in model_paths:
            self.assertTrue(os.path.exists(p))

    def _write_ensemble_meta_and_models(self, feature_names=None):
        """Crea final_ensemble_metadata.yaml + pickles de modelos para tests de scoring."""
        models_dir = self.cfg_dict["train"]["models_dir"]
        ens_dir = os.path.join(models_dir, "ensamble-1")
        os.makedirs(ens_dir, exist_ok=True)

        # Creamos 2 modelos picklables en disco
        model_paths = []
        for seed in [101, 203]:
            path = os.path.join(models_dir, "ensamble-1", f"final_model_seed{seed}.pkl")
            with open(path, "wb") as f:
                pickle.dump(DummyModel(constant=0.4 if seed == 101 else 0.8), f)
            model_paths.append(path)

        if feature_names is None:
            feature_names = ["x1", "x2"]

        meta = {
            "train_months": self.cfg_dict["train"]["train_months"],
            "test_month": self.cfg_dict["train"]["test_month"],
            "binary_target_col": self.cfg_dict["columns"]["binary_target_col"],
            "period_column": self.cfg_dict["columns"]["period_column"],
            "id_column": self.cfg_dict["columns"]["id_column"],
            "best_iteration": 20,
            "seeds_used": [101, 203],
            "model_paths": model_paths,
            "decision_threshold": self.cfg_dict["train"]["decision_threshold"],
            "logloss_in_sample_ensemble": 0.123456,
            "confusion_matrix_in_sample_ensemble": [[1, 2], [3, 4]],
            "feature_names": feature_names,
        }
        meta_path = os.path.join(ens_dir, "final_ensemble_metadata.yaml")
        with open(meta_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)
        return meta_path, model_paths

    # tests/test_trainer_ensemble_and_predict.py  (dentro de TestTrainingAndScoring)
    def test_load_ensemble_models_and_meta(self):
        meta_path, model_paths = self._write_ensemble_meta_and_models()
        cfg = mod.load_config(self.cfg_path)
        models, meta, mp = mod.load_ensemble_models_and_meta(cfg)

        self.assertEqual(os.path.normpath(mp), os.path.normpath(meta_path))
        self.assertEqual(len(models), 2)
        self.assertIn("model_paths", meta)
        self.assertEqual(
            [os.path.normpath(p) for p in meta["model_paths"]],
            [os.path.normpath(p) for p in model_paths]
        )

    def test_build_scoring_matrix(self):
        cfg = mod.load_config(self.cfg_path)
        tr = cfg.train or mod.TrainConfig()
        df2 = mod.ensure_binarias_y_peso(self.df_feat, cfg.columns, tr)
        # usar mes 202104 que está en df_feat
        Xs, info_df, feats = mod.build_scoring_matrix(df2, cfg, 202104)
        self.assertGreaterEqual(Xs.shape[0], 1)
        self.assertIn("numero_de_cliente", info_df.columns)

    def test_score_month_ensemble(self):
        # Prepara metadata + modelos
        self._write_ensemble_meta_and_models(feature_names=["x1", "x2"])

        out = mod.score_month_ensemble(
            month_to_score=202104,
            threshold=0.5,
            config_path=self.cfg_path
        )
        self.assertIn("df_pred", out)
        self.assertIn("df_simple", out)
        self.assertIn("ganancia_total", out)
        # Predicted debe existir y ser int 0/1
        self.assertIn("Predicted", out["df_simple"].columns)
        self.assertTrue(out["df_simple"]["Predicted"].isin([0, 1]).all())

    def test_parse_args(self):
        args = mod._parse_args(["--train", "--mes", "202104", "--threshold", "0.03", "--config", "X.yaml"])
        self.assertTrue(args.train)
        self.assertEqual(args.mes, 202104)
        self.assertAlmostEqual(args.threshold, 0.03, places=6)
        self.assertEqual(args.config, "X.yaml")


class TestMainSmoke(unittest.TestCase):
    """Smoke test de main(): ejercita flujo de scoring con mocks para no entrenar."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmpdir = self.tmp.name

        self.cfg_dict = {
            "paths": {
                "raw_dataset": os.path.join(self.tmpdir, "raw.csv"),
                "processed_dataset": os.path.join(self.tmpdir, "proc.csv"),
                "feature_dataset": os.path.join(self.tmpdir, "feat.csv"),
            },
            "columns": {
                "id_column": "numero_de_cliente",
                "period_column": "foto_mes",
                "target_column_full": "clase_ternaria",
                "binary_target_col": "clase_binaria2",
                "peso_col": "clase_peso",
            },
            "train": {
                "models_dir": os.path.join(self.tmpdir, "models"),
                "train_months": [202101, 202102],
                "test_month": 202104,
                "drop_cols": [],
                "seed": 10,
                "n_models": 1,
                "decision_threshold": 0.25,
                "ganancia_acierto": 100.0,
                "costo_estimulo": 10.0,
            }
        }
        self.cfg_path = os.path.join(self.tmpdir, "config.yaml")
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg_dict, f, sort_keys=False, allow_unicode=True)

        df_feat = pd.DataFrame({
            "numero_de_cliente": [100, 101],
            "foto_mes": [202104, 202104],
            "clase_ternaria": ["BAJA+1", "CONTINUA"],
            "x1": [0.9, 0.1],
            "x2": [1, 0],
        })
        df_feat.to_csv(self.cfg_dict["paths"]["feature_dataset"], index=False)

        os.makedirs(self.cfg_dict["train"]["models_dir"], exist_ok=True)

        # best_params_v2.yaml
        best = {"best_params": {"objective": "binary"}, "best_iteration": 5}
        with open(os.path.join(self.cfg_dict["train"]["models_dir"], "best_params_v2.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(best, f, sort_keys=False, allow_unicode=True)

        # metadata + modelo para scoring
        ens_dir = os.path.join(self.cfg_dict["train"]["models_dir"], "ensamble-1")
        os.makedirs(ens_dir, exist_ok=True)
        model_path = os.path.join(ens_dir, "final_model_seed10.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(DummyModel(constant=0.6), f)

        meta = {
            "model_paths": [model_path],
            "feature_names": ["x1", "x2"],
            "test_month": 202104,
            "train_months": [202101, 202102],
        }
        with open(os.path.join(ens_dir, "final_ensemble_metadata.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

    def test_main_scoring_only(self):
        # Ejecuta main con --mes usando el config temporal
        args = ["--mes", "202104", "--threshold", "0.5", f"--config={self.cfg_path}"]
        # Capturamos la salida de logs opcionalmente
        mod.main(args)

        # Verifica que se hayan creado los CSV en predicciones/
        out_dir = os.path.join(os.getcwd(), "predicciones")
        # Como el script usa un path relativo, el test corre en cwd.
        # Para mantener limpio el repo local, borramos si existe.
        if os.path.exists(out_dir):
            # Debe haber al menos 2 archivos
            files = [f for f in os.listdir(out_dir) if f.endswith(".csv")]
            self.assertTrue(any("pred_" in f for f in files))
            self.assertTrue(any("pred_simple_" in f for f in files))
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
