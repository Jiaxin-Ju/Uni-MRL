"""Canonical fine-tuning entry point for Uni-MRL and its ablations."""

import argparse
import copy
import inspect
import os
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset_test import MolTestDatasetWrapper
from utils.nt_xent import NTXentLoss
from utils.tasks import dataset_task, resolve_targets, rule_subtask, split_paths


PROJECT_ROOT = Path(__file__).resolve().parent
CONTRASTIVE_MODES = {"unimrl", "concat_contrastive"}
SUMMARY_GROUP_COLUMNS = (
    "dataset",
    "target",
    "model_type",
    "llm_model",
    "feature_mode",
    "knowledge_type",
    "num_samples",
    "drop_ratio",
    "metric",
)
RESULT_COLUMNS = (
    "row_type",
    "run",
    "seed",
    *SUMMARY_GROUP_COLUMNS,
    "value",
    "best_epoch",
    "graph_weight",
    "llm_weight",
    "runs",
    "avg",
    "std",
)
MODE_ALIASES = {
    "molclr": "graph_only",
    "plus": "additive",
    "dir_concat": "concat",
    "concat": "concat",
    "wei_sum": "weighted",
    "ws_contrastive": "unimrl",
    "cc_contrastive": "concat_contrastive",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def constructor_kwargs(constructor, values):
    parameters = inspect.signature(constructor.__init__).parameters
    return {key: value for key, value in values.items() if key in parameters}


def build_model(feature_mode, model_type, task, model_config):
    """Build a paper component or the complete Uni-MRL model."""
    feature_mode = MODE_ALIASES.get(feature_mode, feature_mode)
    if feature_mode == "graph_only":
        if model_type == "gin":
            from models.ginet_finetune import GINet as Model
        else:
            from models.gcn_finetune import GCN as Model
    elif feature_mode == "llm_only":
        from models.llm_finetune import LLMOnly as Model
    elif feature_mode == "additive":
        if model_type != "gin":
            raise ValueError("additive fusion is only implemented for GIN")
        from models.ginet_finetune_plus import GINet as Model
    elif feature_mode == "concat":
        if model_type != "gin":
            raise ValueError("direct concatenation is only implemented for GIN")
        from models.ginet_finetune_concat import GINet as Model
    elif feature_mode == "weighted":
        if model_type != "gin":
            raise ValueError("weighted fusion without contrastive loss is only implemented for GIN")
        from models.ginet_finetune_wsum import GINet as Model
    elif feature_mode == "unimrl":
        if model_type == "gin":
            from models.ginet_finetune_ws_contrastive import GINet as Model
        else:
            from models.gcn_finetune_ws_contrastive import GCN as Model
    elif feature_mode == "concat_contrastive":
        if model_type != "gin":
            raise ValueError("concat + contrastive is only implemented for GIN")
        from models.ginet_finetune_cc_contrastive import GINet as Model
    else:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")

    values = dict(model_config)
    values["task"] = task
    return Model(**constructor_kwargs(Model, values))


def load_pretrained_encoder(model, checkpoint_path, device):
    """Load shape-compatible MolCLR parameters and leave new heads untouched."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    current = model.state_dict()
    compatible = {
        name: value
        for name, value in state.items()
        if name in current and current[name].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)}/{len(state)} compatible tensors from {checkpoint_path}")


def fusion_weights(model):
    if not hasattr(model, "w1") or not hasattr(model, "w2"):
        return None, None
    logits = torch.stack((model.w1, model.w2))
    weights = torch.softmax(logits, dim=0).detach().cpu().numpy()
    return float(weights[0]), float(weights[1])


def summarize_results(frame):
    """Summarize repeated runs with mean and sample standard deviation."""
    missing = {"value", *SUMMARY_GROUP_COLUMNS}.difference(frame.columns)
    if missing:
        raise KeyError(f"Cannot summarize results; missing columns: {sorted(missing)}")

    rows = []
    for keys, group in frame.groupby(list(SUMMARY_GROUP_COLUMNS), dropna=False):
        values = pd.to_numeric(group["value"], errors="raise")
        row = dict(zip(SUMMARY_GROUP_COLUMNS, keys))
        row.update(
            {
                "runs": int(values.count()),
                "avg": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


class FineTuner:
    def __init__(self, config, loaders, run_dir):
        self.config = config
        self.train_loader, self.valid_loader, self.test_loader = loaders
        self.device = self._get_device(config["gpu"])
        self.feature_mode = MODE_ALIASES.get(config["feature_mode"], config["feature_mode"])
        self.model = build_model(
            self.feature_mode,
            config["model_type"],
            config["dataset"]["task"],
            config["model"],
        ).to(self.device)
        if config["pretrained"] and self.feature_mode != "llm_only":
            load_pretrained_encoder(self.model, config["pretrained_checkpoint"], self.device)

        self.criterion = (
            nn.CrossEntropyLoss()
            if config["dataset"]["task"] == "classification"
            else nn.MSELoss()
        )
        self.contrastive_loss = NTXentLoss(
            device=self.device,
            temperature=float(config["contrastive_temperature"]),
            use_cosine_similarity=True,
        )
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.run_dir))
        with (self.checkpoint_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    @staticmethod
    def _get_device(requested):
        if requested != "cpu" and torch.cuda.is_available():
            device = torch.device(requested)
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        print(f"Running on: {device}")
        return device

    def _forward(self, data):
        outputs = self.model(data)
        if not isinstance(outputs, tuple):
            raise TypeError("Model forward must return a tuple")
        if len(outputs) == 3:
            graph_h, rule_h, prediction = outputs
        elif len(outputs) == 2:
            graph_h, prediction = outputs
            rule_h = None
        else:
            raise ValueError(f"Unexpected number of model outputs: {len(outputs)}")
        return graph_h, rule_h, prediction

    def _loss(self, data, graph_h, rule_h, prediction):
        if self.config["dataset"]["task"] == "classification":
            task_loss = self.criterion(prediction, data.y.reshape(-1).long())
        else:
            task_loss = self.criterion(prediction.reshape(-1), data.y.reshape(-1).float())
        if self.feature_mode in CONTRASTIVE_MODES:
            if rule_h is None:
                raise RuntimeError("Contrastive mode did not return both modality embeddings")
            alignment = self.contrastive_loss(
                F.normalize(graph_h, dim=1),
                F.normalize(rule_h, dim=1),
            )
            return task_loss + float(self.config["contrastive_weight"]) * alignment
        return task_loss

    def _evaluate(self, loader, split):
        predictions = []
        labels = []
        total_loss = 0.0
        total_examples = 0
        self.model.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                graph_h, rule_h, prediction = self._forward(data)
                loss = self._loss(data, graph_h, rule_h, prediction)
                batch_size = data.y.numel()
                total_loss += loss.item() * batch_size
                total_examples += batch_size
                labels.extend(data.y.detach().cpu().reshape(-1).numpy())
                if self.config["dataset"]["task"] == "classification":
                    scores = F.softmax(prediction, dim=-1)[:, 1]
                    predictions.extend(scores.detach().cpu().numpy())
                else:
                    predictions.extend(prediction.detach().cpu().reshape(-1).numpy())

        if not total_examples:
            raise RuntimeError(f"The {split} loader is empty")
        labels = np.asarray(labels)
        predictions = np.asarray(predictions)
        if self.config["dataset"]["task"] == "classification":
            if np.unique(labels).size < 2:
                raise ValueError(f"{split} split contains only one class; ROC-AUC is undefined")
            metric_name = "roc_auc"
            metric = roc_auc_score(labels, predictions)
        elif self.config["dataset_name"] == "qm9":
            metric_name = "mae"
            metric = mean_absolute_error(labels, predictions)
        else:
            metric_name = "rmse"
            metric = mean_squared_error(labels, predictions, squared=False)
        average_loss = total_loss / total_examples
        print(f"{split}: loss={average_loss:.6f}, {metric_name}={metric:.6f}")
        return average_loss, metric_name, float(metric)

    def train(self):
        prediction_parameters = []
        base_parameters = []
        for name, parameter in self.model.named_parameters():
            if name.startswith("pred_head"):
                prediction_parameters.append(parameter)
            else:
                base_parameters.append(parameter)
        optimizer = torch.optim.Adam(
            [
                {"params": base_parameters, "lr": float(self.config["init_base_lr"])},
                {"params": prediction_parameters, "lr": float(self.config["init_lr"])},
            ],
            weight_decay=float(self.config["weight_decay"]),
        )

        best_metric = -np.inf if self.config["dataset"]["task"] == "classification" else np.inf
        best_epoch = None
        global_step = 0
        checkpoint_path = self.checkpoint_dir / "model.pth"
        for epoch in range(int(self.config["epochs"])):
            self.model.train()
            for batch_index, data in enumerate(self.train_loader):
                data = data.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                graph_h, rule_h, prediction = self._forward(data)
                loss = self._loss(data, graph_h, rule_h, prediction)
                loss.backward()
                optimizer.step()
                if global_step % int(self.config["log_every_n_steps"]) == 0:
                    print(f"epoch={epoch} batch={batch_index} train_loss={loss.item():.6f}")
                    self.writer.add_scalar("loss/train", loss.item(), global_step)
                global_step += 1

            if epoch % int(self.config["eval_every_n_epochs"]) == 0:
                valid_loss, metric_name, metric = self._evaluate(self.valid_loader, "validation")
                self.writer.add_scalar("loss/validation", valid_loss, epoch)
                self.writer.add_scalar(f"metric/validation_{metric_name}", metric, epoch)
                improved = metric > best_metric if metric_name == "roc_auc" else metric < best_metric
                if improved:
                    best_metric = metric
                    best_epoch = epoch
                    torch.save(
                        {"model_state_dict": self.model.state_dict(), "epoch": epoch},
                        checkpoint_path,
                    )

        if not checkpoint_path.is_file():
            torch.save(
                {"model_state_dict": self.model.state_dict(), "epoch": int(self.config["epochs"]) - 1},
                checkpoint_path,
            )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        _, metric_name, test_metric = self._evaluate(self.test_loader, "test")
        graph_weight, llm_weight = fusion_weights(self.model)
        self.writer.close()
        return {
            "metric": metric_name,
            "value": test_metric,
            "best_epoch": best_epoch,
            "graph_weight": graph_weight,
            "llm_weight": llm_weight,
        }


def build_parser():
    parser = argparse.ArgumentParser(description="Train Uni-MRL or a paper ablation")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config_unimrl.yaml"))
    parser.add_argument("--dataset-folder", "--dataset_folder", default=str(PROJECT_ROOT / "dataset" / "scaffold_datasets"))
    parser.add_argument("--dataset", default="esol")
    parser.add_argument("--subtask", default=None)
    parser.add_argument("--all-subtasks", action="store_true")
    parser.add_argument(
        "--llm-model",
        "--model",
        dest="llm_model",
        default=None,
        help="rule generator (default: galactica-6.7b, or galactica-30b for QM9)",
    )
    parser.add_argument(
        "--knowledge-type",
        "--knowledge_type",
        choices=("synthesize", "inference", "all"),
        default="all",
    )
    parser.add_argument(
        "--num-samples",
        "--num_samples",
        type=int,
        choices=(30, 50),
        default=None,
        help="inference-rule sample count (default: 30, or 50 for QM9)",
    )
    parser.add_argument(
        "--feature-mode",
        "--feat_type",
        default="unimrl",
        choices=(
            "graph_only",
            "llm_only",
            "additive",
            "concat",
            "weighted",
            "unimrl",
            "concat_contrastive",
            *MODE_ALIASES.keys(),
        ),
    )
    parser.add_argument("--model-type", "--model_type", choices=("gin", "gcn"), default="gin")
    parser.add_argument(
        "--drop-ratio",
        "--drop_ratio",
        type=float,
        default=None,
        help="GNN dropout",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--contrastive-temperature", type=float, default=None)
    parser.add_argument("--contrastive-weight", type=float, default=None)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    return parser


def load_config(args):
    config_path = Path(args.config).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["gpu"] = args.device
    config["model_type"] = args.model_type
    config["feature_mode"] = MODE_ALIASES.get(args.feature_mode, args.feature_mode)
    config["pretrained"] = args.pretrained
    if args.drop_ratio is not None:
        config["model"]["drop_ratio"] = args.drop_ratio
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["dataset"]["num_workers"] = args.num_workers
    if args.contrastive_temperature is not None:
        config["contrastive_temperature"] = args.contrastive_temperature
    if args.contrastive_weight is not None:
        config["contrastive_weight"] = args.contrastive_weight
    checkpoint_name = "pretrained_gin" if args.model_type == "gin" else "pretrained_gcn"
    config["fine_tune_from"] = checkpoint_name
    config["pretrained_checkpoint"] = str(
        PROJECT_ROOT / "ckpt" / checkpoint_name / "checkpoints" / "model.pth"
    )
    return config


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    dataset_name = args.dataset.lower()
    if args.llm_model is None:
        args.llm_model = "galactica-30b" if dataset_name == "qm9" else "galactica-6.7b"
    if args.num_samples is None:
        args.num_samples = 50 if dataset_name == "qm9" else 30
    targets = resolve_targets(dataset_name, args.subtask, args.all_subtasks)
    base_config = load_config(args)
    if args.drop_ratio is None:
        args.drop_ratio = float(base_config["model"]["drop_ratio"])
    records = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_name = (
        f"{safe_name(dataset_name)}_{safe_name(args.subtask or 'all')}_"
        f"{safe_name(args.model_type)}_{safe_name(args.llm_model)}_"
        f"{safe_name(base_config['feature_mode'])}.csv"
    )
    result_path = output_dir / result_name
    first_completed_run = True

    for target in targets:
        paths = split_paths(args.dataset_folder, dataset_name, target)
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Required scaffold split is missing: {path}")

        for run_index in range(1, args.runs + 1):
            seed = args.seed * run_index
            set_seed(seed)
            config = copy.deepcopy(base_config)
            config["dataset_name"] = dataset_name
            config["task_name"] = dataset_name
            config["dataset"].update(
                {
                    "dataset_name": dataset_name,
                    "subtask": rule_subtask(dataset_name, target),
                    "model": args.llm_model.lower(),
                    "knowledge_type": args.knowledge_type,
                    "num_samples": args.num_samples,
                    "train_data_path": str(paths["train"]),
                    "valid_data_path": str(paths["valid"]),
                    "test_data_path": str(paths["test"]),
                    "task": dataset_task(dataset_name),
                    "target": target,
                    "rule_root": str(PROJECT_ROOT / "dataset" / "eval_code_generation_repo"),
                    "use_llm_features": config["feature_mode"] != "graph_only",
                }
            )

            dataset = MolTestDatasetWrapper(config["batch_size"], **config["dataset"])
            loaders = dataset.get_data_loaders()
            config["model"]["llm4sd_x_dim"] = dataset.feature_dim
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir = (
                PROJECT_ROOT
                / "finetune"
                / f"{timestamp}_{safe_name(dataset_name)}_{safe_name(target)}_{config['feature_mode']}_run{run_index}"
            )
            result = FineTuner(config, loaders, run_dir).train()
            record = {
                "row_type": "run",
                "run": run_index,
                "seed": seed,
                "dataset": dataset_name,
                "target": target,
                "model_type": args.model_type,
                "llm_model": args.llm_model.lower(),
                "feature_mode": config["feature_mode"],
                "knowledge_type": args.knowledge_type,
                "num_samples": args.num_samples,
                "drop_ratio": args.drop_ratio,
                **result,
                "runs": "",
                "avg": "",
                "std": "",
            }
            records.append(record)
            pd.DataFrame([record], columns=RESULT_COLUMNS).to_csv(
                result_path,
                mode="w" if first_completed_run else "a",
                header=first_completed_run,
                index=False,
            )
            first_completed_run = False
            print(f"Saved run {run_index} ({target}) to {result_path}")

    frame = pd.DataFrame(records)
    summary = summarize_results(frame)
    for summary_record in summary.to_dict(orient="records"):
        output_record = {column: "" for column in RESULT_COLUMNS}
        output_record.update(summary_record)
        output_record["row_type"] = "summary"
        pd.DataFrame([output_record], columns=RESULT_COLUMNS).to_csv(
            result_path,
            mode="a",
            header=False,
            index=False,
        )
    print(f"Completed all runs. Final results: {result_path}")
    for row in summary.itertuples(index=False):
        print(
            f"{row.dataset}/{row.target}: {row.avg:.6f} +/- {row.std:.6f} "
            f"({row.metric}, n={row.runs})"
        )
    return frame


if __name__ == "__main__":
    main()
