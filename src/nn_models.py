# ------------------------------
# src/nn_models.py
#
# In dieser Python-Datei wird ein einfacher MLP-Regressor in PyTorch
# bereitgestellt, der sich wie ein sklearn-Regressor verhält und in
# sklearn-Pipelines (inkl. Target-Transformation) integriert werden kann.
# ------------------------------

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from typing import Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """
    Einfacher Feedforward-MLP-Regressor auf Basis von PyTorch.

    Der Regressor implementiert die sklearn-Schnittstelle mit
    ``fit(X, y)`` und ``predict(X)`` und kann daher direkt in
    sklearn-Pipelines verwendet werden.
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (128, 64),
        lr: float = 1e-2,
        batch_size: int = 128,
        max_epochs: int = 1000,
        device: str = "cpu",
        verbose: bool = False,
        random_state: int = 42,
        # Early-Stopping-Parameter
        early_stopping: bool = False,
        val_fraction: float = 0.1,
        patience: int = 20,
        min_delta: float = 0.0,
    ) -> None:
        """
        Initialisiert den MLP-Regressor mit Trainings- und Modell-Hyperparametern.

        Parameter
        ----------
        hidden_dims : Tuple[int, ...], optional
            Anzahl Neuronen pro Hidden Layer, z.B. ``(128, 64)``.
        lr : float, optional
            Lernrate für den Adam-Optimizer.
        batch_size : int, optional
            Batchgröße für das Training.
        max_epochs : int, optional
            Maximale Anzahl Trainingsepochen (Obergrenze, vor Early Stopping).
        device : str, optional
            Zielgerät für das Training, z.B. ``"cpu"`` oder ``"cuda"``.
        verbose : bool, optional
            Wenn ``True``, werden in regelmäßigen Abständen Trainingsinformationen
            (Train-/Val-Loss und Lernrate) auf der Konsole ausgegeben.
        random_state : int, optional
            Seed für NumPy- und PyTorch-Randomness, um reproduzierbare
            Ergebnisse zu erhalten.
        early_stopping : bool, optional
            Ob Early Stopping anhand eines Validierungssets verwendet werden soll.
        val_fraction : float, optional
            Anteil der Trainingsdaten, der als Validierungsset abgetrennt wird,
            falls ``early_stopping=True``.
        patience : int, optional
            Anzahl der Epochen ohne Verbesserung des Validierungs-Loss, bevor
            das Training frühzeitig abgebrochen wird.
        min_delta : float, optional
            Mindestverbesserung des Validierungs-Loss, damit eine Epoche
            als „Verbesserung“ zählt.

        Returns
        -------
        None
            Der Konstruktor gibt keinen Wert zurück. Die eigentliche Netzwerk-
            Architektur wird beim ersten Aufruf von :meth:`fit` erzeugt.
        """
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.device = device
        self.verbose = verbose
        self.random_state = random_state

        # Early-Stopping-Parameter
        self.early_stopping = early_stopping
        self.val_fraction = val_fraction
        self.patience = patience
        self.min_delta = min_delta

        # wird erst in fit() gebaut:
        self._model: nn.Module | None = None

    def _build_model(self, input_dim: int) -> nn.Module:
        """
        Baut ein mehrschichtiges Feedforward-Netz mit ReLU-Aktivierungen.

        Es wird ein vollvernetztes MLP mit den in ``hidden_dims`` angegebenen
        Layer-Größen erzeugt. Der Output-Layer hat genau eine Einheit (Regression).

        Parameter
        ----------
        input_dim : int
            Dimension des Eingangsvektors (Anzahl Features nach Preprocessing).

        Returns
        -------
        torch.nn.Module
            PyTorch-Modell, das die definierte MLP-Architektur repräsentiert.
        """
        layers: list[nn.Module] = []
        in_dim = input_dim

        for h in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        # Output: 1 Wert (Regression)
        layers.append(nn.Linear(in_dim, 1))

        return nn.Sequential(*layers)

    def fit(self, X, y) -> "TorchMLPRegressor":
        """
        Trainiert das MLP auf die übergebenen Trainingsdaten.

        Die Daten werden in NumPy-Arrays konvertiert, in Trainings- und ggf.
        Validierungsteil gesplittet und über einen DataLoader in Mini-Batches
        an das Netzwerk verfüttert. Optional wird Early Stopping basierend
        auf dem Validierungs-Loss verwendet.

        Parameter
        ----------
        X : array-ähnlich, shape (n_samples, n_features)
            Designmatrix mit numerischen Features (nach Preprocessing).
        y : array-ähnlich, shape (n_samples,)
            Zielvariable (z.B. SalePrice oder log-transformierter SalePrice).

        Returns
        -------
        TorchMLPRegressor
            Referenz auf das trainierte Objekt (ermöglicht Method Chaining).
        """
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(self.random_state)

        # nach numpy konvertieren
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        n_samples, input_dim = X.shape

        # Train/Val-Split für Early Stopping
        if self.early_stopping and self.val_fraction > 0.0:
            n_val = max(1, int(n_samples * self.val_fraction))
            indices = rng.permutation(n_samples)
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]

            X_train_np = X[train_idx]
            y_train_np = y[train_idx]
            X_val_np = X[val_idx]
            y_val_np = y[val_idx]
        else:
            X_train_np = X
            y_train_np = y
            X_val_np = None
            y_val_np = None

        n_train = X_train_np.shape[0]

        device = torch.device(self.device)
        self._model = self._build_model(input_dim).to(device)

        # Train-Daten als Tensor + DataLoader
        X_train_tensor = torch.from_numpy(X_train_np).to(device)
        y_train_tensor = torch.from_numpy(y_train_np).to(device)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Val-Daten (falls vorhanden)
        if X_val_np is not None:
            X_val_tensor = torch.from_numpy(X_val_np).to(device)
            y_val_tensor = torch.from_numpy(y_val_np).to(device)
        else:
            X_val_tensor = None
            y_val_tensor = None

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.lr,
            weight_decay=1e-3,
        )

        # Learning-Rate-Scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, self.max_epochs // 10),
            gamma=0.8,
        )

        self.train_losses_: list[float] = []
        self.val_losses_: list[float] = []

        best_val_loss = np.inf
        best_state = None
        epochs_no_improve = 0

        for epoch in range(self.max_epochs):
            self._model.train()
            epoch_loss = 0.0

            # Training über alle Batches
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                preds = self._model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)

            epoch_loss /= n_train
            self.train_losses_.append(epoch_loss)

            if X_val_tensor is not None:
                # Validation-Loss berechnen
                self._model.eval()
                with torch.no_grad():
                    val_preds = self._model(X_val_tensor)
                    val_loss = criterion(val_preds, y_val_tensor).item()
                self.val_losses_.append(val_loss)

                # Early-Stopping-Update
                if val_loss + self.min_delta < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.detach().clone()
                        for k, v in self._model.state_dict().items()
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if self.verbose and (epoch % 10 == 0 or epoch == self.max_epochs - 1):
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"[TorchMLP] Epoch {epoch+1}/{self.max_epochs}, "
                        f"train_loss={epoch_loss:.4f}, "
                        f"val_loss={val_loss:.4f}, "
                        f"lr={current_lr:.2e}"
                    )

                if self.early_stopping and epochs_no_improve >= self.patience:
                    if self.verbose:
                        print(
                            f"[TorchMLP] Early stopping at epoch {epoch+1}, "
                            f"best_val_loss={best_val_loss:.4f}"
                        )
                    break

            else:
                # kein Val-Set → nur Train-Loss loggen
                if self.verbose and (epoch % 10 == 0 or epoch == self.max_epochs - 1):
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"[TorchMLP] Epoch {epoch+1}/{self.max_epochs}, "
                        f"train_loss={epoch_loss:.4f}, "
                        f"lr={current_lr:.2e}"
                    )

            # LR-Scheduler am Ende der Epoche updaten
            scheduler.step()

        # bestes Modell laden, falls Early Stopping mit Val-Set aktiv war
        if best_state is not None:
            self._model.load_state_dict(best_state)

        return self

    def predict(self, X):
        """
        Gibt Vorhersagen des trainierten MLP als NumPy-Array zurück.

        Parameter
        ----------
        X : array-ähnlich, shape (n_samples, n_features)
            Designmatrix mit denselben Features/Preprocessing-Schritten
            wie im Training.

        Returns
        -------
        numpy.ndarray
            1D-Array der Länge ``n_samples`` mit den vorhergesagten
            Zielwerten im Originalraum (ohne log-Transformation).
        """
        if self._model is None:
            raise RuntimeError("Modell wurde noch nicht fit() trainiert.")

        self._model.eval()
        X = np.asarray(X, dtype=np.float32)
        device = torch.device(self.device)
        X_tensor = torch.from_numpy(X).to(device)

        with torch.no_grad():
            preds = self._model(X_tensor).cpu().numpy().reshape(-1)

        return preds


def build_torch_mlp_model(
    preprocessor: ColumnTransformer,
    use_log_target: bool = False,
) -> TransformedTargetRegressor:
    """
    Erzeugt eine sklearn-Pipeline auf Basis des TorchMLPRegressor.

    Aufbau:
    - Preprocessing via ``ColumnTransformer`` (z.B. aus ``build_preprocessor``)
    - Standardisierung der Features mit ``StandardScaler``
    - PyTorch-MLP-Regressor (:class:`TorchMLPRegressor`)
    Optional wird das Target über ``log1p``/``expm1`` transformiert, wenn
    ``use_log_target=True`` gesetzt ist.

    Parameter
    ----------
    preprocessor : sklearn.compose.ColumnTransformer
        ColumnTransformer, der Rohdaten in numerische Features überführt.
    use_log_target : bool, optional
        Wenn ``True``, wird das Target intern mit ``np.log1p`` transformiert
        und die Vorhersagen mit ``np.expm1`` zurücktransformiert.

    Returns
    -------
    sklearn.compose.TransformedTargetRegressor
        Wrapper um die MLP-Pipeline, der optional die log-Transformation
        des Targets kapselt und wie ein normaler sklearn-Regressor
        verwendet werden kann.
    """
    base = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("scale", StandardScaler()),
            ("regressor", TorchMLPRegressor()),
        ]
    )

    if use_log_target:
        func = np.log1p
        inverse_func = np.expm1
    else:
        func = None
        inverse_func = None

    return TransformedTargetRegressor(
        regressor=base,
        func=func,
        inverse_func=inverse_func,
    )
